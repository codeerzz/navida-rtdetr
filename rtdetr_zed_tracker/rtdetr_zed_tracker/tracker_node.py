"""ROS 2 node: RT-DETR Detection2DArray -> ByteTrack -> tracked 2D boxes.

Subscribes ~/detections_input (vision_msgs/Detection2DArray), runs the pure-Python
ByteTracker, and republishes confirmed tracks on ~/tracks_2d with a stable display
id (``red_buoy_2``) in Detection2D.id, boxes in ORIGINAL image pixels.

Kalman dt comes from consecutive header stamps only (never a wall/node clock); a
message stamped older than the previous one is dropped with a warning.
"""
from __future__ import annotations

import time

import rclpy
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .bbox_utils import inverse_letterbox
from .byte_tracker import ByteTracker

DEFAULT_LABELS = {
    0: 'buoy', 1: 'east_buoy', 2: 'green_buoy', 3: 'north_buoy',
    4: 'red_buoy', 5: 'south_buoy', 6: 'west_buoy',
}


def _desc(text):
    return ParameterDescriptor(description=text)


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        p = self.declare_parameter
        self.src_w = p('source_image_width', 1920, _desc('Original image width (px)')).value
        self.src_h = p('source_image_height', 1080, _desc('Original image height (px)')).value
        self.net_w = p('network_width', 640, _desc('Network input width (px)')).value
        self.net_h = p('network_height', 640, _desc('Network input height (px)')).value
        self.coord_space = p('detection_coordinate_space', 'source_pixels',
                             _desc("'source_pixels' (Isaac default) or 'network_padded'")).value
        self.padding_mode = p('padding_mode', 'top_left',
                              _desc("Letterbox pad location if network_padded: top_left|center")).value
        high = p('track_high_thresh', 0.5, _desc('High-score detection threshold')).value
        low = p('track_low_thresh', 0.1, _desc('Low-score detection threshold')).value
        newt = p('new_track_thresh', 0.6, _desc('Min score to spawn a new track')).value
        match = p('match_thresh', 0.8, _desc('Max association cost to accept a match')).value
        max_age = p('max_age', 30, _desc('Frames a track survives without a match')).value
        min_hits = p('min_hits', 3, _desc('Hits before a track is confirmed')).value
        gating = p('class_gating', 'soft', _desc('Class gating: strict|soft|none')).value
        penalty = p('class_mismatch_penalty', 0.3, _desc('Soft-gating class penalty')).value
        labels_file = p('class_labels_file', '', _desc('YAML index->name; empty=built-in buoys')).value
        qos_rel = p('input_qos_reliability', 'reliable',
                    _desc('Subscription reliability: reliable|best_effort')).value

        self.labels = self._load_labels(labels_file)
        self._warned_unmapped = set()

        self.tracker = ByteTracker(
            track_high_thresh=high, track_low_thresh=low, new_track_thresh=newt,
            match_thresh=match, max_age=max_age, min_hits=min_hits,
            class_gating=gating, class_mismatch_penalty=penalty)

        self._last_stamp_ns = None
        self._stat_in = 0
        self._stat_ms = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if qos_rel == 'reliable' else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10, durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(Detection2DArray, '~/tracks_2d', qos)
        self.sub = self.create_subscription(Detection2DArray, '~/detections_input', self.on_dets, qos)
        self.create_timer(5.0, self._log_stats)
        self.get_logger().info(
            f"tracker_node up: coord_space={self.coord_space} gating={gating} "
            f"labels={len(self.labels)} src={self.src_w}x{self.src_h}")

    def _load_labels(self, path):
        if not path:
            self.get_logger().info('class_labels_file empty -> using built-in 7 buoy classes')
            return dict(DEFAULT_LABELS)
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            labels = {int(k): str(v) for k, v in raw.items()}
            self.get_logger().info(f'loaded {len(labels)} class labels from {path}')
            return labels
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'failed to load labels {path}: {e}; using built-in')
            return dict(DEFAULT_LABELS)

    def _name(self, class_index):
        if class_index in self.labels:
            return self.labels[class_index]
        if class_index not in self._warned_unmapped:
            self._warned_unmapped.add(class_index)
            self.get_logger().warn(f'no label for class index {class_index} -> unknown_{class_index}')
        return f'unknown_{class_index}'

    # ---------------------------------------------------------------- callback
    def on_dets(self, msg: Detection2DArray):
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
            self.get_logger().warn(
                f'dropping out-of-order detection (stamp {stamp_ns} < {self._last_stamp_ns})')
            return
        dt = 0.0 if self._last_stamp_ns is None else (stamp_ns - self._last_stamp_ns) / 1e9
        self._last_stamp_ns = stamp_ns

        dets = []
        for d in msg.detections:
            if not d.results:
                continue
            try:
                cls = int(d.results[0].hypothesis.class_id)
            except (ValueError, TypeError):
                cls = -1
            cx = d.bbox.center.position.x
            cy = d.bbox.center.position.y
            w = d.bbox.size_x
            h = d.bbox.size_y
            xyxy = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
            dets.append((xyxy, float(d.results[0].hypothesis.score), cls))

        dets = self._to_source_pixels(dets)

        t0 = time.perf_counter()
        confirmed = self.tracker.update(dets, dt=dt)
        self._stat_ms = (time.perf_counter() - t0) * 1e3
        self._stat_in = len(dets)

        self.pub.publish(self._build_msg(msg.header, confirmed))

    def _to_source_pixels(self, dets):
        if not dets:
            return dets
        out = []
        for xyxy, score, cls in dets:
            if self.coord_space == 'network_padded':
                xyxy = inverse_letterbox([xyxy], (self.src_w, self.src_h),
                                         (self.net_w, self.net_h), self.padding_mode)[0].tolist()
            else:  # source_pixels: clip to bounds (drops pad-region spurious boxes)
                xyxy = [min(max(xyxy[0], 0.0), self.src_w), min(max(xyxy[1], 0.0), self.src_h),
                        min(max(xyxy[2], 0.0), self.src_w), min(max(xyxy[3], 0.0), self.src_h)]
            out.append((xyxy, score, cls))
        return out

    def _build_msg(self, header, tracks):
        out = Detection2DArray()
        out.header = header
        for t in tracks:
            name = self._name(t.frozen_class)
            det = Detection2D()
            det.header = header
            det.id = f'{name}_{t.class_seq}'
            b = t.xyxy
            det.bbox.center.position.x = float((b[0] + b[2]) / 2.0)
            det.bbox.center.position.y = float((b[1] + b[3]) / 2.0)
            det.bbox.size_x = float(b[2] - b[0])
            det.bbox.size_y = float(b[3] - b[1])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(t.score)
            det.results.append(hyp)
            out.detections.append(det)
        return out

    def _log_stats(self):
        n_active = sum(1 for t in self.tracker.tracks if t.is_confirmed)
        self.get_logger().info(
            f'[stats] last_in_dets={self._stat_in} active_tracks={n_active} '
            f'proc={self._stat_ms:.2f}ms')


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
