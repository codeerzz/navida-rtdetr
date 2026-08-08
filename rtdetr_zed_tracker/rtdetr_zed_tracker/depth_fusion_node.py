"""ROS 2 node: fuse ZED depth onto RT-DETR detections to get a 3D position per box.

The whole camera-side pipeline is now three stages:

    RT-DETR  ->  whole-box depth density clustering  ->  world-frame tracking
    (Isaac)      (this node)                             (buoy_mapper_node, host)

This node is the middle stage. It consumes RT-DETR detections DIRECTLY
(~/detections_input = /detections_output) plus the ZED depth image and its
camera_info. For each detected box it scans every depth pixel inside the box,
takes the densest depth cluster (see depth_utils.sample_box_depth), deprojects to
(X,Y,Z) in the camera optical frame, and publishes:

  ~/tracked_objects     TrackedObjectArray  -- primary; invalid-depth objects are
                        STILL published with NaN distances (never fabricated / zero).
  ~/tracks_3d           vision_msgs/Detection3DArray  -- valid-depth objects only.
  ~/tracks_markers      visualization_msgs/MarkerArray -- cube + "class | Z m".
  ~/sync_stats          std_msgs/String  -- match rate / dropped-no-depth.

No image-space tracking, no temporal state
------------------------------------------
There used to be a ByteTrack node (tracker_node + byte_tracker + kalman_box)
between RT-DETR and this one, and this node smoothed each track's position with
an EMA keyed on its track_id. Both are gone, deliberately:

  1. Identity belongs in the map, not the image. buoy_mapper_node associates
     observations by Mahalanobis distance in world coordinates, so an image-space
     track id bought nothing it does not already derive from geometry -- and a
     ByteTrack id is ephemeral anyway (it dies after max_age, and the same buoy
     comes back with a new one).
  2. The EMA actively HURT the downstream estimator. buoy_mapper_node runs an
     information filter, which sums Sigma^-1 and therefore assumes observations
     are independent. Pre-smoothing here made consecutive observations correlated,
     so the filter accumulated confidence faster than the evidence justified.

Emitting one honest, unsmoothed measurement per box per frame is what that filter
actually wants. Outlier rejection moved with it: the mapper's chi-squared gate
rejects a bad depth sample on statistical grounds, which is strictly better than
the fixed max_depth_jump threshold that used to live here.

Dropping the separate tracker process also removes its cost, which per BENCHMARK.md
was ~132 % of a core while the tracking algorithm itself was only ~8 % -- the rest
was Detection2DArray (de)serialisation and UDP loopback, paid 45 times a second.

Depth encoding is handled per the runtime encoding (16UC1 mm / 32FC1 m); intrinsics
are rescaled if the depth image dims differ from the camera_info dims.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .depth_utils import (
    depth_to_metres, deproject, rescale_intrinsics, sample_box_depth, source_box_to_depth_px,
)

# Index -> name for the 7 buoy classes. Lived in tracker_node until that node was
# removed; this is now the only consumer, so it lives here.
DEFAULT_LABELS = {
    0: 'buoy', 1: 'east_buoy', 2: 'green_buoy', 3: 'north_buoy',
    4: 'red_buoy', 5: 'south_buoy', 6: 'west_buoy',
}


def _desc(t):
    return ParameterDescriptor(description=t)


def _marker_id(tag: str) -> int:
    """Stable int id for a marker. Only has to be unique WITHIN one frame -- every
    publish starts with a DELETEALL, so ids are never reused across frames."""
    return abs(hash(tag)) % 2_000_000_000


class DepthFusionNode(Node):
    def __init__(self):
        super().__init__('depth_fusion_node')
        p = self.declare_parameter
        self.src_w = p('source_image_width', 1280, _desc('source image width (px)')).value
        self.src_h = p('source_image_height', 720, _desc('source image height (px)')).value
        self.min_score = p('min_detection_score', 0.0,
                           _desc('drop detections below this score (0 = keep all)')).value
        self.sync_mode = p('sync_mode', 'nearest', _desc('exact|nearest (depth is slower than RGB)')).value
        self.max_dt = p('max_time_diff_sec', 0.05, _desc('nearest-match tolerance (s)')).value
        self.min_valid = p('min_valid_depth', 0.3, _desc('min usable range (m)')).value
        self.max_valid = p('max_valid_depth', 20.0, _desc('max usable range (m)')).value
        self.min_ratio = p('min_valid_ratio', 0.15, _desc('min valid-pixel ratio in box')).value
        self.min_valid_px = p('min_valid_pixels', 15,
                              _desc('min absolute valid-pixel count in box')).value
        self.cluster_window_m = p('depth_cluster_window_m', 0.5,
                                  _desc('max spread (m) of the densest depth cluster')).value
        labels_file = p('class_labels_file', '', _desc('YAML index->name; empty=built-in buoys')).value
        buf_len = p('depth_buffer', 30, _desc('depth frames buffered for stamp matching')).value
        # Headless distance readout. Detections arrive at ~45 Hz, which is unreadable
        # and would flood the log, so the latest frame is held and printed on a timer.
        self.distance_log_hz = p('distance_log_hz', 1.0,
                                 _desc('compact distance log rate (Hz); 0 = off')).value

        self.index_to_name, self.name_to_index = self._load_labels(labels_file)
        self.depth_buf = deque(maxlen=int(buf_len))   # (stamp_ns, depth_m, invalid, w, h)
        self.K = None                                 # (fx,fy,cx,cy) in depth-image space
        self._ci_logged = False
        self._warned_unmapped = set()
        self._last_objects = []                       # latest frame, for the distance log
        self._stat = dict(msgs=0, matched=0, dropped=0, off_sum=0.0)

        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=10, durability=DurabilityPolicy.VOLATILE)
        depth_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=int(buf_len), durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, '~/depth', self.on_depth, depth_qos)
        self.create_subscription(CameraInfo, '~/depth_camera_info', self.on_info, rel)
        self.create_subscription(Detection2DArray, '~/detections_input', self.on_detections, rel)

        self.pub_obj = self.create_publisher(_lazy_tracked_array(), '~/tracked_objects', rel)
        self.pub_3d = self.create_publisher(Detection3DArray, '~/tracks_3d', rel)
        self.pub_mark = self.create_publisher(MarkerArray, '~/tracks_markers', rel)
        self.pub_stats = self.create_publisher(String, '~/sync_stats', rel)
        self.create_timer(5.0, self._log_stats)
        if self.distance_log_hz > 0.0:
            self.create_timer(1.0 / self.distance_log_hz, self._log_distances)
        self.get_logger().info(
            f'depth_fusion_node up: sync={self.sync_mode} src={self.src_w}x{self.src_h} '
            f'cluster_window={self.cluster_window_m}m labels={len(self.index_to_name)} '
            f'dist_log={self.distance_log_hz}Hz (stateless: no image-space tracking, no EMA)')

    # ---------------------------------------------------------------- inputs
    def _load_labels(self, path):
        """(index->name, name->index). RT-DETR emits a numeric class index, so the
        forward map is what turns it into the label the rest of the stack speaks.
        tracker_node used to own this; it is the one job worth inheriting."""
        forward = dict(DEFAULT_LABELS)
        if path:
            try:
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                forward = {int(k): str(v) for k, v in raw.items()}
                self.get_logger().info(f'loaded {len(forward)} class labels from {path}')
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'labels load failed {path}: {e}; using built-in')
                forward = dict(DEFAULT_LABELS)
        return forward, {v: k for k, v in forward.items()}

    def _name(self, class_index):
        if class_index in self.index_to_name:
            return self.index_to_name[class_index]
        if class_index not in self._warned_unmapped:
            self._warned_unmapped.add(class_index)
            self.get_logger().warn(f'no label for class index {class_index} -> unknown_{class_index}')
        return f'unknown_{class_index}'

    def on_info(self, msg: CameraInfo):
        if self.K is not None:
            return
        fx, cx, fy, cy = msg.k[0], msg.k[2], msg.k[4], msg.k[5]
        self.K = (fx, fy, cx, cy)
        self._ci_wh = (msg.width, msg.height)
        self.get_logger().info(f'depth intrinsics from camera_info {msg.width}x{msg.height}: '
                               f'fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}')

    def on_depth(self, msg: Image):
        try:
            arr = _img_to_array(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'depth decode failed ({msg.encoding}): {e}')
            return
        depth_m, invalid = depth_to_metres(arr, msg.encoding)
        ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.depth_buf.append((ns, depth_m, invalid, msg.width, msg.height))
        self._depth_frame_id = msg.header.frame_id

    # ---------------------------------------------------------------- match
    def _match_depth(self, stamp_ns):
        if not self.depth_buf:
            return None, None
        if self.sync_mode == 'exact':
            for e in self.depth_buf:
                if e[0] == stamp_ns:
                    return e, 0.0
            return None, None
        best = min(self.depth_buf, key=lambda e: abs(e[0] - stamp_ns))
        off = abs(best[0] - stamp_ns) / 1e9
        return (best, off) if off <= self.max_dt else (None, None)

    def _intrinsics_for(self, depth_wh):
        fx, fy, cx, cy = self.K
        if depth_wh != self._ci_wh:                        # rescale to actual depth dims
            fx, fy, cx, cy = rescale_intrinsics(fx, fy, cx, cy, self._ci_wh, depth_wh)
            if not self._ci_logged:
                self.get_logger().warn(f'depth image {depth_wh} != camera_info {self._ci_wh}; '
                                       f'rescaled intrinsics -> fx={fx:.1f} cx={cx:.1f}')
        self._ci_logged = True
        return fx, fy, cx, cy

    # ---------------------------------------------------------------- fuse
    def on_detections(self, msg: Detection2DArray):
        self._stat['msgs'] += 1
        if self.K is None:
            return
        entry, off = self._match_depth(
            int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec))
        if entry is None:
            self._stat['dropped'] += 1
            return
        self._stat['matched'] += 1
        self._stat['off_sum'] += off
        _, depth_m, invalid, dw, dh = entry
        fx, fy, cx, cy = self._intrinsics_for((dw, dh))
        frame = getattr(self, '_depth_frame_id', msg.header.frame_id)

        obj_arr = _lazy_tracked_array()()
        obj_arr.header = msg.header
        det3d = Detection3DArray(); det3d.header = msg.header; det3d.header.frame_id = frame
        markers = MarkerArray()
        markers.markers.append(_delete_all(frame))

        for i, d in enumerate(msg.detections):
            if not d.results:
                continue
            score = d.results[0].hypothesis.score
            if score < self.min_score:
                continue
            # RT-DETR emits a NUMERIC class index; tracker_node used to map it to a
            # name before this node ever saw it. Now we do it here.
            try:
                name = self._name(int(d.results[0].hypothesis.class_id))
            except (ValueError, TypeError):
                name = self._name(-1)
            # Per-frame identifier only. It is NOT a track id and carries no meaning
            # across frames -- buoy_mapper_node establishes identity from world
            # geometry. Kept because TrackedObject.track_id exists in the message
            # and an empty string reads worse in logs than an honest frame-local tag.
            tid = f'det_{i}'

            cxb, cyb = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            # Clip to the source image. The Isaac decoder already emits ORIGINAL
            # pixels (verified Phase 0/2), so no letterbox inversion is needed.
            box_src = [max(cxb - w / 2, 0.0), max(cyb - h / 2, 0.0),
                       min(cxb + w / 2, float(self.src_w)), min(cyb + h / 2, float(self.src_h))]
            box_dep = source_box_to_depth_px(box_src, (self.src_w, self.src_h), (dw, dh))
            z, ratio, valid, (uc, vc) = sample_box_depth(
                depth_m, invalid, box_dep, self.min_valid, self.max_valid,
                self.min_ratio, self.min_valid_px, self.cluster_window_m)

            if valid:
                # One honest measurement, published as measured. No EMA, no jump
                # rejection: the mapper's information filter wants independent
                # observations, and its chi-squared gate handles outliers better
                # than a fixed threshold could.
                X, Y, Z = deproject(uc, vc, z, fx, fy, cx, cy)
                obj_arr.objects.append(self._make_object(
                    tid, name, score, X, Y, Z, ratio, True))
                det3d.detections.append(self._make_det3d(tid, name, score, X, Y, Z))
                markers.markers.extend(self._make_markers(frame, tid, name, X, Y, Z))
            else:
                obj_arr.objects.append(self._make_object(
                    tid, name, score, math.nan, math.nan, math.nan, ratio, False))

        self.pub_obj.publish(obj_arr)
        self.pub_3d.publish(det3d)
        self.pub_mark.publish(markers)
        self._last_objects = obj_arr.objects

    # ---------------------------------------------------------------- builders
    def _make_object(self, tid, name, score, x, y, z, ratio, valid):
        TrackedObject = _lazy_tracked_object()
        o = TrackedObject()
        o.track_id = tid
        o.class_name = name
        o.class_index = int(self.name_to_index.get(name, -1))
        o.confidence = float(score)
        o.depth_valid = bool(valid)
        o.depth_valid_ratio = float(ratio)
        # No temporal state, so there is no age to report. Kept at 1 ("seen once,
        # this frame") rather than 0 so the field stays truthful; consumers that
        # used to gate on it (buoy_mapper_node's min_age_frames) must now rely on
        # their own confirmation logic instead.
        o.age_frames = 1
        o.position = Point(x=float(x), y=float(y), z=float(z))
        if valid:
            o.distance_z_m = float(z)
            o.distance_range_m = float(math.sqrt(x * x + y * y + z * z))
        else:
            o.distance_z_m = math.nan
            o.distance_range_m = math.nan
        return o

    def _make_det3d(self, tid, name, score, x, y, z):
        d = Detection3D()
        d.id = tid
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = name
        hyp.hypothesis.score = float(score)
        d.results.append(hyp)
        d.bbox.center.position.x = float(x)
        d.bbox.center.position.y = float(y)
        d.bbox.center.position.z = float(z)
        d.bbox.size.x = d.bbox.size.y = d.bbox.size.z = 0.3
        return d

    def _make_markers(self, frame, tid, name, x, y, z):
        cube = Marker()
        cube.header.frame_id = frame
        cube.ns = 'objects'
        cube.id = _marker_id(tid)
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position = Point(x=float(x), y=float(y), z=float(z))
        cube.pose.orientation.w = 1.0
        cube.scale.x = cube.scale.y = cube.scale.z = 0.3
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = 0.1, 0.8, 1.0, 0.8
        cube.lifetime.nanosec = 300_000_000
        txt = Marker()
        txt.header.frame_id = frame
        txt.ns = 'labels'
        txt.id = _marker_id(tid)
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position = Point(x=float(x), y=float(y), z=float(z) - 0.35)
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.2
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f'{tid} | {name} | {z:.2f} m'
        txt.lifetime.nanosec = 300_000_000
        return [cube, txt]

    def _log_distances(self):
        """Compact distance readout, one line per tick. This is the headless
        substitute for the deleted viewer_node: no GUI, no extra process, just the
        numbers you actually watch on the water, in whichever pane runs this node.

        Nearest first, because the nearest object is the one that matters. Objects
        whose depth failed are listed last and marked -- reporting them is the point,
        since a buoy the detector SEES but cannot range is a different (and more
        dangerous) situation from one it never saw.
        """
        objs = self._last_objects
        if not objs:
            self.get_logger().info('[dist] no detections')
            return
        ranged = sorted((o for o in objs if o.depth_valid), key=lambda o: o.distance_z_m)
        blind = [o for o in objs if not o.depth_valid]
        parts = [f'{o.class_name} {o.distance_z_m:5.2f}m (r={o.distance_range_m:.2f} '
                 f'c={o.confidence:.2f} v={o.depth_valid_ratio:.0%})' for o in ranged]
        parts += [f'{o.class_name} NO-DEPTH (c={o.confidence:.2f} v={o.depth_valid_ratio:.0%})'
                  for o in blind]
        self.get_logger().info(
            f'[dist] {len(ranged)}/{len(objs)} ranged | ' + '  |  '.join(parts))

    def _log_stats(self):
        s = self._stat
        rate = s['matched'] / max(s['msgs'], 1)
        mean_off = (s['off_sum'] / max(s['matched'], 1)) * 1e3
        text = (f"msgs={s['msgs']} matched={s['matched']} ({rate:.0%}) "
                f"dropped_no_depth={s['dropped']} mean_off={mean_off:.1f}ms")
        self.get_logger().info(f'[sync] {text}')
        self.pub_stats.publish(String(data=text))


def _img_to_array(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc in ('16uc1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if enc == '32fc1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f'unsupported depth encoding {msg.encoding}')


def _delete_all(frame):
    m = Marker()
    m.header.frame_id = frame
    m.action = Marker.DELETEALL
    return m


# rtdetr_zed_tracker_msgs is only importable once built+sourced; import lazily so the
# module still imports for unit tests that don't need the custom messages.
def _lazy_tracked_array():
    from rtdetr_zed_tracker_msgs.msg import TrackedObjectArray
    return TrackedObjectArray


def _lazy_tracked_object():
    from rtdetr_zed_tracker_msgs.msg import TrackedObject
    return TrackedObject


def main(args=None):
    rclpy.init(args=args)
    node = DepthFusionNode()
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
