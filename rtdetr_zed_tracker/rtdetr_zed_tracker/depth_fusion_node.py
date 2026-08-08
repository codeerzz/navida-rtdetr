"""ROS 2 node: fuse ZED depth onto the RGB tracks to get a 3D position per ID.

Consumes the RGB-only tracks (~/tracks_input = tracker_node/tracks_2d) plus the
ZED depth image + its camera_info. For each tracked box it samples a robust depth
from the central ROI, deprojects to (X,Y,Z) in the camera optical frame, smooths
per track (EMA + jump rejection), and publishes:

  ~/tracked_objects     TrackedObjectArray  -- primary; invalid-depth objects are
                        STILL published with NaN distances (never fabricated / zero).
  ~/tracks_3d           vision_msgs/Detection3DArray  -- valid-depth objects only.
  ~/tracks_3d_predicted vision_msgs/Detection3DArray  -- state propagated to 'now'.
  ~/tracks_markers      visualization_msgs/MarkerArray -- cube + "id | class | Z m".
  ~/sync_stats          std_msgs/String  -- match rate / dropped-no-depth.

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
from .tracker_node import DEFAULT_LABELS


def _desc(t):
    return ParameterDescriptor(description=t)


def _marker_id(track_id: str) -> int:
    return abs(hash(track_id)) % 2_000_000_000


class DepthFusionNode(Node):
    def __init__(self):
        super().__init__('depth_fusion_node')
        p = self.declare_parameter
        self.src_w = p('source_image_width', 1280, _desc('color/track box width (px)')).value
        self.src_h = p('source_image_height', 720, _desc('color/track box height (px)')).value
        self.sync_mode = p('sync_mode', 'nearest', _desc('exact|nearest (depth is slower than RGB)')).value
        self.max_dt = p('max_time_diff_sec', 0.05, _desc('nearest-match tolerance (s)')).value
        self.min_valid = p('min_valid_depth', 0.3, _desc('min usable range (m)')).value
        self.max_valid = p('max_valid_depth', 20.0, _desc('max usable range (m)')).value
        self.min_ratio = p('min_valid_ratio', 0.15, _desc('min valid-pixel ratio in box')).value
        self.min_valid_px = p('min_valid_pixels', 15,
                              _desc('min absolute valid-pixel count in box')).value
        self.cluster_window_m = p('depth_cluster_window_m', 0.5,
                                  _desc('max spread (m) of the densest depth cluster')).value
        self.ema_alpha = p('depth_ema_alpha', 0.4, _desc('per-track position EMA alpha')).value
        self.max_jump = p('max_depth_jump', 1.0, _desc('reject Z jump > this unless it persists (m)')).value
        labels_file = p('class_labels_file', '', _desc('YAML index->name; empty=built-in buoys')).value
        buf_len = p('depth_buffer', 30, _desc('depth frames buffered for stamp matching')).value

        self.name_to_index = self._load_name_index(labels_file)
        self.depth_buf = deque(maxlen=int(buf_len))   # (stamp_ns, depth_m, invalid, w, h)
        self.K = None                                 # (fx,fy,cx,cy) in depth-image space
        self._ci_logged = False
        self.tracks = {}                              # id -> dict(pos, vel, last_ns, jump, age)
        self._stat = dict(msgs=0, matched=0, dropped=0, off_sum=0.0)

        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=10, durability=DurabilityPolicy.VOLATILE)
        depth_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=int(buf_len), durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, '~/depth', self.on_depth, depth_qos)
        self.create_subscription(CameraInfo, '~/depth_camera_info', self.on_info, rel)
        self.create_subscription(Detection2DArray, '~/tracks_input', self.on_tracks, rel)

        self.pub_obj = self.create_publisher(_lazy_tracked_array(), '~/tracked_objects', rel)
        self.pub_3d = self.create_publisher(Detection3DArray, '~/tracks_3d', rel)
        self.pub_pred = self.create_publisher(Detection3DArray, '~/tracks_3d_predicted', rel)
        self.pub_mark = self.create_publisher(MarkerArray, '~/tracks_markers', rel)
        self.pub_stats = self.create_publisher(String, '~/sync_stats', rel)
        self.create_timer(5.0, self._log_stats)
        self.get_logger().info(
            f'depth_fusion_node up: sync={self.sync_mode} src={self.src_w}x{self.src_h} '
            f'cluster_window={self.cluster_window_m}m labels={len(self.name_to_index)}')

    # ---------------------------------------------------------------- inputs
    def _load_name_index(self, path):
        table = {v: k for k, v in DEFAULT_LABELS.items()}
        if path:
            try:
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                table = {str(v): int(k) for k, v in raw.items()}
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'labels load failed {path}: {e}; using built-in')
        return table

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
    def on_tracks(self, msg: Detection2DArray):
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
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        frame = getattr(self, '_depth_frame_id', msg.header.frame_id)

        obj_arr = _lazy_tracked_array()()
        obj_arr.header = msg.header
        det3d = Detection3DArray(); det3d.header = msg.header; det3d.header.frame_id = frame
        pred3d = Detection3DArray(); pred3d.header = msg.header; pred3d.header.frame_id = frame
        markers = MarkerArray()
        markers.markers.append(_delete_all(frame))
        seen = set()

        for d in msg.detections:
            if not d.results:
                continue
            tid = d.id
            seen.add(tid)
            name = d.results[0].hypothesis.class_id
            score = d.results[0].hypothesis.score
            cxb, cyb = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            box_src = [cxb - w / 2, cyb - h / 2, cxb + w / 2, cyb + h / 2]
            box_dep = source_box_to_depth_px(box_src, (self.src_w, self.src_h), (dw, dh))
            z, ratio, valid, (uc, vc) = sample_box_depth(
                depth_m, invalid, box_dep, self.min_valid, self.max_valid,
                self.min_ratio, self.min_valid_px, self.cluster_window_m)

            st = self.tracks.get(tid)
            if valid:
                X, Y, Z = deproject(uc, vc, z, fx, fy, cx, cy)
                st = self._update_track(tid, st, (X, Y, Z), stamp_ns)
                px, py, pz = st['pos']
                obj_arr.objects.append(self._make_object(
                    tid, name, score, px, py, pz, ratio, st['age'], True))
                det3d.detections.append(self._make_det3d(tid, name, score, px, py, pz))
                markers.markers.extend(self._make_markers(frame, tid, name, px, py, pz))
            else:
                if st is not None:
                    st['age'] += 1
                obj_arr.objects.append(self._make_object(
                    tid, name, score, math.nan, math.nan, math.nan, ratio,
                    st['age'] if st else 0, False))
            # predicted (from any track that has ever had valid depth)
            if st is not None:
                dt = max((now_ns - st['last_ns']) / 1e9, 0.0)
                qx = st['pos'][0] + st['vel'][0] * dt
                qy = st['pos'][1] + st['vel'][1] * dt
                qz = st['pos'][2] + st['vel'][2] * dt
                pd = self._make_det3d(tid, name, score, qx, qy, qz)
                pred3d.detections.append(pd)

        pred3d.header.stamp = self.get_clock().now().to_msg()   # predicted-to-now validity time
        self._gc_tracks(seen)
        self.pub_obj.publish(obj_arr)
        self.pub_3d.publish(det3d)
        self.pub_pred.publish(pred3d)
        self.pub_mark.publish(markers)

    def _update_track(self, tid, st, pos_meas, stamp_ns):
        X, Y, Z = pos_meas
        if st is None:
            st = dict(pos=(X, Y, Z), vel=(0.0, 0.0, 0.0), last_ns=stamp_ns, jump=0, age=1)
            self.tracks[tid] = st
            return st
        if abs(Z - st['pos'][2]) > self.max_jump:
            st['jump'] += 1
            if st['jump'] < 3:                       # transient jump -> reject, keep state
                st['age'] += 1
                return st
        st['jump'] = 0
        dt = max((stamp_ns - st['last_ns']) / 1e9, 1e-3)
        a = self.ema_alpha
        new_pos = tuple(a * m + (1 - a) * o for m, o in zip((X, Y, Z), st['pos']))
        vel_new = tuple((n - o) / dt for n, o in zip(new_pos, st['pos']))
        st['vel'] = tuple(a * vn + (1 - a) * vo for vn, vo in zip(vel_new, st['vel']))
        st['pos'] = new_pos
        st['last_ns'] = stamp_ns
        st['age'] += 1
        return st

    def _gc_tracks(self, seen):
        for tid in [t for t in self.tracks if t not in seen]:
            # keep a little history but drop long-gone ids to bound memory
            self.tracks[tid]['gone'] = self.tracks[tid].get('gone', 0) + 1
            if self.tracks[tid]['gone'] > 60:
                del self.tracks[tid]

    # ---------------------------------------------------------------- builders
    def _make_object(self, tid, name, score, x, y, z, ratio, age, valid):
        TrackedObject = _lazy_tracked_object()
        o = TrackedObject()
        o.track_id = tid
        o.class_name = name
        o.class_index = int(self.name_to_index.get(name, -1))
        o.confidence = float(score)
        o.depth_valid = bool(valid)
        o.depth_valid_ratio = float(ratio)
        o.age_frames = int(age)
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

    def _log_stats(self):
        s = self._stat
        rate = s['matched'] / max(s['msgs'], 1)
        mean_off = (s['off_sum'] / max(s['matched'], 1)) * 1e3
        text = (f"msgs={s['msgs']} matched={s['matched']} ({rate:.0%}) "
                f"dropped_no_depth={s['dropped']} mean_off={mean_off:.1f}ms "
                f"tracks={len(self.tracks)}")
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
