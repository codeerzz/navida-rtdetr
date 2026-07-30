"""ROS 2 node: Open-vocabulary obstacle detection via YOLO-World.

Runs YOLO-World-S (ultralytics) with a configurable text prompt to detect
arbitrary obstacles (default: "a vessel").  Because YOLO-World has no
knowledge of buoys it may fire on them; this node suppresses any detection
that overlaps (IoU >= suppression_iou_threshold) with a confirmed RT-DETR
buoy track coming from ~/buoy_tracks.

Subscribes:
  ~/image               sensor_msgs/Image          — ZED color frame (bgra8/bgr8)
  ~/buoy_tracks         vision_msgs/Detection2DArray — RT-DETR confirmed buoy tracks
  ~/depth               sensor_msgs/Image          — ZED depth image
  ~/depth_camera_info   sensor_msgs/CameraInfo     — ZED depth intrinsics

Publishes:
  ~/detections          vision_msgs/Detection2DArray — filtered obstacle boxes
  ~/sync_stats          std_msgs/String              — diagnostic counters

Parameters (all settable via `ros2 param set` at runtime):
  avoid_prompt              str    "a vessel"
  confidence_threshold      float  0.3
  suppression_iou_threshold float  0.3
  source_image_width        int    1280
  source_image_height       int    720
  model_path                str    "yolov8s-worldv2.pt"
  depth_buffer              int    30
  max_time_diff_sec         float  0.05
  roi_shrink_factor         float  0.4
  min_valid_depth           float  0.3
  max_valid_depth           float  20.0
  min_valid_ratio           float  0.15
  depth_percentile          float  50.0
  depth_ema_alpha           float  0.4
  max_depth_jump            float  1.0
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .depth_utils import depth_to_metres


def _desc(t: str) -> ParameterDescriptor:
    return ParameterDescriptor(description=t)


def _iou(a: list, b: list) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YoloWorldNode(Node):
    """Open-vocabulary obstacle detector using YOLO-World-S.

    Design decisions:
    - YOLO-World-S is chosen for speed (see YOLO_WORLD_README.md).
    - set_classes() is called once at init and again on prompt param change,
      so prompt updates are zero-overhead at inference time.
    - Cross-suppression compares against RT-DETR confirmed tracks; tentative
      tracks are excluded (they may not be buoys yet).
    - Model loading is done in __init__ but guarded so missing ultralytics
      only warns rather than crashing the whole launch graph.
    """

    def __init__(self):
        super().__init__('yolo_world_node')
        p = self.declare_parameter

        # --- detection parameters ---
        self.avoid_prompt: str = p(
            'avoid_prompt', 'a vessel',
            _desc('Text prompt for the obstacle class. '
                  'Change at runtime: ros2 param set /yolo_world_node avoid_prompt "a boat"')
        ).value
        self.conf_thresh: float = p(
            'confidence_threshold', 0.3,
            _desc('YOLO-World confidence threshold (0–1)')
        ).value
        self.supp_iou: float = p(
            'suppression_iou_threshold', 0.3,
            _desc('IoU >= this → YOLO detection suppressed (overlaps an RT-DETR buoy track)')
        ).value
        self.src_w: int = p('source_image_width', 1280, _desc('Color image width (px)')).value
        self.src_h: int = p('source_image_height', 720, _desc('Color image height (px)')).value
        model_path: str = p(
            'model_path', 'yolov8s-worldv2.pt',
            _desc('Path or Ultralytics model name for YOLO-World weights')
        ).value

        # --- depth fusion parameters ---
        buf_len: int = p('depth_buffer', 30, _desc('Depth frames buffered for stamp matching')).value
        self.max_dt: float = p('max_time_diff_sec', 0.05,
                               _desc('Nearest-stamp match tolerance (s)')).value

        # --- state ---
        self._model = None
        self._model_path = model_path
        self._model_lock = threading.Lock()
        self._buoy_tracks: list = []
        self._buoy_lock = threading.Lock()
        self.depth_buf: deque = deque(maxlen=int(buf_len))
        self.K = None
        self._ci_wh = None
        self._ci_logged = False
        self._depth_frame_id = 'zed_left_camera_optical_frame'
        self._stat = dict(frames=0, suppressed=0, no_depth=0)

        self._load_model()

        # --- QoS ---
        rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(buf_len),
            durability=DurabilityPolicy.VOLATILE,
        )

        # --- subscriptions ---
        self.create_subscription(Image, '~/image', self._on_image, rel)
        self.create_subscription(
            Detection2DArray, '~/buoy_tracks', self._on_buoy_tracks, rel)
        self.create_subscription(CameraInfo, '~/depth_camera_info', self._on_camera_info, rel)
        self.create_subscription(Image, '~/depth', self._on_depth, depth_qos)

        # --- publishers ---
        self.pub_det = self.create_publisher(Detection2DArray, '~/detections', rel)
        self.pub_stats = self.create_publisher(String, '~/sync_stats', rel)
        self.create_timer(5.0, self._log_stats)

        # allow runtime updates to avoid_prompt and thresholds
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f'yolo_world_node up | prompt="{self.avoid_prompt}" '
            f'conf={self.conf_thresh} supp_iou={self.supp_iou} model={model_path}')

    # ------------------------------------------------------------------ model
    def _load_model(self):
        """Load YOLO-World-S. Graceful failure so the node starts even without GPU."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            model = YOLO(self._model_path)
            model.set_classes([self.avoid_prompt])
            with self._model_lock:
                self._model = model
            self.get_logger().info(f'YOLO-World loaded: {self._model_path}')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(
                f'Failed to load YOLO-World ({self._model_path}): {e}. '
                f'Install with: pip install ultralytics')

    def _on_param_update(self, params) -> SetParametersResult:
        """Handle ros2 param set at runtime — especially avoid_prompt."""
        for param in params:
            if param.name == 'avoid_prompt' and param.value:
                self.avoid_prompt = param.value
                with self._model_lock:
                    if self._model is not None:
                        try:
                            self._model.set_classes([self.avoid_prompt])
                        except Exception as e:  # noqa: BLE001
                            self.get_logger().warn(f'set_classes failed: {e}')
                self.get_logger().info(f'avoid_prompt updated → "{self.avoid_prompt}"')
            elif param.name == 'confidence_threshold':
                self.conf_thresh = float(param.value)
                self.get_logger().info(f'confidence_threshold → {self.conf_thresh}')
            elif param.name == 'suppression_iou_threshold':
                self.supp_iou = float(param.value)
                self.get_logger().info(f'suppression_iou_threshold → {self.supp_iou}')
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ inputs
    def _on_buoy_tracks(self, msg: Detection2DArray):
        """Cache confirmed RT-DETR buoy tracks as xyxy boxes for suppression."""
        boxes = []
        for d in msg.detections:
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        with self._buoy_lock:
            self._buoy_tracks = boxes

    def _on_camera_info(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
        self._ci_wh = (msg.width, msg.height)
        self.get_logger().info(
            f'Depth intrinsics from camera_info {msg.width}x{msg.height}: '
            f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}')

    def _on_depth(self, msg: Image):
        try:
            arr = _img_to_array(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Depth decode failed ({msg.encoding}): {e}')
            return
        depth_m, invalid = depth_to_metres(arr, msg.encoding)
        ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.depth_buf.append((ns, depth_m, invalid, msg.width, msg.height))
        self._depth_frame_id = msg.header.frame_id

    # ------------------------------------------------------------------ main
    def _on_image(self, msg: Image):
        with self._model_lock:
            model = self._model
        if model is None:
            return

        self._stat['frames'] += 1

        # decode image to BGR numpy
        try:
            frame = _decode_image(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Image decode failed ({msg.encoding}): {e}')
            return

        # YOLO-World inference
        try:
            results = model.predict(frame, conf=self.conf_thresh, verbose=False)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'YOLO-World inference failed: {e}')
            return

        # parse raw detections
        raw: list[tuple[list, float]] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0])
                raw.append((xyxy, score))

        # cross-suppression: drop any YOLO box that overlaps a confirmed buoy track
        with self._buoy_lock:
            buoy_boxes = list(self._buoy_tracks)

        filtered: list[tuple[list, float]] = []
        for xyxy, score in raw:
            suppressed = any(_iou(xyxy, b) >= self.supp_iou for b in buoy_boxes)
            if suppressed:
                self._stat['suppressed'] += 1
            else:
                filtered.append((xyxy, score))

        # build and publish Detection2DArray
        out = Detection2DArray()
        out.header = msg.header
        for i, (xyxy, score) in enumerate(filtered):
            x1, y1, x2, y2 = xyxy
            det = Detection2D()
            det.header = msg.header
            det.id = f'obstacle_{i}'
            det.bbox.center.position.x = float((x1 + x2) / 2)
            det.bbox.center.position.y = float((y1 + y2) / 2)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.avoid_prompt
            hyp.hypothesis.score = score
            det.results.append(hyp)
            out.detections.append(det)

        self.pub_det.publish(out)

    # ------------------------------------------------------------------ helpers
    def _log_stats(self):
        s = self._stat
        text = (f'frames={s["frames"]} suppressed={s["suppressed"]} '
                f'no_depth={s["no_depth"]} prompt="{self.avoid_prompt}"')
        self.get_logger().info(f'[yolo_world] {text}')
        self.pub_stats.publish(String(data=text))


# --------------------------------------------------------------------------- #
# Module-level helpers (used by YoloWorldNode and importable for unit tests)  #
# --------------------------------------------------------------------------- #

def _decode_image(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image → BGR uint8 numpy array."""
    import cv2  # noqa: PLC0415
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('bgra8', 'rgba8'):
        img = raw.reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_BGRA2BGR if enc == 'bgra8' else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(img, code)
    if enc in ('bgr8', 'rgb8'):
        img = raw.reshape(msg.height, msg.width, 3)
        return img if enc == 'bgr8' else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise ValueError(f'Unsupported image encoding: {msg.encoding}')


def _img_to_array(msg: Image) -> np.ndarray:
    """Decode depth sensor_msgs/Image → uint16 or float32 numpy array."""
    enc = msg.encoding.lower()
    if enc in ('16uc1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if enc == '32fc1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f'Unsupported depth encoding: {msg.encoding}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldNode()
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
