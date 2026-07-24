"""ROS 2 node: draw tracked boxes on the original ZED color image.

Alignment verification tool for Phase 2. STAMP-MATCHED: each image is buffered,
and when a tracks message arrives it is composited onto the image with the SAME
stamp (the frame the detections were actually computed from) and published. This
keeps boxes exactly on their objects (no temporal skew / "old+new frame mixing")
and stays smooth because detections run at ~image rate. The output stream is
delayed by the pipeline latency (~120 ms), which is inherent and invisible for
slow scenes.
"""
from __future__ import annotations

import hashlib
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


def _color_for(track_id: str):
    h = hashlib.md5(track_id.encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])  # BGR


class OverlayNode(Node):
    def __init__(self):
        super().__init__('overlay_node')
        self.buffer_len = self.declare_parameter(
            'image_buffer', 15,
            ParameterDescriptor(description='images buffered for stamp matching')).value
        self.match_tol_ms = self.declare_parameter(
            'match_tolerance_ms', 100.0,
            ParameterDescriptor(description='max stamp diff to accept an image<->tracks match (ms)')).value

        self.bridge = CvBridge()
        self.images = deque(maxlen=int(self.buffer_len))   # (stamp_ns, bgr image)

        # RELIABLE so we never drop the exact frame a detection references (best-effort
        # drops fragments of these 8 MB samples over UDP -> stamp matching would miss).
        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=int(self.buffer_len), durability=DurabilityPolicy.VOLATILE)
        track_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=10, durability=DurabilityPolicy.VOLATILE)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=1, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, '~/image', self.on_image, img_qos)
        self.create_subscription(Detection2DArray, '~/tracks_2d', self.on_tracks, track_qos)
        self.pub = self.create_publisher(Image, '~/tracks_overlay', pub_qos)
        self._miss = 0
        self.get_logger().info('overlay_node up (stamp-matched; subs ~/image, ~/tracks_2d)')

    @staticmethod
    def _ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed ({msg.encoding}): {e}')
            return
        self.images.append((self._ns(msg.header.stamp), img))

    def _match_image(self, stamp_ns):
        """Return the buffered image whose stamp equals (or is closest within tol to) stamp_ns."""
        if not self.images:
            return None
        best, best_d = None, None
        for ns, img in self.images:
            d = abs(ns - stamp_ns)
            if ns == stamp_ns:
                return img
            if best_d is None or d < best_d:
                best, best_d = img, d
        if best_d is not None and best_d <= self.match_tol_ms * 1e6:
            return best
        return None

    def on_tracks(self, msg: Detection2DArray):
        canvas = self._match_image(self._ns(msg.header.stamp))
        if canvas is None:
            self._miss += 1
            if self._miss % 30 == 1:
                self.get_logger().warn('no image within tolerance for a tracks stamp '
                                       f'(count={self._miss}); check image buffer / tolerance')
            return
        canvas = canvas.copy()
        for d in msg.detections:
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            color = _color_for(d.id)
            label = d.id if not d.results else f'{d.id} ({d.results[0].hypothesis.score:.2f})'
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OverlayNode()
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
