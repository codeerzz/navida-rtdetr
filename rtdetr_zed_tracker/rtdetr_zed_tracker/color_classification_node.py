"""ROS 2 node: refine a track's color label with YCrCb thresholding.

Bolted onto the pipeline as its own, independent enrichment stage -- same shape
as overlay_node's image<->tracks stamp matching -- so it never touches
tracker_node or depth_fusion_node. Reads the ZED color image + tracker_node's
tracks_2d, classifies each colorable track's crop with color_classifier, and
majority-votes the result per track id (label_vote.LabelVote) before publishing
a refined copy of the tracks with the color-bearing part of the label corrected.

Only labels in ``colorable_labels`` (default: buoy, red_buoy, green_buoy) are
touched -- cardinal-mark classes (east/north/south/west_buoy) pass through
unchanged, since color has no meaning for them. ``buoy`` is included by default
so this pairs directly with class_remap_node: if RT-DETR's 7 trained classes
are collapsed to a single generic "buoy" upstream (no retraining needed -- see
class_remap.yaml), this node is what actually decides red vs green.

Known limitation: ``Detection2D.id`` (e.g. "red_buoy_2") is assigned once by
tracker_node from the ORIGINAL detector class and is left alone here -- only
``results[0].hypothesis.class_id`` is corrected. If this node overrides a track's
color, its display id can look stale (id says "red_buoy_2", refined class_id says
"green_buoy") until tracker_node itself is revisited, which is out of scope here.
"""
from __future__ import annotations

from collections import deque

import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from .color_classifier import classify_color, load_color_ranges
from .label_vote import LabelVote


def _desc(text):
    return ParameterDescriptor(description=text)


class ColorClassificationNode(Node):
    def __init__(self, **kwargs):
        super().__init__('color_classification_node', **kwargs)
        p = self.declare_parameter
        self.buffer_len = p('image_buffer', 15, _desc('images buffered for stamp matching')).value
        self.match_tol_ms = p('match_tolerance_ms', 100.0,
                              _desc('max stamp diff to accept an image<->tracks match (ms)')).value
        self.roi_shrink = p('roi_shrink', 0.6, _desc('central crop fraction sampled for color')).value
        self.min_confidence = p('min_confidence', 0.12,
                                _desc('min in-range pixel ratio to trust a color observation')).value
        self.min_votes = p('min_votes', 3, _desc('observations before a track color is frozen')).value
        color_ranges_file = p('color_ranges_file', '',
                              _desc('YAML: color -> [[y_min,y_max,cr_min,cr_max,cb_min,cb_max], ...]')).value
        colorable = p('colorable_labels', ['buoy', 'red_buoy', 'green_buoy'],
                     _desc('class names eligible for color refinement; others pass through untouched')).value
        self.colorable = set(colorable)

        if not color_ranges_file:
            raise RuntimeError("color_classification_node requires the 'color_ranges_file' parameter")
        self.color_ranges = load_color_ranges(color_ranges_file)
        # 'red' -> 'red_buoy' -- assumes the "<color>_buoy" naming already used by class_labels.yaml.
        self.color_to_label = {c: f'{c}_buoy' for c in self.color_ranges}

        self.bridge = CvBridge()
        self.images = deque(maxlen=int(self.buffer_len))   # (stamp_ns, bgr image)
        self.votes: dict[str, LabelVote] = {}               # track id -> LabelVote

        img_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=int(self.buffer_len), durability=DurabilityPolicy.VOLATILE)
        track_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                               depth=10, durability=DurabilityPolicy.VOLATILE)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             depth=10, durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, '~/image', self.on_image, img_qos)
        self.create_subscription(Detection2DArray, '~/tracks_2d', self.on_tracks, track_qos)
        self.pub = self.create_publisher(Detection2DArray, '~/tracks_color_refined', pub_qos)
        self._miss = 0
        self.get_logger().info(
            f'color_classification_node up: colors={list(self.color_ranges)} '
            f'colorable={sorted(self.colorable)} min_votes={self.min_votes} '
            f'roi_shrink={self.roi_shrink} min_confidence={self.min_confidence}')

    # ---------------------------------------------------------------- image buffer
    @staticmethod
    def _ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed ({msg.encoding}): {e}')
            return
        self.images.append((self._ns(msg.header.stamp), img))

    def _match_image(self, stamp_ns: int):
        """Return the buffered image whose stamp equals (or is closest within
        tolerance to) stamp_ns. Same matching rule as overlay_node."""
        if not self.images:
            return None
        best, best_d = None, None
        for ns, img in self.images:
            if ns == stamp_ns:
                return img
            d = abs(ns - stamp_ns)
            if best_d is None or d < best_d:
                best, best_d = img, d
        return best if best_d is not None and best_d <= self.match_tol_ms * 1e6 else None

    # ---------------------------------------------------------------- main callback
    def on_tracks(self, msg: Detection2DArray):
        image = self._match_image(self._ns(msg.header.stamp))
        if image is None:
            self._miss += 1
            if self._miss % 30 == 1:
                self.get_logger().warn(
                    f'no image within tolerance for a tracks stamp (count={self._miss}); '
                    'passing tracks through unrefined')
            self.pub.publish(msg)   # degrade gracefully rather than drop the whole message
            return

        h_img, w_img = image.shape[:2]
        out = Detection2DArray()
        out.header = msg.header
        for d in msg.detections:
            original_label = d.results[0].hypothesis.class_id if d.results else None
            if original_label not in self.colorable:
                out.detections.append(d)
                continue

            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            x1 = max(0, int(cx - w / 2))
            y1 = max(0, int(cy - h / 2))
            x2 = min(w_img, int(cx + w / 2))
            y2 = min(h_img, int(cy + h / 2))
            crop = image[y1:y2, x1:x2]

            result = classify_color(crop, self.color_ranges, self.roi_shrink, self.min_confidence)
            vote = self.votes.setdefault(d.id, LabelVote(min_votes=self.min_votes))
            vote.add(result.label)

            refined_color = vote.current_best
            if refined_color is not None:
                refined_label = self.color_to_label.get(refined_color)
                if refined_label is not None and refined_label != original_label:
                    d.results[0].hypothesis.class_id = refined_label
            out.detections.append(d)

        self._gc_votes(seen={d.id for d in msg.detections})
        self.pub.publish(out)

    def _gc_votes(self, seen: set):
        """A track id absent from tracks_2d has been dropped by the ByteTracker
        itself (it always republishes every still-alive confirmed track, matched
        or not) -- so it's safe to drop that id's vote state here too."""
        for tid in [t for t in self.votes if t not in seen]:
            del self.votes[tid]


def main(args=None):
    rclpy.init(args=args)
    node = ColorClassificationNode()
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
