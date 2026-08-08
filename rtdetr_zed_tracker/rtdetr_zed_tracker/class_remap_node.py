"""ROS 2 node: RT-DETR Detection2DArray -> class-index remap -> Detection2DArray.

First of the two optional enrichment stages in tracking.launch.py, sitting between
the raw RT-DETR output and whatever consumes it next (color_classification_node
when that stage is on, otherwise depth_fusion_node). Collapses multi-class
detections to a single generic shape WITHOUT retraining the model and without any
downstream node knowing this stage exists -- the launch file just points the next
stage's input at this node's output instead of ``/detections_output``.

Index in, index out: this node never converts to label names, which is what keeps
it compatible with depth_fusion_node's ``int(class_id)``.

Never drops a detection -- class indices missing from the mapping pass through
with their original value unchanged (fail safe, not fail silent).
"""
from __future__ import annotations

import rclpy
import yaml
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

from .class_remap import remap_class


def _desc(text):
    return ParameterDescriptor(description=text)


class ClassRemapNode(Node):
    def __init__(self, **kwargs):
        super().__init__('class_remap_node', **kwargs)
        p = self.declare_parameter
        mapping_file = p('class_remap_file', '',
                         _desc('YAML: original class index -> canonical class index')).value
        qos_rel = str(p('input_qos_reliability', 'reliable',
                       _desc('Subscription reliability: reliable|best_effort')).value).lower()
        if qos_rel not in ('reliable', 'best_effort'):
            raise RuntimeError(
                f"input_qos_reliability must be 'reliable' or 'best_effort' (got: {qos_rel!r}) -- "
                'a typo here silently falling back to best_effort can drop detections unnoticed')

        if not mapping_file:
            raise RuntimeError("class_remap_node requires the 'class_remap_file' parameter")
        with open(mapping_file) as f:
            raw = yaml.safe_load(f) or {}
        self.mapping = {int(k): int(v) for k, v in raw.items()}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if qos_rel == 'reliable' else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10, durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(Detection2DArray, '~/detections_remapped', qos)
        self.sub = self.create_subscription(Detection2DArray, '~/detections_input', self.on_dets, qos)
        self._warned_unmapped = set()
        self.get_logger().info(f'class_remap_node up: {len(self.mapping)} mapped classes from {mapping_file}')

    def on_dets(self, msg: Detection2DArray):
        for d in msg.detections:
            for res in d.results:
                try:
                    original = int(res.hypothesis.class_id)
                except (ValueError, TypeError):
                    continue
                remapped = remap_class(original, self.mapping)
                if remapped == original and original not in self.mapping and original not in self._warned_unmapped:
                    self._warned_unmapped.add(original)
                    self.get_logger().warn(f'class index {original} not in class_remap_file -> passing through unchanged')
                res.hypothesis.class_id = str(remapped)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClassRemapNode()
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
