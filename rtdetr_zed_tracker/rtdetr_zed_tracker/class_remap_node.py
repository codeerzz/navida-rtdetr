"""ROS 2 node: RT-DETR Detection2DArray -> class-index remap -> Detection2DArray.

Sits between the raw RT-DETR output and tracker_node (see class_remap.yaml's
default_value in tracking.launch.py) so multi-class detections can be collapsed
to a single generic shape WITHOUT retraining the model or touching tracker_node
itself: point tracker_node's ``detections_topic`` at this node's output instead
of ``/detections_output`` directly.

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
        qos_rel = p('input_qos_reliability', 'reliable',
                    _desc('Subscription reliability: reliable|best_effort')).value

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
