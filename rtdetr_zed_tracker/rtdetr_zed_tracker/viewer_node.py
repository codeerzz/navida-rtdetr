"""ROS 2 node: refreshing human-readable table of tracked objects.

Subscribes ~/tracked_objects (TrackedObjectArray) and reprints a table sorted by
forward distance (Z) ascending. Run it in its OWN terminal:

  ros2 run rtdetr_zed_tracker viewer_node --ros-args \
    -r /viewer_node/tracked_objects:=/depth_fusion_node/tracked_objects
"""
from __future__ import annotations

import math
import sys

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from rtdetr_zed_tracker_msgs.msg import TrackedObjectArray

HDR = f"{'ID':<14}{'CLASS':<12}{'CONF':>5}  {'Z(m)':>7} {'RANGE(m)':>9} {'VALID%':>7} {'AGE':>6}"


class ViewerNode(Node):
    def __init__(self):
        super().__init__('viewer_node')
        rate = self.declare_parameter(
            'refresh_hz', 4.0, ParameterDescriptor(description='table redraw rate (Hz)')).value
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         depth=5, durability=DurabilityPolicy.VOLATILE)
        self._objs = []
        self._count = 0
        self.create_subscription(TrackedObjectArray, '~/tracked_objects', self.on_objs, qos)
        self.create_timer(1.0 / max(rate, 0.5), self.render)
        self._tty = sys.stdout.isatty()

    def on_objs(self, msg: TrackedObjectArray):
        self._objs = list(msg.objects)
        self._count += 1

    @staticmethod
    def _sort_key(o):
        z = o.distance_z_m
        nan = math.isnan(z)
        return (nan, z if not nan else 0.0)          # valid first, ascending; NaN last

    def render(self):
        objs = sorted(self._objs, key=self._sort_key)
        rows = [HDR, '-' * len(HDR)]
        for o in objs:
            z = f'{o.distance_z_m:7.2f}' if not math.isnan(o.distance_z_m) else '     --'
            r = f'{o.distance_range_m:9.2f}' if not math.isnan(o.distance_range_m) else '       --'
            v = f'{o.depth_valid_ratio * 100:5.0f}%' if o.depth_valid else '   -- '
            rows.append(f'{o.track_id:<14}{o.class_name:<12}{o.confidence:>5.2f}  '
                        f'{z} {r} {v:>7} {o.age_frames:>6}')
        if not objs:
            rows.append('(no tracked objects)')

        out = f'tracked objects: {len(objs)}   (msgs={self._count}, sorted by Z ascending)\n' + '\n'.join(rows)
        if self._tty:
            sys.stdout.write('\033[2J\033[H' + out + '\n')   # clear + home for a live table
            sys.stdout.flush()
        else:
            print(out, flush=True)                            # plain (logs/pipes)


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
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
