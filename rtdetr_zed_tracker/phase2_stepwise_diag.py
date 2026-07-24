#!/usr/bin/env python3
"""STEP 0 + STEP 1 on live data.

Captures /detections_output (raw) and /tracker_node/tracks_2d for a window, then:
  STEP 0: chain the dominant raw detection frame-to-frame (nearest-center) and
          count reversals in its center-y  -> exonerates/implicates RT-DETR.
  STEP 1: for the dominant tracked object, count reversals in center-y and report
          how many distinct track IDs it received (ID changes => association / H-A).

Run inside the container WITH the UDP profile, tracker running. Perform a slow,
steady, ONE-DIRECTION motion (e.g. move a buoy straight down) during the window.
Usage: python3 phase2_stepwise_diag.py [seconds]
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0


def qos(depth=50):
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                      depth=depth, durability=DurabilityPolicy.VOLATILE)


def ns(s):
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


class Cap(Node):
    def __init__(self):
        super().__init__('stepwise_diag')
        self.det = []    # (ns, [(cx,cy,score)])
        self.trk = []    # (ns, [(id,cx,cy)])
        self.create_subscription(Detection2DArray, '/detections_output', self.on_det, qos())
        self.create_subscription(Detection2DArray, '/tracker_node/tracks_2d', self.on_trk, qos())

    def on_det(self, m):
        self.det.append((ns(m.header.stamp),
                         [(d.bbox.center.position.x, d.bbox.center.position.y,
                           d.results[0].hypothesis.score if d.results else 0.0) for d in m.detections]))

    def on_trk(self, m):
        self.trk.append((ns(m.header.stamp),
                         [(d.id, d.bbox.center.position.x, d.bbox.center.position.y) for d in m.detections]))


def count_reversals(cys):
    """Reversals = sign flips vs the dominant direction of travel."""
    if len(cys) < 3:
        return 0, 0.0
    d = np.diff(cys)
    direction = np.sign(np.sum(d)) or 1.0        # dominant direction
    rev = int(np.sum(np.sign(d) == -direction))  # steps against dominant direction
    travel = float(cys[-1] - cys[0])
    return rev, travel


def chain_primary(frames):
    """Greedy nearest-center chain of the dominant raw detection across frames."""
    cys, prev = [], None
    for _, dets in frames:
        if not dets:
            continue
        if prev is None:
            cx, cy, _ = max(dets, key=lambda t: t[2])   # seed = highest score
        else:
            cx, cy, _ = min(dets, key=lambda t: (t[0] - prev[0]) ** 2 + (t[1] - prev[1]) ** 2)
        cys.append(cy); prev = (cx, cy)
    return cys


def main():
    rclpy.init(); n = Cap()
    print(f"capturing {DUR:.0f}s — MOVE ONE OBJECT STEADILY IN ONE DIRECTION NOW...")
    end = time.time() + DUR
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.2)

    print("\n" + "=" * 64)
    print(f"det frames={len(n.det)}  track frames={len(n.trk)}")

    # STEP 0
    raw_cys = chain_primary(n.det)
    rev0, travel0 = count_reversals(raw_cys)
    print("\n--- STEP 0 (RAW detections, dominant object) ---")
    print(f"frames={len(raw_cys)}  vertical_travel={travel0:.1f}px  REVERSALS={rev0}")
    if abs(travel0) < 40:
        print("  WARNING: little vertical travel -> motion too small; redo with a clear one-way move.")

    # STEP 1
    from collections import defaultdict
    by_id = defaultdict(list)
    for _, ts in n.trk:
        for tid, cx, cy in ts:
            by_id[tid].append(cy)
    print("\n--- STEP 1 (TRACKS) ---")
    if not by_id:
        print("  no tracks captured.")
    else:
        dom = max(by_id, key=lambda k: len(by_id[k]))
        rev1, travel1 = count_reversals(by_id[dom])
        print(f"distinct track ids seen: {len(by_id)}  -> {list(by_id.keys())}")
        print(f"dominant id={dom}  frames={len(by_id[dom])}  travel={travel1:.1f}px  REVERSALS={rev1}")
        print(f"\n>>> STEP 1 CASE: "
              f"{'ID UNSTABLE (multiple ids for the motion) -> H-A association' if len(by_id) > 1 else 'ID STABLE -> H-B/H-C/H-D'}")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
