#!/usr/bin/env python3
"""Phase 0 THROWAWAY diagnostic — no implementation code, discovery only.

Subscribes to the color image feeding RT-DETR, /detections_output, and the ZED
depth topic. Over a fixed window it classifies:
  * timestamp propagation of detections vs the source image (A exact / B nearest / C zeroed-or-now)
  * end-to-end data age (now - detection.stamp) = age of data the tracker will see
  * detection<->depth stamp alignment (feasibility of Phase-4 sync)
  * whether depth actually streams, and its encoding/dims
  * the distinct raw class_id values that appear

Run:  python3 phase0_diag.py [duration_sec]
"""
import sys
import time
import statistics as st

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

IMAGE_TOPIC = "/zed_node/left/image_rect_color"
DET_TOPIC = "/detections_output"
DEPTH_TOPIC = "/zed_node/depth/depth_registered"
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def reliable_qos(depth=10):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


class Diag(Node):
    def __init__(self):
        super().__init__("phase0_diag")
        self.img_stamps = []          # (stamp_ns, wall_arrival)
        self.img_stamp_set = set()
        self.depth_stamps = []        # stamp_ns
        self.depth_count = 0
        self.depth_meta = None        # (encoding, w, h)
        self.det_records = []         # dicts
        self.class_ids = {}           # raw class_id -> count
        self.n_det_msgs = 0
        self.n_det_objs = 0

        self.create_subscription(Image, IMAGE_TOPIC, self.on_img, reliable_qos(30))
        self.create_subscription(Detection2DArray, DET_TOPIC, self.on_det, reliable_qos(30))
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, reliable_qos(10))
        self.t0 = time.time()
        self.get_logger().info(
            f"Collecting {DURATION:.0f}s on:\n  img   {IMAGE_TOPIC}\n  det   {DET_TOPIC}\n  depth {DEPTH_TOPIC}")

    def on_img(self, msg: Image):
        ns = stamp_ns(msg.header.stamp)
        self.img_stamps.append((ns, time.time()))
        self.img_stamp_set.add(ns)
        if len(self.img_stamps) > 3000:
            old = self.img_stamps.pop(0)
            self.img_stamp_set.discard(old[0])

    def on_depth(self, msg: Image):
        self.depth_count += 1
        self.depth_stamps.append(stamp_ns(msg.header.stamp))
        if self.depth_meta is None:
            self.depth_meta = (msg.encoding, msg.width, msg.height)

    def on_det(self, msg: Detection2DArray):
        now = time.time()
        ns = stamp_ns(msg.header.stamp)
        self.n_det_msgs += 1
        self.n_det_objs += len(msg.detections)
        for d in msg.detections:
            if d.results:
                cid = d.results[0].hypothesis.class_id
                self.class_ids[cid] = self.class_ids.get(cid, 0) + 1
        # nearest image stamp
        nearest_img = None
        if self.img_stamps:
            nearest_img = min((s for s, _ in self.img_stamps), key=lambda s: abs(s - ns))
        nearest_depth = None
        if self.depth_stamps:
            nearest_depth = min(self.depth_stamps, key=lambda s: abs(s - ns))
        self.det_records.append({
            "ns": ns,
            "age_ms": (now - ns / 1e9) * 1e3,
            "exact_img": ns in self.img_stamp_set,
            "d_img_ms": None if nearest_img is None else (ns - nearest_img) / 1e6,
            "d_depth_ms": None if nearest_depth is None else (ns - nearest_depth) / 1e6,
            "n": len(msg.detections),
        })

    def report(self):
        el = time.time() - self.t0
        p = print
        p("\n" + "=" * 64)
        p("PHASE 0 DIAGNOSTIC REPORT")
        p("=" * 64)
        # rates
        p(f"elapsed: {el:.1f}s")
        p(f"image   msgs: {len(self.img_stamps):5d}  ({len(self.img_stamps)/el:5.1f} Hz)  {IMAGE_TOPIC}")
        p(f"detect  msgs: {self.n_det_msgs:5d}  ({self.n_det_msgs/el:5.1f} Hz)  objs={self.n_det_objs}")
        p(f"depth   msgs: {self.depth_count:5d}  ({self.depth_count/el:5.1f} Hz)  {DEPTH_TOPIC}")
        if self.depth_meta:
            p(f"depth   meta: encoding={self.depth_meta[0]} {self.depth_meta[1]}x{self.depth_meta[2]}")
        else:
            p("depth   meta: >>> NO DEPTH MESSAGES RECEIVED (depth not streaming) <<<")
        # class ids
        p(f"\nclass_id raw values seen: {self.class_ids if self.class_ids else '(none - empty scene?)'}")
        # timestamp classification
        p("\n--- timestamp propagation (detection vs source image) ---")
        if not self.det_records:
            p("INCONCLUSIVE: no detections arrived. /detections_output only publishes")
            p("when >=1 object is detected — need a buoy (or trigger object) in view.")
        else:
            exact = sum(1 for r in self.det_records if r["exact_img"])
            d_img = [r["d_img_ms"] for r in self.det_records if r["d_img_ms"] is not None]
            ages = [r["age_ms"] for r in self.det_records]
            frame_ms = (el / len(self.img_stamps) * 1e3) if self.img_stamps else 33.3
            p(f"detections analysed: {len(self.det_records)}")
            p(f"exact stamp match : {exact}/{len(self.det_records)}")
            if d_img:
                p(f"det-img delta ms  : min={min(d_img):.2f} med={st.median(d_img):.2f} max={max(d_img):.2f}")
            p(f"frame period ms   : {frame_ms:.2f} (half={frame_ms/2:.2f})")
            # classify
            near_zero_stamp = sum(1 for r in self.det_records if r["ns"] < 1_000_000_000)
            if exact == len(self.det_records):
                cls = "A (exact) -> exact-stamp dict lookup"
            elif near_zero_stamp > 0:
                cls = "C (zeroed/now) -> order-based FIFO correlation"
            elif d_img and max(abs(x) for x in d_img) <= frame_ms / 2:
                cls = "B (small consistent offset) -> nearest-stamp within half frame"
            else:
                cls = "B/uncertain -> nearest-stamp; inspect deltas above"
            p(f">>> CLASS: {cls}")
            p(f"\n--- end-to-end data age (now - det.stamp) ---")
            p(f"age ms: min={min(ages):.1f} med={st.median(ages):.1f} max={max(ages):.1f}")
            d_depth = [r["d_depth_ms"] for r in self.det_records if r["d_depth_ms"] is not None]
            if d_depth:
                p(f"\n--- detection vs depth stamp (Phase-4 sync feasibility) ---")
                p(f"det-depth delta ms: min={min(d_depth):.2f} med={st.median(d_depth):.2f} max={max(d_depth):.2f}")
        p("=" * 64 + "\n")


def main():
    rclpy.init()
    node = Diag()
    end = time.time() + DURATION
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
