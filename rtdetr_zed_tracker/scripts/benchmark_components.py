#!/usr/bin/env python3
"""Micro-benchmark every pure compute component of rtdetr_zed_tracker.

Runs each function in isolation with realistic buoy-scene inputs and reports the
per-call cost (median of many runs) plus the implied max frequency. Pure Python /
numpy only -- no ROS, no hardware. Run inside the container:

  python3 src/rtdetr_zed_tracker/scripts/benchmark_components.py
"""
import statistics
import sys
import timeit

import numpy as np

sys.path.insert(0, '/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker')
from rtdetr_zed_tracker import bbox_utils as bb          # noqa: E402
from rtdetr_zed_tracker.kalman_box import KalmanBox       # noqa: E402
from rtdetr_zed_tracker.byte_tracker import ByteTracker   # noqa: E402
from rtdetr_zed_tracker import depth_utils as du          # noqa: E402

SRC_WH = (1280, 720)
NET_WH = (640, 640)
DEPTH_WH = (1280, 720)


def bench(label, stmt, number, repeat=7, extra=''):
    """Time `stmt` (callable). Report median per-call in us + implied Hz."""
    times = timeit.repeat(stmt, number=number, repeat=repeat)
    per_call = statistics.median(times) / number          # seconds
    us = per_call * 1e6
    hz = 1.0 / per_call if per_call > 0 else float('inf')
    print(f'{label:<34} {us:>10.2f} us/call   {hz:>12,.0f} Hz   {extra}')


def make_boxes(n, wh=SRC_WH):
    w, h = wh
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, w - 60, n); y1 = rng.uniform(0, h - 60, n)
    return np.stack([x1, y1, x1 + 50, y1 + 50], axis=1)


def make_detections(n):
    boxes = make_boxes(n)
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.6, 0.95, n)
    classes = rng.integers(0, 7, n)
    return [(boxes[i], float(scores[i]), int(classes[i])) for i in range(n)]


def warm_tracker(n_obj, frames=6):
    """Return a ByteTracker with ~n_obj confirmed, stable tracks."""
    trk = ByteTracker()
    base = make_detections(n_obj)
    for _ in range(frames):
        jit = [(b[0] + np.random.default_rng().normal(0, 0.5, 4), b[1], b[2]) for b in base]
        trk.update(jit, dt=1 / 30)
    return trk, base


print('=' * 92)
print('rtdetr_zed_tracker component micro-benchmark   (median per-call, buoy-scene inputs)')
print('=' * 92)

# ---- bbox / geometry ------------------------------------------------------
print('\n[ bbox_utils / geometry ]')
for n in (7, 20, 50):
    A = make_boxes(n); B = make_boxes(n)
    bench(f'iou_matrix  {n}x{n}', lambda A=A, B=B: bb.iou_matrix(A, B), number=2000,
          extra=f'({n} tracks x {n} dets)')
boxes7 = make_boxes(7, NET_WH)
bench('inverse_letterbox  7 boxes', lambda: bb.inverse_letterbox(boxes7, SRC_WH, NET_WH), number=5000)
bench('xyxy_to_xyah  7 boxes', lambda: bb.xyxy_to_xyah(boxes7), number=10000)
bench('xyah_to_xyxy  7 boxes', lambda: bb.xyah_to_xyxy(bb.xyxy_to_xyah(boxes7)), number=10000)

# ---- Kalman ---------------------------------------------------------------
print('\n[ kalman_box ]')
kf = KalmanBox(np.array([640., 360., 1.0, 50.]))
bench('KalmanBox.predict (1 track)', lambda: kf.predict(1 / 30), number=20000)
kf2 = KalmanBox(np.array([640., 360., 1.0, 50.]))
bench('KalmanBox.update  (1 track)',
      lambda: kf2.update(np.array([641., 361., 1.0, 50.])), number=20000)


def kf_bank(n):
    banks = [KalmanBox(np.array([100. + i, 200., 1.0, 50.])) for i in range(n)]
    m = np.array([101., 201., 1.0, 50.])
    def step():
        for k in banks:
            k.predict(1 / 30); k.update(m)
    return step


bench('KF bank predict+update  7', kf_bank(7), number=3000, extra='(7 tracks, full cycle)')

# ---- ByteTracker (the hot path) ------------------------------------------
print('\n[ byte_tracker  (full frame: predict + 2-stage assoc + KF) ]')
for n in (7, 20, 50):
    trk, base = warm_tracker(n)
    def frame(trk=trk, base=base):
        jit = [(b[0] + np.random.default_rng().normal(0, 0.5, 4), b[1], b[2]) for b in base]
        trk.update(jit, dt=1 / 30)
    bench(f'ByteTracker.update  {n} objs', frame, number=1000, extra=f'({n} tracked objects/frame)')

# ---- depth ----------------------------------------------------------------
print('\n[ depth_utils ]')
depth16 = (np.random.default_rng(2).uniform(300, 8000, DEPTH_WH[::-1])).astype(np.uint16)
depth16[::7, ::7] = 0                                   # scatter invalid
bench('depth_to_metres  1280x720 16UC1',
      lambda: du.depth_to_metres(depth16, '16uc1'), number=500,
      extra=f'({DEPTH_WH[0]}x{DEPTH_WH[1]} = {DEPTH_WH[0]*DEPTH_WH[1]/1e6:.2f} MPix)')
depth_m, invalid = du.depth_to_metres(depth16, '16uc1')
box_d = np.array([600., 320., 700., 420.])
bench('sample_box_depth  (1 box ROI)',
      lambda: du.sample_box_depth(depth_m, invalid, box_d), number=5000)


def sample_n(n):
    bxs = make_boxes(n, DEPTH_WH)
    def step():
        for b in bxs:
            du.sample_box_depth(depth_m, invalid, b)
    return step


bench('sample_box_depth  7 boxes', sample_n(7), number=2000, extra='(per-frame depth sampling)')
bench('rescale_intrinsics', lambda: du.rescale_intrinsics(700, 700, 640, 360, SRC_WH, DEPTH_WH),
      number=20000)
bench('source_box_to_depth_px', lambda: du.source_box_to_depth_px(box_d, SRC_WH, DEPTH_WH),
      number=20000)
bench('deproject  (1 point)', lambda: du.deproject(640, 360, 3.2, 700, 700, 640, 360), number=50000)

# ---- end-to-end pure pipeline per frame ----------------------------------
print('\n[ end-to-end pure per-frame (tracker + 7-box depth sample), excludes ROS/DDS ]')
trk7, base7 = warm_tracker(7)
bxs7 = make_boxes(7, DEPTH_WH)


def e2e():
    jit = [(b[0] + np.random.default_rng().normal(0, 0.5, 4), b[1], b[2]) for b in base7]
    trk7.update(jit, dt=1 / 30)
    for b in bxs7:
        du.sample_box_depth(depth_m, invalid, b)


bench('tracker.update + 7-box depth', e2e, number=800, extra='(compute budget/frame)')
print('=' * 92)
