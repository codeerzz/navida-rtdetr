#!/usr/bin/env python3
"""Micro-benchmark the pure compute in rtdetr_zed_tracker (no ROS, no DDS).

  python3 src/rtdetr_zed_tracker/scripts/benchmark_components.py

Only depth_utils is left to measure: the ByteTrack layer (bbox_utils, kalman_box,
byte_tracker) was removed when identity moved to the map, so the container's whole
per-frame compute budget is now depth sampling and deprojection.

The number that matters is sample_box_depth. It scans EVERY valid pixel in the box,
so cost scales with box AREA rather than with object count -- a big near buoy is the
expensive case, not a crowded scene. The densest-cluster sweep is vectorised
(searchsorted); the scalar two-pointer loop it replaced cost 108 ms on a 320x480 box
and could not keep up with a 45 Hz pipeline.
"""
import statistics
import sys
import timeit

import numpy as np

sys.path.insert(0, '/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker')
from rtdetr_zed_tracker import depth_utils as du          # noqa: E402


def bench(label, fn, number=200, extra=''):
    times = timeit.repeat(fn, number=number, repeat=7)
    per = statistics.median(times) / number
    unit, val = ('µs', per * 1e6) if per < 1e-3 else ('ms', per * 1e3)
    hz = 1.0 / per
    print(f'  {label:<44} {val:8.2f} {unit}   {hz:10,.0f} Hz  {extra}')


def scene(box_w, box_h, obj_frac=0.45, obj_z=6.0, bg_z=17.0, invalid_frac=0.25):
    """A depth patch shaped like a real detection: a buoy at obj_z occupying part of
    the box, water/background at bg_z behind it, plus a band of invalid pixels."""
    rng = np.random.default_rng(0)
    depth = np.full((box_h, box_w), bg_z, dtype=np.float32)
    depth += rng.normal(0, 0.8, depth.shape).astype(np.float32)
    oh, ow = int(box_h * obj_frac), int(box_w * obj_frac)
    y0, x0 = (box_h - oh) // 2, (box_w - ow) // 2
    depth[y0:y0 + oh, x0:x0 + ow] = obj_z + rng.normal(0, 0.05, (oh, ow)).astype(np.float32)
    invalid = rng.random(depth.shape) < invalid_frac
    return depth, invalid


print('rtdetr_zed_tracker component micro-benchmark   (median per-call)')
print('=' * 92)

print('\n[ depth_utils / geometry ]')
bench('rescale_intrinsics', lambda: du.rescale_intrinsics(658.8, 658.8, 651.9, 343.5,
                                                          (1280, 720), (640, 360)))
bench('source_box_to_depth_px', lambda: du.source_box_to_depth_px(
    [100.0, 100.0, 260.0, 340.0], (1280, 720), (640, 360)))
bench('deproject (1 pt)', lambda: du.deproject(320.0, 180.0, 6.0, 658.8, 658.8, 651.9, 343.5))

print('\n[ depth_to_metres  (full frame decode) ]')
raw16 = (np.random.default_rng(1).integers(0, 20000, (720, 1280))).astype(np.uint16)
bench('depth_to_metres 1280x720 16UC1', lambda: du.depth_to_metres(raw16, '16UC1'), number=50)

print('\n[ sample_box_depth  (whole-box scan + vectorised densest-cluster sweep) ]')
print('  cost tracks BOX AREA, not object count -- every valid pixel is scanned')
for w, h, note in ((40, 60, 'far/small buoy   ~2.4 k px'),
                   (80, 120, 'mid buoy         ~9.6 k px'),
                   (160, 240, 'near buoy        ~38 k px'),
                   (320, 480, 'very near buoy   ~154 k px')):
    d, inv = scene(w, h)
    box = [0.0, 0.0, float(w), float(h)]
    z, ratio, ok, _ = du.sample_box_depth(d, inv, box)
    got = f'z={z:.2f}m ratio={ratio:.2f}' if ok else f'INVALID ratio={ratio:.2f}'
    bench(f'{w}x{h}  {note}', lambda d=d, inv=inv, box=box: du.sample_box_depth(d, inv, box),
          number=(200 if w < 200 else 40), extra=got)

print('\n[ per-frame estimate ]')
d, inv = scene(80, 120)
box = [0.0, 0.0, 80.0, 120.0]


def per_frame_7():
    du.depth_to_metres(raw16, '16UC1')
    for _ in range(7):
        du.sample_box_depth(d, inv, box)
        du.deproject(320.0, 180.0, 6.0, 658.8, 658.8, 651.9, 343.5)


bench('decode + 7 mid-size boxes', per_frame_7, number=30, extra='(compute budget/frame)')
print('\nNote: depth_to_metres dominates and is per-FRAME, not per-box.')
