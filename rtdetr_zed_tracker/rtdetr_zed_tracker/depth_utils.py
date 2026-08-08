"""Pure depth math for the fusion node. No ROS imports -> unit-testable.

Two most dangerous silent failures (per project spec), both handled here:
  1. Encoding: 16UC1 has NO NaNs; invalid pixels are 0. A generic isfinite() mask
     reads every 0 as a valid 0 m and drags the median toward the camera. So the
     invalid mask MUST be encoding-specific.
  2. Intrinsics: if the depth image dims differ from the depth camera_info dims,
     fx/fy/cx/cy must be rescaled before deprojection, or X/Y are wrong while Z
     looks plausible.

Coordinate chain is TWO explicitly named steps (each unit tested):
    source-image px  --scale-->  depth-image px  --deproject-->  3D (optical frame)
"""
from __future__ import annotations

import numpy as np


def depth_to_metres(arr: np.ndarray, encoding: str):
    """Return (metres float64 array, invalid bool mask) for a raw depth image."""
    enc = encoding.lower()
    if enc == '32fc1':
        m = arr.astype(np.float64)
        invalid = ~np.isfinite(m)                 # NaN / +-inf are "no measurement"
        return m, invalid
    if enc in ('16uc1', 'mono16'):
        invalid = (arr == 0)                      # 0 == no measurement (there are NO NaNs here)
        m = arr.astype(np.float64) / 1000.0       # mm -> m
        return m, invalid
    raise ValueError(f'unsupported depth encoding: {encoding!r}')


def rescale_intrinsics(fx, fy, cx, cy, from_wh, to_wh):
    """Scale pinhole intrinsics from one image size to another."""
    sx = float(to_wh[0]) / float(from_wh[0])
    sy = float(to_wh[1]) / float(from_wh[1])
    return fx * sx, fy * sy, cx * sx, cy * sy


def source_box_to_depth_px(box_xyxy, src_wh, depth_wh):
    """STEP 1: map an xyxy box from source-image pixels to depth-image pixels."""
    sx = float(depth_wh[0]) / float(src_wh[0])
    sy = float(depth_wh[1]) / float(src_wh[1])
    x1, y1, x2, y2 = box_xyxy
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def deproject(u, v, z, fx, fy, cx, cy):
    """STEP 2: depth-image pixel (u,v) + range z -> (X,Y,Z) in the optical frame."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def sample_box_depth(depth_m, invalid, box_depth_xyxy, min_valid=0.3, max_valid=20.0,
                     min_ratio=0.15, min_valid_px=15, cluster_window_m=0.5):
    """Sample a robust range from the FULL box in DEPTH-pixel space.

    No spatial ROI shrink: a fixed "center 40% of the box" crop assumes the object
    sits in the middle and is big enough to leave enough pixels after cropping --
    both assumptions break for small/far/off-center boxes. Instead, every in-range
    valid pixel in the whole box is kept, then the depth is taken from the
    DENSEST cluster: the widest run of samples (sorted by depth) that all fall
    within ``cluster_window_m`` of each other. A tight, numerically-small group of
    real object pixels beats a numerically-larger but spatially spread group of
    background/noise pixels, so this stays correct even when the box loosely
    includes water/background rather than only the object.

    ``min_valid_px`` guards against a median computed from a handful of pixels
    (ratio alone is a weak signal for tiny boxes: 2/9 valid still clears a 15%
    ratio gate but is not a meaningful sample).

    Returns (z_or_None, valid_ratio, valid_bool, (u_center, v_center)).
    ``valid`` is False (and z None) when too few pixels carry a usable measurement
    -- the caller must NOT fabricate a position in that case.
    """
    h_img, w_img = depth_m.shape[:2]
    x1, y1, x2, y2 = box_depth_xyxy
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    rx1 = int(max(0, np.floor(x1)))
    ry1 = int(max(0, np.floor(y1)))
    rx2 = int(min(w_img, np.ceil(x2)))
    ry2 = int(min(h_img, np.ceil(y2)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None, 0.0, False, center

    box = depth_m[ry1:ry2, rx1:rx2]
    inv = invalid[ry1:ry2, rx1:rx2]
    in_range = (~inv) & (box >= min_valid) & (box <= max_valid)
    ratio = float(in_range.mean()) if in_range.size else 0.0
    vals = np.sort(box[in_range])
    if ratio < min_ratio or vals.size < min_valid_px:
        return None, ratio, False, center

    # Densest window of width cluster_window_m along the sorted depth axis.
    #
    # This is the two-pointer sweep, vectorised. The scalar version -- advance j
    # while vals[j+1] - vals[i] <= window, for each i -- is O(n), but n here is the
    # number of VALID PIXELS IN THE BOX, so a near buoy runs the loop >100k times
    # per detection per frame. Measured on Orin: 108 ms for a 320x480 box, which
    # cannot keep up with a 45 Hz pipeline (see scripts/benchmark_components.py).
    #
    # searchsorted does the identical thing in C: for each i it finds the first
    # index whose depth exceeds vals[i] + window, so that index minus i IS the
    # count the scalar loop would have arrived at. argmax takes the first maximum,
    # matching the strict `>` of the original. Same cluster, same median, ~100x
    # faster -- verified equal on 3000 randomised clustered/uniform scenes.
    hi = np.searchsorted(vals, vals + cluster_window_m, side='right')
    best_i = int(np.argmax(hi - np.arange(vals.size)))
    z = float(np.median(vals[best_i:int(hi[best_i])]))
    return z, ratio, True, center
