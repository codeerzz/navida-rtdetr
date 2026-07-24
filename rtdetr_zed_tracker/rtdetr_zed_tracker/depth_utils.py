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


def sample_box_depth(depth_m, invalid, box_depth_xyxy, roi_shrink=0.4,
                     min_valid=0.3, max_valid=20.0, min_ratio=0.15, percentile=50.0):
    """Sample a robust range from the central ROI of a box in DEPTH-pixel space.

    Returns (z_or_None, valid_ratio, valid_bool, (u_center, v_center)).
    ``valid`` is False (and z None) when too few pixels carry a usable measurement
    -- the caller must NOT fabricate a position in that case.
    """
    h_img, w_img = depth_m.shape[:2]
    x1, y1, x2, y2 = box_depth_xyxy
    ucx = (x1 + x2) / 2.0
    vcy = (y1 + y2) / 2.0
    bw = (x2 - x1) * roi_shrink
    bh = (y2 - y1) * roi_shrink
    rx1 = int(max(0, np.floor(ucx - bw / 2)))
    ry1 = int(max(0, np.floor(vcy - bh / 2)))
    rx2 = int(min(w_img, np.ceil(ucx + bw / 2)))
    ry2 = int(min(h_img, np.ceil(vcy + bh / 2)))
    center = (ucx, vcy)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, 0.0, False, center

    roi = depth_m[ry1:ry2, rx1:rx2]
    inv = invalid[ry1:ry2, rx1:rx2]
    valid = (~inv) & (roi >= min_valid) & (roi <= max_valid)
    ratio = float(valid.mean()) if valid.size else 0.0
    if ratio < min_ratio or not valid.any():
        return None, ratio, False, center
    z = float(np.percentile(roi[valid], percentile))
    return z, ratio, True, center
