"""Vectorized bounding-box utilities. Pure numpy, zero ROS imports.

Box convention throughout the tracker: xyxy = [x1, y1, x2, y2] in image pixels.
The Kalman filter works in xyah = [cx, cy, aspect, height] where aspect = w / h.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """IoU between every pair. boxes_a (Na,4), boxes_b (Nb,4) xyxy -> (Na,Nb)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])   # (Na,Nb,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def xyxy_to_xyah(boxes) -> np.ndarray:
    """(N,4) xyxy -> (N,4) [cx, cy, aspect=w/h, height]."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    cx = b[:, 0] + w / 2.0
    cy = b[:, 1] + h / 2.0
    h_safe = np.where(h == 0, 1e-6, h)
    return np.stack([cx, cy, w / h_safe, h], axis=1)


def xyah_to_xyxy(z) -> np.ndarray:
    """(N,4) [cx, cy, aspect, height] -> (N,4) xyxy."""
    a = np.asarray(z, dtype=np.float64).reshape(-1, 4)
    h = a[:, 3]
    w = a[:, 2] * h
    x1 = a[:, 0] - w / 2.0
    y1 = a[:, 1] - h / 2.0
    x2 = a[:, 0] + w / 2.0
    y2 = a[:, 1] + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def inverse_letterbox(boxes, src_wh, net_wh, padding_mode: str = 'top_left') -> np.ndarray:
    """Map boxes from padded network space back to original image pixels.

    The forward pipeline resizes the source image by a uniform scale
    ``s = min(net_w/src_w, net_h/src_h)`` (aspect preserved) then pads to
    net_w x net_h. ``padding_mode`` says where the padding went:
      * 'top_left'  : image anchored top-left, pad on bottom/right (Isaac ROS
                      PadNode 'BOTTOM_RIGHT') -> zero offset.
      * 'center'    : image centered, pad split on all sides (classic letterbox).
    Boxes are clipped to the source image bounds.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if b.shape[0] == 0:
        return b

    src_w, src_h = float(src_wh[0]), float(src_wh[1])
    net_w, net_h = float(net_wh[0]), float(net_wh[1])
    s = min(net_w / src_w, net_h / src_h)

    if padding_mode == 'top_left':
        pad_x = pad_y = 0.0
    elif padding_mode == 'center':
        pad_x = (net_w - src_w * s) / 2.0
        pad_y = (net_h - src_h * s) / 2.0
    else:
        raise ValueError(f"padding_mode must be 'top_left' or 'center', got {padding_mode!r}")

    b[:, [0, 2]] = (b[:, [0, 2]] - pad_x) / s
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_y) / s
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, src_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, src_h)
    return b
