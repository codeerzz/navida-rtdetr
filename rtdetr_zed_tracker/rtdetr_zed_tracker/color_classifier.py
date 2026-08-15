"""Lighting-robust buoy color classification. Pure numpy/OpenCV, zero ROS imports.

Classifies a cropped BGR image region by thresholding in YCrCb rather than HSV/RGB:
luma (Y) carries almost all of the change caused by glare and shadow, while chroma
(Cr, Cb) stays comparatively stable. Keeping the Y range wide and deciding mostly on
Cr/Cb makes the color decision far less sensitive to lighting than a straight HSV
(with Value) or RGB threshold -- this is the fix for reflective-water / shadowed-buoy
misclassification.

Config format (see config/color_ranges.yaml)::

    <color_name>:
      - [y_min, y_max, cr_min, cr_max, cb_min, cb_max]
      - [y_min, y_max, cr_min, cr_max, cb_min, cb_max]   # optional 2nd+ range

A color may declare more than one range. This generalizes the "two ranges for one
troublesome color" fix into a single mechanism that works for any color, rather than
special-casing red the way plain-HSV segmentation typically does (there the 2nd range
is needed because Hue wraps at 0/180; here it earns its keep because a red buoy under
strong glare desaturates toward white, shifting Cr/Cb near the edge of a single tight
range -- same symptom, different cause, same fix).
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

# (y_min, y_max, cr_min, cr_max, cb_min, cb_max), each in [0, 255] (cv2.COLOR_BGR2YCrCb convention).
ColorRange = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ColorResult:
    """Result of classifying one crop. ``label`` is None when no color cleared
    ``min_confidence`` -- callers (e.g. a per-track vote) must treat that as
    "no observation", never as a wrong-but-confident guess."""

    label: str | None
    confidence: float
    ratios: dict[str, float] = field(default_factory=dict)


def load_color_ranges(path: str) -> dict[str, list[ColorRange]]:
    """Load ``{color: [[y_min,y_max,cr_min,cr_max,cb_min,cb_max], ...]}`` from YAML.

    Accepts either a single range (flat 6-element list) or a list of ranges per
    color, so a color that doesn't need the multi-range mechanism can stay terse.

    Validates each range is a 6-element list right here rather than letting a
    malformed one (a hand-edited calibration line missing a value, an emptied
    list, ...) surface later as a confusing unpack error deep inside
    ``classify_color`` on the first live frame.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    ranges: dict[str, list[ColorRange]] = {}
    for color, entry in raw.items():
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(
                f'color_ranges.yaml: "{color}" must be a 6-element range or a list of '
                f'6-element ranges, got: {entry!r}')
        rows = entry if isinstance(entry[0], (list, tuple)) else [entry]
        parsed: list[ColorRange] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                raise ValueError(
                    f'color_ranges.yaml: "{color}" range must have exactly 6 values '
                    f'[y_min,y_max,cr_min,cr_max,cb_min,cb_max], got: {row!r}')
            parsed.append(tuple(int(v) for v in row))
        ranges[color] = parsed
    return ranges


def _central_roi(shape: tuple[int, int], roi_shrink: float) -> tuple[int, int, int, int]:
    """Bounding box of the central ``roi_shrink`` fraction of a (h, w) shape.

    Same idea as depth_utils.sample_box_depth's ROI shrink: sampling only the
    middle of the box keeps background/reflection bleeding in at the box edges
    from ever getting a vote.
    """
    h, w = shape[:2]
    bw, bh = w * roi_shrink, h * roi_shrink
    x1 = int(max(0, round((w - bw) / 2)))
    y1 = int(max(0, round((h - bh) / 2)))
    x2 = int(min(w, round(x1 + bw)))
    y2 = int(min(h, round(y1 + bh)))
    return x1, y1, x2, y2


DEFAULT_MIN_CONFIDENCE = 0.12


def threshold_for(color: str, min_confidence) -> float:
    """The confidence a given color must clear.

    ``min_confidence`` is either one float for every color, or a mapping of
    per-color thresholds. Per-color exists because the colors are not equally
    risky to get wrong: a color whose range hugs neutral chroma (black) can be
    imitated by any dark, washed-out patch, so it should have to show more
    evidence than one sitting far out on the Cr axis (red, green).
    """
    if isinstance(min_confidence, Mapping):
        return float(min_confidence.get(color, DEFAULT_MIN_CONFIDENCE))
    return float(min_confidence)


def classify_color(bgr_crop: np.ndarray, color_ranges: dict[str, list[ColorRange]],
                   roi_shrink: float = 0.6,
                   min_confidence=DEFAULT_MIN_CONFIDENCE) -> ColorResult:
    """Classify the dominant configured color inside ``bgr_crop``.

    ``confidence`` is the winning color's in-range pixel ratio within the sampled
    ROI, in [0, 1]. If it doesn't clear the threshold for that color, ``label`` is
    None (uncertain) rather than a forced, possibly-wrong guess.

    ``min_confidence`` may be a single float (same bar for every color) or a
    mapping ``{color: threshold}`` -- see ``threshold_for``.

    Note the order: the highest-scoring color is picked FIRST, and only then does
    its threshold apply. So raising one color's threshold never lets a different
    color win by default; it only makes that color abstain when its own evidence
    is thin.
    """
    if bgr_crop is None or bgr_crop.size == 0 or not color_ranges:
        return ColorResult(None, 0.0, {})

    x1, y1, x2, y2 = _central_roi(bgr_crop.shape, roi_shrink)
    roi = bgr_crop[y1:y2, x1:x2]
    if roi.size == 0:
        return ColorResult(None, 0.0, {})

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    total = ycrcb.shape[0] * ycrcb.shape[1]

    ratios: dict[str, float] = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(ycrcb.shape[:2], dtype=bool)
        for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
            lo = np.array([y_min, cr_min, cb_min])
            hi = np.array([y_max, cr_max, cb_max])
            mask |= cv2.inRange(ycrcb, lo, hi).astype(bool)
        ratios[color] = float(mask.sum()) / total if total else 0.0

    best_color = max(ratios, key=ratios.get)
    best_ratio = ratios[best_color]
    label = best_color if best_ratio >= threshold_for(best_color, min_confidence) else None
    return ColorResult(label, best_ratio, ratios)


def suggest_range_from_samples(bgr_crops: list[np.ndarray], roi_shrink: float = 0.6,
                               percentile_low: float = 5.0, percentile_high: float = 95.0,
                               pad: int = 3) -> ColorRange:
    """Derive one YCrCb range that covers a set of REAL captured crops of the same object.

    This is the fix for placeholder ranges (guessed from synthetic swatches) not
    matching a real camera + real object: a camera's white balance/exposure and a
    real surface's pigment reflectance shift Y/Cr/Cb away from a flat synthetic
    patch, which shows up as low ``classify_color`` confidence on real footage.

    Pass one crop per lighting condition you want covered (direct light, shadow,
    glare, ...) -- each crop's per-channel [percentile_low, percentile_high] band
    is computed independently (robust to a few stray background/edge pixels
    within that one crop), then the bands are unioned across crops and padded by
    ``pad`` on each side. More/more-varied crops -> a wider, more robust range;
    a single crop just calibrates for that one lighting condition.
    """
    if not bgr_crops:
        raise ValueError('suggest_range_from_samples needs at least one crop')

    los, his = [], []
    for i, crop in enumerate(bgr_crops):
        if crop is None or crop.size == 0:
            warnings.warn(f'suggest_range_from_samples: skipping empty crop #{i}', stacklevel=2)
            continue
        x1, y1, x2, y2 = _central_roi(crop.shape, roi_shrink)
        roi = crop[y1:y2, x1:x2]
        if roi.size == 0:
            warnings.warn(
                f'suggest_range_from_samples: skipping crop #{i} -- roi_shrink={roi_shrink} '
                f'leaves an empty region for this crop size', stacklevel=2)
            continue
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        los.append(np.percentile(ycrcb, percentile_low, axis=0))
        his.append(np.percentile(ycrcb, percentile_high, axis=0))

    if not los:
        raise ValueError('suggest_range_from_samples: every provided crop was empty')

    lo = np.min(los, axis=0) - pad
    hi = np.max(his, axis=0) + pad
    lo = np.clip(lo, 0, 255)
    hi = np.clip(hi, 0, 255)
    y_min, cr_min, cb_min = (int(v) for v in lo)
    y_max, cr_max, cb_max = (int(v) for v in hi)
    return (y_min, y_max, cr_min, cr_max, cb_min, cb_max)


def ranges_overlap(a: ColorRange, b: ColorRange) -> bool:
    """True if two YCrCb ranges share any (Y, Cr, Cb) volume.

    A wide range from a contaminated calibration sample (background/other-object
    pixels pulled into the percentile band) tends to show up as overlap with
    another color's range -- e.g. a "red" range whose Cr lower bound sinks into
    "green" territory. Checking this catches that failure mode before it ships,
    rather than discovering it as "everything classifies as red" on the water.
    """
    (a_y0, a_y1, a_cr0, a_cr1, a_cb0, a_cb1) = a
    (b_y0, b_y1, b_cr0, b_cr1, b_cb0, b_cb1) = b
    return (max(a_y0, b_y0) <= min(a_y1, b_y1)
            and max(a_cr0, b_cr0) <= min(a_cr1, b_cr1)
            and max(a_cb0, b_cb0) <= min(a_cb1, b_cb1))
