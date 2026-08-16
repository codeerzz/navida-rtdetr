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

Additions validated against real buoy photos + hand-labeled ground truth
(scripts/test_buoy_folder.py --ground-truth) before landing here, NONE of which touch
the calibrated numbers in color_ranges.yaml -- they change how ``classify_color``
reads pixels, not what counts as "red"/"green"/"black":

  * Glare exclusion: a pixel whose Y is near-blown-out (specular highlight) is
    dropped from every color's mask AND from the ratio's denominator, instead of
    being allowed to silently drag Cr/Cb toward neutral and cost a color its vote.
  * Two-color-space voting: red gets an HSV hue check as a rescue/confirmation (OR)
    -- YCrCb alone can miss a red buoy that's desaturated toward white under glare.
    Green gets an HSV hue check as a REQUIRED confirmation (AND) -- water/algae can
    accidentally fall inside green's YCrCb band, and this cuts that false-positive
    without touching the calibrated range itself.
  * Gray-world white balance + CLAHE (``gray_world_white_balance``, ``clahe_luma``):
    optional whole-frame preprocessing a caller (color_classification_node) can run
    before cropping, to correct the sea's blue-green color cast and boost contrast
    in dark water -- these do not run inside ``classify_color`` itself, since they
    need the full frame, not just a crop, to work well.

Every addition is a keyword argument that defaults to "on" (it measurably helped,
never hurt, in that benchmark) but can be switched off individually, and the
previous, simpler implementation is kept directly below as a comment for a fast
revert if a real deployment ever disagrees with the benchmark.
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


# ── HSV yardımcıları: iki-uzay oylaması için ──────────────────────────────
# classify_color'ın YCrCb kararına ek kanıt/onay olarak kullanılır (aşağıya
# bak). Burada, tek başına HSV'ye geçmek için değil -- YCrCb hâlâ birincil
# karar mekanizması, HSV sadece belirli durumlarda devreye giriyor.

def _hsv_red_ratio(bgr: np.ndarray) -> float:
    """HSV uzayında kırmızı piksel oranı (Hue 0/180 civarındaki iki bant)."""
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0
    lo_s, lo_v = 60, 40
    mask1 = cv2.inRange(hsv, (0, lo_s, lo_v), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (165, lo_s, lo_v), (180, 255, 255))
    return float(cv2.countNonZero(cv2.bitwise_or(mask1, mask2))) / total


def _hsv_green_ratio(bgr: np.ndarray) -> float:
    """HSV uzayında yeşil piksel oranı (Hue 35-85 bandı)."""
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 0.0
    lo_s, lo_v = 60, 40
    mask = cv2.inRange(hsv, (35, lo_s, lo_v), (85, 255, 255))
    return float(cv2.countNonZero(mask)) / total


DEFAULT_GLARE_Y_THRESH = 245
DEFAULT_HSV_RED_MIN_RATIO = 0.10
DEFAULT_HSV_GREEN_MIN_RATIO = 0.10


def classify_color(bgr_crop: np.ndarray, color_ranges: dict[str, list[ColorRange]],
                   roi_shrink: float = 0.6,
                   min_confidence=DEFAULT_MIN_CONFIDENCE,
                   use_glare_mask: bool = True,
                   glare_y_thresh: int = DEFAULT_GLARE_Y_THRESH,
                   use_hsv_vote: bool = True,
                   hsv_red_min_ratio: float = DEFAULT_HSV_RED_MIN_RATIO,
                   hsv_green_min_ratio: float = DEFAULT_HSV_GREEN_MIN_RATIO) -> ColorResult:
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

    Two additions on top of the plain YCrCb decision (both default ON; see the
    module docstring for why -- validated on real buoy photos with zero precision
    cost). Both are no-ops for any color name other than "red"/"green", so a
    color_ranges.yaml without those two colors behaves exactly as before:

      use_glare_mask: pixels with Y >= glare_y_thresh (a blown-out specular
        highlight) are dropped from every color's mask AND from the ratio's
        denominator, instead of counting toward whichever color their
        washed-out, near-neutral chroma happens to land closest to.

      use_hsv_vote: an HSV hue check on the same (glare-filtered) region, used
        two different ways depending on the color, because red and green fail in
        opposite directions:
          - red:   RESCUE/CONFIRM (OR). If YCrCb is uncertain or already says
            red, and HSV agrees, that's accepted (or boosts confidence) -- a red
            buoy desaturated by glare can fall outside the tight YCrCb range
            while still reading clearly red in hue.
          - green: CONFIRM-ONLY (AND). If YCrCb's top pick is green but HSV does
            NOT agree, green is rejected -- water/algae can drift into green's
            YCrCb band on its own. On rejection, the runner-up color is tried
            (and used if IT independently clears its own threshold) before
            giving up and returning "uncertain": otherwise a real second-place
            red/black reading would be thrown away along with the bad green
            guess, for no reason -- it was never the problem.
    """
    if bgr_crop is None or bgr_crop.size == 0 or not color_ranges:
        return ColorResult(None, 0.0, {})

    x1, y1, x2, y2 = _central_roi(bgr_crop.shape, roi_shrink)
    roi = bgr_crop[y1:y2, x1:x2]
    if roi.size == 0:
        return ColorResult(None, 0.0, {})

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # --- ESKİ SÜRÜM (glare maskesi yok, HSV oylaması yok) --------------------
    # Geri dönmek istersen: bu bloğu aç, aşağıdaki YENİ bloğu (return'e kadar)
    # kapat. Ya da kod değiştirmeden use_glare_mask=False, use_hsv_vote=False
    # geçerek aynı eski davranışı elde edebilirsin.
    #
    # total = ycrcb.shape[0] * ycrcb.shape[1]
    # ratios: dict[str, float] = {}
    # for color, ranges in color_ranges.items():
    #     mask = np.zeros(ycrcb.shape[:2], dtype=bool)
    #     for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
    #         lo = np.array([y_min, cr_min, cb_min])
    #         hi = np.array([y_max, cr_max, cb_max])
    #         mask |= cv2.inRange(ycrcb, lo, hi).astype(bool)
    #     ratios[color] = float(mask.sum()) / total if total else 0.0
    #
    # best_color = max(ratios, key=ratios.get)
    # best_ratio = ratios[best_color]
    # label = best_color if best_ratio >= threshold_for(best_color, min_confidence) else None
    # return ColorResult(label, best_ratio, ratios)
    # --------------------------------------------------------------------------

    # --- YENİ: glare (parlama) piksellerini pay ve paydadan hariç tut ---
    if use_glare_mask:
        valid = ycrcb[:, :, 0] < glare_y_thresh
    else:
        valid = np.ones(ycrcb.shape[:2], dtype=bool)
    total = int(valid.sum())
    if total == 0:
        # ROI'nin tamamı parlama -- güvenilir bir okuma yok.
        return ColorResult(None, 0.0, {})

    ratios: dict[str, float] = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(ycrcb.shape[:2], dtype=bool)
        for y_min, y_max, cr_min, cr_max, cb_min, cb_max in ranges:
            lo = np.array([y_min, cr_min, cb_min])
            hi = np.array([y_max, cr_max, cb_max])
            mask |= cv2.inRange(ycrcb, lo, hi).astype(bool)
        mask &= valid
        ratios[color] = float(mask.sum()) / total

    base_colors = set(ratios)
    best_color = max(ratios, key=ratios.get)
    best_ratio = ratios[best_color]
    label = best_color if best_ratio >= threshold_for(best_color, min_confidence) else None
    conf = best_ratio

    # --- YENİ: iki-uzay (YCrCb + HSV) oylaması ---
    if use_hsv_vote:
        if 'red' in base_colors and (label == 'red' or label is None):
            hsv_ratio = _hsv_red_ratio(roi)
            if hsv_ratio >= hsv_red_min_ratio:
                if label is None:
                    label, conf = 'red', hsv_ratio
                    ratios = {**ratios, 'red_hsv': round(hsv_ratio, 3)}
                elif label == 'red':
                    conf = (conf + hsv_ratio) / 2.0
                    ratios = {**ratios, 'red_hsv': round(hsv_ratio, 3)}

        if 'green' in base_colors and label == 'green':
            hsv_ratio = _hsv_green_ratio(roi)
            ratios = {**ratios, 'green_hsv': round(hsv_ratio, 3)}
            if hsv_ratio < hsv_green_min_ratio:
                runner_up = max((c for c in base_colors if c != 'green'),
                                key=lambda c: ratios[c], default=None)
                if runner_up is not None and ratios[runner_up] >= threshold_for(runner_up, min_confidence):
                    label, conf = runner_up, ratios[runner_up]
                else:
                    label, conf = None, 0.0
            else:
                conf = (conf + hsv_ratio) / 2.0

    return ColorResult(label, conf, ratios)


# ── Ön işleme: gray-world beyaz dengesi + CLAHE ────────────────────────────
# classify_color'ın İÇİNDE kullanılmazlar (bir crop değil, TAM kareye ihtiyaç
# duyarlar) -- çağıran taraf (color_classification_node.on_image) kare gelir
# gelmez, crop'lanıp classify_color'a gitmeden ÖNCE bunları uygular.

def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Gray-world varsayımıyla kanal ortalamalarını eşitler.

    Deniz ortamında baskın mavi-yeşil renk cast'i kırmızıyı bastırır (kamera
    otomatik beyaz dengesi suyun rengini "nötr" sanıp kırmızıyı söndürür). Her
    kanalı, üç kanalın ortalaması ortak bir gri değere denk gelecek şekilde
    ölçekleyerek bu cast'i giderir.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    b *= gray_mean / max(b_mean, 1e-6)
    g *= gray_mean / max(g_mean, 1e-6)
    r *= gray_mean / max(r_mean, 1e-6)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def clahe_luma(bgr: np.ndarray, clip_limit: float = 2.0,
               tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE'yi SADECE Y (parlaklık) kanalına uygular, Cr/Cb dokunulmadan kalır.

    Düşük ışıkta kırmızı/yeşil arasındaki fark Cr/Cb'de zaten var ama Y çok
    düşükken piksel değerleri sıkışıp ayırt edilmesi zorlaşır. Y'yi yerel
    kontrastla açmak bunu düzeltir; Cr/Cb'ye dokunmamak renk kararının bu ön
    işlemden etkilenmemesini garanti eder.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    y_eq = clahe.apply(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)


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
