"""Unit tests for the YCrCb color classifier -- the fix for buoy color flipping
under water reflections/shadows/specular glare.

Runs anywhere with numpy/opencv/pyyaml/pytest installed. No ROS, no ZED, no GPU:

    pip install numpy opencv-python pyyaml pytest
    cd rtdetr_zed_tracker && pytest test/test_color_classifier.py -v

Most tests below load fixtures/color_ranges_test.yaml, NOT the real
config/color_ranges.yaml -- that one gets recalibrated against the actual robot
camera on the water (scripts/webcam_color_demo.py --calibrate) and must be free
to change without breaking this suite. Swatch BGR values were picked and their
actual YCrCb converted with cv2 (see the numbers in each test's comment) so the
assertions are grounded in real colorspace math, not guesses -- if you change
the fixture file, re-derive them the same way rather than hand-tuning until
tests pass.

The one exception is test_production_color_ranges_do_not_overlap, which
deliberately DOES load the real config/color_ranges.yaml: it's a regression
test for the "everything classifies as red" bug a contaminated calibration
sample caused in the field (a too-wide red range whose Cr lower bound sank into
green's Cr band). Keep it pointed at the real file.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import yaml
from rtdetr_zed_tracker.color_classifier import (
    classify_color,
    load_color_ranges,
    ranges_overlap,
    suggest_range_from_samples,
)

TEST_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(TEST_DIR, 'fixtures', 'color_ranges_test.yaml')
PRODUCTION_CONFIG_PATH = os.path.join(TEST_DIR, '..', 'config', 'color_ranges.yaml')
PRODUCTION_LABELS_PATH = os.path.join(TEST_DIR, '..', 'config', 'class_labels.yaml')


def solid_bgr(bgr, size=(120, 160)):
    """A solid-color synthetic crop, like a buoy filling most of its own bbox."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


@pytest.fixture(scope='module')
def color_ranges():
    return load_color_ranges(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Lighting robustness -- the actual bug this module fixes.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('bgr,expected', [
    ((30, 30, 200), 'red'),      # normal daylight red
    ((140, 140, 235), 'red'),    # same red, washed out by moderate glare
    ((10, 10, 90), 'red'),       # same red, in shadow
    ((40, 170, 40), 'green'),    # normal daylight green
    ((150, 220, 150), 'green'),  # same green, moderate glare
    ((10, 60, 10), 'green'),     # same green, in shadow
])
def test_same_buoy_classifies_the_same_across_lighting(color_ranges, bgr, expected):
    """THE bug: one buoy, one true color, three lighting conditions -> one label."""
    result = classify_color(solid_bgr(bgr), color_ranges)
    assert result.label == expected, f'BGR={bgr} misclassified under this lighting variant'


def test_specular_highlight_needs_the_second_red_range(color_ranges):
    """A near-blown-out glare spot on a red hull (Y~230, Cr/Cb pulled toward
    neutral 128) is exactly the case a single HSV/YCrCb range misses."""
    highlight_bgr = (230, 220, 250)
    result = classify_color(solid_bgr(highlight_bgr), color_ranges)
    assert result.label == 'red'

    # Prove *why*: with only the first (normal-light) red range, the same crop
    # must fail -- otherwise this test isn't demonstrating the fix does anything.
    single_range_only = {'red': [color_ranges['red'][0]], 'green': color_ranges['green']}
    degraded = classify_color(solid_bgr(highlight_bgr), single_range_only)
    assert degraded.label != 'red', 'single-range red should NOT catch the specular highlight'


# --------------------------------------------------------------------------- #
# Confidence / "uncertain" gating.
# --------------------------------------------------------------------------- #

def test_neutral_glare_patch_is_uncertain_not_a_wrong_color(color_ranges):
    """A fully blown-out white/gray patch (sky reflection, sun glint on water)
    must not be forced into red or green -- it must come back as 'uncertain'."""
    result = classify_color(solid_bgr((250, 250, 250)), color_ranges)
    assert result.label is None
    assert result.confidence < 0.12


def test_water_blue_is_not_confused_with_either_buoy_color(color_ranges):
    result = classify_color(solid_bgr((120, 60, 20)), color_ranges)  # murky blue-ish water
    assert result.label is None


# --------------------------------------------------------------------------- #
# Central-ROI shrink -- ignore a contaminated box edge.
# --------------------------------------------------------------------------- #

def test_roi_shrink_ignores_a_contaminating_border():
    """A red buoy whose bbox edge caught a strip of green (reflection/neighbor
    object) must still read as red once the center-only ROI is sampled."""
    ranges = load_color_ranges(CONFIG_PATH)
    img = solid_bgr((30, 30, 200), size=(100, 100))
    border = 15
    img[:border, :] = (40, 170, 40)
    img[-border:, :] = (40, 170, 40)
    img[:, :border] = (40, 170, 40)
    img[:, -border:] = (40, 170, 40)

    contaminated_full_frame = classify_color(img, ranges, roi_shrink=1.0)
    center_only = classify_color(img, ranges, roi_shrink=0.5)

    assert center_only.label == 'red'
    assert center_only.confidence > contaminated_full_frame.confidence


# --------------------------------------------------------------------------- #
# Config loading -- both terse (single range) and multi-range forms.
# --------------------------------------------------------------------------- #

def test_load_color_ranges_accepts_flat_and_nested_forms(tmp_path):
    yaml_text = """
    red:
      - [0, 255, 150, 255, 0, 140]
      - [0, 255, 130, 150, 118, 135]
    solid_flat:
      - [0, 255, 0, 255, 0, 255]
    """
    p = tmp_path / 'ranges.yaml'
    p.write_text(yaml_text)
    ranges = load_color_ranges(str(p))
    assert len(ranges['red']) == 2
    assert len(ranges['solid_flat']) == 1


def test_load_color_ranges_accepts_a_truly_flat_top_level_list(tmp_path):
    """`color: [y_min, y_max, cr_min, cr_max, cb_min, cb_max]` with no extra
    list-of-lists wrapper -- the terse single-range form from the docstring."""
    p = tmp_path / 'ranges.yaml'
    p.write_text('red: [0, 255, 150, 255, 0, 140]\n')
    ranges = load_color_ranges(str(p))
    assert ranges['red'] == [(0, 255, 150, 255, 0, 140)]


def test_load_color_ranges_rejects_an_empty_range_list(tmp_path):
    """Regression guard for a hand-edited calibration line left empty (e.g.
    `red: []` after a botched paste) -- must fail loudly at load time, not
    silently produce a 0-length range that only breaks on the first live frame."""
    p = tmp_path / 'ranges.yaml'
    p.write_text('red: []\ngreen: [20, 245, 60, 122, 75, 125]\n')
    with pytest.raises(ValueError, match='red'):
        load_color_ranges(str(p))


def test_load_color_ranges_rejects_a_range_with_wrong_length(tmp_path):
    """Regression guard for the most plausible real typo: miscounting values
    while hand-typing a calibrated range (e.g. 4 or 7 instead of 6)."""
    p = tmp_path / 'ranges.yaml'
    p.write_text('red: [[0, 255, 150, 255]]\n')   # only 4 of 6 values
    with pytest.raises(ValueError, match='red'):
        load_color_ranges(str(p))


def test_empty_or_missing_crop_returns_uncertain(color_ranges):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert classify_color(empty, color_ranges).label is None
    assert classify_color(None, color_ranges).label is None


# --------------------------------------------------------------------------- #
# Calibration from real captures -- the fix for placeholder ranges not
# matching a real camera + real object (this is what dropped confidence to
# ~0.3 on the actual robot buoy: synthetic-swatch ranges vs. real sensor color).
# --------------------------------------------------------------------------- #

def test_suggested_range_covers_the_samples_it_was_built_from():
    """A range built from N crops of the same real object must classify those
    same crops with much higher confidence than an unrelated guessed range."""
    samples = [
        solid_bgr((25, 35, 195)),   # slightly off from the placeholder guess
        solid_bgr((60, 55, 170)),   # a duller, more desaturated capture
        solid_bgr((15, 20, 140)),   # a darker capture (mild shadow)
    ]
    learned_range = suggest_range_from_samples(samples)
    learned_ranges = {'red': [learned_range]}

    for crop in samples:
        result = classify_color(crop, learned_ranges)
        assert result.label == 'red'
        assert result.confidence > 0.9, 'a range built from these exact samples should nearly saturate'


def test_more_varied_samples_widen_the_learned_range():
    narrow = suggest_range_from_samples([solid_bgr((30, 30, 200))])
    wide = suggest_range_from_samples([solid_bgr((30, 30, 200)), solid_bgr((140, 140, 235))])
    narrow_span = (narrow[1] - narrow[0]) + (narrow[3] - narrow[2]) + (narrow[5] - narrow[4])
    wide_span = (wide[1] - wide[0]) + (wide[3] - wide[2]) + (wide[5] - wide[4])
    assert wide_span >= narrow_span


def test_suggest_range_from_samples_rejects_empty_input():
    with pytest.raises(ValueError):
        suggest_range_from_samples([])


def test_suggest_range_from_samples_skips_empty_or_none_crops_with_a_warning():
    """A degenerate crop (None, or a zero-size ndarray from a bad ROI select)
    must be skipped, not crash cv2.cvtColor -- the good samples still count."""
    good = solid_bgr((30, 30, 200))
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.warns(UserWarning, match='skipping'):
        learned = suggest_range_from_samples([good, empty, None])
    assert learned == suggest_range_from_samples([good])


def test_suggest_range_from_samples_raises_when_every_crop_is_empty():
    with pytest.raises(ValueError, match='every provided crop was empty'):
        suggest_range_from_samples([np.zeros((0, 0, 3), dtype=np.uint8), None])


def test_suggest_range_from_samples_skips_a_crop_too_small_for_roi_shrink():
    """A tiny crop can have a non-empty shape but still shrink to an empty ROI
    -- must be skipped the same way, not raise from inside cv2.cvtColor."""
    good = solid_bgr((30, 30, 200))
    tiny = solid_bgr((30, 30, 200), size=(1, 1))
    with pytest.warns(UserWarning, match='skipping'):
        learned = suggest_range_from_samples([good, tiny], roi_shrink=0.1)
    assert learned == suggest_range_from_samples([good], roi_shrink=0.1)


# --------------------------------------------------------------------------- #
# Range-overlap detection -- catches a contaminated calibration sample before
# it ships as "everything on the water classifies as red".
# --------------------------------------------------------------------------- #

def test_disjoint_ranges_do_not_overlap():
    red = (20, 210, 155, 225, 85, 120)
    green = (20, 245, 60, 122, 75, 125)
    assert not ranges_overlap(red, green)


def test_a_too_wide_calibrated_range_is_flagged_as_overlapping():
    """Regression case: a real calibration run produced red Cr as low as 97,
    which crosses into green's Cr band (60-122) -- exactly the "everything is
    red" bug this check exists to catch before it reaches color_ranges.yaml."""
    contaminated_red = (33, 202, 97, 208, 108, 174)
    green = (20, 245, 60, 122, 75, 125)
    assert ranges_overlap(contaminated_red, green)


def test_production_color_ranges_do_not_overlap():
    """The real, field-calibrated config/color_ranges.yaml must never have two
    different colors' ranges overlapping -- that's what makes every buoy read
    as whichever color's range happens to be widest (the exact bug this test
    guards against; see test_a_too_wide_calibrated_range_is_flagged_as_overlapping
    for the incident that prompted it). Recalibrate with
    scripts/webcam_color_demo.py --calibrate <color> until this passes again."""
    ranges = load_color_ranges(PRODUCTION_CONFIG_PATH)
    colors = list(ranges)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1:]:
            for range_a in ranges[color_a]:
                for range_b in ranges[color_b]:
                    assert not ranges_overlap(range_a, range_b), (
                        f'"{color_a}" range {range_a} overlaps "{color_b}" range {range_b} '
                        f'in config/color_ranges.yaml -- recalibrate, see docstring above')


def test_every_production_colour_has_a_class_index():
    """Every colour in config/color_ranges.yaml needs a matching "<colour>_buoy"
    entry in config/class_labels.yaml, or the refinement is computed and then
    thrown away.

    color_classification_node decides in name space and writes the class INDEX
    back out. A colour with no configured index hits the node's fail-safe: it
    keeps the original class id and warns once (never writing a name onto an
    index-space topic, which would break depth_fusion_node's int()). That is the
    right behaviour, but from the outside it looks exactly like the colour
    refinement doing nothing at all -- silent except for one log line.

    This is a config-pairing test, not a code test: adding a colour to
    color_ranges.yaml is only half the job, and the other half is easy to forget.
    """
    ranges = load_color_ranges(PRODUCTION_CONFIG_PATH)
    with open(PRODUCTION_LABELS_PATH) as f:
        names = {str(v) for v in (yaml.safe_load(f) or {}).values()}

    missing = sorted(f'{c}_buoy' for c in ranges if f'{c}_buoy' not in names)
    assert not missing, (
        f'config/color_ranges.yaml defines {missing} but config/class_labels.yaml has no '
        f'index for them, so those refinements would be silently dropped. Append a new '
        f'index (do NOT reuse 0-6 -- those are what the model emits).')
