"""Unit tests for the depth-fusion math (encoding masks + coordinate chain)."""
import numpy as np

from rtdetr_zed_tracker.depth_utils import (
    depth_to_metres, deproject, rescale_intrinsics, sample_box_depth, source_box_to_depth_px,
)


def test_encoding_16uc1_zero_is_invalid_not_zero_metres():
    """THE dangerous case: all-zeros 16UC1 ROI must be 'no depth', never 0.0 m."""
    raw = np.zeros((20, 20), dtype=np.uint16)
    m, invalid = depth_to_metres(raw, '16uc1')
    assert invalid.all(), "every 0 pixel must be flagged invalid in 16-bit mode"
    z, ratio, valid, _ = sample_box_depth(m, invalid, [0, 0, 20, 20])
    assert valid is False and z is None and ratio == 0.0


def test_encoding_16uc1_scales_mm_to_m():
    raw = np.full((10, 10), 2500, dtype=np.uint16)   # 2500 mm
    m, invalid = depth_to_metres(raw, '16uc1')
    assert not invalid.any()
    assert np.isclose(m[0, 0], 2.5)


def test_encoding_32fc1_nan_is_invalid():
    raw = np.full((10, 10), 3.0, dtype=np.float32)
    raw[0, 0] = np.nan
    raw[1, 1] = np.inf
    m, invalid = depth_to_metres(raw, '32fc1')
    assert invalid[0, 0] and invalid[1, 1]
    assert not invalid[5, 5]
    assert np.isclose(m[5, 5], 3.0)


def test_sample_ignores_out_of_range_and_uses_median():
    raw = np.full((40, 40), 3000, dtype=np.uint16)   # 3 m
    raw[:, :20] = 0                                    # left half invalid
    m, invalid = depth_to_metres(raw, '16uc1')
    z, ratio, valid, center = sample_box_depth(m, invalid, [0, 0, 40, 40])
    assert valid and np.isclose(z, 3.0)
    assert 0.4 < ratio < 0.6                           # ~half valid


def test_min_ratio_gate_rejects_sparse_depth():
    raw = np.zeros((40, 40), dtype=np.uint16)
    raw[0, 0] = 3000                                   # a single valid pixel
    m, invalid = depth_to_metres(raw, '16uc1')
    z, ratio, valid, _ = sample_box_depth(m, invalid, [0, 0, 40, 40], min_ratio=0.15)
    assert valid is False and z is None


def test_min_valid_px_gate_rejects_tiny_sample_even_if_ratio_passes():
    """A 3x3 box with 1 valid pixel is ratio=0.11 (already fails), but a 3x3 box
    with ALL 9 pixels valid still isn't a meaningful sample -- min_valid_px must
    reject it independent of ratio."""
    raw = np.full((3, 3), 3000, dtype=np.uint16)       # 9/9 valid, ratio=1.0
    m, invalid = depth_to_metres(raw, '16uc1')
    z, ratio, valid, _ = sample_box_depth(m, invalid, [0, 0, 3, 3], min_valid_px=15)
    assert valid is False and z is None and ratio == 1.0


def test_dense_cluster_wins_over_larger_spread_group():
    """The whole box is scanned (no center-crop), so it can include a spread-out
    background/noise group alongside the real object. A plain median over all
    pixels would land inside the spread group's range and be wrong; picking the
    densest cluster instead should lock onto the tight buoy group even though it
    has fewer pixels than the spread group."""
    vals = np.concatenate([
        np.full(15, 2.0),                      # buoy: tight cluster at 2.0 m
        np.linspace(5.0, 10.0, 20),             # background/noise: spread thin
    ]).astype(np.float32).reshape(5, 7)
    invalid = np.zeros_like(vals, dtype=bool)
    m = vals.astype(np.float64)
    z, ratio, valid, _ = sample_box_depth(m, invalid, [0, 0, 7, 5])
    assert valid
    assert np.isclose(z, 2.0, atol=0.05), f'expected the dense buoy cluster (2.0m), got {z}'


def test_coordinate_step1_source_to_depth_scale():
    # source 1280x720 box -> depth 640x360 (half res)
    box = source_box_to_depth_px([100, 200, 300, 400], (1280, 720), (640, 360))
    assert np.allclose(box, [50, 100, 150, 200])


def test_coordinate_step2_deproject_center_is_zero_xy():
    # a point at the principal point deprojects to (0,0,z)
    x, y, z = deproject(320.0, 240.0, 5.0, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert np.isclose(x, 0.0) and np.isclose(y, 0.0) and np.isclose(z, 5.0)


def test_deproject_offaxis_sign():
    # a pixel right of center -> +X ; below center -> +Y (optical frame)
    x, y, _ = deproject(420.0, 340.0, 4.0, fx=400.0, fy=400.0, cx=320.0, cy=240.0)
    assert x > 0 and y > 0


def test_rescale_intrinsics_half():
    fx, fy, cx, cy = rescale_intrinsics(1000.0, 1000.0, 640.0, 360.0, (1280, 720), (640, 360))
    assert (fx, fy, cx, cy) == (500.0, 500.0, 320.0, 180.0)
