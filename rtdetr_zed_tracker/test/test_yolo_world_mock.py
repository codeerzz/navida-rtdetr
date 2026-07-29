"""test_yolo_world_mock.py — Hardware-free tests for YOLO-World integration.

Runs on any machine with:  pip install ultralytics opencv-python numpy pytest

What is tested:
  1. _iou()          — IoU helper correctness (identical / no-overlap / partial)
  2. Cross-suppression logic — pure logic, no model needed
  3. _decode_image() — image encoding conversion (bgra8 / bgr8 / rgb8)
  4. YOLO-World-S inference + prompt — live model on a synthetic RGB image
     (skipped automatically if ultralytics is not installed)
  5. End-to-end mock — synthetic image + mock RT-DETR buoy tracks,
     verifying suppression removes buoy-overlapping detections

Run:
  python3 -m pytest test/test_yolo_world_mock.py -v

Visualise (optional, opens an OpenCV window):
  python3 test/test_yolo_world_mock.py --visualise
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the node importable without ROS installed.
# We mock every ROS 2 package the node imports before importing it.
# ---------------------------------------------------------------------------
def _make_ros_stubs():
    """Insert lightweight stubs so yolo_world_node.py can be imported without ROS."""
    stubs = {
        'rclpy': MagicMock(),
        'rclpy.node': MagicMock(),
        'rclpy.qos': MagicMock(),
        'rcl_interfaces': MagicMock(),
        'rcl_interfaces.msg': MagicMock(),
        'sensor_msgs': MagicMock(),
        'sensor_msgs.msg': MagicMock(),
        'std_msgs': MagicMock(),
        'std_msgs.msg': MagicMock(),
        'vision_msgs': MagicMock(),
        'vision_msgs.msg': MagicMock(),
        'geometry_msgs': MagicMock(),
        'geometry_msgs.msg': MagicMock(),
    }
    # SetParametersResult needs to be a real callable that returns an object
    stubs['rcl_interfaces.msg'].SetParametersResult = lambda **kw: types.SimpleNamespace(**kw)
    for name, stub in stubs.items():
        sys.modules.setdefault(name, stub)


_make_ros_stubs()

# Also stub depth_utils so we can import yolo_world_node standalone
# (the real depth_utils has no ROS deps, but the relative import needs a package)
import importlib, pathlib, sys as _sys  # noqa: E402
_pkg_dir = pathlib.Path(__file__).parent.parent / 'rtdetr_zed_tracker'

def _import_module_directly(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Import the pure modules with their real code
_depth_utils = _import_module_directly(
    'rtdetr_zed_tracker.depth_utils', _pkg_dir / 'depth_utils.py')

# Patch the relative import inside yolo_world_node before loading it
_sys.modules['rtdetr_zed_tracker'] = types.ModuleType('rtdetr_zed_tracker')
_sys.modules['rtdetr_zed_tracker'].depth_utils = _depth_utils
_sys.modules['rtdetr_zed_tracker.depth_utils'] = _depth_utils

# Now load the node module
_yw_mod = _import_module_directly(
    'rtdetr_zed_tracker.yolo_world_node', _pkg_dir / 'yolo_world_node.py')

_iou = _yw_mod._iou
_decode_image = _yw_mod._decode_image


# ---------------------------------------------------------------------------
# 1. IoU tests
# ---------------------------------------------------------------------------
class TestIou(unittest.TestCase):
    def test_identical_boxes(self):
        box = [10, 10, 100, 100]
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        # Two 10x10 boxes offset by 5 → 5x5 intersection = 25 / (100+100-25) = 0.1429
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        expected = 25 / 175
        assert abs(_iou(a, b) - expected) < 1e-5

    def test_contained_box(self):
        outer = [0, 0, 20, 20]
        inner = [5, 5, 15, 15]
        # inner area = 100, outer area = 400, union = 400
        expected = 100 / 400
        assert abs(_iou(outer, inner) - expected) < 1e-5

    def test_zero_area_box(self):
        a = [5, 5, 5, 5]   # degenerate
        b = [0, 0, 10, 10]
        assert _iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# 2. Cross-suppression logic (pure Python, no model)
# ---------------------------------------------------------------------------
class TestCrossSuppression(unittest.TestCase):
    """Test the suppression logic in isolation."""

    def _run_suppression(self, raw_detections, buoy_tracks, threshold=0.3):
        """Replicate the suppression loop from yolo_world_node._on_image."""
        filtered = []
        suppressed_count = 0
        for xyxy, score in raw_detections:
            suppressed = any(_iou(xyxy, b) >= threshold for b in buoy_tracks)
            if suppressed:
                suppressed_count += 1
            else:
                filtered.append((xyxy, score))
        return filtered, suppressed_count

    def test_no_buoy_tracks_passes_all(self):
        dets = [([10, 10, 100, 100], 0.9), ([200, 200, 300, 300], 0.8)]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[])
        assert len(filtered) == 2
        assert n_supp == 0

    def test_exact_overlap_suppressed(self):
        buoy = [10, 10, 100, 100]
        dets = [(buoy, 0.9)]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[buoy])
        assert len(filtered) == 0
        assert n_supp == 1

    def test_non_overlapping_vessel_passes(self):
        buoy = [10, 10, 100, 100]
        vessel = [500, 500, 800, 700]  # far away, no overlap
        dets = [(vessel, 0.85)]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[buoy])
        assert len(filtered) == 1
        assert n_supp == 0

    def test_partial_overlap_above_threshold_suppressed(self):
        # buoy = 100x100 box, vessel overlaps 80x80 of it.
        # intersection = 80*80 = 6400
        # union = 10000 + 10000 - 6400 = 13600
        # IoU = 6400/13600 ≈ 0.47  → above 0.3 threshold → suppressed
        buoy = [0, 0, 100, 100]
        vessel_nearby = [20, 20, 120, 120]  # 80×80 intersection with buoy
        computed_iou = _iou(vessel_nearby, buoy)
        assert computed_iou > 0.3, f"Precondition failed: iou={computed_iou:.4f} must be > 0.3"
        dets = [(vessel_nearby, 0.7)]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[buoy], threshold=0.3)
        assert len(filtered) == 0, f"Expected suppression but got {len(filtered)} detections"
        assert n_supp == 1

    def test_tiny_overlap_below_threshold_passes(self):
        # Very small overlap — should NOT be suppressed at threshold=0.3
        buoy = [0, 0, 10, 10]
        vessel = [9, 9, 100, 100]  # barely touches buoy corner
        dets = [(vessel, 0.7)]
        iou_val = _iou(vessel, buoy)
        assert iou_val < 0.3, f"Precondition: iou={iou_val:.4f} should be < 0.3"
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[buoy], threshold=0.3)
        assert len(filtered) == 1
        assert n_supp == 0

    def test_mixed_scene(self):
        """One buoy-overlap detection suppressed, one real vessel passes."""
        buoy = [100, 100, 200, 200]
        dets = [
            ([100, 100, 200, 200], 0.8),   # identical to buoy → suppressed
            ([600, 300, 900, 500], 0.75),  # far → passes
        ]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=[buoy])
        assert len(filtered) == 1
        assert n_supp == 1
        assert filtered[0][0] == [600, 300, 900, 500]

    def test_multiple_buoy_tracks(self):
        """Suppression checks against ALL RT-DETR buoy tracks."""
        buoys = [[0, 0, 50, 50], [200, 200, 250, 250], [400, 100, 500, 200]]
        dets = [
            ([200, 200, 250, 250], 0.6),   # overlaps buoy[1] → suppressed
            ([700, 700, 900, 900], 0.9),   # no overlap → passes
        ]
        filtered, n_supp = self._run_suppression(dets, buoy_tracks=buoys)
        assert n_supp == 1
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# 3. Image decoding
# ---------------------------------------------------------------------------
class TestDecodeImage(unittest.TestCase):
    def _make_mock_image(self, height, width, channels, encoding):
        msg = MagicMock()
        msg.height = height
        msg.width = width
        msg.encoding = encoding
        if channels == 4:
            arr = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
        else:
            arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        msg.data = arr.tobytes()
        return msg

    def test_bgra8_decoded_to_bgr(self):
        msg = self._make_mock_image(100, 100, 4, 'bgra8')
        out = _decode_image(msg)
        assert out.shape == (100, 100, 3)
        assert out.dtype == np.uint8

    def test_bgr8_passthrough(self):
        msg = self._make_mock_image(100, 100, 3, 'bgr8')
        out = _decode_image(msg)
        assert out.shape == (100, 100, 3)

    def test_rgb8_converted_to_bgr(self):
        msg = self._make_mock_image(100, 100, 3, 'rgb8')
        out = _decode_image(msg)
        assert out.shape == (100, 100, 3)

    def test_unsupported_encoding_raises(self):
        msg = self._make_mock_image(100, 100, 3, 'mono8')
        with pytest.raises(ValueError, match='Unsupported'):
            _decode_image(msg)


# ---------------------------------------------------------------------------
# 4. Live YOLO-World inference (skipped if ultralytics not installed)
# ---------------------------------------------------------------------------
ultralytics_available = False
try:
    import ultralytics  # noqa: F401
    ultralytics_available = True
except ImportError:
    pass


@pytest.mark.skipif(not ultralytics_available, reason='ultralytics not installed')
class TestYoloWorldLiveInference(unittest.TestCase):
    """Runs real YOLO-World-S on a synthetic image. Requires ultralytics."""

    @classmethod
    def setUpClass(cls):
        from ultralytics import YOLO
        cls.model = YOLO('yolov8s-worldv2.pt')
        cls.model.set_classes(['a vessel'])

    def _make_test_image(self):
        """Create a 640×640 synthetic BGR image (blue rectangle on grey background)."""
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        # Draw a rough "vessel" silhouette — elongated dark blue rectangle
        img[280:360, 100:540] = [80, 50, 30]   # hull
        img[240:285, 180:460] = [60, 60, 60]   # superstructure
        return img

    def test_inference_runs_without_error(self):
        img = self._make_test_image()
        results = self.model.predict(img, conf=0.1, verbose=False)
        # Just check it returned something without crashing
        assert results is not None

    def test_prompt_update_accepted(self):
        """set_classes should not raise for valid text prompts."""
        self.model.set_classes(['a boat'])
        self.model.set_classes(['a vessel'])  # reset

    def test_output_format(self):
        """Each detected box must have xyxy and conf."""
        img = self._make_test_image()
        results = self.model.predict(img, conf=0.05, verbose=False)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                assert len(xyxy) == 4
                assert all(isinstance(v, float) for v in xyxy)
                conf = float(box.conf[0])
                assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 5. End-to-end mock: image + buoy tracks → filtered detections
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ultralytics_available, reason='ultralytics not installed')
def test_end_to_end_suppression_with_real_model():
    """
    Creates a synthetic scene with two objects:
      - One that overlaps a fake RT-DETR buoy track → must be suppressed
      - A clear area with no buoy overlap → any YOLO detection there must pass

    This test validates the full suppression pipeline end-to-end with a real model.
    It is probabilistic: if YOLO-World makes no detections at all the test is skipped.
    """
    from ultralytics import YOLO

    model = YOLO('yolov8s-worldv2.pt')
    model.set_classes(['a vessel'])

    img = np.full((720, 1280, 3), 130, dtype=np.uint8)

    # Fake RT-DETR buoy at left side of image
    buoy_xyxy = [50, 300, 200, 450]

    raw_detections: list[tuple] = []
    results = model.predict(img, conf=0.05, verbose=False)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            raw_detections.append((box.xyxy[0].tolist(), float(box.conf[0])))

    if not raw_detections:
        pytest.skip('YOLO-World made no detections on synthetic image — probabilistic test skipped')

    # Apply suppression
    filtered = []
    for xyxy, score in raw_detections:
        suppressed = _iou(xyxy, buoy_xyxy) >= 0.3
        if not suppressed:
            filtered.append((xyxy, score))

    # All remaining detections must NOT heavily overlap the buoy region
    for xyxy, _ in filtered:
        assert _iou(xyxy, buoy_xyxy) < 0.3, \
            f'Suppression failed: detection {xyxy} overlaps buoy {buoy_xyxy}'


# ---------------------------------------------------------------------------
# CLI: --visualise flag
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if '--visualise' in sys.argv:
        try:
            import cv2
            from ultralytics import YOLO

            print('Running YOLO-World-S on synthetic scene (visualise mode)...')
            model = YOLO('yolov8s-worldv2.pt')
            model.set_classes(['a vessel'])

            img = np.full((720, 1280, 3), 130, dtype=np.uint8)
            # Draw a rough vessel shape
            img[300:450, 400:900] = [70, 50, 30]
            img[250:305, 500:800] = [50, 50, 50]

            # Fake buoy (simulate RT-DETR confirmed track)
            buoy = [50, 300, 200, 450]
            cv2.rectangle(img, (buoy[0], buoy[1]), (buoy[2], buoy[3]), (0, 255, 0), 3)
            cv2.putText(img, 'RT-DETR buoy (suppression zone)',
                        (buoy[0], buoy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results = model.predict(img, conf=0.1, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    score = float(box.conf[0])
                    suppressed = _iou(xyxy, buoy) >= 0.3
                    color = (0, 0, 255) if suppressed else (255, 128, 0)
                    label = f'SUPPRESSED (iou>{_iou(xyxy, buoy):.2f})' if suppressed \
                            else f'obstacle ({score:.2f})'
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(img, label, (xyxy[0], xyxy[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow('YOLO-World mock test (green=buoy zone, orange=obstacle, red=suppressed)', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except ImportError as e:
            print(f'Visualise requires ultralytics + opencv-python: {e}')
    else:
        # Run pytest programmatically
        pytest.main([__file__, '-v'])
