"""Phase 1 acceptance tests for the pure-Python ByteTracker.

Run: pytest test/  (inside the container: numpy 1.26 + scipy 1.15).
Covers the five scenarios required by the project spec.
"""
import numpy as np

from rtdetr_zed_tracker.bbox_utils import inverse_letterbox, iou_matrix
from rtdetr_zed_tracker.byte_tracker import ByteTracker


def box(cx, cy, w=50.0, h=50.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def det(cx, cy, score, cls, w=50.0, h=50.0):
    return (box(cx, cy, w, h), score, cls)


def confirmed_by_id(tracker):
    return {t.global_id: t for t in tracker.tracks if t.is_confirmed}


def nearest_track_id(tracks, cx, cy):
    best, bestd = None, 1e18
    for t in tracks:
        b = t.xyxy
        tx, ty = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = (tx - cx) ** 2 + (ty - cy) ** 2
        if d < bestd:
            best, bestd = t.global_id, d
    return best


# --------------------------------------------------------------------------- #
def test_bbox_utils_sanity():
    a = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([[100, 100, 110, 110]]))[0, 0] == 0.0
    # top_left inverse letterbox: 1920x1080 -> 640, scale = 640/1920
    inv = inverse_letterbox([[0, 0, 640, 360]], (1920, 1080), (640, 640), 'top_left')[0]
    assert np.allclose(inv, [0, 0, 1920, 1080], atol=1e-6)


def test_two_boxes_crossing_keep_ids():
    """Two same-class objects crossing paths keep stable, distinct IDs."""
    tr = ByteTracker(min_hits=3, class_gating='soft')
    ya, yb = 200.0, 216.0        # slight vertical offset so they never fully coincide
    xa, xb = 100.0, 400.0
    va, vb = 25.0, -25.0
    id_a = id_b = None
    for f in range(14):
        cxa, cxb = xa + va * f, xb + vb * f
        tr.update([det(cxa, ya, 0.9, 0), det(cxb, yb, 0.9, 0)], dt=1.0)
        conf = [t for t in tr.tracks if t.is_confirmed]
        if len(conf) == 2 and id_a is None:
            id_a = nearest_track_id(conf, cxa, ya)
            id_b = nearest_track_id(conf, cxb, yb)
        if id_a is not None:
            assert nearest_track_id(conf, cxa, ya) == id_a, f"A id swapped at frame {f}"
            assert nearest_track_id(conf, cxb, yb) == id_b, f"B id swapped at frame {f}"
    assert id_a is not None and id_b is not None and id_a != id_b


def test_reappear_after_gap_recovers_id():
    """Object vanishes for 5 frames, reappears near predicted position -> same ID."""
    tr = ByteTracker(min_hits=3, max_age=30, class_gating='soft')
    x, y, v = 100.0, 200.0, 20.0
    f = 0
    for _ in range(5):                         # establish + confirm a moving track
        tr.update([det(x + v * f, y, 0.9, 0)], dt=1.0); f += 1
    orig = list(confirmed_by_id(tr).keys())
    assert len(orig) == 1
    orig_id = orig[0]
    for _ in range(5):                         # 5-frame gap, no detections
        tr.update([], dt=1.0); f += 1
    assert orig_id in confirmed_by_id(tr), "track dropped during gap"
    tr.update([det(x + v * f, y, 0.9, 0)], dt=1.0)   # reappear at predicted position
    conf = confirmed_by_id(tr)
    assert orig_id in conf, "did not recover original ID"
    assert len(conf) == 1


def test_spurious_single_frame_never_confirmed():
    """A one-frame detection must never be confirmed."""
    tr = ByteTracker(min_hits=3, class_gating='soft')
    tr.update([det(300, 300, 0.9, 0)], dt=1.0)
    for _ in range(6):
        tr.update([], dt=1.0)
        assert all(not t.is_confirmed for t in tr.tracks)
    assert len(confirmed_by_id(tr)) == 0


def test_label_frozen_at_confirmation():
    """Class A for 10 frames then B for 2 -> keeps ID and keeps label A."""
    tr = ByteTracker(min_hits=3, class_gating='none')   # gating off so class-B dets still match
    for _ in range(10):
        tr.update([det(250, 250, 0.9, 0)], dt=1.0)      # class 0 == 'buoy'
    conf = confirmed_by_id(tr)
    assert len(conf) == 1
    tid = next(iter(conf))
    assert conf[tid].frozen_class == 0
    for _ in range(2):
        tr.update([det(250, 250, 0.9, 4)], dt=1.0)      # class 4 == 'red_buoy'
    conf = confirmed_by_id(tr)
    assert tid in conf, "ID changed after class flip"
    assert conf[tid].frozen_class == 0, "frozen label must stay class 0"
    assert conf[tid].class_votes[4] == 2                # counter keeps updating for diagnostics


def _run_two_overlapping_diff_class(gating):
    tr = ByteTracker(min_hits=3, class_gating=gating)
    cxa, cxb, y = 300.0, 330.0, 300.0    # overlapping (IoU ~0.5), different classes, stationary
    ids_seen = []
    for _ in range(20):
        tr.update([det(cxa, y, 0.9, 4), det(cxb, y, 0.9, 2)], dt=1.0)
        ids_seen.append(tuple(sorted(t.global_id for t in tr.tracks if t.is_confirmed)))
    return tr, ids_seen


def test_two_overlapping_diff_class_soft_no_swap():
    """Overlapping boxes of different classes under 'soft' -> 2 stable, class-correct tracks."""
    tr, ids_seen = _run_two_overlapping_diff_class('soft')
    conf = [t for t in tr.tracks if t.is_confirmed]
    assert len(conf) == 2
    # each track kept a single, correct class (no cross-class contamination)
    classes = sorted(t.frozen_class for t in conf)
    assert classes == [2, 4]
    for t in conf:
        assert set(t.class_votes.keys()) == {t.frozen_class}
    # ids stable once confirmed
    stable = [s for s in ids_seen if len(s) == 2]
    assert len(set(stable)) == 1, f"IDs changed over time: {set(stable)}"

    # document the difference: 'none' runs the same scenario and also yields 2 tracks,
    # but without class protection during crossings (see Phase 2 visual gate).
    tr_none, ids_none = _run_two_overlapping_diff_class('none')
    conf_none = [t for t in tr_none.tracks if t.is_confirmed]
    assert len(conf_none) == 2
