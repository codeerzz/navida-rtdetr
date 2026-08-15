"""Integration test for color_classification_node -- exercises the ROS glue
(stamp matching, cv_bridge decode, Detection2DArray in/out) that
test_color_classifier.py deliberately leaves untested by testing pure functions
only.

Needs rclpy + cv_bridge + vision_msgs (a sourced ROS 2 install -- the Isaac ROS
container / the robot, or a ROS 2 desktop install). Skips itself entirely if
those aren't importable, so it's safe to keep in the same test/ directory as the
ROS-free suite without breaking `pytest test/` on a machine without ROS (a
laptop):

    # inside a sourced ROS 2 Humble environment:
    cd rtdetr_zed_tracker && pytest test/test_color_classification_node.py -v

No live ZED or camera needed -- images and tracks are constructed in-process and
delivered to the node's callbacks directly (not spun through a real DDS graph),
so this only needs the ROS 2 Python libraries to be importable, not a running
zed_node or Isaac ROS pipeline.

Uses test/fixtures/color_ranges_test.yaml, NOT the real config/color_ranges.yaml
-- same reason as test_color_classifier.py: that file gets recalibrated against
the real robot camera and must be free to change without breaking this suite.

NOTE: this file could not be executed in the environment this PR was written in
(no rclpy available there -- see test_color_classifier.py's docstring for the
part that WAS run and verified). Review the ROS API usage carefully before
trusting it; run it on the robot / in the Isaac ROS container before relying on
it as a merge gate.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
cv_bridge = pytest.importorskip('cv_bridge')
pytest.importorskip('vision_msgs')

from builtin_interfaces.msg import Time
from rtdetr_zed_tracker.color_classification_node import (
    ColorClassificationNode,
)
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'color_ranges_test.yaml')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'class_labels_test.yaml')


class _FakePublisher:
    """Swapped in for node.pub so a test can inspect what would have been
    published without needing a real DDS graph or an executor spin."""

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def _stamp(sec: int, nsec: int = 0) -> Time:
    t = Time()
    t.sec = sec
    t.nanosec = nsec
    return t


def _image_msg(bgr: np.ndarray, stamp: Time) -> Image:
    bridge = cv_bridge.CvBridge()
    msg = bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
    msg.header.stamp = stamp
    msg.header.frame_id = 'zed_left_camera_optical_frame'
    return msg


def _tracks_msg(stamp: Time, boxes) -> Detection2DArray:
    """boxes: iterable of (track_id, class_id, cx, cy, w, h)."""
    msg = Detection2DArray()
    msg.header.stamp = stamp
    for tid, cls, cx, cy, w, h in boxes:
        d = Detection2D()
        d.id = tid
        d.bbox.center.position.x, d.bbox.center.position.y = float(cx), float(cy)
        d.bbox.size_x, d.bbox.size_y = float(w), float(h)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = cls
        hyp.hypothesis.score = 0.9
        d.results.append(hyp)
        msg.detections.append(d)
    return msg


def _solid_bgr(bgr, size=(200, 300)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    # rclpy.init() tries to create a log dir under $ROS_LOG_DIR / $ROS_HOME /
    # ~/.ros -- on some CI runners that resolves to a path the job can't write
    # to (e.g. "Failed to create log directory: /home/runner ...") and
    # rclpy.init() raises before any test even runs. Point it at a directory
    # this process definitely owns instead of relying on the runner's HOME.
    os.environ.setdefault('ROS_LOG_DIR', tempfile.mkdtemp(prefix='ros_log_'))
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_node(**overrides):
    params = {'color_ranges_file': CONFIG_PATH, 'min_votes': 1}  # decide on the first frame here
    params.update(overrides)
    parameter_overrides = [rclpy.parameter.Parameter(k, value=v) for k, v in params.items()]
    node = ColorClassificationNode(parameter_overrides=parameter_overrides)
    node.pub = _FakePublisher()
    return node


def test_matches_image_by_stamp_and_corrects_a_wrong_color_label():
    node = _make_node()
    try:
        red_bgr = _solid_bgr((30, 30, 200))  # BGR: a red buoy
        stamp = _stamp(10, 0)
        node.on_image(_image_msg(red_bgr, stamp))

        # detector said green_buoy, but the crop is unambiguously red.
        tracks = _tracks_msg(stamp, [('green_buoy_1', 'green_buoy', 150, 100, 100, 100)])
        node.on_tracks(tracks)

        refined = node.pub.messages[0].detections[0].results[0].hypothesis.class_id
        assert refined == 'red_buoy'
    finally:
        node.destroy_node()


def test_generic_buoy_label_is_colorable_by_default():
    """Pairs with class_remap_node: when RT-DETR's classes are collapsed to a
    single generic "buoy" upstream, this node must still be able to assign a
    color to it -- 'buoy' is colorable by default for exactly this reason."""
    node = _make_node()
    try:
        red_bgr = _solid_bgr((30, 30, 200))
        stamp = _stamp(15, 0)
        node.on_image(_image_msg(red_bgr, stamp))

        tracks = _tracks_msg(stamp, [('buoy_1', 'buoy', 150, 100, 100, 100)])
        node.on_tracks(tracks)

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
    finally:
        node.destroy_node()


def test_non_colorable_label_passes_through_untouched():
    """Cardinal marks (north/south/east/west) have no color meaning -- must
    never be touched even if a crop happens to fall in a color range."""
    node = _make_node()
    try:
        red_bgr = _solid_bgr((30, 30, 200))
        stamp = _stamp(20, 0)
        node.on_image(_image_msg(red_bgr, stamp))

        tracks = _tracks_msg(stamp, [('north_buoy_1', 'north_buoy', 150, 100, 100, 100)])
        node.on_tracks(tracks)

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'north_buoy'
    finally:
        node.destroy_node()


def test_no_matching_image_degrades_to_pass_through_not_a_drop():
    """No image within tolerance -> publish tracks unrefined rather than stall
    the pipeline waiting for one."""
    node = _make_node()
    try:
        tracks = _tracks_msg(_stamp(99, 0), [('red_buoy_1', 'red_buoy', 150, 100, 100, 100)])
        node.on_tracks(tracks)
        assert len(node.pub.messages) == 1
        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
    finally:
        node.destroy_node()


def test_uncertain_color_leaves_the_original_label_alone():
    """A neutral/blown-out crop must not overwrite a possibly-correct existing
    label with a forced guess."""
    node = _make_node()
    try:
        white_bgr = _solid_bgr((250, 250, 250))
        stamp = _stamp(30, 0)
        node.on_image(_image_msg(white_bgr, stamp))

        tracks = _tracks_msg(stamp, [('red_buoy_1', 'red_buoy', 150, 100, 100, 100)])
        node.on_tracks(tracks)

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
    finally:
        node.destroy_node()


def test_vote_state_is_garbage_collected_when_a_track_disappears():
    node = _make_node()
    try:
        stamp = _stamp(40, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('red_buoy_1', 'red_buoy', 150, 100, 100, 100)]))
        assert 'red_buoy_1' in node.votes

        node.on_tracks(_tracks_msg(_stamp(41, 0), []))  # track no longer reported
        assert 'red_buoy_1' not in node.votes
    finally:
        node.destroy_node()


# --------------------------------------------------------------------- index space
# With class_labels_file set, class_id is the numeric RT-DETR index on the wire in
# BOTH directions. This is the mode tracking.launch.py runs, and getting it wrong
# is not a silent no-op: writing a name back out makes depth_fusion_node's
# int(class_id) raise on every frame.

def _index_node(**overrides):
    return _make_node(class_labels_file=LABELS_PATH, **overrides)


def test_numeric_class_index_is_translated_decided_and_written_back_as_an_index():
    node = _index_node()
    try:
        stamp = _stamp(50, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))   # a red crop

        # '0' is the generic "buoy" -- colorable, so the red crop must win and be
        # written back as index '2' (red_buoy), NOT as the string 'red_buoy'.
        node.on_tracks(_tracks_msg(stamp, [('', '0', 150, 100, 100, 100)]))

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == '2'
    finally:
        node.destroy_node()


def test_unmapped_class_index_passes_through_untouched():
    """Index 5 has no name in the fixture -> it cannot be shown to be colorable,
    so it must pass through rather than be guessed at."""
    node = _index_node()
    try:
        stamp = _stamp(55, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('', '5', 150, 100, 100, 100)]))

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == '5'
    finally:
        node.destroy_node()


def test_non_colorable_index_passes_through_untouched():
    node = _index_node()
    try:
        stamp = _stamp(58, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('', '3', 150, 100, 100, 100)]))  # north_buoy

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == '3'
    finally:
        node.destroy_node()


def test_refined_colour_with_no_configured_index_keeps_the_original_class_id():
    """'green_buoy' has no index in the fixture. The refinement must be dropped and
    the original index kept -- never emitted as a name, which would break the
    int(class_id) contract every downstream node on this graph relies on."""
    node = _index_node()
    try:
        stamp = _stamp(60, 0)
        # (40, 170, 40) is the "normal daylight green" swatch test_color_classifier
        # pins against these same fixture ranges.
        node.on_image(_image_msg(_solid_bgr((40, 170, 40)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('', '0', 150, 100, 100, 100)]))

        out = node.pub.messages[0].detections[0].results[0].hypothesis.class_id
        assert out == '0'
        int(out)   # the actual contract: downstream must still be able to parse it
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------- vote keying

def test_grid_vote_key_groups_by_box_centre_not_by_id():
    """There are no track ids on the live graph -- every detection carries id ''.
    Under vote_key='grid' two boxes in different cells must still vote separately,
    which is the whole reason the mode exists."""
    node = _make_node(vote_key='grid', vote_cell_px=64.0)
    try:
        stamp = _stamp(70, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200), size=(400, 600)), stamp))
        node.on_tracks(_tracks_msg(stamp, [
            ('', 'buoy', 100, 100, 60, 60),
            ('', 'buoy', 400, 300, 60, 60),
        ]))

        assert set(node.votes) == {'1:1', '6:4'}
    finally:
        node.destroy_node()


def test_grid_vote_accumulates_across_frames_for_a_box_that_stays_in_its_cell():
    node = _make_node(vote_key='grid', vote_cell_px=64.0, min_votes=3)
    try:
        for i in range(3):
            stamp = _stamp(80 + i, 0)
            node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
            # Drifts a few px per frame but never leaves cell 2:1.
            node.on_tracks(_tracks_msg(stamp, [('', 'buoy', 150 + i * 3, 100, 60, 60)]))

        assert list(node.votes) == ['2:1']
        assert node.votes['2:1'].is_frozen
        assert node.votes['2:1'].frozen == 'red'
    finally:
        node.destroy_node()


def test_grid_vote_is_dropped_as_soon_as_its_cell_goes_unseen():
    """A frozen cell must not outlive the object that froze it by more than one
    frame, or a different buoy entering that cell would inherit its colour."""
    node = _make_node(vote_key='grid', vote_cell_px=64.0)
    try:
        stamp = _stamp(90, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('', 'buoy', 150, 100, 60, 60)]))
        assert '2:1' in node.votes

        stamp2 = _stamp(91, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp2))
        node.on_tracks(_tracks_msg(stamp2, [('', 'buoy', 400, 100, 60, 60)]))
        assert '2:1' not in node.votes
    finally:
        node.destroy_node()


def test_vote_key_none_keeps_no_state_and_still_refines():
    node = _make_node(vote_key='none')
    try:
        stamp = _stamp(95, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('red_buoy_1', 'green_buoy', 150, 100, 100, 100)]))

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
        assert node.votes == {}
    finally:
        node.destroy_node()


def test_a_frozen_vote_survives_a_frame_with_no_matching_image():
    """The pass-through path must still apply what the vote already decided.
    Otherwise the published label flickers back to the detector's own guess on
    every dropped frame -- which is exactly what freezing a vote is meant to
    prevent, and it happens on ~16% of frames on the real rig."""
    node = _make_node(vote_key='grid', vote_cell_px=64.0, min_votes=1)
    try:
        stamp = _stamp(100, 0)
        node.on_image(_image_msg(_solid_bgr((30, 30, 200)), stamp))
        node.on_tracks(_tracks_msg(stamp, [('', 'buoy', 150, 100, 60, 60)]))
        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
        assert node.votes['2:1'].is_frozen

        # Same box, but far enough ahead in time that no buffered image matches.
        node.on_tracks(_tracks_msg(_stamp(200, 0), [('', 'buoy', 150, 100, 60, 60)]))
        assert node.pub.messages[1].detections[0].results[0].hypothesis.class_id == 'red_buoy'
    finally:
        node.destroy_node()


def test_no_image_and_no_vote_still_leaves_the_label_alone():
    """The fallback only applies knowledge that exists -- it must never invent a
    colour for a detection the node has never managed to look at."""
    node = _make_node(vote_key='grid', vote_cell_px=64.0)
    try:
        node.on_tracks(_tracks_msg(_stamp(120, 0), [('', 'green_buoy', 150, 100, 60, 60)]))
        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'green_buoy'
        assert node.votes == {}
    finally:
        node.destroy_node()


# ------------------------------------------------------- per-colour confidence

def _partly_red_bgr():
    """Neutral grey canvas with a red square covering 1/4 of the sampled ROI.

    A _solid_bgr crop is useless for threshold tests: it scores exactly 1.0, which
    no threshold in [0, 1] can reject. Grey is deliberate -- at Cr=Cb=128 it falls
    in neither fixture range, so it contributes to no colour's ratio.

    Geometry: the bbox used below is 100x100, classify_color samples the central
    roi_shrink=0.6 of it (60x60 = 3600 px), and a 30x30 red patch at the image
    centre puts 900 of those in range -> red ratio 0.25.
    """
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    img[85:115, 135:165] = (30, 30, 200)
    return img


def test_a_raised_bar_makes_that_colour_abstain_rather_than_elect_another():
    """The winner is picked BEFORE any threshold is applied, so failing the bar
    must yield "uncertain" -- never a vote for a different colour."""
    node = _make_node(min_confidence_overrides='red:0.5')   # 0.25 observed < 0.5
    try:
        stamp = _stamp(130, 0)
        node.on_image(_image_msg(_partly_red_bgr(), stamp))
        node.on_tracks(_tracks_msg(stamp, [('t1', 'buoy', 150, 100, 100, 100)]))

        # Not red (bar not cleared), and NOT flipped to green either.
        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'buoy'
        assert node.votes['t1'].current_best is None
    finally:
        node.destroy_node()


def test_the_same_observation_passes_when_the_bar_is_low():
    """Companion to the test above: proves the abstention came from the threshold
    and not from the crop being unclassifiable in the first place."""
    node = _make_node(min_confidence_overrides='red:0.2')   # 0.25 observed > 0.2
    try:
        stamp = _stamp(135, 0)
        node.on_image(_image_msg(_partly_red_bgr(), stamp))
        node.on_tracks(_tracks_msg(stamp, [('t1', 'buoy', 150, 100, 100, 100)]))

        assert node.pub.messages[0].detections[0].results[0].hypothesis.class_id == 'red_buoy'
    finally:
        node.destroy_node()


def test_colours_without_an_override_keep_the_global_min_confidence():
    node = _make_node(min_confidence=0.05, min_confidence_overrides='red:0.9')
    try:
        assert node.min_confidence_per_color['red'] == pytest.approx(0.9)
        assert node.min_confidence_per_color['green'] == pytest.approx(0.05)
    finally:
        node.destroy_node()


def test_empty_override_string_leaves_every_colour_on_the_global_bar():
    node = _make_node(min_confidence=0.2, min_confidence_overrides='')
    try:
        assert set(node.min_confidence_per_color) == {'red', 'green'}
        assert all(v == pytest.approx(0.2) for v in node.min_confidence_per_color.values())
    finally:
        node.destroy_node()


@pytest.mark.parametrize('bad', ['red', 'red:high', 'red:30'])
def test_malformed_override_is_rejected_at_startup(bad):
    """Missing colon, non-numeric, and out-of-[0,1] are unambiguous mistakes.
    Strict on purpose: an override exists to make a colour HARDER to claim, so one
    that silently reverted to the low global bar would be invisible in exactly the
    situation it was added to guard against."""
    with pytest.raises(RuntimeError, match='min_confidence_overrides'):
        _make_node(min_confidence_overrides=bad)


def test_override_for_an_unconfigured_colour_is_ignored_not_fatal():
    """The shipped default is "black:0.30", but a config with only red and green is
    perfectly valid -- a default must never stop the node from starting. So this
    one degrades to a warning (the test fixture has no black)."""
    node = _make_node(min_confidence=0.2, min_confidence_overrides='black:0.30')
    try:
        assert set(node.min_confidence_per_color) == {'red', 'green'}
        assert all(v == pytest.approx(0.2) for v in node.min_confidence_per_color.values())
    finally:
        node.destroy_node()


def test_invalid_vote_key_is_rejected_at_startup():
    """A typo must not degrade silently into per-frame decisions that look like a
    working vote."""
    with pytest.raises(RuntimeError, match='vote_key'):
        _make_node(vote_key='per_track')
