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
