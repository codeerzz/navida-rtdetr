"""Unit tests for class_remap.py -- collapsing RT-DETR's trained classes into
a generic shape without retraining. Pure Python, no ROS/ZED needed."""
from __future__ import annotations

from rtdetr_zed_tracker.class_remap import collapse_mapping, remap_class


def test_mapped_class_is_remapped():
    mapping = {4: 0, 2: 0}   # red_buoy, green_buoy -> buoy
    assert remap_class(4, mapping) == 0
    assert remap_class(2, mapping) == 0


def test_unmapped_class_passes_through_unchanged():
    """Fail safe, not fail silent: a class index the mapping doesn't mention
    must keep its original value, never get dropped or defaulted to 0."""
    mapping = {4: 0}
    assert remap_class(3, mapping) == 3


def test_collapse_mapping_sends_every_listed_class_to_the_canonical_one():
    mapping = collapse_mapping(range(7), canonical=0)
    assert all(remap_class(i, mapping) == 0 for i in range(7))


def test_collapse_mapping_can_target_a_subset():
    """Collapsing only the colored classes leaves cardinal marks untouched --
    the "keep cardinal marks distinct" option mentioned in class_remap.yaml."""
    mapping = collapse_mapping([2, 4], canonical=0)   # green_buoy, red_buoy only
    assert remap_class(2, mapping) == 0
    assert remap_class(4, mapping) == 0
    assert remap_class(1, mapping) == 1   # east_buoy untouched
    assert remap_class(3, mapping) == 3   # north_buoy untouched
