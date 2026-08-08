"""Class-index remapping for RT-DETR detections. Pure Python, zero ROS imports.

Lets you treat all (or a chosen subset of) RT-DETR's trained classes as one
generic shape WITHOUT retraining the model: remap every incoming class index to
a canonical one before it reaches the rest of the graph, then let a separate
stage (e.g. color_classification_node) decide the color-bearing part of the label
independently. Same shape/semantics split all_seaing_perception's LiDAR+camera
pipeline uses (see the object-tracking comparison report) -- applied here to an
already-trained multi-class detector via a lookup table instead of a retrain.
"""
from __future__ import annotations


def remap_class(class_index: int, mapping: dict[int, int]) -> int:
    """Look up ``class_index`` in ``mapping``; anything not listed passes
    through unchanged (never silently dropped or defaulted to 0)."""
    return mapping.get(class_index, class_index)


def collapse_mapping(class_indices, canonical: int = 0) -> dict[int, int]:
    """Build a mapping that sends every one of ``class_indices`` to ``canonical``.

    ``collapse_mapping(range(7))`` collapses all 7 trained buoy classes to a
    single generic class 0 ("buoy" in class_labels.yaml); pass a subset (e.g.
    only the colored classes) to leave the rest -- cardinal marks, say --
    passing through unchanged instead.
    """
    return {int(c): canonical for c in class_indices}
