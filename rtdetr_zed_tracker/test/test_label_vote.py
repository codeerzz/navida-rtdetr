"""Unit tests for the reusable LabelVote component."""
from __future__ import annotations

from rtdetr_zed_tracker.label_vote import LabelVote


def test_freezes_the_majority_after_min_votes():
    v = LabelVote(min_votes=3)
    v.add('red')
    v.add('red')
    assert not v.is_frozen
    v.add('red')
    assert v.is_frozen
    assert v.current_best == 'red'


def test_a_single_bad_frame_cannot_flip_an_already_frozen_label():
    """THE point of this component: one wrong observation (e.g. a glare frame)
    after freezing must never change the decided label."""
    v = LabelVote(min_votes=2)
    v.add('red')
    v.add('red')
    assert v.current_best == 'red'
    v.add('green')  # a single bad frame
    v.add('green')
    assert v.current_best == 'red', 'frozen label must not move'


def test_uncertain_none_observations_never_count_toward_the_vote():
    v = LabelVote(min_votes=2)
    v.add(None)
    v.add(None)
    v.add(None)
    assert not v.is_frozen
    assert v.current_best is None
    v.add('green')
    v.add('green')
    assert v.current_best == 'green'


def test_majority_wins_even_with_a_noisy_minority():
    v = LabelVote(min_votes=5)
    for label in ['red', 'green', 'red', 'red', 'red']:
        v.add(label)
    assert v.current_best == 'red'
