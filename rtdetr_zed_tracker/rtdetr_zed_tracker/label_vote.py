"""Majority-vote label freezing. Pure Python, zero ROS imports.

Same idea as ``Track.class_votes``/``frozen_class`` in byte_tracker.py (vote every
observation, freeze the winner once enough evidence has accumulated so a single bad
frame can't flip a decided label) factored out into a standalone, reusable component
so any per-track label decision -- not just the detector's shape class -- can use it
without duplicating the logic.
"""
from __future__ import annotations

from collections import Counter


class LabelVote:
    """Accumulates votes for a label and freezes the winner after ``min_votes``.

    ``None`` observations (caller is uncertain) are never counted -- an unsure frame
    should never be able to drag the vote toward a wrong answer.
    """

    def __init__(self, min_votes: int = 3):
        self.min_votes = int(min_votes)
        self._votes: Counter = Counter()
        self.frozen: str | None = None

    def add(self, label: str | None) -> None:
        """Register one observation. No-op once frozen, or if ``label`` is None."""
        if self.frozen is not None or label is None:
            return
        self._votes[label] += 1
        if sum(self._votes.values()) >= self.min_votes:
            self.frozen = self._votes.most_common(1)[0][0]

    @property
    def current_best(self) -> str | None:
        """Frozen value if decided; otherwise the current leading vote (or None)."""
        if self.frozen is not None:
            return self.frozen
        if not self._votes:
            return None
        return self._votes.most_common(1)[0][0]

    @property
    def is_frozen(self) -> bool:
        return self.frozen is not None
