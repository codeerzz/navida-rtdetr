"""ByteTrack multi-object tracker. Pure numpy + scipy, ZERO ROS imports.

Two-stage association (high-score then low-score detections), Hungarian matching
(``scipy.optimize.linear_sum_assignment``), soft class gating, per-track class
voting with a label frozen at confirmation, and stable IDs (a global monotonic
id for uniqueness plus a per-class display sequence like ``4_2``).

Detections are passed to :meth:`update` as an iterable of ``(xyxy, score, class_index)``.
``dt`` (seconds between this and the previous frame) is passed explicitly.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bbox_utils import iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from .kalman_box import KalmanBox

TENTATIVE = 0
CONFIRMED = 1
DELETED = 2


class Track:
    def __init__(self, xyxy, score, class_index, global_id):
        self.kf = KalmanBox(xyxy_to_xyah([xyxy])[0])
        self.global_id = int(global_id)
        self.state = TENTATIVE

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.score = float(score)
        self.class_index = int(class_index)          # most recent observed class
        self.class_votes = Counter([int(class_index)])

        self.frozen_class = None                      # set at confirmation
        self.class_seq = None                         # per-class display number
        self.display_id = None                        # e.g. "4_2" (class_index_seq)

    # --- geometry -----------------------------------------------------------
    @property
    def xyxy(self) -> np.ndarray:
        return xyah_to_xyxy([self.kf.xyah])[0]

    def predict(self, dt: float):
        self.kf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, xyxy, score, class_index):
        self.kf.update(xyxy_to_xyah([xyxy])[0])
        self.hits += 1
        self.time_since_update = 0
        self.score = float(score)
        self.class_index = int(class_index)
        self.class_votes[int(class_index)] += 1

    # --- labelling ----------------------------------------------------------
    @property
    def voted_class(self) -> int:
        return self.class_votes.most_common(1)[0][0]

    @property
    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED


class ByteTracker:
    def __init__(self,
                 track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 match_thresh: float = 0.8,
                 max_age: int = 30,
                 min_hits: int = 3,
                 class_gating: str = 'soft',
                 class_mismatch_penalty: float = 0.3):
        assert class_gating in ('strict', 'soft', 'none')
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.class_gating = class_gating
        self.class_mismatch_penalty = class_mismatch_penalty

        self.tracks: List[Track] = []
        self._next_global_id = 1
        self._class_seq_counters: Counter = Counter()   # frozen_class -> count issued
        self.frame_count = 0

    # ------------------------------------------------------------------ cost
    def _cost_matrix(self, tracks: Sequence[Track], boxes: np.ndarray,
                     classes: np.ndarray) -> np.ndarray:
        """cost = (1 - iou) + penalty * (class mismatch). Strict => inf on mismatch."""
        if len(tracks) == 0 or len(boxes) == 0:
            return np.zeros((len(tracks), len(boxes)), dtype=np.float64)
        tboxes = np.array([t.xyxy for t in tracks], dtype=np.float64)
        tcls = np.array([t.class_index for t in tracks], dtype=np.int64)
        cost = 1.0 - iou_matrix(tboxes, boxes)
        mismatch = tcls[:, None] != classes[None, :]
        if self.class_gating == 'strict':
            cost = np.where(mismatch, np.inf, cost)
        elif self.class_gating == 'soft':
            cost = cost + self.class_mismatch_penalty * mismatch
        # 'none' -> no change
        return cost

    def _match(self, tracks: Sequence[Track], boxes: np.ndarray, classes: np.ndarray):
        """Hungarian assignment gated by match_thresh. Returns matches, un_t, un_d."""
        cost = self._cost_matrix(tracks, boxes, classes)
        nt, nd = cost.shape
        if nt == 0 or nd == 0:
            return [], list(range(nt)), list(range(nd))

        big = 1e9
        solver_cost = np.where(np.isfinite(cost), cost, big)
        rows, cols = linear_sum_assignment(solver_cost)

        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.match_thresh:      # inf/mismatch rejected here
                matches.append((int(r), int(c)))
                matched_t.add(int(r))
                matched_d.add(int(c))
        un_t = [i for i in range(nt) if i not in matched_t]
        un_d = [j for j in range(nd) if j not in matched_d]
        return matches, un_t, un_d

    # ------------------------------------------------------------------ main
    def update(self, detections: Sequence[Tuple], dt: float = 1.0) -> List[Track]:
        """Advance one frame. ``detections`` = iterable of (xyxy, score, class_index).
        Returns the list of currently CONFIRMED tracks."""
        self.frame_count += 1

        if len(detections) > 0:
            boxes = np.array([np.asarray(d[0], dtype=np.float64).reshape(4) for d in detections])
            scores = np.array([float(d[1]) for d in detections])
            classes = np.array([int(d[2]) for d in detections])
        else:
            boxes = np.zeros((0, 4)); scores = np.zeros((0,)); classes = np.zeros((0,), dtype=np.int64)

        high = scores >= self.track_high_thresh
        low = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)
        hi_idx = np.nonzero(high)[0]
        lo_idx = np.nonzero(low)[0]

        # 1) predict all existing tracks forward
        for t in self.tracks:
            t.predict(dt)

        # 2) first association: all tracks vs high-score detections
        m1, un_t1, un_hi = self._match(self.tracks, boxes[hi_idx], classes[hi_idx])
        for ti, di in m1:
            gd = hi_idx[di]
            self.tracks[ti].update(boxes[gd], scores[gd], classes[gd])

        # 3) second association: still-unmatched tracks vs low-score detections
        remaining_tracks = [self.tracks[i] for i in un_t1]
        m2, un_t2_local, _ = self._match(remaining_tracks, boxes[lo_idx], classes[lo_idx])
        for ti, di in m2:
            gd = lo_idx[di]
            remaining_tracks[ti].update(boxes[gd], scores[gd], classes[gd])

        # tracks that matched nothing in either stage
        matched_remaining = {ti for ti, _ in m2}
        lost_tracks = [remaining_tracks[i] for i in range(len(remaining_tracks))
                       if i not in matched_remaining]

        # 4) promote / prune
        for t in self.tracks:
            if t.state == TENTATIVE and t.hits >= self.min_hits and t.time_since_update == 0:
                self._confirm(t)
        for t in lost_tracks:
            if t.time_since_update > self.max_age:
                t.state = DELETED
            elif t.state == TENTATIVE:
                # a tentative track that missed a frame never survives -> drop it
                t.state = DELETED
        self.tracks = [t for t in self.tracks if t.state != DELETED]

        # 5) spawn new tracks from unmatched HIGH detections above new_track_thresh
        for di in un_hi:
            gd = hi_idx[di]
            if scores[gd] >= self.new_track_thresh:
                self.tracks.append(Track(boxes[gd], scores[gd], classes[gd], self._next_global_id))
                self._next_global_id += 1

        return [t for t in self.tracks if t.is_confirmed]

    def _confirm(self, t: Track):
        t.state = CONFIRMED
        t.frozen_class = t.voted_class            # freeze label at confirmation
        self._class_seq_counters[t.frozen_class] += 1
        t.class_seq = self._class_seq_counters[t.frozen_class]
        t.display_id = f"{t.frozen_class}_{t.class_seq}"
