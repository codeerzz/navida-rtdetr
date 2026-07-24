#!/usr/bin/env python3
"""Synthetic reversal discriminator (H-A / H-D). No ROS, no bag, no motion needed.

Feeds the tracker a CLEAN, noiseless, strictly-monotonic downward trajectory
(center-y increasing) at several speeds, and reports for each:
  * reversals in the PUBLISHED track center-y (should be 0 for H-D to hold)
  * which frames the track COASTED (no match) vs matched  -> H-A signature
  * the velocity state estimate, and whether the track ID stayed stable

A reversal that lines up with a coast frame is the H-A "hold-then-snap" pattern.
"""
import numpy as np

from rtdetr_zed_tracker.byte_tracker import ByteTracker


def box(cy, cx=640.0, w=60.0, h=60.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def run(step, frames=25, drops=frozenset(), **kw):
    tr = ByteTracker(min_hits=3, class_gating='none', **kw)
    rows = []
    for f in range(frames):
        cy_true = 100.0 + step * f            # strictly increasing (downward)
        dets = [] if f in drops else [(box(cy_true), 0.9, 0)]
        tr.update(dets, dt=1.0)
        conf = [t for t in tr.tracks if t.is_confirmed]
        if conf:
            t = conf[0]
            b = t.xyxy
            rows.append(dict(f=f, cy_true=cy_true, pub_cy=(b[1] + b[3]) / 2,
                             gid=t.global_id, matched=(t.time_since_update == 0),
                             vy=float(t.kf.mean[5])))
    return rows


def reversals(seq):
    d = np.diff(seq)
    idx = [i + 1 for i, dd in enumerate(d) if dd < -1e-6]   # went UP when it should go DOWN
    return idx


def analyse(label, **kw):
    print(f"\n### {label}")
    for step in (10, 25, 40, 60):
        rows = run(step, **kw)
        pub = [r['pub_cy'] for r in rows]
        ids = sorted({r['gid'] for r in rows})
        coast = [r['f'] for r in rows if not r['matched']]
        rev = reversals(pub)
        print(f"step={step:3d}px/frame  frames={len(rows):2d}  reversals={len(rev)}  "
              f"coast_frames={coast}  distinct_ids={len(ids)}")
        for i in rev:
            r = rows[i]
            print(f"     reversal @track-frame {i}: {pub[i-1]:.1f} -> {pub[i]:.1f}  "
                  f"matched={r['matched']} vy={r['vy']:.2f} id={r['gid']}")


if __name__ == '__main__':
    print("=" * 70)
    print("SYNTHETIC REVERSAL DISCRIMINATOR (tracker-level H-A/H-D)")
    print("clean monotonic-down input; any reversal is introduced BY THE TRACKER")
    print("=" * 70)
    analyse("A) continuous clean detections (H-D baseline)")

    # H-A: confirmed track, then intermittent detection dropouts (RT-DETR misses)
    print("\n### B) with detection DROPOUTS after confirmation (H-A realistic trigger)")
    drops = frozenset({7, 8, 12, 16, 17})
    for step in (10, 25, 40):
        rows = run(step, frames=25, drops=drops)
        pub = [r['pub_cy'] for r in rows]
        coast = [r['f'] for r in rows if not r['matched']]
        rev = reversals(pub)
        print(f"step={step:3d}  drops@{sorted(drops)}  reversals={len(rev)}  coast_frames={coast}  "
              f"ids={len({r['gid'] for r in rows})}")
        for i in rev:
            r = rows[i]
            print(f"     reversal @track-frame {i}: {pub[i-1]:.1f} -> {pub[i]:.1f}  "
                  f"matched={r['matched']} coast_lag vy={r['vy']:.2f} id={r['gid']}")
