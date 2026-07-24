# Phase 7 — Performance measurements

Measured live on Jetson AGX Orin, Isaac ROS container, 2026-07-24, with the full stack running
(ZED `zed_mono_rect` + RT-DETR + tracker + depth fusion + viewer). All ROS measurements taken through
the UDP-only Fast DDS profile.

## Throughput (`ros2 topic hz`, ~10 s windows)

| Topic | Rate | Notes |
|-------|------|-------|
| `/zed_node/left/image_rect_color` (RT-DETR input) | ~22 Hz* | *subscriber-limited — see caveat |
| `/zed_node/depth/depth_registered` | ~20 Hz* | *subscriber-limited |
| `/detections_output` (RT-DETR) | **~45 Hz** | small msg → accurate |
| `/tracker_node/tracks_2d` | **~45 Hz** | tracks 1:1 with detections |
| `/depth_fusion_node/tracked_objects` (**final output**) | **~29 Hz** | capped by depth sync |

\* **Caveat:** the color/depth topics carry 3.69 MB images. A single Python `ros2 topic hz` subscriber
over UDP loopback drops frames, so 22/20 Hz **undercounts** the true camera rate. The small topics
(detections, tracks) are measured accurately at ~45 Hz — that's the real detector/tracker rate. The
fused output is a genuine ~29 Hz (limited by nearest-stamp depth matching, not the transport).

## Bandwidth (`ros2 topic bw`)

| Topic | BW | Msg size |
|-------|-----|----------|
| `/zed_node/depth/depth_registered` | 73.5 MB/s | 3.69 MB |
| `/zed_node/left/image_rect_color` | 27.9 MB/s | 3.69 MB |
| `/depth_fusion_node/tracked_objects` | negligible | tiny struct |

Aggregate over loopback is modest (~100 MB/s); nowhere near a transport bottleneck.

## RGB↔depth sync (`/depth_fusion_node/sync_stats`)

```
msgs=13207 matched=10617 (80%) dropped_no_depth=2539 mean_off=3.4ms tracks=1
```
- **80 % of RGB tracks get a depth match**, mean stamp offset **3.4 ms** (well inside the 50 ms window).
- The 20 % dropped = frames where the ~45 Hz RGB track had no ~20 Hz depth frame close enough — expected
  given the depth rate is roughly half the RGB rate. Per-object `depth_valid` / `VALID%` (visible live in
  the viewer) reflects this at the object level.

## System load (`tegrastats`, 6 samples)

| Metric | Value |
|--------|-------|
| **GPU (GR3D_FREQ)** | **70–96 %, ~85 % avg** ← the bottleneck |
| CPU | 2–3 cores pegged 100 %, remainder 40–60 % |
| RAM | 12.8 / 62.8 GB |
| Power | GPU_SOC 21.8 W · CPU 6.7 W · SYS 8.4 W → **~37 W** |
| Temps | ~64 °C (cpu/gpu/tj) — healthy, no throttling |

The Orin GPU is shared by RT-DETR (TensorRT) and ZED `NEURAL_LIGHT` depth; at ~85 % it's the limiting
resource. ~15 % headroom remains.

## Verdict

**Comfortably real-time — no optimization needed.** Final per-object 3D output at ~29 Hz with tracking
at ~45 Hz is well above what buoy tracking requires. GPU is the ceiling (~85 %), not CPU, RAM, or
transport. UDP-loopback transport is fine; **SHM / `--ipc=host` is not worth pursuing** (bandwidth is low
and latency already ~3 ms sync).

If more fused-3D rate is ever needed, the only real lever is the depth stage (raise depth rate via
`depth_downsample_factor`, or drop `NEURAL_LIGHT` to a lighter mode) — at the cost of GPU load or depth
quality. Not recommended unless a downstream consumer demands >29 Hz.

**Phase 8 (C++ port): not justified.** The Python nodes are not the bottleneck (they ride at 45/29 Hz on
a fraction of CPU; the GPU vision stages dominate). Skip it.
