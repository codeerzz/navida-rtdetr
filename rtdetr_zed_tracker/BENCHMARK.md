# Component Benchmark — rtdetr_zed_tracker

Per-component profiling of every pure compute unit, plus live per-process cost. Complements
`PERF.md` (which measures end-to-end topic rates). Measured on Jetson AGX Orin, Isaac ROS container,
2026-07-24.

- **Micro-benchmarks:** `scripts/benchmark_components.py` — each function in isolation, median of many
  runs, realistic buoy-scene inputs (7 objects, source 1280×720, net 640×640, depth 1280×720).
  Reproduce: `python3 src/rtdetr_zed_tracker/scripts/benchmark_components.py`.
- **Live per-process:** `ps` on the running stack.
- Note: micro-benchmarks ran *while the live pipeline was up* (GPU + tracker at load), so absolute
  numbers are slightly pessimistic — the ranking and orders of magnitude are what matter.

---

## 1. Pure-component compute cost (single-thread, no ROS/DDS)

| Component | Function | Per call | Ceiling | Scales with |
|-----------|----------|---------:|--------:|-------------|
| **bbox_utils** | `iou_matrix` 7×7 | 42.8 µs | 23,300 Hz | O(T·D) |
| | `iou_matrix` 20×20 | 57.8 µs | 17,300 Hz | |
| | `iou_matrix` 50×50 | 130.7 µs | 7,600 Hz | |
| | `inverse_letterbox` 7 boxes | 44.8 µs | 22,300 Hz | O(boxes) |
| | `xyxy_to_xyah` 7 | 21.4 µs | 46,800 Hz | |
| | `xyah_to_xyxy` 7 | 40.1 µs | 24,900 Hz | |
| **kalman_box** | `predict` (1) | 17.4 µs | 57,500 Hz | O(tracks) |
| | `update` (1) | 61.8 µs | 16,200 Hz | |
| | 7-track predict+update cycle | 543.9 µs | 1,840 Hz | |
| **byte_tracker** ⭐ | `update` **7 objs** | **1.76 ms** | **569 Hz** | O(T·D) + Hungarian |
| | `update` 20 objs | 4.91 ms | 204 Hz | |
| | `update` 50 objs | 11.75 ms | 85 Hz | |
| **depth_utils** | `depth_to_metres` 1280×720 16UC1 | 2.83 ms | 354 Hz | O(pixels) |
| | `sample_box_depth` (1 box) | 199 µs | 5,000 Hz | O(ROI px) |
| | `sample_box_depth` 7 boxes | 1.21 ms | 828 Hz | O(boxes·ROI) |
| | `rescale_intrinsics` | 0.59 µs | 1.68 M Hz | O(1) |
| | `source_box_to_depth_px` | 1.47 µs | 680 k Hz | O(1) |
| | `deproject` (1 pt) | 0.41 µs | 2.47 M Hz | O(1) |
| **end-to-end** | tracker.update + 7-box depth | **3.65 ms** | **274 Hz** | per frame |

⭐ = the hot path. **Total pure compute budget per frame ≈ 3.65 ms → a 274 Hz ceiling** for a 7-object
scene. Everything downstream is far above the ~30 Hz the system actually needs.

## 2. Live per-process cost (whole stack running, ~45 Hz tracking / ~29 Hz output)

| Process | CPU | RSS | Role |
|---------|----:|----:|------|
| `component_container` | ~183 % | 1864 MB | isaac RT-DETR + ZED NITROS (GPU orchestration) |
| **`tracker_node`** | **~132 %** | 97 MB | RGB ByteTrack |
| `depth_fusion_node` | ~18 % | 388 MB | depth sampling + deproject + smoothing |
| `viewer_node` | ~3 % | 60 MB | stdout table @ 4 Hz |

(On a 12-core Orin, 100 % = one core. `component_container` at 183 % is the vision stack; our three
Python nodes together use <1.6 cores.)

---

## 3. Analysis — where the time actually goes

**The algorithms are not the bottleneck.** ByteTrack on a 7-object scene is 1.76 ms (569 Hz ceiling) but
only needs to run at ~45 Hz. Depth fusion's whole per-frame math is ~4 ms. The GPU vision stages
(RT-DETR + NEURAL_LIGHT depth, ~85 % GPU per `PERF.md`) set the real ceiling, not any Python compute.

**`tracker_node` costs ~132 % CPU while its algorithm is 1.76 ms/frame.** At 45 Hz the algorithm needs
only 45 × 1.76 ms ≈ **79 ms/s ≈ 8 % of one core**. The other ~124 % is **ROS/DDS overhead**: `Detection2DArray`
deserialization, message construction, per-callback Python/numpy setup, and UDP-loopback traffic —
paid every one of the 45 frames/s. So the glue costs ~15× the tracker itself. It is *not* limiting
throughput (GPU is), and 1.3 cores on a 12-core Orin is comfortable, but it's the one number that stands
out.

**Scaling headroom is large.** ByteTrack stays real-time to ~50 objects (85 Hz ceiling). IoU is O(T·D)
and grows to 131 µs at 50×50 — still negligible. Buoy scenes are ≤10 objects, deep inside the safe zone.

**Biggest single pure op = `depth_to_metres` (2.83 ms).** It converts the full 0.92 MPix image to
float64 every frame. That's the only pure function worth optimizing if CPU ever mattered — see below.

## 4. Optional levers (none currently needed)

Ordered by payoff; **do not apply unless a real constraint appears** — the system is comfortably
real-time and GPU-bound.

1. **Throttle `tracker_node` to the consumed rate.** The fused output is 29 Hz and the viewer 4 Hz;
   running the tracker at 45 Hz wastes ~40 % of its CPU. Rate-limiting input (or matching detector rate)
   would cut tracker CPU roughly proportionally. Highest payoff, zero quality cost.
2. **`depth_to_metres` in float32, not float64** (or keep 16UC1 in mm and scale only sampled pixels).
   ~2.8 ms → roughly halved; removes the largest pure op. Only matters if depth fusion CPU grows.
3. **Downsample depth** (`depth_downsample_factor`) — cuts `depth_to_metres` and `sample_box_depth`
   quadratically and lowers GPU depth cost, at the price of depth spatial resolution.

**C++ port (Phase 8): still not justified.** The Python algorithms have 6–19× headroom over the required
rate; the cost that stands out (`tracker_node` glue) is ROS/DDS message handling, which a C++ rewrite
reduces but does not eliminate — and it isn't limiting the system. Rewriting buys nothing measurable
until the GPU ceiling is raised first.

## 5. Reproduce

```bash
# pure-component micro-benchmarks (no hardware needed)
python3 /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/benchmark_components.py

# live per-process cost (stack must be running)
ps -eo pcpu,pmem,rss,comm --sort=-pcpu | grep -E "tracker_node|depth_fusion_node|viewer_node|component_container"
```
