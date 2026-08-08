# rtdetr_zed_tracker

RT-DETR (custom buoy detector) → **ByteTrack** stable IDs → **ZED depth fusion** → per-object 3D
distance. Output is a stream where every tracked object has a stable ID, a class name, and an honest
distance — e.g. `green_buoy_1 | green_buoy | 2.62 m`.

Runs inside the Isaac ROS container on Jetson AGX Orin (ROS 2 Humble, Fast DDS).

---

## Architecture

```
                RT-DETR (rgb) ── /detections_output ─┐
                                                     ▼
                                          tracker_node  ── /tracker_node/tracks_2d ─┐
   ZED ── /zed_node/depth/depth_registered ───────────────────────────────────────┤
       ── /zed_node/depth/camera_info ─────────────────────────────────────────────┤
                                                                                    ▼
                                                                        depth_fusion_node
                                                                                    │
                     /depth_fusion_node/tracked_objects  (per-object ID+class+distance)
                     /depth_fusion_node/tracks_3d, _predicted, tracks_markers, sync_stats
                                                                                    │
                                                                              viewer_node
                                                                          (live stdout table)
```

**Color refinement is a third, optional, independent stage.** `color_classification_node` subscribes to
the same ZED color image + `/tracker_node/tracks_2d` and re-checks `red_buoy`/`green_buoy` tracks with a
lighting-robust YCrCb threshold (see `rtdetr_zed_tracker/color_classifier.py`), majority-voting per track
before correcting the label on `~/tracks_color_refined`. It fixes color flips under water reflections and
shadows — the RGB detector's own color guess is otherwise trusted as-is. Toggle with the
`enable_color_refinement` launch arg; disabling it does not affect `tracker_node` or `depth_fusion_node`.

**Optional: decouple shape from color without retraining.** RT-DETR was trained on 7 classes that bundle
shape and color together (`red_buoy`, `green_buoy`, `north_buoy`, ...). `class_remap_node` (default off —
`enable_class_remap` launch arg) sits *before* `tracker_node` and collapses those to a single generic
`buoy` (see `config/class_remap.yaml`), so `tracker_node` never needs to be touched. Pair it with
`color_classification_node` (which treats `buoy` as colorable by default) to have YCrCb decide red vs.
green instead of trusting the class RT-DETR was trained to predict.

**RGB tracking and depth are separate by design.** `tracker_node` is pure RGB (Kalman + ByteTrack on
detections — no depth). `depth_fusion_node` takes each RGB track box, samples ZED depth in its central
ROI, deprojects to (X, Y, Z), and smooths per-track. Depth is *fused onto* tracks, never used to track.

---

## Run it — one command (single terminator window, 4 panes)

Needs **terminator** (once): `sudo apt-get update && sudo apt-get install -y terminator`.

From the **host** (Jetson):
```bash
bash .../scripts/start_all.sh            # FULL: 3x2 grid, overlay + rviz
bash .../scripts/start_all.sh headless   # HEADLESS: 2x2, no GUI
bash .../scripts/start_all_headless.sh   # same as `headless`
```
**FULL** — one terminator window, 2×2: **1** pipeline · **2** tracker+fusion **(+overlay)** · **3**
distance table · **4** rviz. rviz shows both the 3D markers *and* the overlay image (the "Overlay"
image display, enabled by default) — so no separate rqt window is needed.
**HEADLESS** — 2×2, no graphical viz: **1** pipeline · **2** tracker+fusion **(no overlay)** · **3**
distance table · **4** free shell. Use this on the robot / over SSH / for performance runs.

Both cold-start the container first (via the unmodified `run_dev.sh`, in a separate "container host"
window you keep open) if it isn't already running.

Panes attach with `docker exec` directly rather than through `run_dev.sh`'s argument passthrough — the
latter ends in `/bin/bash $@` (unquoted), which word-splits a multi-word command so **nothing runs**
(panes open into the container but launch nothing). See `scripts/_attach.sh`.

> **Overlay note:** the overlay is only produced when Terminal 2 runs `run_tracking.sh`
> **`enable_overlay:=true`** — that flag is what launches `overlay_node`. `start_all.sh` passes it
> automatically. If you start things by hand without it, `/overlay_node/tracks_overlay` never appears
> and the view stays blank (no error).

## Run it manually (each block is its own terminal, all inside the container)

`docker exec -it isaac_ros_dev-aarch64-container bash`

### 1. ZED + RT-DETR pipeline
```bash
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=zed_mono_rect,rtdetr \
  engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/rtdetr/best.plan \
  interface_specs_file:=/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_rtdetr/zed_quickstart_interface_specs.json \
  confidence_threshold:=0.3
```
Wait for RT-DETR detections to start flowing before the next step.

### 2. Tracker + depth fusion
```bash
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_tracking.sh
#   add  enable_overlay:=true   for the RGB overlay (rqt_image_view /overlay_node/tracks_overlay)
```

### 3. Live distance table (Phase 5 viewer)
```bash
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_viewer.sh
```
```
tracked objects: 2   (sorted by Z ascending)
ID            CLASS        CONF     Z(m)  RANGE(m)  VALID%    AGE
------------------------------------------------------------------
green_buoy_1  green_buoy   0.90     2.34      2.41     87%    142
red_buoy_2    red_buoy     0.90     5.10      5.25     41%     23
```

### Optional — RViz 3D markers
```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml
rviz2 -d /workspaces/isaac_ros-dev/install/rtdetr_zed_tracker/share/rtdetr_zed_tracker/config/tracker.rviz
```
If RViz shows a Fixed Frame error, confirm the depth frame and update it:
`ros2 topic echo /zed_node/depth/depth_registered --field header.frame_id --once`

---

## Outputs

| Topic | Type | Meaning |
|-------|------|---------|
| `/depth_fusion_node/tracked_objects` | `rtdetr_zed_tracker_msgs/TrackedObjectArray` | **primary output** — ID, class, `distance_z_m`, `distance_range_m`, position, validity |
| `/depth_fusion_node/tracks_3d` | `vision_msgs/Detection3DArray` | measured 3D boxes |
| `/depth_fusion_node/tracks_3d_predicted` | `vision_msgs/Detection3DArray` | Kalman-predicted 3D |
| `/depth_fusion_node/tracks_markers` | `visualization_msgs/MarkerArray` | RViz cubes + `id \| class \| Z m` |
| `/depth_fusion_node/sync_stats` | `std_msgs/String` | RGB↔depth match rate / stamp offset |
| `/tracker_node/tracks_2d` | `vision_msgs/Detection2DArray` | 2D tracks (pre-fusion) |

`distance_z_m` = forward (optical Z). `distance_range_m` = Euclidean range √(X²+Y²+Z²). They differ
30 %+ off-axis, so both are published — **never a bare "distance".** Invalid depth → **NaN**, never 0.

---

## Gotchas (full detail in `NOTES.md`)

1. **UDP profile is mandatory.** Any node/CLI joining the pipeline must export
   `FASTRTPS_DEFAULT_PROFILES_FILE=.../udp_only_profile.xml`. Without it: **zero messages, no error.**
   The `run_*.sh` scripts set it; manual `ros2` commands don't.
2. **Never two `tracker_node` instances** — duplicate publishers look like ID swaps / reversals.
   `run_tracking.sh` pre-kills stale ones. Verify: `ros2 topic info /tracker_node/tracks_2d`
   (Publisher count must be 1).
3. **The isaac `zed_mono_rect` bundle reads `zed.yaml`, not `common_stereo.yaml`** — depth is enabled
   there (`depth_mode: NEURAL_LIGHT`, `openni_depth_mode: true`).
4. Effective framerate is compute-bound (RT-DETR + NEURAL_LIGHT share the Orin GPU), not the 60 Hz
   grab/pub setting.

---

## Build

```bash
cd /workspaces/isaac_ros-dev
colcon build --symlink-install --packages-select rtdetr_zed_tracker rtdetr_zed_tracker_msgs
```

## Test (15 unit tests, no hardware needed)

```bash
cd /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker && python3 -m pytest test/ -v
```

---

## Package layout

- `rtdetr_zed_tracker/byte_tracker.py`, `kalman_box.py`, `bbox_utils.py` — pure tracker (no ROS).
- `rtdetr_zed_tracker/tracker_node.py` — RGB tracking node → `tracks_2d`.
- `rtdetr_zed_tracker/depth_utils.py`, `depth_fusion_node.py` — depth math + fusion.
- `rtdetr_zed_tracker/overlay_node.py` — stamp-matched box overlay for rqt.
- `rtdetr_zed_tracker/viewer_node.py` — live distance table.
- `config/` — `tracker_params.yaml`, `class_labels.yaml` (7 buoy classes), `tracker.rviz`.
- `launch/tracking.launch.py` — tracker + fusion (+ optional overlay).
- `scripts/` — `start_all.sh` (host launcher, opens all gnome-terminals), `pipeline.sh`,
  `run_tracking.sh`, `run_viewer.sh`, `run_overlay_view.sh`, `run_phase2_test.sh`,
  `benchmark_components.py`.
- `udp_only_profile.xml` — required Fast DDS profile.
