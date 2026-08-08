# rtdetr_zed_tracker

RT-DETR (custom buoy detector) → **ZED depth fusion** → per-object 3D distance. Output is a stream
where every detected object has a class name and an honest distance — e.g. `green_buoy | 2.62 m`.

Runs inside the Isaac ROS container on Jetson AGX Orin (ROS 2 Humble, Fast DDS).

---

## Architecture

```
                RT-DETR (rgb) ── /detections_output ─┐
                                                     ▼
                                          [ class_remap_node ]        optional
                                                     ▼
                                     [ color_classification_node ]    optional
                                                     │
   ZED ── /zed_node/depth/depth_registered ──────────┤
       ── /zed_node/depth/camera_info ───────────────┤
                                                     ▼
                                            depth_fusion_node
                                                     │
                     /depth_fusion_node/tracked_objects  (per-object class+distance)
                     /depth_fusion_node/tracks_3d, tracks_markers, sync_stats
                                                     │
                                        [dist] log line, 1 Hz, in the node's own pane
```

**There is no image-space tracker.** `depth_fusion_node` consumes RT-DETR detections directly and is
stateless across frames: one honest, unsmoothed measurement per box per frame. Identity is established
downstream on the **host** by `buoy_mapper_node`, which associates observations by Mahalanobis distance
in world coordinates. See the commit message for `996e2ac` for why the ByteTrack stage and the per-track
EMA were removed (short version: the EMA correlated observations that the mapper's information filter
assumes are independent).

**Two optional enrichment stages, both off by default.** Each is a plain topic-in/topic-out filter, and
`tracking.launch.py` computes the next stage's input topic — so a disabled stage is simply skipped and
nothing else on the graph needs to know. With both off you get exactly the bare pipeline above.

1. **`class_remap_node`** (`enable_class_remap:=true`) — RT-DETR was trained on 7 classes that bundle
   shape and color together (`red_buoy`, `green_buoy`, `north_buoy`, ...). This collapses them to a
   single generic `buoy` (see `config/class_remap.yaml`) without retraining. Index in, index out.
2. **`color_classification_node`** (`enable_color_refinement:=true`) — re-checks each colorable
   detection's crop against a lighting-robust YCrCb threshold (see `color_classifier.py`) and
   majority-votes the result (`label_vote.py`) before correcting the class. Fixes color flips under
   water reflections and shadows.

Pair them (`enable_class_remap:=true enable_color_refinement:=true`) to decouple shape from color: every
trained class collapses to a shape-only `buoy`, and YCrCb decides red vs. green instead of trusting the
color RT-DETR was trained to predict.

**Everything on this graph speaks class *indices*, not names.** RT-DETR puts the numeric class index in
`class_id` (`"4"`), and `depth_fusion_node` does `int(class_id)`. `class_remap_node` maps index→index.
`color_classification_node` is handed `class_labels.yaml` so it can decide in name space but write the
index back out — without that it would compare `"4"` against `{"buoy", "red_buoy", ...}`, match nothing,
and silently refine nothing.

**Colour voting has no track ids to key on.** `LabelVote` was written against the deleted tracker's
per-track `Detection2D.id`; raw RT-DETR detections carry an empty id. The launch therefore runs
`color_vote_key:=grid`, which keys votes on the box centre quantised to `color_vote_cell_px` (64 px).
A buoy holds its cell far longer than the ~3 frames a vote needs at 45 Hz, and a cell that goes unseen
for one frame is dropped, so a frozen vote can't outlive its object. This substitutes proximity for
identity, which is weaker than a real track id — the right long-term home for label voting is
`buoy_mapper_node` on the host, where genuine world-frame identity exists. `LabelVote` is deliberately
pure Python with zero ROS imports so it can move there unchanged.

---

## Run it — one command (single terminator window, 4 panes)

Needs **terminator** (once): `sudo apt-get update && sudo apt-get install -y terminator`.

From the **host** (Jetson):
```bash
bash .../scripts/start_all.sh            # FULL: 2x2 grid, overlay + rviz
bash .../scripts/start_all.sh headless   # HEADLESS: 2x2, no GUI
bash .../scripts/start_all_headless.sh   # minimal: 2 panes, nothing else

bash .../scripts/start_all_color.sh          # FULL, + colour refinement
bash .../scripts/start_all_color.sh headless # HEADLESS, + colour refinement
```
**FULL** — one terminator window, 2×2: **1** pipeline · **2** fusion **(+overlay)** · **3**
live topic dump · **4** rviz. rviz shows both the 3D markers *and* the overlay image (the "Overlay"
image display, enabled by default) — so no separate rqt window is needed.
**HEADLESS** — 2 panes, no graphical viz: **1** pipeline · **2** fusion **(no overlay)**. Distances
appear in pane 2 as `[dist]` lines once a second, nearest first. Use this on the robot / over SSH /
for performance runs.

### Colour is a separate launcher, on purpose

`start_all.sh` and `start_all_headless.sh` are **colour-free**: they run the bare
`RT-DETR → depth fusion` pipeline. `start_all_color.sh` is the same window, same panes and same cold
start, with the two optional stages added to pane 2 — it reuses `start_all.sh` rather than copying it,
so the launchers cannot drift apart.

The split exists because colour is not free. Measured on the Orin with the ZED live: enabling the two
stages drops `depth_fusion_node`'s depth match rate from **~100 % to ~71 %** (watch
`[sync] ... dropped_no_depth`), because two more nodes subscribe to the full-res colour image
(3.7 MB @ 30 Hz) over the UDP loopback. Roughly a quarter of the range measurements is the price of
the colour label — worth paying when you need colour, not by default.

All of them cold-start the container first (via the unmodified `run_dev.sh`, in a separate
"container host" window you keep open) if it isn't already running.

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

### 2. Depth fusion
```bash
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_tracking.sh
#   add  enable_overlay:=true   for the RGB overlay (rqt_image_view /overlay_node/tracks_overlay)
```

**With colour refinement** (both optional stages, the intended pairing):
```bash
bash .../scripts/run_tracking.sh enable_class_remap:=true enable_color_refinement:=true
```
Every trained class collapses to a shape-only `buoy`, then YCrCb decides red vs. green. Add
`enable_overlay:=true` to watch the correction happen — the overlay draws the same detections the
fusion node consumes, with names resolved from `class_labels.yaml`, so a recoloured box visibly
changes on screen instead of only showing up in a log line.

Both default to **off**: with no arguments this launch is the bare
`RT-DETR → depth fusion` pipeline, byte for byte what it was before these stages existed.

Other arguments: `color_ranges_file` (YAML overrides), `color_vote_key` (`id`|`grid`|`none`,
default `grid`), `color_vote_cell_px` (default 64), `class_remap_file`. Full list:
```bash
ros2 launch rtdetr_zed_tracker tracking.launch.py --show-args
```

### 3. Live distances
`depth_fusion_node` prints them itself, once a second, in its own pane — there is no separate viewer
process any more (`distance_log_hz` in `fusion_params.yaml`; `0` turns it off):
```
[dist] 2/3 ranged | red_buoy 4.21m (r=4.35 c=0.88 v=82%)  |  green_buoy 6.90m (...)  |  buoy NO-DEPTH (...)
```
Nearest first. Objects the detector saw but could **not** range are listed last and marked
`NO-DEPTH` — a buoy you can see but not range is a different, more dangerous situation than one you
never saw, so it is reported rather than dropped.

For every field, or a readable running dump:
```bash
ros2 topic echo /depth_fusion_node/tracked_objects
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_topics.sh
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
| `/depth_fusion_node/tracks_markers` | `visualization_msgs/MarkerArray` | RViz cubes + `id \| class \| Z m` |
| `/depth_fusion_node/sync_stats` | `std_msgs/String` | RGB↔depth match rate / stamp offset |
| `/class_remap_node/detections_remapped` | `vision_msgs/Detection2DArray` | only when `enable_class_remap:=true` |
| `/color_classification_node/detections_color_refined` | `vision_msgs/Detection2DArray` | only when `enable_color_refinement:=true` |

`distance_z_m` = forward (optical Z). `distance_range_m` = Euclidean range √(X²+Y²+Z²). They differ
30 %+ off-axis, so both are published — **never a bare "distance".** Invalid depth → **NaN**, never 0.

---

## Gotchas (full detail in `NOTES.md`)

1. **UDP profile is mandatory.** Any node/CLI joining the pipeline must export
   `FASTRTPS_DEFAULT_PROFILES_FILE=.../udp_only_profile.xml`. Without it: **zero messages, no error.**
   The `run_*.sh` scripts set it; manual `ros2` commands don't.
2. **Never two `depth_fusion_node` instances** — duplicate publishers make one object look like two,
   with fighting distances. `run_tracking.sh` pre-kills stale ones, and `start_all*.sh` refuse to stack
   a second pipeline on a running one. Verify:
   `ros2 topic info /depth_fusion_node/tracked_objects` (Publisher count must be 1).
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

## Test (no hardware needed)

```bash
cd /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker && python3 -m pytest test/ -v
```

`test_color_classification_node.py` needs `rclpy` + `cv_bridge` + `vision_msgs` and **skips itself**
when they aren't importable, so the ROS-free suite (`test_color_classifier.py`, `test_label_vote.py`,
`test_class_remap.py`, `test_depth_utils.py`) still runs on a laptop. Run the full suite **inside the
container** — the Jetson host's `cv_bridge` is built against a different numpy ABI and fails to import
(`_ARRAY_API not found`), which silently reduces the run to the ROS-free tests.

---

## Package layout

- `rtdetr_zed_tracker/depth_utils.py`, `depth_fusion_node.py` — depth math + fusion (the core stage).
- `rtdetr_zed_tracker/color_classifier.py`, `label_vote.py` — YCrCb classification + vote freezing.
  Both are pure Python/numpy with **zero ROS imports**, so they are unit-testable on a laptop and
  `label_vote.py` can be lifted into `buoy_mapper_node` on the host unchanged.
- `rtdetr_zed_tracker/color_classification_node.py` — optional colour refinement stage (ROS glue).
- `rtdetr_zed_tracker/class_remap.py`, `class_remap_node.py` — optional class-collapse stage.
- `rtdetr_zed_tracker/overlay_node.py` — stamp-matched box overlay for rqt.
- `config/` — `fusion_params.yaml`, `class_labels.yaml` (7 buoy classes), `class_remap.yaml`,
  `color_ranges.yaml`, `tracker.rviz`.
- `launch/tracking.launch.py` — fusion + the two optional stages (+ optional overlay).
- `scripts/` — `start_all.sh` (host launcher), `start_all_color.sh` (same, with colour),
  `start_all_headless.sh`, `_attach.sh` (per-pane stage dispatch), `pipeline.sh`,
  `run_tracking.sh`, `run_topics.sh`, `run_overlay_view.sh`, `run_rviz.sh`,
  `benchmark_components.py`, `webcam_color_demo.py` (colour range calibration).
- `udp_only_profile.xml` — required Fast DDS profile.
