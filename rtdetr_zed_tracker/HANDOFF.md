# HANDOFF — RT-DETR + ByteTrack + ZED depth 3D tracking

**Resume point for 2026-07-24.** Start here. Deep discovery details are in `NOTES.md`.

Goal: per-object stream where each tracked object has a stable ID, a class name, and a
distance — e.g. `green_buoy_1 | green_buoy | 2.62 m`. **Working end-to-end today.**

---

## 1. Where we are

| Phase | State |
|-------|-------|
| 0 Discovery | ✅ done (`NOTES.md`) |
| 1 Pure-Python ByteTrack | ✅ done, 6/6 tests |
| 2 tracker_node + overlay | ✅ done, alignment verified |
| 3 Message package | ✅ done (`rtdetr_zed_tracker_msgs`) |
| 4 Depth fusion | ✅ **done + verified live** (distance-per-track) |
| 5 Human-readable viewer | ✅ done (`viewer_node`) |
| 6 Integration (README + rviz, isaac bundle kept) | ✅ done (`README.md`, `config/tracker.rviz`) |
| 7 Measure / optimize | ✅ done (`PERF.md` topic rates, `BENCHMARK.md` per-component) — GPU-bound ~85%, no opt needed |
| 8 C++ port | ❌ not justified (Python not the bottleneck; GPU dominates) |

**Verified live today:** RGB tracking 58 Hz · depth 27 Hz · fusion 90% depth-match, 0.9 ms stamp
offset · sample output `green_buoy_1: distance_z_m=2.62, distance_range_m=2.69`. All 15 unit tests pass.

---

## 2. How to run it (inside the container)

```bash
# Terminal 1 — pipeline (only if not already running):
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=zed_mono_rect,rtdetr \
  engine_file_path:=/workspaces/isaac_ros-dev/isaac_ros_assets/models/rtdetr/best.plan \
  interface_specs_file:=/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_rtdetr/zed_quickstart_interface_specs.json \
  confidence_threshold:=0.3

# Terminal 2 — tracker + depth fusion (one command; sets UDP profile, kills stale dups):
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_tracking.sh
#   add  enable_overlay:=true  for the RGB overlay

# Terminal 3 — live distance table (its own terminal; clears screen):
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_viewer.sh

# Other views (MUST export the profile first, else zero msgs silently):
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml
ros2 topic echo /depth_fusion_node/tracked_objects        # raw distances per object
# rqt_image_view /overlay_node/tracks_overlay             # (needs enable_overlay:=true)
# rviz2  -> MarkerArray on /depth_fusion_node/tracks_markers
```

Rebuild after code edits: `cd /workspaces/isaac_ros-dev && colcon build --symlink-install
--packages-select rtdetr_zed_tracker rtdetr_zed_tracker_msgs`

---

## 3. Architecture (RGB tracking is separate from depth — by design)

```
RGB:   RT-DETR(rgb) -> /detections_output -> tracker_node -> /tracker_node/tracks_2d
DEPTH: ZED depth ----------------------------\
                                              depth_fusion_node -> /depth_fusion_node/tracked_objects
                          /tracker_node/tracks_2d ------/          (+ tracks_3d, _predicted, markers, sync_stats)
```
tracker_node = pure RGB (Kalman/ByteTrack on detections). depth_fusion_node = takes each RGB track box,
samples ZED depth in its central ROI, deprojects to (X,Y,Z), smooths per-track (EMA + jump reject).

---

## 4. File map (`src/rtdetr_zed_tracker/`)

- `rtdetr_zed_tracker/byte_tracker.py` `kalman_box.py` `bbox_utils.py` — pure tracker (no ROS).
- `rtdetr_zed_tracker/tracker_node.py` — RGB tracking node -> tracks_2d.
- `rtdetr_zed_tracker/overlay_node.py` — stamp-matched box overlay for rqt.
- `rtdetr_zed_tracker/depth_utils.py` — pure depth math (encoding, deproject, ROI sample).
- `rtdetr_zed_tracker/depth_fusion_node.py` — RGB tracks + depth -> 3D distance per ID.
- `config/tracker_params.yaml` `config/class_labels.yaml` (7 buoy classes).
- `launch/tracking.launch.py` — tracker + fusion (+ optional overlay).
- `scripts/run_tracking.sh` — one-command launcher. `scripts/run_phase2_test.sh` — tracker+overlay only.
- `test/test_byte_tracker.py` `test/test_depth_utils.py` — 15 tests.
- `udp_only_profile.xml` — REQUIRED Fast DDS profile (see gotchas).
- `phase0_diag.py` `phase2_*_diag.py` `phase2_overlay*.jpg` — one-off debug artifacts (user chose to keep).
- `rtdetr_zed_tracker_msgs/` — TrackedObject / TrackedObjectArray.

---

## 5. Config state (edited files — git-tracked, a `git pull` could revert)

| File | Setting | Value |
|------|---------|-------|
| `.../isaac_ros_rtdetr/.../isaac_ros_rtdetr_core.launch.py` | `scale` | **True** (÷255 norm — makes custom model detect) |
| `zed-ros2-wrapper/.../config/zed.yaml` | grab | HD720 @ 60; `pub_frame_rate 60`; `pub_resolution NATIVE` |
| `zed.yaml` depth section | `depth_mode` NEURAL_LIGHT, `openni_depth_mode` true | **enables depth** (isaac bundle reads zed.yaml, NOT common_stereo.yaml) |
| `zed-ros2-wrapper/.../config/common_stereo.yaml` | pub_frame_rate 60, openni true | (only used if ZED run standalone; isaac bundle ignores it) |
| `isaac_ros_assets/.../zed_quickstart_interface_specs.json` | camera_resolution | **1280×720** (must match ZED res or NITROS pool crash) |
| `config/tracker_params.yaml` | source_image_* | 1280 / 720 |

---

## 6. Gotchas (must-know — full detail in NOTES.md §7/§10)

1. **UDP profile is mandatory** for any node/CLI that joins the pipeline
   (`FASTRTPS_DEFAULT_PROFILES_FILE=.../udp_only_profile.xml`). Without it: **zero messages, no error.**
   Fast DDS SHM does not deliver to separate participants here.
2. **Duplicate publishers = fake reversals / ID swaps.** Never leave two tracker_node instances up.
   `run_tracking.sh` pre-kills stale ones. Check with `ros2 topic info /tracker_node/tracks_2d`
   (Publisher count must be 1).
3. **Host vs container:** my shell = host (Jetson); ROS runs in the container. Workspace is bind-mounted
   (host `/mnt/nova_ssd/workspaces/isaac_ros-dev` == container `/workspaces/isaac_ros-dev`).
   Killing container node procs needs `docker exec -u root`.
4. **isaac zed_mono_rect bundle reads `zed.yaml`, not `common_stereo.yaml`** — put ZED overrides in zed.yaml.
5. Framerate effective ~ (grab/pub 60 but compute-bound). Depth NEURAL_LIGHT costs grab throughput.

---

## 7. NEXT

- **Phase 6 — DONE**: user chose to keep the isaac `zed_mono_rect` bundle (depth via zed.yaml, verified).
  Added `README.md` (full run recipe + outputs table + gotchas) and `config/tracker.rviz`
  (MarkerArray on `/depth_fusion_node/tracks_markers`, Fixed Frame `zed_left_camera_optical_frame` —
  fix if `depth_registered` header frame differs). No new launch file needed; the 3-terminal flow
  (isaac pipeline · run_tracking.sh · run_viewer.sh) is the integration.
- **Phase 7 — measure**: `ros2 topic bw/hz`, e2e latency, tegrastats, depth_valid distribution; decide if
  the UDP-loopback transport / ~25 fps compute cap needs addressing (SHM/`--ipc=host`).

## 8. Open questions for the user
- Depth fusion output: is `distance_z_m` per track what you want downstream, or a different fusion product?
- Phase 6: keep single isaac launch (simplest) or split ZED into its own process (spec's design)?
