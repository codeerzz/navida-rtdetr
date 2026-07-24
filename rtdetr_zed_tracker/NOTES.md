# Phase 0 — Discovery Notes (rtdetr_zed_tracker)

Date: 2026-07-23. Verified from the running system, not from memory.
Status: **Phase 0 COMPLETE.** Static + runtime discovery done. Caveats: depth is currently OFF (§6),
and joining nodes need the UDP Fast DDS profile (§7).

---

## 1. CRITICAL: host / container split (affects everything)

- My tools / shell run on the **host** (Jetson, hostname `ubuntu`, user `jetson`, paths `/mnt/nova_ssd/...`).
- The ROS 2 pipeline runs **inside a Docker container**: `isaac_ros_dev-aarch64-container` (id `bdbd380e0cdc`).
- The workspace is bind-mounted: host `/mnt/nova_ssd/workspaces/isaac_ros-dev` == container `/workspaces/isaac_ros-dev`.
  So file edits/builds on either path are shared.
- `vision_msgs`, `isaac_ros_*`, `zed_*`, and a working ROS Python env exist **only in the container**.
  The host has no `vision_msgs` and a different (JetPack) python.
- ⇒ **All project ROS nodes, colcon builds, and ros2 CLI for this project MUST run inside the container**
  (`docker exec bdbd380e0cdc bash -lc '...'`). Host python is only for the standalone TRT scripts
  (`test_trt.py`, `capture_image.py`), which use the host's TensorRT/pycuda.

## 2. Environment (authoritative = CONTAINER)

| | Container (code runs here) | Host (standalone TRT only) |
|---|---|---|
| python | 3.10.12 | 3.10 (JetPack) |
| numpy | **1.26.4** | 2.2.6 |
| scipy | **1.15.3 — `linear_sum_assignment` WORKS** | broken (numpy 2.2 conflict) |
| cv_bridge | **OK** | OK |
| vision_msgs | **OK** | absent |
| filterpy | **MISSING** | — |
| tensorrt | (n/a) | 10.3.0 |

- `ROS_DISTRO=humble`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, default `ROS_DOMAIN_ID`, **no custom Fast DDS profile**, `/dev/shm` = 31G (healthy).
- **Correction vs approved plan:** scipy is available in the container ⇒ Hungarian assignment is viable
  with no environment changes. The "greedy because scipy broken" premise was a host artifact.
  **DECISION (user, 2026-07-23): Hungarian matching via `scipy.optimize.linear_sum_assignment`.**
  Kalman filter stays **pure-numpy** (filterpy absent).

## 3. Topics (verified via `ros2 topic info -v`)

ZED naming is `/zed_node/...` (NOT `/zed/zed_node/...`).

| Role | Topic | Type | QoS |
|------|-------|------|-----|
| Detections | `/detections_output` | `vision_msgs/msg/Detection2DArray` | RELIABLE / VOLATILE |
| Color (feeds RT-DETR; registered to depth) | `/zed_node/left/image_rect_color` | `sensor_msgs/msg/Image` | RELIABLE / VOLATILE |
| Depth | `/zed_node/depth/depth_registered` | `sensor_msgs/msg/Image` | RELIABLE / VOLATILE |
| Depth intrinsics | `/zed_node/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | RELIABLE / VOLATILE |

Default rclpy QoS (RELIABLE/VOLATILE/KEEP_LAST) matches all four → no QoS gymnastics needed.

## 4. Class ID format (decided)

- `rtdetr_decoder_node.cpp:102`: `hyp.hypothesis.class_id = std::to_string(labels.at(i))`
  → **integer index strings** → labels file REQUIRED.
- Mapping (user-confirmed, from `isaac_ros_assets/models/rtdetr/test_trt.py`):
  `0=buoy 1=east_buoy 2=green_buoy 3=north_buoy 4=red_buoy 5=south_buoy 6=west_buoy`.

## 5. RT-DETR geometry (from `isaac_ros_rtdetr_core.launch.py`, unambiguous)

- source = `interface_specs['camera_resolution']` (currently 1920×1080); network = 640×640.
- resize: `keep_aspect_ratio=True`, `disable_padding=True`; pad: `padding_type='BOTTOM_RIGHT'`.
- ⇒ inverse letterbox: `s = min(640/W, 640/H)`, image anchored **top-left**, pad bottom/right
  ⇒ `x_orig = x_net/s`, `y_orig = y_net/s`, clip to `[0,W]×[0,H]`. (`padding_mode='top_left'`, zero offset.)
- Prereq already applied this session: `image_to_tensor_node` `scale: True` (÷255 → [0,1]) so the custom
  model detects at all.
- No `use_intra_process_comms` set anywhere in the launch stack.

## 6. Timestamp propagation / latency / depth — MEASURED (live, buoy in view)

Ran `phase0_diag.py` (30 s, UDP profile — see §7) with buoys in view. Results:

- **Rates:** image `/zed_node/left/image_rect_color` = **13.7 Hz**; `/detections_output` = **13.8 Hz**
  (≈1 detection msg per frame when objects present); depth = **0 Hz**.
- **Pipeline runs at ~13.7 Hz** at HD1080 + RT-DETR on the Orin (not 30). Frame period ≈ 73 ms.
- **Color image encoding: `bgra8`, 1920×1080** (4-channel — matters for cv_bridge in overlay/fusion).
- **Timestamp propagation ≈ CLASS A (exact).** median det↔image stamp delta = **0.00 ms**; 318/414 exact
  matches; the non-exact tail (max 267 ms) is diagnostic buffer-eviction races, not a real offset.
  ⇒ **`sync_mode` default = `exact`**, with `nearest` (≤ half-frame ≈ 36 ms) as safety fallback.
- **End-to-end data age (now − det.stamp):** min 87 / **median 122** / max 760 ms. This is the staleness
  the tracker sees ⇒ the Kalman `tracks_3d_predicted` output matters.
- **class_id values observed:** `{'4':405, '2':114, '6':14}` = red_buoy / green_buoy / west_buoy.
  Confirms integer-index strings and the §4 label map.
- **DEPTH IS OFF:** 0 depth msgs even over UDP ⇒ the `zed_mono_rect` example disables depth. Phase 4 must
  enable it (separate ZED launch: `depth_mode` ≠ NONE, plus the openni/downsample payload cuts). Depth
  encoding/dims and det↔depth stamp offset still TBD once depth is enabled.

## 7. Fast DDS transport gotcha (was misread as a "hung pipeline")

Symptom: a fresh `docker exec` participant received **zero** data from the long-running pipeline
(all topics silent, incl. `/tf_static`, heartbeat) — even though `ros2 node list` worked and an in-exec
self pub→sub ran at 5 Hz. Initially misdiagnosed as a stalled GXF graph.

Root cause: **the pipeline was fine.** Proof: from the **host** (separate `/dev/shm` ⇒ forced onto UDP),
`/zed_node/left/image_rect_color` read **10–13 Hz** and a live 1080p frame was captured. Fresh
same-container participants default to Fast DDS **SHM/data-sharing**, which did not deliver pipeline data
here; forcing **UDPv4-only** fixed it instantly (412 img + 414 det msgs in 30 s).

**Workaround (REQUIRED for all project nodes that join the running pipeline):**
`export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml`
(UDPv4 transport, `useBuiltinTransports=false`). Applies to the Phase-2 tracker, Phase-4 fusion, viewer,
and any `ros2 topic` debugging via `docker exec`. Revisit properly in Phase 7 (SHM tuning / `--ipc=host`)
before claiming SHM zero-copy anywhere. localhost UDP easily handles these rates.

## 7. BLOCKER: pipeline is hung (as of run at ~15:00)

- Launch `ros2 launch isaac_ros_examples ... zed_mono_rect,rtdetr` (PID 20583) started **14:04**, still alive;
  all nodes discoverable (`ros2 node list` OK).
- But **every topic is silent** — `/tf_static` (transient_local, guaranteed), `/zed_node/status/heartbeat`,
  `/rosout`, color, detections, depth — all 0 msgs.
- Not a transport/QoS/camera issue: a self pub→sub inside the container hit `average rate: 5.001`;
  `/dev/shm` healthy; ZED still on USB (`2b03:f582`). ⇒ the GXF graph / ZED grab has **stalled**.
- **Action needed from user:** restart the pipeline and place a buoy (or trigger object) in view, then
  re-run `phase0_diag.py` in the container to fill in §6.

## 8. Open items to close at execution (remaining after Phase 0)
- Depth encoding/dims/rate + det↔depth stamp offset — deferred to Phase 4 (depth must be enabled first).
- ~~Empirical pad-mode confirmation via the Phase-2 overlay gate.~~ **DONE (Phase 2):** decoder emits
  ORIGINAL 1920×1080 pixels (preprocessor uses `orig_target_sizes=[max(w,h)]²`, `preprocessor:82`);
  verified live — box cx≈1326/cy≈470 and the overlay boxes sit exactly on the buoys. Tracker uses
  `detection_coordinate_space='source_pixels'` (no inverse letterbox); `network_padded`+inverse kept as
  a fallback param.
- Phase 7: proper SHM vs UDP transport decision (see §7).
- Model quality (not a tracker issue): the custom model mislabels the orange ball as `green_buoy` and
  fires ~0.51 on the robot arm — noted for later retraining, does not affect tracking/geometry.

## 10. GOTCHAS learned in Phase 2 (read before debugging visuals)

- **Duplicate publishers = fake "reversals / old+new mixing / ID swaps".** The reversal bug chased in
  Phase 2 was NOT the tracker — it was **6 stray `tracker_node` instances all publishing to
  `/tracker_node/tracks_2d`** (leftover from repeated debug launches; `ros2 run ... &` inside a
  `docker exec` leaves the node child alive when the shell exits — killing the wrapper PID is not
  enough). Six independent Kalman states + ID counters interleaving on one topic renders downstream as
  alternating positions and swapping IDs. Symptoms to check FIRST: `ros2 topic info <topic>` Publisher
  count > 1; track-msg rate ≈ N× detection rate; per-class display id climbing fast (e.g. `green_buoy_26`).
  Cleanup that works: `docker exec -u root <c> bash -c 'for p in $(ps -eo pid,cmd | grep "[l]ib/rtdetr_zed_tracker/tracker_node" | awk "{print \$1}"); do kill -9 $p; done'`.
  `run_phase2_test.sh` now pre-kills stragglers and kills node executables on exit.
- **The tracker algorithm is clean** (proven by `phase2_reversal_diag.py`): 0 reversals on clean
  monotonic input AND on inputs with detection dropouts/coasting; KF velocity carries coasts forward.
- **Overlay must be stamp-matched**, not image-driven-latest: drawing the latest tracks on the latest
  image trails boxes by the ~120 ms pipeline latency ("old+new mixing"). `overlay_node.py` composites
  each track set onto the image with the SAME stamp (RELIABLE image intake so the exact frame isn't
  dropped) — publishes the aligned, ~120 ms-delayed pair.
- **Framerate:** grab=60 (HD720) but effective ~25. `pub_frame_rate` raised 30→60 in common_stereo.yaml,
  but 25 < 30 means the real cap is upstream (NEURAL_LIGHT depth compute per grab, RT-DETR inference at
  720p, or USB) — not a config knob. Diagnose by comparing `hz` of the raw image vs `/detections_output`.
- **Kill needs the right user:** stray node procs are owned by container `admin`; host `pkill` (user
  `jetson`) fails silently on permission. Use `docker exec -u root`.
- **`run_dev.sh` drops forwarded commands.** Its attach path ends in `docker exec ... /bin/bash $@`
  with `$@` **unquoted**, so `run_dev.sh -- -c 'bash foo.sh; exec bash'` word-splits and effectively
  runs nothing (terminals open into the container but launch nothing). Do NOT route multi-word commands
  through run_dev.sh. `scripts/_attach.sh` attaches with `docker exec -it -u admin -w $WS <container>
  bash -lc "<cmd>; exec bash"` directly (same as run_dev's line ~195, correctly quoted). `run_dev.sh`
  is still used, unmodified, only to CREATE the container (`scripts/_coldstart.sh`).
- **One-window launcher:** `scripts/start_all.sh` builds a 2×2 terminator layout (needs
  `apt install terminator`) → panes = pipeline / tracker+fusion(`enable_overlay:=true`) / viewer /
  overlay-view. Overlay is blank unless `enable_overlay:=true` is passed (it's what starts overlay_node).

## 9. Design defaults locked by Phase 0 (feed into Phase 1/2/4)
- Association: **Hungarian** (`scipy.optimize.linear_sum_assignment`). Kalman: **pure-numpy**.
- `padding_mode='top_left'`, `source=1280x720` (switched from 1080p on 2026-07-23 to cut UDP-loopback
  overlay load; keep zed.yaml grab_resolution, interface_specs.json, and tracker source dims in sync),
  `network=640`, inverse scale `min(640/W,640/H)`.
- `sync_mode` default **`exact`** (fallback `nearest`, half-frame ≈ 36 ms).
- Color image is **bgra8** — convert to bgr/rgb in overlay & fusion.
- Labels file = 7 buoy classes (index order in §4).
- All joining nodes launch with the **UDP profile** env var (§7).
- Class-id are strings of ints; unmapped ⇒ `unknown_<idx>` + warn-once.
