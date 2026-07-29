# YOLO-World Obstacle Detector

Open-vocabulary obstacle detection for the navida-rtdetr pipeline.  
Lets the vehicle avoid **any** described object (e.g. `"a vessel"`) without retraining.

---

## Why YOLO-World?

The competition may present an obstacle whose exact appearance is unknown in advance.
RT-DETR (the buoy detector) can only recognise the 7 buoy classes it was fine-tuned on.
YOLO-World is an open-vocabulary detector: it accepts a **text prompt** at runtime and
finds any object matching that description in the image.

### Why not SAM / Grounded-SAM?

| Model | Verdict |
|-------|---------|
| SAM (Meta, 2023/2024) | No text prompting — needs a box or point; useless without a detector upstream |
| Grounded-SAM (Grounding DINO + SAM) | Has text → box, but also produces pixel masks we don't need; very heavy (~3–5× slower than YOLO-World) |
| OWL-ViT / OWL-v2 | Text → box only, but slower than YOLO-World-S and harder to export to TensorRT |
| **YOLO-World-S** | Text → box, fast, ONNX/TensorRT exportable, well-maintained by Ultralytics ✅ |

Obstacle avoidance only needs a **bounding box + distance** (from ZED depth).  
Pixel-level masks add computation with no benefit here.

### Why the Small (S) variant?

The Jetson AGX Orin GPU is already running at **~85 % utilisation** with RT-DETR inference
(TensorRT) and ZED NEURAL_LIGHT depth running in parallel.

Benchmark numbers on Orin (measured 2026-07-24):

| Variant | Params | FPS estimate (Orin, FP16) | Notes |
|---------|--------|--------------------------|-------|
| `yolov8s-worldv2` | ~11 M | ~25–30 Hz | ✅ chosen |
| `yolov8m-worldv2` | ~26 M | ~15 Hz | acceptable if needed |
| `yolov8l-worldv2` | ~44 M | ~8 Hz | too slow |

The avoidance controller does not need > 20 Hz.  
If accuracy matters more than speed in a future iteration, switch to the M variant in
`config/yolo_world_params.yaml → model_path`.

---

## System Integration

```
ZED Camera
    │
    ├──► RT-DETR → tracker_node ──► /tracker_node/tracks_2d ─────┐
    │         buoy detection (7 fixed classes)                     │
    │                                                             ▼
    └──► yolo_world_node ◄─── /tracker_node/tracks_2d  (cross-suppression)
              │   text prompt: "a vessel"
              │
              ▼
    /yolo_world_node/detections   (Detection2DArray, filtered obstacle boxes)
    /yolo_world_node/sync_stats   (diagnostic counters)
```

### Cross-Suppression (Duba ↔ Vessel Conflict Resolution)

YOLO-World has no buoy knowledge, so it may detect a buoy as a vessel.
RT-DETR confirmed buoy tracks are used to suppress overlapping YOLO-World detections:

- If `IoU(YOLO_detection, RT-DETR_buoy_track) ≥ suppression_iou_threshold (0.3)`:
  → The detection is discarded (that object is a confirmed buoy, not an obstacle).

This means **RT-DETR is the authority on buoys**.  
YOLO-World handles everything RT-DETR doesn't know.

---

## Topic Map

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/zed_node/left/image_rect_color` (remapped via launch) | `sensor_msgs/Image` |
| Subscribe | `/tracker_node/tracks_2d` (remapped via launch) | `vision_msgs/Detection2DArray` |
| Subscribe | `/zed_node/depth/depth_registered` (remapped via launch) | `sensor_msgs/Image` |
| Subscribe | `/zed_node/depth/camera_info` (remapped via launch) | `sensor_msgs/CameraInfo` |
| Publish | `/yolo_world_node/detections` | `vision_msgs/Detection2DArray` |
| Publish | `/yolo_world_node/sync_stats` | `std_msgs/String` |

---

## Configuration

All parameters live in `config/yolo_world_params.yaml`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `avoid_prompt` | `"a vessel"` | The object class to detect. Change at runtime. |
| `confidence_threshold` | `0.3` | Minimum YOLO-World confidence. |
| `suppression_iou_threshold` | `0.3` | IoU needed to suppress a YOLO detection as a buoy. |
| `source_image_width/height` | `1280 / 720` | Must match the ZED resolution. |
| `model_path` | `"yolov8s-worldv2.pt"` | Ultralytics model name or local path. |

### Change the prompt at runtime (no restart)

```bash
# Inside the container, with UDP profile exported:
ros2 param set /yolo_world_node avoid_prompt "a red motorboat"
# Takes effect on the next inference frame. No rebuild needed.
```

### Disable YOLO-World (buoy-only run)

```bash
bash scripts/run_tracking.sh enable_yolo_world:=false
```

---

## Build

```bash
cd /workspaces/isaac_ros-dev

# Install ultralytics (first time only)
pip install ultralytics

colcon build --symlink-install \
  --packages-select rtdetr_zed_tracker rtdetr_zed_tracker_msgs
source install/setup.bash
```

---

## Run

```bash
# Full stack (inside the container, UDP profile must be set — see run_tracking.sh)
bash /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/run_tracking.sh

# With custom prompt:
bash .../scripts/run_tracking.sh avoid_prompt:="a ship"

# Without YOLO-World (buoys only):
bash .../scripts/run_tracking.sh enable_yolo_world:=false
```

### Watch obstacle output

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=.../udp_only_profile.xml
ros2 topic echo /yolo_world_node/detections
ros2 topic echo /yolo_world_node/sync_stats
```

---

## TensorRT Export (optional, for maximum speed on Orin)

```bash
# Export to ONNX first, then to TensorRT engine
yolo export model=yolov8s-worldv2.pt format=engine device=0 half=True

# Update model_path in yolo_world_params.yaml:
#   model_path: "/path/to/yolov8s-worldv2.engine"
```

> ⚠️ After TensorRT export, `set_classes()` (runtime prompt changes) may not work.
> Lock in your prompt before exporting if you use this path.

---

## Test (no hardware needed)

```bash
cd /workspaces/isaac_ros-dev/src/rtdetr_zed_tracker
pip install ultralytics opencv-python numpy
python3 -m pytest test/test_yolo_world_mock.py -v
```

See `test/test_yolo_world_mock.py` for details on what is tested.

---

## Gotchas

1. **UDP profile is still required** — same as all other nodes. `run_tracking.sh` sets it.
2. **Model download on first run** — `yolov8s-worldv2.pt` is downloaded from Ultralytics hub (~23 MB). Ensure internet access or pre-download.
3. **TensorRT export freezes prompt** — after `.engine` conversion, `set_classes()` no longer works. Use ONNX or PyTorch weights if you need runtime prompt changes.
4. **bgra8 encoding** — the ZED publishes 4-channel images. `yolo_world_node` converts to BGR automatically before inference.
