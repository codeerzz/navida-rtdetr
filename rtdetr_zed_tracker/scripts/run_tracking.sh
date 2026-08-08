#!/usr/bin/env bash
# Start depth fusion (+ optional overlay). This is the container half of the stack:
# it consumes RT-DETR detections directly, scans the depth inside each box, and
# publishes 3D positions. There is no image-space tracker any more.
#
# Run INSIDE the container, with the ZED + RT-DETR pipeline already up.
# Usage:
#   bash run_tracking.sh                       # depth fusion only
#   bash run_tracking.sh enable_overlay:=true  # + RGB overlay for rqt_image_view
#
#   # both optional enrichment stages (collapse classes, then let YCrCb pick the colour):
#   bash run_tracking.sh enable_class_remap:=true enable_color_refinement:=true
#
# Every argument is forwarded to tracking.launch.py -- see its docstring, or
#   ros2 launch rtdetr_zed_tracker tracking.launch.py --show-args
set -e

source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

# REQUIRED: our nodes are separate participants and can only get pipeline data over
# UDP here (Fast DDS SHM doesn't deliver to them). See NOTES.md §7.
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml

# Avoid duplicate publishers if a previous run is still alive (see NOTES.md §10).
# The two optional stages are listed too: they are off by default, but a previous
# run started WITH them leaves them behind, and a stale class_remap_node still
# publishing would feed the new fusion node a second, competing detection stream.
for n in depth_fusion_node overlay_node color_classification_node class_remap_node; do
  pkill -9 -f "lib/rtdetr_zed_tracker/$n" 2>/dev/null || true
done
sleep 1

exec ros2 launch rtdetr_zed_tracker tracking.launch.py "$@"
