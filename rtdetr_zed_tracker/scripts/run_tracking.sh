#!/usr/bin/env bash
# Start the RGB tracker + depth fusion (+ optional overlay).
# Run INSIDE the container, with the ZED + RT-DETR pipeline already up.
# Usage:
#   bash run_tracking.sh                     # tracker + depth fusion
#   bash run_tracking.sh enable_overlay:=true  # + RGB overlay for rqt_image_view
set -e

source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

# REQUIRED: our nodes are separate participants and can only get pipeline data over
# UDP here (Fast DDS SHM doesn't deliver to them). See NOTES.md §7.
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml

# Avoid duplicate publishers if a previous run is still alive (see NOTES.md §10).
pkill -9 -f "lib/rtdetr_zed_tracker/tracker_node" 2>/dev/null || true
pkill -9 -f "lib/rtdetr_zed_tracker/depth_fusion_node" 2>/dev/null || true
pkill -9 -f "lib/rtdetr_zed_tracker/overlay_node" 2>/dev/null || true
sleep 1

exec ros2 launch rtdetr_zed_tracker tracking.launch.py "$@"
