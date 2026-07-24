#!/usr/bin/env bash
# Phase 2 test helper — runs tracker_node + overlay_node against the live pipeline.
# Run INSIDE the container, with the ZED+RT-DETR pipeline already up and a buoy in view.
set -e

source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

# REQUIRED: fresh participants can't get pipeline data over Fast DDS SHM here (see NOTES.md §7).
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml

SHARE="$(ros2 pkg prefix rtdetr_zed_tracker)/share/rtdetr_zed_tracker"

# Guarantee a SINGLE instance: kill any stragglers from previous runs. Multiple
# tracker_node instances all publish to /tracker_node/tracks_2d and their
# interleaved states look exactly like ID swaps / position reversals downstream.
kill_nodes() {
  pkill -9 -f "lib/rtdetr_zed_tracker/tracker_node" 2>/dev/null || true
  pkill -9 -f "lib/rtdetr_zed_tracker/overlay_node" 2>/dev/null || true
}
kill_nodes
sleep 1

ros2 run rtdetr_zed_tracker tracker_node --ros-args \
  --params-file "$SHARE/config/tracker_params.yaml" \
  -p class_labels_file:="$SHARE/config/class_labels.yaml" \
  -r __node:=tracker_node \
  -r /tracker_node/detections_input:=/detections_output &

ros2 run rtdetr_zed_tracker overlay_node --ros-args \
  -r __node:=overlay_node \
  -r /overlay_node/image:=/zed_node/left/image_rect_color \
  -r /overlay_node/tracks_2d:=/tracker_node/tracks_2d &

# Kill the actual node executables on exit (killing the `ros2 run` wrapper alone
# leaves the node child alive — that is how the duplicates accumulated).
trap kill_nodes EXIT INT TERM
echo
echo "  tracker + overlay running. To view, in another container shell (source install + export the"
echo "  same FASTRTPS profile):"
echo "     rqt_image_view /overlay_node/tracks_overlay"
echo "     ros2 topic echo /tracker_node/tracks_2d"
echo "  Ctrl-C to stop."
wait
