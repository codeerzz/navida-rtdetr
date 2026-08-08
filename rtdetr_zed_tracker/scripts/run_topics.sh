#!/usr/bin/env bash
# Live distance readout for the container half of the stack. No GUI, no X server.
#
# Replaces the old `run_viewer.sh` / viewer_node, which was deleted along with the
# ByteTrack layer: viewer_node rendered a table keyed on track ids, and there are
# no track ids any more.
#
# Two ways to watch distances, and you usually want the first:
#
#   1. depth_fusion_node prints a compact `[dist]` line itself, once a second, in
#      whichever pane runs it. That works in every mode including headless, costs
#      nothing extra, and is tuned with the distance_log_hz parameter.
#         [dist] 2/3 ranged | red_buoy 4.21m (r=4.35 c=0.88 v=82%) | buoy NO-DEPTH ...
#
#   2. This pane: the full message stream, when you need every field (position in
#      the camera optical frame, class_index, per-object valid ratio).
#
# Run INSIDE the container, with pipeline.sh and run_tracking.sh already up.
set -u

source /opt/ros/humble/setup.bash
source "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" 2>/dev/null || true
export FASTRTPS_DEFAULT_PROFILES_FILE="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/rtdetr_zed_tracker/udp_only_profile.xml"

TOPIC=/depth_fusion_node/tracked_objects

echo "Waiting for $TOPIC …"
until ros2 topic list 2>/dev/null | grep -qx "$TOPIC"; do sleep 1; done
echo "Found it. Publish rate first, then the live stream. Ctrl-C stops it; the pane stays."
echo
timeout 5 ros2 topic hz "$TOPIC" 2>/dev/null || true
echo
# depth_valid=false with NaN distances is CORRECT and expected: the box had too few
# usable depth pixels. A fabricated 0.0 there would be far worse -- it would place a
# phantom obstacle right on the bow.
exec ros2 topic echo "$TOPIC"
