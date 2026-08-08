#!/usr/bin/env bash
# Live RGB overlay (boxes + class/score) in rqt_image_view. Runs INSIDE
# the container. Requires run_tracking.sh to have been started WITH enable_overlay:=true
# (that is what launches overlay_node — without it there is nothing to view).
source /opt/ros/humble/setup.bash
source "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" 2>/dev/null || true
export FASTRTPS_DEFAULT_PROFILES_FILE="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/rtdetr_zed_tracker/udp_only_profile.xml"

echo "Waiting for /overlay_node/tracks_overlay to appear (needs run_tracking.sh enable_overlay:=true)…"
until ros2 topic list 2>/dev/null | grep -q '^/overlay_node/tracks_overlay$'; do sleep 1; done
echo "Found it — opening rqt_image_view."
exec ros2 run rqt_image_view rqt_image_view /overlay_node/tracks_overlay
