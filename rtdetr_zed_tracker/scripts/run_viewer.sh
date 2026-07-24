#!/usr/bin/env bash
# Live table of tracked objects sorted by distance. Run in its OWN terminal
# (it clears the screen), while run_tracking.sh is up.
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/udp_only_profile.xml
exec ros2 run rtdetr_zed_tracker viewer_node --ros-args \
  -r /viewer_node/tracked_objects:=/depth_fusion_node/tracked_objects "$@"
