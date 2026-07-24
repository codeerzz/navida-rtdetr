#!/usr/bin/env bash
# rviz2 showing the 3D track markers (/depth_fusion_node/tracks_markers). Runs INSIDE
# the container. Needs the pipeline + tracker + fusion up.
source /opt/ros/humble/setup.bash
source "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" 2>/dev/null || true
export FASTRTPS_DEFAULT_PROFILES_FILE="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/src/rtdetr_zed_tracker/udp_only_profile.xml"
exec rviz2 -d "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/rtdetr_zed_tracker/share/rtdetr_zed_tracker/config/tracker.rviz"
