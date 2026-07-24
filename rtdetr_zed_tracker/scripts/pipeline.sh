#!/usr/bin/env bash
# Terminal 1 payload: the ZED + RT-DETR pipeline. Runs INSIDE the container.
source /opt/ros/humble/setup.bash
source "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" 2>/dev/null || true

WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
exec ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=zed_mono_rect,rtdetr \
  engine_file_path:="${WS}/isaac_ros_assets/models/rtdetr/best.plan" \
  interface_specs_file:="${WS}/isaac_ros_assets/isaac_ros_rtdetr/zed_quickstart_interface_specs.json" \
  confidence_threshold:=0.3
