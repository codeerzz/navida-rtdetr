#!/usr/bin/env bash
# Attach ONE terminator pane to the running Isaac ROS container and run a stage.
#
# Uses `docker exec` directly (correctly quoted). This is the same attach that
# run_dev.sh performs on its line ~195, but WITHOUT its bug: run_dev.sh ends with
# `/bin/bash $@` (unquoted), which word-splits any multi-word forwarded command, so
# nothing actually runs. That is why panes opened into the container but launched
# nothing. Attaching here instead makes the command survive intact.
set -u
CONTAINER="isaac_ros_dev-aarch64-container"
WS="/workspaces/isaac_ros-dev"          # path INSIDE the container
S="$WS/src/rtdetr_zed_tracker/scripts"

case "${1:-}" in
  pipeline) CMD="bash '$S/pipeline.sh'" ;;
  tracking) CMD="bash '$S/run_tracking.sh' enable_overlay:=true" ;;   # tracker + fusion + overlay
  core)     CMD="bash '$S/run_tracking.sh'" ;;                        # tracker + fusion, NO overlay
  viewer)   CMD="bash '$S/run_viewer.sh'" ;;
  overlay)  CMD="bash '$S/run_overlay_view.sh'" ;;
  rviz)     CMD="bash '$S/run_rviz.sh'" ;;
  shell)    CMD="source /opt/ros/humble/setup.bash; source \$ISAAC_ROS_WS/install/setup.bash 2>/dev/null; export FASTRTPS_DEFAULT_PROFILES_FILE=\$ISAAC_ROS_WS/src/rtdetr_zed_tracker/udp_only_profile.xml; echo 'sourced + UDP profile set — try: ros2 topic echo /depth_fusion_node/tracked_objects'" ;;
  *)        echo "unknown stage '${1:-}' (use: pipeline|tracking|core|viewer|overlay|rviz|shell)"; exec bash ;;
esac

if [ -z "$(docker ps -q --filter "name=$CONTAINER" --filter status=running 2>/dev/null)" ]; then
  echo "Container '$CONTAINER' is not running — start it first (start_all.sh handles this)."
  exec bash
fi

# -it: interactive tty (terminator provides it). Trailing `exec bash` keeps the pane
# alive as a shell after the stage is stopped (Ctrl-C), so you can restart it in place.
exec docker exec -it -u admin -w "$WS" "$CONTAINER" bash -lc "$CMD; exec bash"
