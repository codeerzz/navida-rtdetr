#!/usr/bin/env bash
# Attach ONE terminator pane to the running Isaac ROS container and run a stage.
#
# Uses `docker exec` directly (correctly quoted). This is the same attach that
# run_dev.sh performs on its line ~195, but WITHOUT its bug: run_dev.sh ends with
# `/bin/bash $@` (unquoted), which word-splits any multi-word forwarded command, so
# nothing actually runs. That is why panes opened into the container but launched
# nothing. Attaching here instead makes the command survive intact.
#
# Stages, in the order a run needs them:
#
#   pipeline     ZED driver + RT-DETR (TensorRT). Everything else waits on this.
#   fusion       depth_fusion_node + overlay_node  -- the container half of the stack
#   core         depth_fusion_node only, no overlay -- same thing minus the GUI cost
#   color        fusion + the two optional colour stages (see below)
#   color-core   color, minus the overlay -- same GUI-free trade as `core`
#   topics       live dump of the node's output, the readable "is it working" pane
#   overlay      rqt_image_view on the overlay image (needs `fusion`/`color`, not `core`)
#   rviz         rviz2 with the 3D marker config
#   shell        sourced shell, nothing running
#
# The `color*` stages turn on class_remap_node + color_classification_node, which
# collapse RT-DETR's 7 trained classes to a shape-only "buoy" and then let a
# lighting-robust YCrCb threshold decide red vs green. They are a PAIR: remapping
# without the colour stage would throw the detector's colour away and put nothing
# back. Started only by start_all_color.sh -- start_all.sh and
# start_all_headless.sh deliberately use `fusion`/`core` and stay colour-free.
#
# Measured cost of the colour stages on the Orin: depth_fusion_node's depth match
# rate drops from ~100% to ~71%, because two more nodes subscribe to the full-res
# colour image (3.7 MB @ 30 Hz) over the UDP loopback. Pay it when you need colour,
# not by default -- which is exactly why this is a separate launcher.
#
# There is no `tracker` stage any more: the image-space ByteTrack node is gone and
# depth_fusion_node consumes RT-DETR detections directly. Identity is established
# on the HOST by buoy_mapper_node from world geometry, not in this container.
set -u
CONTAINER="isaac_ros_dev-aarch64-container"
WS="/workspaces/isaac_ros-dev"          # path INSIDE the container
S="$WS/src/rtdetr_zed_tracker/scripts"

# Sourcing + the UDP profile, needed by every stage that talks to the pipeline.
# Our nodes are separate DDS participants and only receive pipeline data over UDP
# here (Fast DDS SHM does not deliver to them). See NOTES.md §7.
ENV_PREAMBLE="source /opt/ros/humble/setup.bash; \
source \$ISAAC_ROS_WS/install/setup.bash 2>/dev/null; \
export FASTRTPS_DEFAULT_PROFILES_FILE=\$ISAAC_ROS_WS/src/rtdetr_zed_tracker/udp_only_profile.xml"

COLOUR_ARGS="enable_class_remap:=true enable_color_refinement:=true"

case "${1:-}" in
  pipeline)   CMD="bash '$S/pipeline.sh'" ;;
  fusion)     CMD="bash '$S/run_tracking.sh' enable_overlay:=true" ;;
  core)       CMD="bash '$S/run_tracking.sh'" ;;
  color)      CMD="bash '$S/run_tracking.sh' $COLOUR_ARGS enable_overlay:=true" ;;
  color-core) CMD="bash '$S/run_tracking.sh' $COLOUR_ARGS" ;;
  topics)     CMD="$ENV_PREAMBLE; bash '$S/run_topics.sh'" ;;
  overlay)    CMD="bash '$S/run_overlay_view.sh'" ;;
  rviz)       CMD="bash '$S/run_rviz.sh'" ;;
  shell)      CMD="$ENV_PREAMBLE; echo 'sourced + UDP profile set — try: ros2 topic echo /depth_fusion_node/tracked_objects'" ;;
  *)          echo "unknown stage '${1:-}' (use: pipeline|fusion|core|color|color-core|topics|overlay|rviz|shell)"; exec bash ;;
esac

if [ -z "$(docker ps -q --filter "name=$CONTAINER" --filter status=running 2>/dev/null)" ]; then
  echo "Container '$CONTAINER' is not running — start it first (start_all.sh handles this)."
  exec bash
fi

# -it: interactive tty (terminator provides it). Trailing `exec bash` keeps the pane
# alive as a shell after the stage is stopped (Ctrl-C), so you can restart it in place.
exec docker exec -it -u admin -w "$WS" "$CONTAINER" bash -lc "$CMD; exec bash"
