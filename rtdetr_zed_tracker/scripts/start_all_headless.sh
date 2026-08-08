#!/usr/bin/env bash
# Minimum-overhead launcher: ZED + RT-DETR pipeline, and depth fusion. Nothing
# else. No overlay, no rviz, no topic dump.
#
#   bash start_all_headless.sh            # 2 panes, no GUI at all
#   bash start_all_headless.sh overlay    # 2 panes + overlay_node, still no rviz
#
# `overlay` starts overlay_node alongside depth_fusion_node in pane 2 and opens
# rqt_image_view in a third pane, so you get the annotated camera image without
# paying for rviz. rviz is never started by this script in either mode -- that is
# what start_all.sh is for.
#
# Two panes, because on a real run you want the frame budget spent on detection,
# not on rendering. You still get live distances without any GUI: depth_fusion_node
# prints a compact line once a second in pane 2 --
#
#   [dist] 2/3 ranged | red_buoy 4.21m (r=4.35 c=0.88 v=82%) | buoy NO-DEPTH (...)
#
# nearest first, with objects the detector saw but could not range marked NO-DEPTH.
# Rate is the distance_log_hz parameter (0 turns it off). For every field:
#
#   ros2 topic echo /depth_fusion_node/tracked_objects
#
# Use start_all.sh when you want the full view (overlay, rviz, live topic pane).
#
# NOT started here: the host half. After this comes up, on the host run
#     ros2 launch usv_bringup nav2.launch.py use_sim_time:=false
# which brings up buoy_mapper_node (plus TF, localization, Nav2).
#
# X/DISPLAY is still required: panes are terminator windows, same as start_all.sh.
set -euo pipefail

MODE="${1:-plain}"
case "$MODE" in
  plain|"") MODE="plain" ;;
  overlay|gui) MODE="overlay" ;;
  *) echo "usage: start_all_headless.sh [plain|overlay]"; exit 2 ;;
esac

WS_HOST="/mnt/nova_ssd/workspaces/isaac_ros-dev"
SDIR="$WS_HOST/src/rtdetr_zed_tracker/scripts"
CONTAINER="isaac_ros_dev-aarch64-container"

if ! command -v terminator >/dev/null 2>&1; then
  echo "terminator is not installed. Install it once with:"
  echo "    sudo apt-get update && sudo apt-get install -y terminator"
  exit 1
fi

is_up(){ [ -n "$(docker ps -q --filter "name=$CONTAINER" --filter status=running 2>/dev/null)" ]; }

# Refuse to stack a second pipeline on a running one. start_all.sh skips the
# cold start when the container is up but does not stop the panes already
# attached to it, so a second launch leaves two zed_node instances fighting over
# the camera device. Same trap applies here.
if is_up; then
  if docker exec "$CONTAINER" pgrep -f "isaac_ros_examples.launch.py" >/dev/null 2>&1; then
    echo "A pipeline is already running in $CONTAINER."
    echo "Stop it first (Ctrl-C in its pane), or stop the container:"
    echo "    docker stop $CONTAINER"
    exit 1
  fi
else
  echo "Container not running — cold-starting (keep the 'container host' window open)…"
  terminator --no-dbus --title="container host (keep open)" -e "bash '$SDIR/_coldstart.sh'" &
  printf 'waiting for container'
  for _ in $(seq 1 600); do is_up && break; printf '.'; sleep 1; done
  is_up || { echo; echo "ERROR: container did not come up."; exit 1; }
  echo " up."; sleep 3
fi

CFG="$(mktemp --suffix=-rtdetr-headless.terminator)"
trap 'rm -f "$CFG"' EXIT

term() {  # $1=name $2=type $3=parent $4=order  [$5=command]
  printf '    [[[%s]]]\n      type = %s\n      parent = %s\n      order = %s\n' "$1" "$2" "$3" "$4"
  if [ "$2" = Terminal ]; then
    printf '      profile = default\n      command = bash %s/_attach.sh %s\n' "$SDIR" "$5"
  fi
  return 0
}

{
  cat <<'HDR'
[global_config]
[keybindings]
[profiles]
  [[default]]
    scrollback_infinite = True
    exit_action = hold
[layouts]
  [[headless]]
HDR
  term root Window '""' 0
  if [ "$MODE" = overlay ]; then
    # pipeline | fusion(+overlay_node) over the rqt_image_view pane. No rviz.
    term vA   VPaned root 0
    term row  HPaned vA   0
    term t1 Terminal row 0 pipeline
    term t2 Terminal row 1 fusion
    term t3 Terminal vA  1 overlay
  else
    term row  HPaned root 0
    term t1 Terminal row 0 pipeline
    term t2 Terminal row 1 core      # depth_fusion_node, no overlay
  fi
  echo "[plugins]"
} > "$CFG"

echo "Launching terminator [headless/$MODE]… give the pipeline ~15 s."
echo "Distances appear in pane 2 as [dist] lines, once a second, nearest first."
echo "Then, on the HOST: ros2 launch usv_bringup nav2.launch.py use_sim_time:=false"
terminator --no-dbus -g "$CFG" -l headless
