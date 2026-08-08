#!/usr/bin/env bash
# One-command launcher — opens EVERYTHING that has to run inside the container, in a
# SINGLE terminator window with split panes. Runs on the HOST (Jetson).
#
#   bash start_all.sh              # FULL: 2x2, with overlay + rviz (default)
#   bash start_all.sh headless     # HEADLESS: 2x2, no GUI
#
# FULL panes:      1 pipeline (ZED + RT-DETR)   2 depth fusion + overlay
#                  3 full topic stream          4 rviz (3D markers + overlay image)
# HEADLESS panes:  1 pipeline (ZED + RT-DETR)   2 depth fusion, NO overlay
#                  3 full topic stream          4 free shell
#
# Live distances need no GUI and no dedicated pane: depth_fusion_node prints a
# compact line once a second in pane 2, nearest object first --
#
#   [dist] 2/3 ranged | red_buoy 4.21m (r=4.35 c=0.88 v=82%) | buoy NO-DEPTH (...)
#
# Rate is the distance_log_hz parameter in config/fusion_params.yaml (0 = off).
#
# The pipeline is three stages and only the first two live in this container:
#
#     RT-DETR  ->  whole-box depth density clustering  ->  world-frame tracking
#     (pane 1)     (pane 2, depth_fusion_node)            (HOST, buoy_mapper_node)
#
# There is no tracker pane any more — the image-space ByteTrack node is gone and
# depth_fusion_node consumes RT-DETR detections directly.
#
# NOT started here: the host half. After this comes up, on the host run
#     ros2 launch usv_bringup nav2.launch.py use_sim_time:=false
# which brings up buoy_mapper_node (plus TF, localization, Nav2).
#
# Cold-starts the container first (via the unmodified run_dev.sh) if it isn't running.
# Panes attach with `docker exec` directly (see _attach.sh). Overlay is enabled only in
# FULL mode (that flag is what starts overlay_node).
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  full|headless) ;;
  --no-viz|noviz|core) MODE="headless" ;;
  *) echo "usage: start_all.sh [full|headless]"; exit 2 ;;
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

# 1) make sure the container is up (own terminator process via --no-dbus so it does not
#    hijack the grid launch below)
# Refuse to stack a second pipeline on a running one: this script skips the cold
# start when the container is up but does not stop panes already attached to it, so
# a second launch leaves two zed_node instances fighting over the camera device.
if is_up && docker exec "$CONTAINER" pgrep -f "isaac_ros_examples.launch.py" >/dev/null 2>&1; then
  echo "A pipeline is already running in $CONTAINER."
  echo "Stop it first (Ctrl-C in its pane), or stop the container:"
  echo "    docker stop $CONTAINER"
  exit 1
fi

if ! is_up; then
  echo "Container not running — cold-starting (keep the 'container host' window open)…"
  terminator --no-dbus --title="container host (keep open)" -e "bash '$SDIR/_coldstart.sh'" &
  printf 'waiting for container'
  for _ in $(seq 1 600); do is_up && break; printf '.'; sleep 1; done
  is_up || { echo; echo "ERROR: container did not come up."; exit 1; }
  echo " up."; sleep 3
fi

# 2) generate the terminator layout for this mode
CFG="$(mktemp --suffix=-rtdetr.terminator)"
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
  [[tracking]]
HDR
  term root Window '""' 0
  term vA   VPaned root 0
  term rowT HPaned vA   0
  term rowB HPaned vA   1
  term t1 Terminal rowT 0 pipeline
  if [ "$MODE" = full ]; then
    # 2x2 with overlay (viewed in rviz's image display) + rviz markers
    term t2 Terminal rowT 1 fusion
    term t3 Terminal rowB 0 topics
    term t4 Terminal rowB 1 rviz
  else
    # 2x2, no GUI
    term t2 Terminal rowT 1 core
    term t3 Terminal rowB 0 topics
    term t4 Terminal rowB 1 shell
  fi
  echo "[plugins]"
} > "$CFG"

echo "Launching terminator [$MODE]…"
if [ "$MODE" = full ]; then
  echo "Give the pipeline ~15 s. In rviz, enable the 'Overlay' image display to see boxes."
  echo "Then, on the HOST: ros2 launch usv_bringup nav2.launch.py use_sim_time:=false"
fi
# --no-dbus so THIS invocation reads our -g config (its 'tracking' layout) rather than
# being served by an already-running terminator that never loaded it.
terminator --no-dbus -g "$CFG" -l tracking
