#!/usr/bin/env bash
# Minimum-overhead launcher: ZED + RT-DETR pipeline and tracker + depth fusion.
# Nothing else. No overlay, no rviz, no distance table.
#
#   bash start_all_headless.sh
#
# This used to be `exec start_all.sh headless`, which still opened four panes:
# the two below plus viewer_node (the live distance table, which clears and
# repaints the screen continuously) and an idle shell. Both are gone here — the
# table is a debugging view, and on a run you want the frame budget spent on
# detection. Read the tracks off the topic instead when you need them:
#
#   ros2 topic echo /depth_fusion_node/tracked_objects
#
# start_all.sh is untouched; use it when you want the full view with rviz.
#
# X/DISPLAY is still required: panes are terminator windows, same as start_all.sh.
set -euo pipefail

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
  term row  HPaned root 0
  term t1 Terminal row 0 pipeline
  term t2 Terminal row 1 core      # run_tracking.sh with no overlay
  echo "[plugins]"
} > "$CFG"

echo "Launching terminator [headless]… give the pipeline ~15 s."
terminator --no-dbus -g "$CFG" -l headless
