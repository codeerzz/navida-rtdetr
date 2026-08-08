#!/usr/bin/env bash
# One-command launcher WITH colour refinement. Runs on the HOST (Jetson).
#
#   bash start_all_color.sh            # FULL: 2x2, overlay + rviz (default)
#   bash start_all_color.sh headless   # HEADLESS: 2x2, no GUI
#
# Same window, same panes, same cold start as start_all.sh -- the only difference
# is that pane 2 also runs the two optional enrichment stages:
#
#     RT-DETR -> class_remap_node -> color_classification_node -> depth_fusion_node
#
# class_remap_node collapses RT-DETR's 7 trained classes to a shape-only "buoy",
# and color_classification_node decides red vs green from a lighting-robust YCrCb
# threshold instead of trusting the colour the detector was trained to predict.
# They are a pair: remapping alone would discard the detector's colour and put
# nothing back in its place.
#
# WHY THIS IS A SEPARATE SCRIPT rather than a flag on start_all.sh: colour is not
# free. Measured on the Orin with the ZED running, enabling these two stages drops
# depth_fusion_node's depth match rate from ~100% to ~71% (`[sync] ... dropped_no_depth`),
# because two more nodes subscribe to the full-res colour image -- 3.7 MB at 30 Hz --
# over the UDP loopback. So roughly a quarter of the range measurements are the
# price of the colour label. Keeping the default launchers colour-free means you
# only pay that when you actually asked for colour.
#
# Everything else -- the container cold start, the duplicate-pipeline guard, the
# terminator layout -- is start_all.sh's, reused rather than copied, so the two
# launchers cannot drift apart. This script only chooses which _attach.sh stage
# fills pane 2.
#
# CALIBRATION WARNING: config/color_ranges.yaml has `red` range 1 calibrated from
# real footage, but `green` (and red's specular-highlight range) are still
# placeholders derived from synthetic swatches. Verified on the rig: a red buoy
# reads red=0.30 / green=0.00, comfortably over the 0.12 threshold -- but that
# measurement says nothing about whether a GREEN buoy would read green. Put one in
# front of the camera before trusting this on the water; if it does not come back
# `green_buoy`, recalibrate with:
#
#     python3 scripts/webcam_color_demo.py --calibrate green
#
# NOT started here: the host half. After this comes up, on the host run
#     ros2 launch usv_bringup nav2.launch.py use_sim_time:=false
set -euo pipefail

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-full}"
case "$MODE" in
  full|headless) ;;
  --no-viz|noviz|core) MODE="headless" ;;
  *) echo "usage: start_all_color.sh [full|headless]"; exit 2 ;;
esac

# The colour flavours of start_all.sh's two fusion panes (see _attach.sh).
export RTDETR_FUSION_STAGE=color        # FULL:     fusion + colour + overlay
export RTDETR_CORE_STAGE=color-core     # HEADLESS: fusion + colour, no overlay
export RTDETR_BANNER="$MODE + colour"

echo "Colour refinement ON (class_remap + YCrCb colour vote)."
echo "Expect depth match rate ~71% instead of ~100% — see this script's header."
exec bash "$SDIR/start_all.sh" "$MODE"
