#!/usr/bin/env bash
# Terminal 1 payload: the ZED + RT-DETR pipeline. Runs INSIDE the container.
#
# start_all.sh gets here through _attach.sh, which is already inside. But this is
# also the script you rerun by hand to restart just the pipeline, and from a host
# shell every path below is wrong: /opt/ros/humble/share/isaac_ros_zed does not
# exist there, and isaac_ros_examples is not on the host's AMENT_PREFIX_PATH. It
# used to fail with four confusing sed/grep errors and a "package not found".
# So: if we are on the host, hop into the container and rerun there.
if [ ! -f /.dockerenv ]; then
  CONTAINER="isaac_ros_dev-aarch64-container"
  if [ -z "$(docker ps -q --filter "name=$CONTAINER" --filter status=running 2>/dev/null)" ]; then
    echo "Container '$CONTAINER' is not running — start it with start_all.sh." >&2
    exit 1
  fi
  echo "(host shell — re-running inside $CONTAINER)"
  # Same attach as _attach.sh: -u admin, workspace cwd, login shell.
  exec docker exec -it -u admin -w /workspaces/isaac_ros-dev "$CONTAINER" \
    bash -lc "bash '/workspaces/isaac_ros-dev/src/rtdetr_zed_tracker/scripts/pipeline.sh'"
fi

source /opt/ros/humble/setup.bash
source "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" 2>/dev/null || true

WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"

# ZED TF: zed_node must not publish odom -> zed_camera_link or map -> odom, or
# it fights usv_localization for edges the host already owns (odom -> base_link
# from GNSS, base_link -> zed_camera_link static). Two publishers on one edge do
# not error — TF answers with whichever wrote last, so the camera frame's parent
# flaps and reads as sensor noise rather than as a fault.
#
# zed_wrapper/config/zed.yaml sets all three to false and IS loaded after the
# Isaac bundle's zed_mono.yaml, so in principle it already wins. This forces the
# bundle file off as well: it is the file that actually ships them as true, the
# load order is the bundle's to change (a launch-file edit upstream silently
# flips the outcome), and the failure mode is silent. Idempotent, and a no-op if
# the file is absent.
for _zcfg in /opt/ros/humble/share/isaac_ros_zed/config/zed_mono.yaml; do
  [ -f "$_zcfg" ] || continue
  sudo sed -i -E 's/^([[:space:]]*publish_(map_|imu_)?tf[[:space:]]*:[[:space:]]*)true/\1false/' \
    "$_zcfg" 2>/dev/null || \
  sed -i -E 's/^([[:space:]]*publish_(map_|imu_)?tf[[:space:]]*:[[:space:]]*)true/\1false/' \
    "$_zcfg" || echo "WARNING: could not patch $_zcfg (relying on zed.yaml override)" >&2
done
unset _zcfg

exec ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=zed_mono_rect,rtdetr \
  engine_file_path:="${WS}/isaac_ros_assets/models/rtdetr/best.plan" \
  interface_specs_file:="${WS}/isaac_ros_assets/isaac_ros_rtdetr/zed_quickstart_interface_specs.json" \
  confidence_threshold:=0.8
