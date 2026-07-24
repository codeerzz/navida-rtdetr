#!/usr/bin/env bash
# Cold-start the Isaac ROS container using the UNMODIFIED run_dev.sh (the docker
# starter). This window owns the container's lifecycle (run_dev uses `docker run
# --rm`), so keep it open. It lands at an idle container shell; start_all.sh then
# opens the 4 working panes that attach to this same container.
ICR="/mnt/nova_ssd/workspaces/isaac_ros-dev/src/isaac_ros_common"
ZED_ARGS='-v /usr/local/zed/settings:/usr/local/zed/settings -v /usr/local/zed/resources:/usr/local/zed/resources'
cd "$ICR" || { echo "missing $ICR"; exec bash; }
exec ./scripts/run_dev.sh -b -a "$ZED_ARGS"
