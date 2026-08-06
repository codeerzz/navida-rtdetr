#!/usr/bin/env bash
# Cold-start the Isaac ROS container using the UNMODIFIED run_dev.sh (the docker
# starter). This window owns the container's lifecycle (run_dev uses `docker run
# --rm`), so keep it open. It lands at an idle container shell; start_all.sh then
# opens the 4 working panes that attach to this same container.
ICR="/mnt/nova_ssd/workspaces/isaac_ros-dev/src/isaac_ros_common"
ZED_ARGS='-v /usr/local/zed/settings:/usr/local/zed/settings -v /usr/local/zed/resources:/usr/local/zed/resources'

# Guard the image tag before launching.
#
# build_image_layers.sh ends with, when its image key resolves to no Dockerfile
# layers at all:
#
#     docker tag ${BASE_IMAGE_NAME} ${TARGET_IMAGE_NAME}
#     print_info "Nothing to build, retagged ..."
#
# which repoints isaac_ros_dev-aarch64 at the bare nvcr.io base — an image with
# zero isaac_ros packages. run_dev.sh then starts the container from it and the
# pipeline dies on "Package 'isaac_ros_examples' not found", with the ZED config
# files missing too. The good image stays on disk under GOOD_IMAGE below; only
# the tag is lost, so restoring the tag is the whole repair.
#
# Checked on every cold start rather than fixed once: a manual `docker tag`
# survives only until the next build attempt retags it again.
IMAGE="isaac_ros_dev-aarch64"
GOOD_IMAGE="isaac_ros_dev-aarch64:backup-preyolo"

if ! docker run --rm --entrypoint bash "$IMAGE" \
       -c '[ -d /opt/ros/humble/share/isaac_ros_examples ]' 2>/dev/null; then
  echo "WARNING: $IMAGE has no isaac_ros_examples — the tag was overwritten."
  if docker image inspect "$GOOD_IMAGE" >/dev/null 2>&1; then
    echo "Restoring it from $GOOD_IMAGE."
    docker tag "$GOOD_IMAGE" "$IMAGE"
  else
    echo "ERROR: $GOOD_IMAGE is gone too. Cannot start a working container." >&2
    echo "Find an image with isaac_ros_examples and tag it as $IMAGE." >&2
    exec bash
  fi
fi

cd "$ICR" || { echo "missing $ICR"; exec bash; }
exec ./scripts/run_dev.sh -b -a "$ZED_ARGS"
