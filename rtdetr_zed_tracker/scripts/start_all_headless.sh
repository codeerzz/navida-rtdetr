#!/usr/bin/env bash
# Headless (no-visualization) launcher: pipeline + tracker + fusion + distance table,
# no overlay / rqt / rviz. Thin wrapper over start_all.sh headless.
exec bash "$(dirname "$0")/start_all.sh" headless
