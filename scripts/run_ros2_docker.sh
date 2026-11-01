#!/usr/bin/env bash
# Build and run the ROS 2 closed loop in the official Humble image.
set -euo pipefail
cd "$(dirname "$0")/.."
DUR=${1:-180}
[[ "$DUR" =~ ^[1-9][0-9]*$ ]] || { echo 'Duration must be a positive number of seconds.' >&2; exit 2; }
runtime_options=()
if [[ -n "${DOCKER_NETWORK:-}" ]]; then
  runtime_options+=(--network "$DOCKER_NETWORK")
fi
"${DOCKER_BIN:-docker}" run --rm "${runtime_options[@]}" -v "$PWD":/work -w /work \
  -e RUN_SECONDS="$DUR" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e ROS_LOCALHOST_ONLY=1 \
  ros:humble-ros-base bash scripts/run_ros2_container.sh
cat results/ros2_metrics.json
