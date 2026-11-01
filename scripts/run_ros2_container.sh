#!/usr/bin/env bash
set -eo pipefail
cd /work
source /opt/ros/humble/setup.bash
mkdir -p results
# Use a new output for this run, then promote it after all nodes have produced data.
rm -f results/ros2_metrics.current.json
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
export PYTHONPATH="/work:${PYTHONPATH:-}"
ros2 pkg executables sensor_fusion_sim
set +e
# Signal the launcher once; it forwards SIGINT to its children itself.
timeout --foreground --signal=INT --kill-after=10s "${RUN_SECONDS:-180}s" \
  ros2 launch sensor_fusion_sim sim.launch.py \
  model_path:=/work/models/bc_policy.npz metrics_out:=/work/results/ros2_metrics.current.json
status=$?
set -e
if [[ "$status" != 0 && "$status" != 124 ]]; then
  exit "$status"
fi
python3 - <<'PY'
import json
import math
import os
from pathlib import Path
path = Path('results/ros2_metrics.current.json')
result = json.loads(path.read_text())
assert result['wall_time_s'] >= int(os.environ.get('RUN_SECONDS', '180')) * 0.9
assert result['samples_10hz'] > 0
assert all(rate > 0 for rate in result['topic_rates_hz'].values())
assert all(math.isfinite(result[name]) for name in ('rmse_fused_m', 'rmse_raw_gps_m'))
path.replace('results/ros2_metrics.json')
PY
chown -R "${HOST_UID:-0}:${HOST_GID:-0}" /work/ros2_ws /work/results
