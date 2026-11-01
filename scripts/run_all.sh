#!/usr/bin/env bash
# Fusion Monte-Carlo -> behavior cloning -> tests of the newly trained policy -> plot.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
mkdir -p results models .tmp .cache
export TMPDIR="$PWD/.tmp"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PY=${PYTHON:-python3}
"$PY" -m fusion_core.evaluate_fusion --episodes 50 --out results/fusion_metrics.json
"$PY" -m fusion_core.imitation --demos 200 --eval 200 --out results/policy_metrics.json
"$PY" -m pytest -q --capture=sys
"$PY" -m fusion_core.plot
