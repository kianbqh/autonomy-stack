#!/usr/bin/env bash
# tests -> scenario-matrix evaluation (checkpointed per scenario; pass RESUME=1 to continue) -> results/
# Env: PYTHON (numpy, pandas, pytest), SEEDS (default 20), WORKERS (default 3)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/..:$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-python3}
mkdir -p results
"$PY" -m pytest --capture=sys -q tests 2>&1 | tee results/pytest.log
"$PY" -m avsafety.evaluate --seeds "${SEEDS:-20}" --workers "${WORKERS:-3}" ${RESUME:+--resume} 2>&1 | tee results/evaluate.log
