#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$task_root/world_model:$task_root/manipulation${PYTHONPATH:+:$PYTHONPATH}"
exec "${M3_PYTHON:-python}" "$task_root/world_model/scripts/run_all.py" "$@"
