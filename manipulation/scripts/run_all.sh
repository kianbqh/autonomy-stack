#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec "${M3_PYTHON:-python}" "$(dirname "$0")/run_all.py" "$@"
