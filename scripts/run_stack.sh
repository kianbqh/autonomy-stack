#!/usr/bin/env bash
# Regenerate the two safety episodes, then open the replay application.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/safety:$PWD/perception${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON_BIN:-python}" -m avsafety.export
cd viz
if [[ ! -d node_modules ]]; then npm ci; fi
npm run dev -- --host 127.0.0.1 --port "${VIZ_PORT:-5176}" --strictPort
