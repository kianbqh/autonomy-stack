#!/usr/bin/env bash
# Optionally export a run, then typecheck, test, build and run headless SwiftShader checks.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "${FUSION_REPO:-../ros2-sensor-fusion-sim}" ]; then ${PYTHON:-python3} tools/export_run.py --fusion-repo "${FUSION_REPO:-../ros2-sensor-fusion-sim}"; fi
npm ci
mkdir -p results
npm run typecheck
npx vitest run --reporter=default --reporter=junit --outputFile=results/vitest-junit.xml | tee results/vitest.log
npm run build | tee results/build.log
node scripts/bench.mjs | tee results/bench.log
