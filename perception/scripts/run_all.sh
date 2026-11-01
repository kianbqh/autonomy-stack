#!/usr/bin/env bash
# Default CPU pipeline: tests -> subset conversion -> fine-tune -> ONNX export + latency + gap heuristic.
# Uses the bundled 48-train / 16-validation subset by default.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
python -m pytest --capture=sys -q
./scripts/prepare_subset.sh
python scripts/train.py --epochs 20 --imgsz 512 --batch 4 --warmup-epochs 1 --name smoke
python scripts/export_bench.py --weights runs/smoke/weights/best.pt --imgsz 512 --threads 2 --n 8
python scripts/render_gap_example.py
