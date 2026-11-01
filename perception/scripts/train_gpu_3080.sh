#!/usr/bin/env bash
# Full official split; prepare_full.py creates this YAML without resizing or resampling.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-yolo11s.pt}; EPOCHS=${EPOCHS:-100}; IMGSZ=${IMGSZ:-640}; BATCH=${BATCH:-8}
DATA=${DATA:-SKU-110K.yaml}; RUN_PROJECT=${RUN_PROJECT:-runs}
RUN_NAME=${RUN_NAME:-sku110k_full_yolo11s}
resume_args=()
if [[ -n "${RESUME:-}" ]]; then resume_args=(--resume "$RESUME"); fi
python scripts/train.py --model "$MODEL" --data "$DATA" --device 0 --epochs "$EPOCHS" --imgsz "$IMGSZ" \
  --batch "$BATCH" --workers 8 --seed 0 --name "$RUN_NAME" --project "$RUN_PROJECT" --out results/train_full_3080.json "${resume_args[@]}"
python scripts/export_bench.py --weights "$RUN_PROJECT/$RUN_NAME/weights/best.pt" \
  --data "$DATA" --imgsz "$IMGSZ" --out results/onnx_bench_full.json
