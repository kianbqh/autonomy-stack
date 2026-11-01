#!/usr/bin/env bash
# Full official split; prepare_full.py creates this YAML without resizing or resampling.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-yolo11s.pt}; EPOCHS=${EPOCHS:-100}; IMGSZ=${IMGSZ:-640}; BATCH=${BATCH:-8}
DATA=${DATA:-SKU-110K.yaml}; RUN_PROJECT=${RUN_PROJECT:-runs}
python scripts/train.py --model "$MODEL" --data "$DATA" --device 0 --epochs "$EPOCHS" --imgsz "$IMGSZ" \
  --batch "$BATCH" --workers 8 --seed 0 --name sku110k_full_yolo11s --project "$RUN_PROJECT" --out results/train_full_3080.json
# Pass the official validation image directory explicitly when images are not in data/.
python scripts/export_bench.py --weights "$RUN_PROJECT/sku110k_full_yolo11s/weights/best.pt" \
  --images "${VAL_IMAGES:-data/sku110k_yolo/images/val}" --out results/onnx_bench_full.json
