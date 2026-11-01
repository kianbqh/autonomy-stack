#!/usr/bin/env bash
# Full SKU-110K training on a single CUDA GPU under WSL2.
# Uses Ultralytics' built-in SKU-110K.yaml, which downloads & converts the official dataset (~13.6 GB:
# 8,219 train / 588 val / 2,936 test images, ~1.7M boxes). Model, epochs, image size and batch are configurable below.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-yolo11s.pt}; EPOCHS=${EPOCHS:-100}; IMGSZ=${IMGSZ:-640}; BATCH=${BATCH:-16}
python scripts/train.py --model "$MODEL" --data SKU-110K.yaml --device 0 --epochs "$EPOCHS" --imgsz "$IMGSZ" \
  --batch "$BATCH" --workers 8 --name sku110k_full_${MODEL%.pt} --out results/train_full_3080.json
python scripts/export_bench.py --weights runs/sku110k_full_${MODEL%.pt}/weights/best.pt --out results/onnx_bench_full.json
