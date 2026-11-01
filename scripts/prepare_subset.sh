#!/usr/bin/env bash
# Download two small parquet shards of SKU-110K and convert a subset to YOLO format.
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$repo_dir"
N_TRAIN=${N_TRAIN:-48}; N_VAL=${N_VAL:-16}
PY=${PYTHON:-python3}
mkdir -p data/parquet results
BASE=https://huggingface.co/datasets/finedet/sku110k/resolve/main/data
download() {
  local destination=$1 source=$2
  if [[ ! -s "$destination" ]]; then
    "$PY" - "$source" "$destination" <<'PY'
import os, shutil, sys, urllib.request
source, destination = sys.argv[1:]
with urllib.request.urlopen(source, timeout=60) as response, open(destination + '.part', 'wb') as output:
    shutil.copyfileobj(response, output, length=1024 * 1024)
os.replace(destination + '.part', destination)
PY
  fi
}
download data/parquet/train-00008.parquet "$BASE/train-00008-of-00009.parquet"
download data/parquet/validation.parquet "$BASE/validation-00000-of-00001.parquet"
rm -rf -- "$repo_dir/data/sku110k_yolo"
"$PY" -m shelfvision.convert --parquet data/parquet/train-00008.parquet --split train --limit "$N_TRAIN" | tee results/dataset_train.json
"$PY" -m shelfvision.convert --parquet data/parquet/validation.parquet --split val --limit "$N_VAL" | tee results/dataset_val.json
