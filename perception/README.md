# Retail perception

I group detected product boxes into shelf rows and mark unusually wide spaces between neighboring facings. The detector is trained on real retail photographs from the full official SKU-110K release; its gap stage uses box geometry rather than SKU identities.

This module runs offline. It does not feed the robot's EKF or obstacle controller. SKU-110K labels products, not empty shelf spaces or navigation obstacles, so the red regions are geometric gap candidates.

## Full dataset and training

The official archive comes from the authors' [SKU-110K repository](https://github.com/eg4000/SKU110K_CVPR19), using the fixed archive URL also distributed in Ultralytics' dataset configuration. The authors state academic and non-commercial use. Raw photographs and model weights are kept outside Git.

The downloaded archive is 12,196,152,718 bytes, SHA-256 `fd40deee21044af97804d6583e62d43f7b317c46c70153a5fb427bd26438c226`. The converter preserves all official splits and full-size images:

| Split | Listed images | Annotation boxes |
|---|---:|---:|
| Train | 8,219 | 1,208,482 |
| Validation | 588 | 90,968 |
| Test | 2,936 | 431,546 |

The test split is prepared but is not used for model selection or the validation metrics. Conversion creates single-class normalized YOLO boxes from the official CSV dimensions, with five decimals, without image resizing or sample selection. Missing files and questionable boxes were both zero. [Archive, split and conversion provenance](results/sku110k_full_provenance.json).

Ultralytics verification accepted **8,216 training images / 1,207,996 boxes** and all **588 validation images / 90,968 boxes**. It rejected three truncated training JPEGs (`train_4222`, `train_882`, `train_924`, containing 483 labels), removed three duplicate labels, and automatically restored/re-saved 494 training and 33 validation JPEGs at their original dimensions. The downloaded archive remains unchanged. [Exact rejection reasons and loader counts](results/dataset_verification.json).

The full local run uses COCO-pretrained YOLO11s, 640-pixel training input, 100 epochs, train batch 8, eight training workers, seed 0, deterministic training and three warmup epochs. The per-epoch validation loader uses batch 16 and 16 workers. It runs on a 12 GB RTX 3080 under Ubuntu 24.04 / WSL2, Ultralytics 8.4.196 and PyTorch 2.15.0+cu132. The primary pretrained model SHA-256 is `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.

An initial batch-16 attempt reached batch 502/514 of its first epoch, then was stopped after measured dedicated/shared GPU memory pressure produced multi-minute batches. It completed no epoch or validation metric and logged no OOM. The batch-8 run starts independently from the same pretrained checkpoint for the full 100 epochs. [Attempt and resource record](results/batch16_attempt.json).

Training is currently running. Raw epoch/validation results and ONNX parity will be added after that run finishes; the earlier 48-image CPU experiment is retained in source history rather than used as the full-data result.

## Run

Create an environment and install `requirements.txt`, then download the [official fixed archive](https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz) into a separate cache/data directory:

```bash
python scripts/prepare_full.py --archive /path/to/SKU110K_fixed.tar.gz \
  --root /path/to/datasets \
  --out results/sku110k_full_provenance.json
python scripts/train.py --model yolo11s.pt --data /path/to/datasets/SKU110K_fixed/data.yaml \
  --device 0 --epochs 100 --imgsz 640 --batch 8 --workers 8 --seed 0 \
  --name sku110k_full_yolo11s --project /path/to/runs --out results/train_full_3080.json
python scripts/export_bench.py --weights /path/to/runs/sku110k_full_yolo11s/weights/best.pt \
  --imgsz 640 --image-list /path/to/datasets/SKU110K_fixed/val.txt --out results/onnx_bench_full.json
```

The preparation script safely resumes extraction by reusing complete files, then atomically rewrites labels so repeated conversion does not append duplicates. It hashes the supplied archive; `--download-record` can reuse an existing downloader's URL, size and checksum record. The training script saves exact arguments, actual loader counts, each epoch's elapsed time and metrics, the full raw CSV, best-weight hash and a separate validation of the best model with `max_det=1000`.

ONNX export compares raw outputs against PyTorch on the same seed-0 input. Raw CPU forward measurements use five warmup calls followed by 50 timed calls, batch 1 and four threads. The separate validation-image calls include image loading, preprocessing, inference and NMS, including first-call initialization; gap grouping and drawing are outside that timer. The script records all timings, model hashes and per-image detections/candidates.

## Code and checks

- `scripts/prepare_full.py`: official CSV conversion and safe private archive extraction.
- `scripts/train.py`: fixed-seed training, actual data counts, epoch timings and best validation.
- `scripts/export_bench.py`: ONNX export, parity, timing samples and candidate rendering.
- `shelfvision/gaps.py`: vertical-overlap row grouping and spaces wider than 1.5 times the row's median product width.
- `shelfvision/convert.py`: optional Parquet-mirror converter, retained with its five unit checks.

From the repository root, `PYTHONPATH=.:perception python -m pytest --capture=sys -q perception/tests` runs the converter and gap-geometry tests. The root workflow includes these checks; full training is a local GPU run.
