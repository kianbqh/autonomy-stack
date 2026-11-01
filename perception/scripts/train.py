"""Train / fine-tune an Ultralytics YOLO detector on SKU-110K (single class).
CPU smoke:  python scripts/train.py --data configs/sku110k_subset.yaml --device cpu --epochs 5 --imgsz 640 --batch 8 --name smoke
3080 full:  see scripts/train_gpu_3080.sh (official SKU-110K.yaml, ~13.6 GB download)."""
import argparse, json, os, time
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n.pt"); ap.add_argument("--data", default="configs/sku110k_subset.yaml")
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--device", default="cpu")
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--name", default="smoke")
ap.add_argument("--warmup-epochs", type=float, default=3.0); ap.add_argument("--out", default="results/train_smoke.json")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
data_path = str(Path(a.data).resolve()) if Path(a.data).is_file() else a.data

model = YOLO(a.model)
t0 = time.time()
model.train(data=data_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
            project=os.path.abspath("runs"), name=a.name, exist_ok=True, max_det=1000, plots=False, verbose=False, amp=a.device != "cpu", warmup_epochs=a.warmup_epochs,
            seed=a.seed, deterministic=True)
train_s = time.time() - t0
best = os.path.join("runs", a.name, "weights", "best.pt")
m = YOLO(best).val(data=data_path, imgsz=a.imgsz, batch=a.batch, device=a.device, max_det=1000, plots=False, verbose=False,
                  project=os.path.abspath("runs"), name=a.name + "_val", exist_ok=True)
res = {"model": a.model, "seed": a.seed, "deterministic": True, "batch": a.batch, "workers": a.workers, "warmup_epochs": a.warmup_epochs, "data": a.data, "epochs": a.epochs, "imgsz": a.imgsz, "device": a.device, "train_seconds": round(train_s, 1),
       "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
       "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4), "weights": best}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2))
