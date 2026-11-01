"""Export a trained YOLO model to ONNX and benchmark CPU inference latency with onnxruntime vs PyTorch,
then run the shelf-gap heuristic on validation images and write an annotated example."""
import argparse, glob, hashlib, json, os, statistics as st, time
import numpy as np, onnxruntime as ort
from PIL import Image, ImageDraw
from ultralytics import YOLO
from shelfvision.gaps import find_gaps

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/smoke/weights/best.pt"); ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--images", default="data/sku110k_yolo/images/val"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default="results/onnx_bench.json")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)

onnx_path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False)
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
x = np.random.default_rng(0).random((1, 3, a.imgsz, a.imgsz), dtype=np.float32)
for _ in range(5): sess.run(None, {inp: x})
ort_ms = []
for _ in range(50):
    t = time.perf_counter(); sess.run(None, {inp: x}); ort_ms.append((time.perf_counter() - t) * 1e3)

import torch; torch.set_num_threads(a.threads)
pt = YOLO(a.weights).model.eval(); xt = torch.from_numpy(x)
with torch.inference_mode():
    for _ in range(5): pt(xt)
    pt_ms = []
    for _ in range(50):
        t = time.perf_counter(); pt(xt); pt_ms.append((time.perf_counter() - t) * 1e3)
    pt_output = pt(xt)
    pt_output = pt_output[0] if isinstance(pt_output, (tuple, list)) else pt_output
    pt_output = pt_output.cpu().numpy()
ort_output = sess.run(None, {inp: x})[0]
assert np.isfinite(pt_output).all() and np.isfinite(ort_output).all()
np.testing.assert_allclose(ort_output, pt_output, rtol=1e-3, atol=1e-3)

# end-to-end (pre + infer + NMS) through ultralytics with the ONNX model, plus gap heuristic
imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[: a.n]
assert len(imgs) == a.n, f'Expected {a.n} validation images, found {len(imgs)}'
om = YOLO(onnx_path, task="detect"); e2e, n_det, n_gaps = [], [], 0
per_image = []
for i, p in enumerate(imgs):
    t = time.perf_counter(); r = om.predict(p, imgsz=a.imgsz, conf=0.25, max_det=1000, verbose=False, device="cpu")[0]
    e2e.append((time.perf_counter() - t) * 1e3)
    b = r.boxes.xyxy.cpu().numpy(); gaps = find_gaps(b); n_det.append(len(b)); n_gaps += len(gaps)
    per_image.append({"image": os.path.basename(p), "inference_ms": e2e[-1], "detections": len(b), "gaps": gaps})
    if i == 0:
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im)
        for bb in b: d.rectangle(bb.tolist(), outline=(0, 255, 0), width=2)
        for g in gaps: d.rectangle([g["x0"], g["y0"], g["x1"], g["y1"]], outline=(255, 0, 0), width=4)
        d.rectangle([0, 0, 320, 22], fill=(0, 0, 0))
        d.text((5, 5), "Green: products   Red: candidate gaps", fill=(255, 255, 255))
        im.save("results/example_detections.jpg", quality=85)

pct = lambda v, q: float(np.percentile(v, q))
res = {"weights": a.weights, "onnx": onnx_path, "imgsz": a.imgsz, "threads": a.threads,
       "input_seed": 0, "warmup_runs": 5, "timed_runs": 50, "batch": 1,
       "onnx_size_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
       "onnxruntime_ms": {"p50": round(pct(ort_ms, 50), 1), "p95": round(pct(ort_ms, 95), 1)},
       "pytorch_ms": {"p50": round(pct(pt_ms, 50), 1), "p95": round(pct(pt_ms, 95), 1)},
       "speedup_p50": round(pct(pt_ms, 50) / pct(ort_ms, 50), 2),
       "end_to_end_onnx_ms": {"p50": round(pct(e2e, 50), 1), "p95": round(pct(e2e, 95), 1), "n_images": len(imgs)},
       "mean_detections_per_image": round(st.mean(n_det), 1), "gaps_flagged_total": n_gaps,
       "onnxruntime_samples_ms": ort_ms, "pytorch_samples_ms": pt_ms, "per_image": per_image,
       "numeric_parity": {"shape": list(ort_output.shape), "max_abs_error": float(np.abs(ort_output - pt_output).max()), "rtol": 1e-3, "atol": 1e-3, "allclose_passed": True},
       "end_to_end_protocol": "Ultralytics predict: image loading, preprocessing, inference and NMS; first image includes session initialization/warmup; excludes gap calculation and drawing",
       "end_to_end_session_threads": {"intra_op": om.predictor.model.backend.session.get_session_options().intra_op_num_threads,
                                      "inter_op": om.predictor.model.backend.session.get_session_options().inter_op_num_threads,
                                      "zero_means": "ONNX Runtime default"},
       "end_to_end_torch_threads": torch.get_num_threads(),
       "raw_session_threads": {"intra_op": so.intra_op_num_threads, "inter_op": so.inter_op_num_threads},
       "weights_sha256": hashlib.sha256(open(a.weights, 'rb').read()).hexdigest(),
       "onnx_sha256": hashlib.sha256(open(onnx_path, 'rb').read()).hexdigest()}
json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2))
