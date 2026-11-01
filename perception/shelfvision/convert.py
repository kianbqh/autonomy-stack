"""Convert SKU-110K parquet shards (Hugging Face `finedet/sku110k`, COCO-style absolute [x, y, w, h] boxes)
into an Ultralytics YOLO dataset (images/ + labels/ with normalized cx cy w h, single class 'object').
Images are downscaled so the long side is <= --max-side to save disk; labels are normalized so they are unaffected."""
from __future__ import annotations
import argparse, io, json, os
import pyarrow.parquet as pq
from PIL import Image


def coco_to_yolo(bbox, w, h):
    """[x, y, bw, bh] absolute -> (cx, cy, bw, bh) normalized, clipped to the image; None if degenerate."""
    x, y, bw, bh = bbox
    x0, y0 = max(0.0, x), max(0.0, y); x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def convert(parquet: str, out_dir: str, split: str, limit: int | None, max_side: int, offset: int = 0) -> dict:
    img_dir = os.path.join(out_dir, "images", split); lab_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    pf = pq.ParquetFile(parquet); n_img = n_box = seen = n_bad = 0
    for rg in range(pf.num_row_groups):
        for row in pf.read_row_group(rg).to_pylist():
            seen += 1
            if seen <= offset:
                continue
            if limit is not None and n_img >= limit:
                return {"images": n_img, "boxes": n_box, "skipped_corrupt": n_bad}
            w, h = row["width"], row["height"]
            name = os.path.splitext(os.path.basename(row["image"]["path"] or f"{split}_{seen}.jpg"))[0]
            try:
                im = Image.open(io.BytesIO(row["image"]["bytes"])); im.load(); im = im.convert("RGB")
            except OSError:  # SKU-110K ships a few truncated JPEGs; skip them
                n_bad += 1; continue
            if im.size != (w, h):  # raw pixel orientation must match the annotation frame
                n_bad += 1; continue
            s = min(1.0, max_side / max(im.size))
            if s < 1:
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
            im.save(os.path.join(img_dir, name + ".jpg"), quality=90)
            lines = [b for b in (coco_to_yolo(bb, w, h) for bb in row["objects"]["bbox"]) if b]
            with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
                f.writelines(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in lines)
            n_img += 1; n_box += len(lines)
    return {"images": n_img, "boxes": n_box, "skipped_corrupt": n_bad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", default="data/sku110k_yolo")
    ap.add_argument("--split", required=True); ap.add_argument("--limit", type=int); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()
    result = convert(a.parquet, a.out, a.split, a.limit, a.max_side, a.offset)
    print(json.dumps({"split": a.split, "source": a.parquet, "max_side": a.max_side, "offset": a.offset, **result}))
