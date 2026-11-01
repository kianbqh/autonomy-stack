import io, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from shelfvision.convert import coco_to_yolo, convert
from shelfvision.gaps import find_gaps, group_rows


def test_coco_to_yolo_normalizes_and_clips():
    assert coco_to_yolo([10, 20, 30, 40], 100, 200) == (0.25, 0.2, 0.3, 0.2)
    cx, cy, w, h = coco_to_yolo([-10, 0, 30, 50], 100, 100)  # clipped at left edge
    assert (round(cx, 3), w) == (0.1, 0.2)
    assert coco_to_yolo([100, 0, 10, 10], 100, 100) is None  # fully outside


def _parquet(tmp_path, ok=True):
    buf = io.BytesIO(); Image.new("RGB", (200, 100), "white").save(buf, "JPEG")
    data = buf.getvalue() if ok else buf.getvalue()[:50]
    t = pa.table({"image": [{"bytes": data, "path": "a.jpg"}], "width": [200], "height": [100],
                  "objects": [{"bbox": [[0, 0, 50, 50], [150, 50, 50, 50]], "category": [0, 0]}]})
    p = tmp_path / "x.parquet"; pq.write_table(t, p); return str(p)


def test_convert_writes_yolo_labels(tmp_path):
    out = tmp_path / "ds"; r = convert(_parquet(tmp_path), str(out), "train", None, max_side=100)
    assert r == {"images": 1, "boxes": 2, "skipped_corrupt": 0}
    lines = open(out / "labels/train/a.txt").read().split("\n")
    assert lines[0] == "0 0.125000 0.250000 0.250000 0.500000"
    assert Image.open(out / "images/train/a.jpg").size == (100, 50)  # downscaled; labels unaffected


def test_convert_skips_truncated(tmp_path):
    r = convert(_parquet(tmp_path, ok=False), str(tmp_path / "ds"), "train", None, 1280)
    assert r["images"] == 0 and r["skipped_corrupt"] == 1


def _shelf(xs, y0, w=10, h=20):
    return [[x, y0, x + w, y0 + h] for x in xs]


def test_rows_and_gap_detection():
    top = _shelf([0, 11, 22, 33, 80, 91], 0)       # gap 43..80 = 3.7x width
    bottom = _shelf([0, 11, 22, 33, 44, 55], 50)   # fully stocked
    b = np.array(top + bottom, dtype=float)
    rows = group_rows(b)
    assert len(rows) == 2 and all(len(r) == 6 for r in rows)
    gaps = find_gaps(b)
    assert len(gaps) == 1 and gaps[0]["row"] == 0 and gaps[0]["x0"] == 43 and gaps[0]["x1"] == 80


def test_no_gaps_on_sparse_rows_or_empty():
    assert find_gaps(np.zeros((0, 4))) == []
    assert find_gaps(np.array(_shelf([0, 100], 0), dtype=float)) == []  # < min_row items
