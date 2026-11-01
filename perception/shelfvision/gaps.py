"""Out-of-stock / empty-facing heuristic: cluster detected products into shelf rows by vertical overlap,
then flag horizontal gaps between neighbours that are wider than k x the row's median product width.
Unsupervised post-processing on detector output; SKU-110K has no gap labels, so it is unit-tested, not scored."""
from __future__ import annotations
import numpy as np


def group_rows(boxes: np.ndarray, min_overlap: float = 0.5) -> list[np.ndarray]:
    """boxes (N,4) xyxy -> list of index arrays, one per shelf row (sorted top to bottom, left to right)."""
    if len(boxes) == 0:
        return []
    order = np.argsort((boxes[:, 1] + boxes[:, 3]) / 2)
    rows: list[list[int]] = []
    for i in order:
        y0, y1 = boxes[i, 1], boxes[i, 3]
        placed = False
        for r in rows:
            ry0, ry1 = np.median(boxes[r, 1]), np.median(boxes[r, 3])
            ov = max(0.0, min(y1, ry1) - max(y0, ry0)) / max(1e-6, min(y1 - y0, ry1 - ry0))
            if ov >= min_overlap:
                r.append(i); placed = True; break
        if not placed:
            rows.append([i])
    out = [np.array(sorted(r, key=lambda j: boxes[j, 0])) for r in rows]
    return sorted(out, key=lambda r: float(np.median(boxes[r, 1])))


def find_gaps(boxes: np.ndarray, k: float = 1.5, min_row: int = 3) -> list[dict]:
    """Return gaps [{'row', 'x0', 'x1', 'y0', 'y1', 'width_ratio'}] wider than k x median width in their row."""
    gaps = []
    for ri, r in enumerate(group_rows(boxes)):
        if len(r) < min_row:
            continue
        b = boxes[r]; med_w = float(np.median(b[:, 2] - b[:, 0]))
        y0, y1 = float(np.median(b[:, 1])), float(np.median(b[:, 3]))
        for a, c in zip(b[:-1], b[1:]):
            g = c[0] - a[2]
            if g > k * med_w:
                gaps.append({"row": ri, "x0": float(a[2]), "x1": float(c[0]), "y0": y0, "y1": y1, "width_ratio": float(g / med_w)})
    return gaps
