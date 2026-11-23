"""Read the Ultralytics ndjson export directly.

Line 1 is the dataset record (``class_names``: {"0": name, ...}). Every other line
is one image: ``file, url, width, height, split, annotations: {boxes: [[cls, cx, cy, w, h], ...]}``
with box coordinates normalised to 0..1.

The ``url`` values are signed and private. Nothing in this module prints or saves them.
"""
from __future__ import annotations

import json
from typing import Iterator

import pandas as pd


def iter_lines(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def class_names(path: str) -> dict[int, str]:
    first = next(iter_lines(path))
    return {int(k): v for k, v in first.get("class_names", {}).items()}


def image_records(path: str) -> Iterator[dict]:
    """Image lines only (skips the dataset line)."""
    for i, r in enumerate(iter_lines(path)):
        if i == 0 and "file" not in r:
            continue
        if r.get("file"):
            yield r


def load_tables(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(images, boxes) without URLs.

    images: image, width, height, shipped_split, n_boxes
    boxes:  image, cls_name, cx, cy, w, h   (normalised)
    """
    names = class_names(path)
    imgs, rows = [], []
    for r in image_records(path):
        bxs = r.get("annotations", {}).get("boxes", []) or []
        imgs.append((r["file"], r.get("width"), r.get("height"), r.get("split"), len(bxs)))
        for b in bxs:
            c = int(b[0])
            rows.append((r["file"], names.get(c, str(c)), *map(float, b[1:5])))
    images = pd.DataFrame(imgs, columns=["image", "width", "height", "shipped_split", "n_boxes"])
    boxes = pd.DataFrame(rows, columns=["image", "cls_name", "cx", "cy", "w", "h"])
    return images, boxes


def image_urls(path: str) -> dict[str, str]:
    """filename -> signed URL. Keep in memory only; never print or save."""
    return {r["file"]: r["url"] for r in image_records(path) if r.get("url")}
