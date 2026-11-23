"""Build a YOLO dataset folder from the split CSV, the image folder and the ndjson labels.

Layout written to ``--out`` (put it on fast local disk, e.g. /content/yolo on Colab,
so training does not read every image from Drive):

    <out>/images/{train,val,test}/<image>
    <out>/labels/{train,val,test}/<image stem>.txt   one "cls cx cy w h" line per box
    <out>/dataset.yaml                               six class names, paths

Classes follow ``src.split.CORE_CLASSES`` (decisions b and f): "healthy images" is
merged into healthy maize, "none healthy" boxes are dropped and those images get
an empty label file (background negatives).

Usage:
    python -m src.make_yolo_dataset --split-csv splits/split_v1.csv \
        --images "/content/drive/Shared drives/<team>/kara-agro/images_1280" \
        --ndjson data/kara-agro-ai-maize.ndjson --out /content/yolo
    --neg-fraction 0.5   keep half the negatives in train (val/test keep all)
    --link               symlink instead of copy (only if --images is already on local disk)
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.ndjson_io import load_tables
from src.split import CORE_CLASSES, CORE_NAMES, MERGE, SEED


def run(split_csv: str, images_dir: str, ndjson: str, out: str, link: bool = False,
        neg_fraction: float = 1.0, seed: int = SEED) -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    if neg_fraction < 1.0:  # subsample negatives in train only; evaluation sets stay complete
        rng = np.random.default_rng(seed)
        neg_train = split.index[(split["split"] == "train") & (split["is_negative"] == 1)]
        drop = rng.choice(neg_train, size=round(len(neg_train) * (1 - neg_fraction)), replace=False)
        split = split.drop(index=drop)
        print(f"negatives in train: kept {len(neg_train) - len(drop)} of {len(neg_train)}")

    _, boxes = load_tables(ndjson)
    boxes["cls_name"] = boxes["cls_name"].replace(MERGE)
    boxes = boxes[boxes["cls_name"].isin(CORE_NAMES) & boxes["image"].isin(split["image"])]
    boxes["cls"] = boxes["cls_name"].map({n: i for i, n in enumerate(CORE_NAMES)})
    by_image = {k: v for k, v in boxes.groupby("image")}

    missing, written = [], 0
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    for r in split.itertuples():
        src = os.path.join(images_dir, r.image)
        if not os.path.exists(src):
            missing.append(r.image)
            continue
        dst = os.path.join(out, "images", r.split, r.image)
        if not os.path.exists(dst):
            if link:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
        b = by_image.get(r.image)
        lines = [] if b is None else [
            f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for c, x, y, w, h in b[["cls", "cx", "cy", "w", "h"]].itertuples(index=False)]
        if len(lines) != r.n_boxes:
            raise ValueError(f"{r.image}: {len(lines)} boxes in ndjson, {r.n_boxes} in split CSV")
        stem = os.path.splitext(r.image)[0]
        with open(os.path.join(out, "labels", r.split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    names = "\n".join(f"  {i}: {n}" for i, (n, _) in enumerate(CORE_CLASSES))
    with open(os.path.join(out, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n{names}\n")

    summary = split[~split["image"].isin(missing)].groupby("split").agg(
        images=("image", "size"), negatives=("is_negative", "sum"), boxes=("n_boxes", "sum"))
    print(summary.to_string())
    print(f"wrote {written} images and label files to {out}")
    if missing:
        print(f"WARNING: {len(missing)} images in the CSV are not in {images_dir} "
              f"(e.g. {missing[:3]}); run fetch_images again, then rebuild")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-csv", default="splits/split_v1.csv")
    p.add_argument("--images", required=True)
    p.add_argument("--ndjson", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--link", action="store_true")
    p.add_argument("--neg-fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    run(a.split_csv, a.images, a.ndjson, a.out, a.link, a.neg_fraction, a.seed)


if __name__ == "__main__":
    main()
