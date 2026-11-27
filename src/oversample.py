"""Repeat factor sampling (RFS) for run R02: build an oversampled copy of the YOLO dataset.

Method (LVIS, Gupta et al. 2019, arXiv 1908.03195, section 3):
  f_c   = share of train images that contain class c (background images count in the total)
  r_c   = max(1, sqrt(t / f_c))                      class repeat factor, t = 0.3
  r_i   = max over the classes in image i of r_c     image repeat factor (1 for background images)
  k_i   = floor(r_i) + Bernoulli(r_i - floor(r_i))   stochastic rounding, seed 42

Image i then appears k_i times in train: the original plus (k_i - 1) copies named
``<stem>__rep1<ext>``, ``<stem>__rep2<ext>``. Copies are symlinks to the original image,
and the label file is copied. Every copy gets its own random mosaic, flip, colour and scale
augmentation during training, so rare-class images also get more augmented views.

Val and test are not changed: they are symlinked as whole folders, so evaluation uses
exactly the same images and labels as every other run.

We use separate file names rather than repeating lines in a .txt image list, because we
have not checked that Ultralytics keeps duplicate entries in a list.

Usage (Colab, after make_yolo_dataset has built /content/yolo):
    python -m src.oversample --src /content/yolo --out /content/yolo_rfs --t 0.3
Then train R02 with data=/content/yolo_rfs/dataset.yaml.

Writes <out>/rfs_manifest.csv (one row per original train image: classes, r_i, k_i) and
prints the class factors and how many times each class's images appear before and after.
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import yaml

SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_train_labels(src: str) -> pd.DataFrame:
    """One row per train image: image file name, label file stem, set of class ids."""
    img_dir = os.path.join(src, "images", "train")
    lbl_dir = os.path.join(src, "labels", "train")
    rows = []
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXT:
            continue
        lp = os.path.join(lbl_dir, stem + ".txt")
        classes = set()
        if os.path.exists(lp):
            with open(lp) as f:
                classes = {int(float(line.split()[0])) for line in f if line.strip()}
        rows.append((name, stem, ext, frozenset(classes)))
    return pd.DataFrame(rows, columns=["image", "stem", "ext", "classes"])


def repeat_factors(table: pd.DataFrame, names: list[str], t: float = 0.3) -> tuple[pd.Series, pd.DataFrame]:
    """Image repeat factor r_i, plus a per-class table of f_c and r_c."""
    n = len(table)
    counts = np.zeros(len(names))
    for cs in table["classes"]:
        for c in cs:
            counts[c] += 1
    f = counts / n
    with np.errstate(divide="ignore"):
        r_c = np.where(f > 0, np.maximum(1.0, np.sqrt(t / np.where(f > 0, f, 1))), 1.0)
    r_i = table["classes"].map(lambda cs: max((r_c[c] for c in cs), default=1.0))
    per_class = pd.DataFrame({"class": names, "train_images": counts.astype(int),
                              "f_c": f.round(4), "r_c": r_c.round(3)})
    return r_i, per_class


def stochastic_round(r: pd.Series, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.floor(r.to_numpy())
    return pd.Series((base + (rng.random(len(r)) < (r.to_numpy() - base))).astype(int), index=r.index)


def _link_dir(src: str, dst: str):
    if os.path.lexists(dst):
        return
    os.symlink(os.path.abspath(src), dst, target_is_directory=True)


def build(src: str, out: str, t: float = 0.3, seed: int = SEED, copy_images: bool = False) -> pd.DataFrame:
    with open(os.path.join(src, "dataset.yaml")) as f:
        ds = yaml.safe_load(f)
    names_map = ds["names"]
    names = [names_map[i] for i in sorted(names_map)] if isinstance(names_map, dict) else list(names_map)

    table = read_train_labels(src)
    if table.empty:
        raise FileNotFoundError(f"no train images found in {src}/images/train")
    table["r"], per_class = repeat_factors(table, names, t)
    table["k"] = stochastic_round(table["r"], seed)

    if os.path.exists(os.path.join(out, "images", "train")):
        raise FileExistsError(f"{out} already has a train folder; delete it or pick another --out")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(out, sub, "train"), exist_ok=True)
        for s in ("val", "test"):
            if os.path.isdir(os.path.join(src, sub, s)):
                _link_dir(os.path.join(src, sub, s), os.path.join(out, sub, s))

    src_img = os.path.join(src, "images", "train")
    src_lbl = os.path.join(src, "labels", "train")
    dst_img = os.path.join(out, "images", "train")
    dst_lbl = os.path.join(out, "labels", "train")
    place = shutil.copy2 if copy_images else (lambda a, b: os.symlink(os.path.abspath(a), b))
    for r in table.itertuples():
        lbl = os.path.join(src_lbl, r.stem + ".txt")
        for j in range(r.k):
            stem = r.stem if j == 0 else f"{r.stem}__rep{j}"
            place(os.path.join(src_img, r.image), os.path.join(dst_img, stem + r.ext))
            if os.path.exists(lbl):
                shutil.copyfile(lbl, os.path.join(dst_lbl, stem + ".txt"))
            else:  # background image: keep an empty label file, as make_yolo_dataset does
                open(os.path.join(dst_lbl, stem + ".txt"), "w").close()

    ds_out = dict(ds)
    ds_out["path"] = os.path.abspath(out)
    with open(os.path.join(out, "dataset.yaml"), "w") as f:
        yaml.safe_dump(ds_out, f, sort_keys=False)

    manifest = table.assign(classes=table["classes"].map(lambda cs: " ".join(map(str, sorted(cs)))))
    manifest[["image", "classes", "r", "k"]].to_csv(os.path.join(out, "rfs_manifest.csv"), index=False)
    with open(os.path.join(out, "rfs_settings.yaml"), "w") as f:
        yaml.safe_dump({"method": "repeat factor sampling (Gupta et al. 2019)", "t": t, "seed": seed,
                        "source": os.path.abspath(src)}, f, sort_keys=False)

    # Report: how often each class's images are seen per epoch, before and after.
    after = np.zeros(len(names))
    for cs, k in zip(table["classes"], table["k"]):
        for c in cs:
            after[c] += k
    per_class["images_after"] = after.astype(int)
    per_class["actual_x"] = (after / np.maximum(per_class["train_images"], 1)).round(2)
    print(f"t = {t}, seed = {seed}")
    print(per_class.to_string(index=False))
    print(f"\ntrain images: {len(table)} -> {int(table['k'].sum())} "
          f"(+{int(table['k'].sum()) - len(table)} copies); val and test unchanged (symlinked)")
    print(f"wrote {out}/dataset.yaml and rfs_manifest.csv")
    return per_class


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="YOLO dataset built by make_yolo_dataset, e.g. /content/yolo")
    p.add_argument("--out", required=True, help="new folder for the oversampled dataset, e.g. /content/yolo_rfs")
    p.add_argument("--t", type=float, default=0.3, help="RFS threshold (plan: 0.3)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking (e.g. on Windows)")
    a = p.parse_args()
    build(a.src, a.out, a.t, a.seed, a.copy)


if __name__ == "__main__":
    main()
