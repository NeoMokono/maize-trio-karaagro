"""Build our own grouped, stratified train/val/test split (audit sections 10 and 11).

Steps:
1. Read the ndjson (``src/ndjson_io.py``) and drop the images with no boxes
   (the 99 unlabelled images of the shipped test split).
2. Map classes: the six core classes stay; "healthy images" merges into
   healthy maize; "none healthy" boxes are dropped and their images become
   background negatives.
3. Group images by the loose filename stem, so an original and its Roboflow
   copies always land in the same split (or by capture date, as a sensitivity check).
4. Split the groups about 70/15/15 with a greedy, multi-label stratifier that
   also balances image size (see ``stratified_group_split``), seed 42.
5. Write ``splits/split_v1.csv`` and print the checks report.

Usage (from the repo root):
    python -m src.split --ndjson data/kara-agro-ai-maize.ndjson --out splits/split_v1.csv
    python -m src.split --ndjson ... --group-by date --out splits/split_v1_dategroups.csv
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from src.ndjson_io import load_tables

# Final class list, in YOLO index order (0..5). Short names are used for CSV columns.
CORE_CLASSES = [
    ("fall armyworm egg", "egg"),
    ("fall armyworm larva", "larva"),
    ("fall armyworm frass", "frass"),
    ("fall armyworm larval damage", "damage"),
    ("healthy maize", "healthy"),
    ("maize streak disease", "streak"),
]
MERGE = {"healthy images": "healthy maize"}   # decision (b)
NEGATIVE = {"none healthy"}                   # decision (b): boxes dropped, image kept as background
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_EGG_IMAGES = 40
SEED = 42

CORE_NAMES = [n for n, _ in CORE_CLASSES]
SHORT = dict(CORE_CLASSES)


# --- grouping ----------------------------------------------------------------

_EXT = r"(jpe?g|png|bmp|webp)"


def group_stem(name: str, rule: str = "loose") -> str:
    """Filename stem shared by an original and its augmented copies.

    strict: drop ``ds__``, the extension, the trailing ``__<hash>``, a Roboflow
    ``_jpg.rf.<hash>`` suffix and any leftover extension endings (``.jpg``,
    ``_jpeg``, ``_jpg``; some names carry two, e.g. ``..._jpeg_jpg.rf.<hash>``).
    loose (default, audit decision): also drop a trailing ``-<h>_<m>_<s>`` time
    suffix and a ``-<n>`` copy number, and normalise ``uuid_``/lower-case UUIDs.
    """
    s = re.sub(r"^ds__", "", str(name))
    s = re.sub(rf"\.{_EXT}$", "", s, flags=re.I)
    s = re.sub(r"__[0-9a-f]{6,}$", "", s)
    s = re.sub(rf"[_.]{_EXT}\.rf\.[0-9a-zA-Z]+$", "", s, flags=re.I)
    s = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", s)
    while True:  # leftover extension endings, possibly doubled
        t = re.sub(rf"[_.]{_EXT}$", "", s, flags=re.I)
        if t == s:
            break
        s = t
    if rule == "loose":
        s = re.sub(r"-\d{1,2}_\d{1,2}_\d{1,2}$", "", s)
        s = re.sub(r"-\d{1,2}$", "", s)
        s = re.sub(r"^uuid_", "", s, flags=re.I)
        if re.fullmatch(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", s):
            s = s.upper()
    elif rule != "strict":
        raise ValueError("rule must be 'strict' or 'loose'")
    return s


def capture_date(name: str) -> str | None:
    """Date from the filename (13-digit epoch ms after ``ds__``, or IMG_YYYYMMDD)."""
    m = re.match(r"^ds__(\d{13})", str(name))
    if m:
        return pd.to_datetime(int(m.group(1)), unit="ms").strftime("%Y-%m-%d")
    m = re.search(r"(?:IMG|PXL|VID)[_-](\d{8})", str(name))
    if m:
        d = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
        return None if pd.isna(d) else d.strftime("%Y-%m-%d")
    return None


def assign_groups(names: pd.Series, by: str = "stem") -> pd.Series:
    stems = names.map(group_stem)
    if by == "stem":
        return stems
    if by == "date":  # images without a date fall back to their stem
        dates = names.map(capture_date)
        return ("date:" + dates).where(dates.notna(), "stem:" + stems)
    raise ValueError("by must be 'stem' or 'date'")


# --- class mapping -----------------------------------------------------------

def map_classes(images: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply decisions (b) and (f). Returns (image table with flags, core boxes)."""
    known = set(CORE_NAMES) | set(MERGE) | NEGATIVE
    unknown = set(boxes["cls_name"]) - known
    if unknown:
        raise ValueError(f"unexpected class names in the ndjson: {sorted(unknown)}")

    labelled = images[images["n_boxes"] > 0].copy()       # drops the unlabelled shipped test
    b = boxes[boxes["image"].isin(labelled["image"])].copy()
    b["cls_name"] = b["cls_name"].replace(MERGE)
    neg_imgs = set(b.loc[b["cls_name"].isin(NEGATIVE), "image"])
    core = b[b["cls_name"].isin(CORE_NAMES)].copy()
    core["cls"] = core["cls_name"].map({n: i for i, n in enumerate(CORE_NAMES)})

    counts = core.groupby(["image", "cls_name"]).size().unstack(fill_value=0)
    t = labelled.set_index("image")
    for name, short in CORE_CLASSES:
        n = counts[name] if name in counts else pd.Series(dtype=int)
        t[f"n_{short}"] = n.reindex(t.index, fill_value=0).astype(int)
        t[f"has_{short}"] = (t[f"n_{short}"] > 0).astype(int)
    t["n_boxes"] = t[[f"n_{s}" for _, s in CORE_CLASSES]].sum(axis=1)
    t["is_negative"] = (t["n_boxes"] == 0).astype(int)
    odd = t.index[(t["is_negative"] == 1) & ~t.index.isin(neg_imgs)]
    if len(odd):
        raise ValueError(f"{len(odd)} images lost all boxes but are not 'none healthy'")
    return t.reset_index(), core


# --- splitting ---------------------------------------------------------------

def size_bucket(table: pd.DataFrame, top: int = 8) -> pd.Series:
    """Image size as ``WxH``; all but the ``top`` most common sizes become "other"."""
    size = table["width"].astype(str) + "x" + table["height"].astype(str)
    keep = size.value_counts().index[:top]
    return size.where(size.isin(keep), "other")


def stratified_group_split(table: pd.DataFrame, fractions: dict[str, float] = SPLITS,
                           seed: int = SEED, res_weight: float = 0.5) -> pd.Series:
    """Greedy multi-label stratification over groups (after Sechidis et al., 2011).

    Why not StratifiedGroupKFold: it stratifies one label per sample, while an
    image here can hold several classes at once, and it makes equal folds rather
    than 70/15/15. The greedy version handles both:
      * each group gets a label vector = images per stratum (six classes + negatives)
        and a size vector = images per image-size bucket;
      * groups are visited rarest-stratum first (eggs before damage), in a seeded
        random order within a stratum (not biggest first: that sent nearly all
        the large Roboflow-copy groups, mostly 1536x2048, to train);
      * each group goes to the split that is least full *relative to its target*:
        remaining demand / target for the group's rarest stratum, plus
        ``res_weight`` x the same for the group's image sizes, plus a small
        term for images overall. Relative demand fills every split in proportion
        to its share, instead of always favouring train (largest absolute demand).
    So the rare classes are placed while there is still room, each split gets a
    similar resolution mix, and a whole group always lands in one split.
    """
    strata = [f"has_{s}" for _, s in CORE_CLASSES] + ["is_negative"]
    g = table.groupby("group")[strata].sum()
    g["n_images"] = table.groupby("group").size()
    res = pd.crosstab(table["group"], size_bucket(table)).reindex(g.index, fill_value=0)
    totals = g[strata].sum()
    names = list(fractions)
    frac = np.array([fractions[s] for s in names])
    want = np.outer(frac, totals.to_numpy()).astype(float)      # split x stratum
    want_res = np.outer(frac, res.sum().to_numpy()).astype(float)  # split x size bucket
    want_imgs = frac * len(table)
    have = np.zeros_like(want)
    have_res = np.zeros_like(want_res)
    have_imgs = np.zeros(len(names))

    order = g.sample(frac=1, random_state=seed)                  # seeded random order
    present = order[strata].to_numpy() > 0
    rarity = np.where(present, totals.to_numpy()[None, :], np.inf).min(axis=1)
    rarest = np.where(present, totals.to_numpy()[None, :], np.inf).argmin(axis=1)
    order = order.assign(_rarity=rarity, _rarest=rarest)
    order = order.sort_values("_rarity", kind="stable")         # rarest first, no size sort
    res_vec = res.loc[order.index].to_numpy(dtype=float)
    lab_vec = order[strata].to_numpy(dtype=float)
    n_img = order["n_images"].to_numpy(dtype=float)
    rk = order["_rarest"].to_numpy()

    out = {}
    for i, grp in enumerate(order.index):
        k = int(rk[i])
        rel_k = (want[:, k] - have[:, k]) / want[:, k]
        r = res_vec[i]
        rel_res = ((want_res - have_res) / np.maximum(want_res, 1e-9)) @ r / r.sum()
        rel_img = (want_imgs - have_imgs) / want_imgs
        score = rel_k + res_weight * rel_res + 0.1 * rel_img
        s = int(np.argmax(score))
        out[grp] = names[s]
        have[s] += lab_vec[i]
        have_res[s] += r
        have_imgs[s] += n_img[i]
    return table["group"].map(out)


# --- report ------------------------------------------------------------------

def report(table: pd.DataFrame, min_egg: int = MIN_EGG_IMAGES, strict: bool = True) -> bool:
    """Print the section-10 checks. Returns True if all hard checks pass."""
    ok = True
    print(f"images: {len(table)}   groups: {table['group'].nunique()}   "
          f"negatives: {int(table['is_negative'].sum())}")

    leak = table.groupby("group")["split"].nunique()
    n_leak = int((leak > 1).sum())
    print(f"\n[{'PASS' if n_leak == 0 else 'FAIL'}] groups in more than one split: {n_leak}")
    ok &= n_leak == 0

    shares = table["split"].value_counts(normalize=True).reindex(list(SPLITS)).round(3)
    print("\nimage share per split:", shares.to_dict(), " target:", SPLITS)

    shorts = [s for _, s in CORE_CLASSES]
    imgs = table.groupby("split")[[f"has_{s}" for s in shorts] + ["is_negative"]].sum()
    imgs.columns = shorts + ["negative"]
    bxs = table.groupby("split")[[f"n_{s}" for s in shorts]].sum()
    bxs.columns = shorts
    print("\nimages per class and split:\n", imgs.reindex(list(SPLITS)).to_string())
    print("\nboxes per class and split:\n", bxs.reindex(list(SPLITS)).to_string())
    pct = (100 * imgs / imgs.sum()).reindex(list(SPLITS)).round(1)
    print("\n% of each class's images per split (target 70/15/15):\n", pct.to_string())

    eggs = imgs["egg"].reindex(["val", "test"])
    egg_ok = bool((eggs >= min_egg).all())
    print(f"\n[{'PASS' if egg_ok else 'FAIL'}] egg images in val/test >= {min_egg}: {eggs.to_dict()}")
    ok &= egg_ok

    res = pd.crosstab(table["split"], size_bucket(table).rename("size"), normalize="index")
    res = (100 * res).reindex(list(SPLITS)).round(1)
    spread = float((res.max() - res.min()).max())
    print(f"\n[{'PASS' if spread <= 5 else 'WARN'}] resolution mix per split (% of images), "
          f"largest gap between splits {spread:.1f} points:\n", res.to_string())

    if strict and not ok:
        raise AssertionError("split checks failed, see report above")
    return ok


# --- main --------------------------------------------------------------------

def run(ndjson: str, out: str = "splits/split_v1.csv", group_by: str = "stem",
        seed: int = SEED, strict: bool | None = None) -> pd.DataFrame:
    images, boxes = load_tables(ndjson)
    print(f"ndjson: {len(images)} images, {len(boxes)} boxes "
          f"(audit: 16,543 and 92,932); {int((images['n_boxes'] == 0).sum())} without boxes dropped")
    table, _ = map_classes(images, boxes)
    table["group"] = assign_groups(table["image"], by=group_by)
    table["split"] = stratified_group_split(table, seed=seed)
    report(table, strict=(group_by == "stem") if strict is None else strict)

    cols = (["image", "group", "split", "width", "height", "n_boxes", "is_negative"]
            + [f"has_{s}" for _, s in CORE_CLASSES] + [f"n_{s}" for _, s in CORE_CLASSES])
    table = table[cols].sort_values(["split", "group", "image"]).reset_index(drop=True)
    if out:
        import os
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(table)} rows)")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndjson", required=True)
    p.add_argument("--out", default="splits/split_v1.csv")
    p.add_argument("--group-by", choices=["stem", "date"], default="stem")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    run(a.ndjson, a.out, a.group_by, a.seed)


if __name__ == "__main__":
    main()
