"""Reusable checks for the data audit notebook (notebooks/data-audit.ipynb).

Every function takes the two tables returned by ``src.audit.load_ndjson``:

* ``images``: one row per image, columns ``image, width, height, split, n_boxes``
* ``boxes``:  one row per box, columns ``image, cls, w_px, h_px, area_share, ...``

and returns a pandas object, so the notebook only has to call and display.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Published figures we sanity-check against.
EXPECTED_IMAGES = 16_543   # Hawa's earlier load of the Ultralytics ndjson
EXPECTED_BOXES = 92_932
PUBLISHED_IMAGES = 16_552  # dataset page / README

# Per-class table from the dataset page (images, boxes).
PUBLISHED_PER_CLASS = {
    "maize streak disease": (6_204, 13_607),
    "fall armyworm larval damage": (5_138, 38_076),
    "healthy maize": (4_575, 22_707),
    "fall armyworm frass": (3_376, 15_218),
    "fall armyworm larva": (1_454, 1_512),
    "fall armyworm egg": (301, 1_026),
    "none healthy": (187, 470),
    "healthy images": (32, 318),
}


def find_class(boxes: pd.DataFrame, keyword: str, exclude: str | None = None) -> str:
    """Return the single class name containing ``keyword`` (case-insensitive).

    ``exclude`` drops names containing that word, e.g. "larva" without "damage".
    """
    hits = [c for c in boxes["cls"].unique() if keyword.lower() in str(c).lower()
            and not (exclude and exclude.lower() in str(c).lower())]
    if len(hits) != 1:
        raise ValueError(f"'{keyword}' matches {hits}; pass the exact class name")
    return hits[0]


def _with_image_cols(boxes: pd.DataFrame, images: pd.DataFrame, cols: list[str],
                     how: str = "inner") -> pd.DataFrame:
    """Join image-level ``cols`` onto boxes, taking them from ``images``.

    ``load_ndjson`` may already put some of these columns (e.g. ``split``) on boxes;
    those copies are dropped first so the merge never produces ``split_x``/``split_y``.
    """
    return (boxes.drop(columns=cols, errors="ignore")
            .merge(images[["image", *cols]], on="image", how=how))


# --- 2. overview and sanity checks -------------------------------------------

def sanity_checks(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.Series:
    """Counts and integrity checks, as one labelled Series."""
    per_image = boxes.groupby("image").size()
    n_boxes_col = images.set_index("image")["n_boxes"]
    recount = per_image.reindex(n_boxes_col.index, fill_value=0)
    return pd.Series({
        "images": len(images),
        "images (expected)": EXPECTED_IMAGES,
        "images (published)": PUBLISHED_IMAGES,
        "boxes": len(boxes),
        "boxes (expected)": EXPECTED_BOXES,
        "duplicate image names": int(images["image"].duplicated().sum()),
        "images with no boxes": int((n_boxes_col == 0).sum()),
        "boxes pointing at unknown images": int((~boxes["image"].isin(images["image"])).sum()),
        "images where n_boxes != box rows": int((recount != n_boxes_col).sum()),
        "boxes with zero or negative size": int(((boxes["w_px"] <= 0) | (boxes["h_px"] <= 0)).sum()),
        "boxes with area_share > 1": int((boxes["area_share"] > 1).sum()),
        "missing split": int(images["split"].isna().sum()),
    })


def compare_to_published(boxes: pd.DataFrame) -> pd.DataFrame:
    """Our per-class image/box counts next to the dataset page's table."""
    ours = boxes.groupby("cls").agg(images=("image", "nunique"), boxes=("image", "size"))
    pub = pd.DataFrame(PUBLISHED_PER_CLASS, index=["images_pub", "boxes_pub"]).T
    out = ours.join(pub, how="outer")
    out["images_diff"] = out["images"] - out["images_pub"]
    out["boxes_diff"] = out["boxes"] - out["boxes_pub"]
    return out.sort_values("boxes", ascending=False)


# --- 3. class imbalance ------------------------------------------------------

def class_summary(boxes: pd.DataFrame) -> pd.DataFrame:
    """Images and boxes per class, shares, boxes per image and imbalance ratio."""
    s = boxes.groupby("cls").agg(images=("image", "nunique"), boxes=("image", "size"))
    s["image_share_%"] = 100 * s["images"] / boxes["image"].nunique()
    s["box_share_%"] = 100 * s["boxes"] / len(boxes)
    s["boxes_per_image"] = s["boxes"] / s["images"]
    s["box_ratio_to_largest"] = s["boxes"].max() / s["boxes"]
    return s.sort_values("boxes", ascending=False).round(2)


def boxes_per_image_dist(boxes: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of how many boxes of a class sit in an image that has that class."""
    counts = boxes.groupby(["cls", "image"]).size()
    q = counts.groupby("cls").quantile([.5, .9, .99]).unstack()
    q.columns = ["p50", "p90", "p99"]
    q["max"] = counts.groupby("cls").max()
    return q


# --- 4. box size -------------------------------------------------------------

def box_size_by_class(boxes: pd.DataFrame) -> pd.DataFrame:
    """10/50/90th percentiles of box area (% of image) and side length (px)."""
    side = np.sqrt(boxes["w_px"] * boxes["h_px"])
    area = (boxes["area_share"] * 100).groupby(boxes["cls"]).quantile([.1, .5, .9]).unstack()
    area.columns = ["area%_p10", "area%_p50", "area%_p90"]
    px = side.groupby(boxes["cls"]).quantile([.1, .5, .9]).unstack()
    px.columns = ["side_px_p10", "side_px_p50", "side_px_p90"]
    return area.join(px).sort_values("area%_p50").round(2)


# --- 5. co-occurrence --------------------------------------------------------

def presence_matrix(boxes: pd.DataFrame) -> pd.DataFrame:
    """Image x class table of 0/1: does the image contain the class?"""
    return (boxes.assign(v=1)
            .pivot_table(index="image", columns="cls", values="v", aggfunc="max", fill_value=0))


def cooccurrence(boxes: pd.DataFrame) -> pd.DataFrame:
    """Class x class count of images containing both (diagonal = images per class)."""
    p = presence_matrix(boxes)
    return p.T.dot(p)


# --- 6. per-split counts -----------------------------------------------------

def counts_per_split(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Images and boxes per class in each split, with column MultiIndex (unit, split)."""
    b = _with_image_cols(boxes, images, ["split"], how="left")
    imgs = b.groupby(["cls", "split"])["image"].nunique().unstack(fill_value=0)
    bxs = b.groupby(["cls", "split"]).size().unstack(fill_value=0)
    return pd.concat({"images": imgs, "boxes": bxs}, axis=1)


def flag_thin_splits(per_split: pd.DataFrame, min_images: int = 30) -> pd.DataFrame:
    """Rows (class, split) where a class has fewer than ``min_images`` images."""
    imgs = per_split["images"].stack()
    thin = imgs[imgs < min_images].rename("images").to_frame()
    thin["boxes"] = per_split["boxes"].stack().reindex(thin.index)
    return thin


# --- 7. image resolution -----------------------------------------------------

def resolution_counts(images: pd.DataFrame) -> pd.DataFrame:
    """Images per (width x height), most common first, with a share column."""
    r = images.assign(size=images["width"].astype(str) + "x" + images["height"].astype(str))
    out = r["size"].value_counts().rename("images").to_frame()
    out["share_%"] = (100 * out["images"] / len(images)).round(2)
    return out


def resolution_by_split(images: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Images per split for the ``top`` most common resolutions (rest grouped as 'other')."""
    size = images["width"].astype(str) + "x" + images["height"].astype(str)
    keep = size.value_counts().index[:top]
    size = size.where(size.isin(keep), "other")
    return pd.crosstab(size.rename("size"), images["split"], margins=True)


def resolution_by_class(images: pd.DataFrame, boxes: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Share (%) of each class's images at each of the ``top`` resolutions."""
    size = (images["width"].astype(str) + "x" + images["height"].astype(str)).rename("size")
    keep = size.value_counts().index[:top]
    size = size.where(size.isin(keep), "other")
    m = boxes[["image", "cls"]].drop_duplicates().merge(
        pd.concat([images["image"], size], axis=1), on="image")
    return (100 * pd.crosstab(m["cls"], m["size"], normalize="index")).round(1)


# --- 8. oddities -------------------------------------------------------------

def class_overview(boxes: pd.DataFrame, cls: str) -> pd.Series:
    """Image/box counts and the other classes that share images with ``cls``."""
    imgs = boxes.loc[boxes["cls"] == cls, "image"].unique()
    others = boxes[boxes["image"].isin(imgs) & (boxes["cls"] != cls)]
    return pd.Series({
        "images": len(imgs),
        "boxes": int((boxes["cls"] == cls).sum()),
        "images also containing other classes": others["image"].nunique(),
        "other classes in those images": others["cls"].value_counts().to_dict(),
    })


def cross_class_examples(boxes: pd.DataFrame, host: str, guest: str, n: int = 10) -> pd.DataFrame:
    """Images that contain both ``host`` and ``guest`` boxes, with box counts.

    Use for manual inspection, e.g. frass boxes inside healthy-maize images.
    """
    both = boxes[boxes["cls"].isin([host, guest])]
    tab = both.groupby(["image", "cls"]).size().unstack(fill_value=0)
    tab = tab[(tab.get(host, 0) > 0) & (tab.get(guest, 0) > 0)]
    print(f"{len(tab)} images contain both '{host}' and '{guest}' "
          f"({int(tab[guest].sum())} '{guest}' boxes)")
    return tab.sort_values(guest, ascending=False).head(n)


def sample_images(boxes: pd.DataFrame, cls: str, n: int = 10, seed: int = 42) -> list[str]:
    """``n`` random image names containing ``cls``, for manual inspection."""
    imgs = pd.Series(boxes.loc[boxes["cls"] == cls, "image"].unique())
    return imgs.sample(min(n, len(imgs)), random_state=seed).tolist()


# --- 2b. filename clues (Roboflow copies, capture dates) ----------------------

def filename_metadata(images: pd.DataFrame) -> pd.DataFrame:
    """Per image: Roboflow flag, filename pattern and capture date parsed from the name.

    Names look like ``ds__1629436142698__dc6ff7b49e.jpg`` (13-digit epoch milliseconds),
    ``ds__IMG_20220630_101130_115__...jpg`` or ``ds__IMG-20210612-WA0166...jpg``
    (date in the name), or ``ds__1622342660033_jpg.rf.<hash>.jpg`` (Roboflow export).
    The date is a rough proxy for a capture session, not a confirmed one.
    """
    name = images["image"].astype(str)
    epoch = name.str.extract(r"^ds__(\d{13})", expand=False)
    ymd = name.str.extract(r"(?:IMG|PXL|VID)[_-](\d{8})", expand=False)
    date = pd.to_datetime(epoch.astype(float), unit="ms", errors="coerce").dt.normalize()
    date = date.fillna(pd.to_datetime(ymd, format="%Y%m%d", errors="coerce"))
    pattern = np.select([epoch.notna(), ymd.notna()], ["epoch_ms", "date_in_name"], "other")
    return pd.DataFrame({
        "image": images["image"],
        "split": images["split"],
        "roboflow": name.str.contains(".rf.", regex=False),
        "pattern": pattern,
        "date": date,
    })


def dates_across_splits(meta: pd.DataFrame) -> pd.Series:
    """How many capture dates have images in more than one split (a leakage hint)."""
    m = meta.dropna(subset=["date"])
    per_date = m.groupby("date")["split"].nunique()
    return pd.Series({
        "images with a parsed date": len(m),
        "distinct dates": len(per_date),
        "dates appearing in 2+ splits": int((per_date > 1).sum()),
        "images on those dates": int(m["date"].isin(per_date[per_date > 1].index).sum()),
    })


# --- 4b. box size after resizing ---------------------------------------------

def box_side_at_imgsz(images: pd.DataFrame, boxes: pd.DataFrame, imgsz: int = 640) -> pd.DataFrame:
    """Median box side in pixels after letterboxing each image so its long side is ``imgsz``."""
    b = _with_image_cols(boxes, images, ["width", "height"])
    scale = imgsz / b[["width", "height"]].max(axis=1)
    side = np.sqrt(b["w_px"] * b["h_px"]) * scale
    return side.groupby(b["cls"]).quantile([.1, .5, .9]).unstack().round(0).rename(
        columns={.1: "p10", .5: "p50", .9: "p90"}).sort_values("p50")


# --- 7b. resolution shortcut -------------------------------------------------

def resolution_shortcut_test(images: pd.DataFrame, boxes: pd.DataFrame, classes: list[str],
                             max_depth: int = 4, seed: int = 42) -> pd.Series:
    """Can width and height alone predict 'image contains any of ``classes``'?

    Fits a small decision tree with 5-fold cross-validation on labelled images and
    compares it with always guessing the majority answer.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    labelled = images[images["n_boxes"] > 0]
    has = labelled["image"].isin(boxes.loc[boxes["cls"].isin(classes), "image"])
    X = labelled[["width", "height"]].to_numpy()
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    acc = cross_val_score(tree, X, has, cv=cv, scoring="accuracy")
    majority = max(has.mean(), 1 - has.mean())
    return pd.Series({
        "images": len(labelled),
        "share with the classes": round(has.mean(), 3),
        "majority-baseline accuracy": round(majority, 3),
        "tree accuracy (5-fold mean)": round(acc.mean(), 3),
        "tree accuracy (5-fold std)": round(acc.std(), 3),
        "gain over baseline": round(acc.mean() - majority, 3),
    })


# --- 8b. visual inspection from the ndjson's signed URLs --------------------
# The URLs are signed and private: these helpers never print or save them.

def _ndjson_lines(path: str):
    import json
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ndjson_class_names(path: str) -> dict[int, str]:
    """Class index -> name, from the first (dataset) line of the ndjson."""
    first = next(_ndjson_lines(path))
    return {int(k): v for k, v in first.get("class_names", {}).items()}


def image_urls(path: str) -> dict[str, str]:
    """Image filename -> signed URL, read straight from the ndjson.

    ``load_ndjson`` drops the URLs, so this reads the file again. Do not print
    or save the result: the URLs give access to the data.
    """
    return {r["file"]: r["url"] for r in _ndjson_lines(path)
            if r.get("file") and r.get("url")}


def ndjson_boxes(path: str, files: list[str]) -> pd.DataFrame:
    """Box positions for ``files`` from the ndjson (normalised cx, cy, w, h).

    The boxes table from ``load_ndjson`` has sizes but not positions, so drawing
    needs the raw annotations.
    """
    names = ndjson_class_names(path)
    wanted, rows = set(files), []
    for r in _ndjson_lines(path):
        if r.get("file") in wanted:
            for b in r.get("annotations", {}).get("boxes", []):
                c = int(b[0])
                rows.append((r["file"], names.get(c, str(c)), *map(float, b[1:5])))
    return pd.DataFrame(rows, columns=["image", "cls", "cx", "cy", "w", "h"])


def _fetch_image(url: str, timeout: int = 30):
    """Download one image into PIL. Raises RuntimeError without the URL in it."""
    import io
    import urllib.error
    import urllib.request
    from PIL import Image
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    except urllib.error.HTTPError as e:
        hint = " (the signed URL has probably expired; re-export the ndjson)" if e.code in (400, 401, 403) else ""
        raise RuntimeError(f"HTTP {e.code}{hint}") from None
    except Exception as e:  # network error, timeout, not an image
        raise RuntimeError(type(e).__name__) from None


# Fixed colours per class, so every sheet uses the same key.
CLASS_COLOURS = {
    "fall armyworm egg": "#e6194b", "fall armyworm larva": "#f58231",
    "fall armyworm frass": "#ffe119", "fall armyworm larval damage": "#911eb4",
    "healthy maize": "#3cb44b", "maize streak disease": "#4363d8",
    "none healthy": "#f032e6", "healthy images": "#42d4f4",
}


def draw_boxes(img, img_boxes: pd.DataFrame, long_side: int = 600):
    """Downsize ``img`` to ``long_side`` px and draw its boxes with a class label each."""
    from PIL import ImageDraw
    scale = long_side / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    W, H = img.size
    d = ImageDraw.Draw(img)
    for _, b in img_boxes.iterrows():
        col = CLASS_COLOURS.get(b["cls"], "#ffffff")
        x0, y0 = (b["cx"] - b["w"] / 2) * W, (b["cy"] - b["h"] / 2) * H
        x1, y1 = (b["cx"] + b["w"] / 2) * W, (b["cy"] + b["h"] / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = str(b["cls"]).replace("fall armyworm ", "FAW ")
        tw = d.textlength(label)
        d.rectangle([x0, max(0, y0 - 12), x0 + tw + 4, max(12, y0)], fill=col)
        d.text((x0 + 2, max(0, y0 - 12)), label, fill="black")
    return img


def contact_sheet(files: list[str], ndjson_path: str, title: str, out_dir: str,
                  ncols: int = 5, long_side: int = 600):
    """Fetch ``files`` via their signed URLs, draw their boxes and show them in a grid.

    Each annotated image is saved as a PNG in ``out_dir`` and the grid as
    ``out_dir/_sheet.png``. Put ``out_dir`` under ``data/`` so it stays gitignored.
    Images that fail to download are listed with the reason; the rest still show.
    Returns the matplotlib figure.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)
    urls = image_urls(ndjson_path)
    pos = ndjson_boxes(ndjson_path, files)
    shown, failed = [], []
    for f in files:
        if f not in urls:
            failed.append((f, "no URL in the ndjson"))
            continue
        try:
            img = draw_boxes(_fetch_image(urls[f]), pos[pos["image"] == f], long_side)
        except RuntimeError as e:
            failed.append((f, str(e)))
            continue
        img.save(os.path.join(out_dir, os.path.splitext(f)[0] + ".png"))
        shown.append((f, img))
    del urls

    for f, why in failed:
        print(f"could not load {f}: {why}")
    if not shown:
        print("no images loaded")
        return None
    nrows = -(-len(shown) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (f, img) in zip(axes.flat, shown):
        ax.imshow(img)
        counts = pos[pos["image"] == f]["cls"].str.replace("fall armyworm ", "FAW ").value_counts()
        ax.set_title(f"{f[:34]}\n" + ", ".join(f"{k} {v}" for k, v in counts.items()), fontsize=7)
    present = sorted(pos[pos["image"].isin([f for f, _ in shown])]["cls"].unique())
    fig.legend(handles=[Patch(color=CLASS_COLOURS.get(c, "#ffffff"), label=c) for c in present],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{title} ({len(shown)} of {len(files)} loaded)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "_sheet.png"), dpi=120)
    return fig


# --- 2c. group stems (augmented copies of one photo) ------------------------
# Real name patterns seen in the export (Hawa's run 3):
#   ds__1629436142698__dc6ff7b49e.jpg                         epoch ms + __<hash>
#   ds__1622342660033_jpg.rf.<32 hex>__<hash>.jpg              Roboflow copy of an epoch name
#   ds__8F7FFA29-...-5406CFE5AB81-12_12_54_jpeg.rf.<hex>__<hash>.jpg   UUID + time-like suffix
#   ds__uuid_d78e15ab-....jpg__<hash>.jpg                      'uuid_' name with an inner .jpg
#   ds__IMG_20210507_095650-17_48_9__<hash>.jpg, ds__IMG_1286-2__<hash>.jpg

def group_stem(name: str, rule: str = "strict") -> str:
    """The part of a filename shared by an original and its augmented copies.

    ``strict``: remove ``ds__``, the extension, the trailing ``__<hash>``, a Roboflow
    ``_jpg.rf.<hash>`` suffix and any inner extension. Both
    ``ds__1622342660033_jpg.rf.<hex>__<hash>.jpg`` and ``ds__1622342660033__<hash>.jpg``
    become ``1622342660033``.

    ``loose``: as strict, then also treat these as the same photo: a trailing
    ``-<h>_<m>_<s>`` time-like suffix (``...CBF9850-5_29_36``, ``IMG_..._095650-17_48_9``),
    a trailing copy number ``-<n>`` (``IMG_1286-2``), and the ``uuid_`` prefix and
    letter case of UUIDs. Use it if those suffixes turn out to mark crops or
    re-exports of one original; that is an assumption, not yet checked.
    """
    import re
    s = re.sub(r"^ds__", "", str(name))
    s = re.sub(r"\.(jpe?g|png|bmp|webp)$", "", s, flags=re.I)
    s = re.sub(r"__[0-9a-f]{6,}$", "", s)                                  # __<hash>
    s = re.sub(r"[_.](jpe?g|png|bmp|webp)\.rf\.[0-9a-zA-Z]+$", "", s, flags=re.I)  # _jpg.rf.<hash>
    s = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", s)
    s = re.sub(r"\.(jpe?g|png|bmp|webp)$", "", s, flags=re.I)              # inner extension
    if rule == "loose":
        s = re.sub(r"-\d{1,2}_\d{1,2}_\d{1,2}$", "", s)                     # -h_m_s
        s = re.sub(r"-\d{1,2}$", "", s)                                     # -2 copy number
        s = re.sub(r"^uuid_", "", s, flags=re.I)
        if re.fullmatch(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", s):
            s = s.upper()
    elif rule != "strict":
        raise ValueError("rule must be 'strict' or 'loose'")
    return s


def group_stats(images: pd.DataFrame, rule: str = "strict") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Summary of groups formed by ``group_stem(name, rule)``.

    Returns (summary, images-per-group value counts, examples). ``examples`` has
    three names per filename pattern with their full name and both stems, so the
    rule can be checked by eye; show it with ``display.max_colwidth`` set to None.
    Prints a warning if no group has 2+ images although many names are Roboflow
    copies, which would mean the rule is not matching the names.
    """
    g = images.assign(stem=images["image"].map(lambda n: group_stem(n, rule)))
    size = g.groupby("stem").size()
    splits = g.groupby("stem")["split"].nunique()
    multi = size[size > 1]
    summary = pd.Series({
        "rule": rule,
        "images": len(g),
        "groups": len(size),
        "groups with 2+ images": len(multi),
        "images in such groups": int(multi.sum()),
        "largest group": int(size.max()),
        "groups spanning 2+ splits": int((splits > 1).sum()),
        "images in groups spanning 2+ splits": int(size[splits > 1].sum()),
    })
    meta = filename_metadata(images)
    rf_share = meta["roboflow"].mean()
    if len(multi) == 0 and rf_share > 0.05:
        print(f"WARNING: no group has 2+ images, yet {rf_share:.0%} of names contain '.rf.'. "
              "The stem rule is probably not matching these names; check the examples.")
    ex = (images.assign(pattern=meta["pattern"].values, roboflow=meta["roboflow"].values)
          .groupby(["pattern", "roboflow"]).head(3)[["pattern", "roboflow", "image"]]
          .sort_values(["pattern", "roboflow"]))
    ex["stem_strict"] = ex["image"].map(lambda n: group_stem(n, "strict"))
    ex["stem_loose"] = ex["image"].map(lambda n: group_stem(n, "loose"))
    return summary, size.value_counts().sort_index().rename("groups"), ex.reset_index(drop=True)
