"""Download the images from the signed URLs in the ndjson into a folder (e.g. the shared Drive).

* Parallel downloads, with resume: files already in the target folder are skipped
  (never overwritten, so a re-run cannot create duplicates such as Drive's
  ``name (1).jpg`` copies).
* Optional resize (default: long side 1280 px) to cut the ~31 GB set down to a few GB.
  YOLO labels are normalised to 0..1, so resizing does not change them. The EXIF
  orientation tag is kept, so the image is read the same way as the original.
* Each file is written to a hidden temporary name (``.name.<pid>.part``) and renamed
  only when complete, so an interrupted run never leaves a half-written image
  under its real name. Leftover ``.part`` files from an earlier run are ignored.
* A bad ``--out``/``--ndjson``/``--split-csv`` path stops the run with a clear
  message instead of creating a stray folder tree.
* The signed URLs are never printed or saved. If they have expired (HTTP 403),
  re-export the ndjson from Ultralytics and run again; finished files are kept.

Usage (from the repo root; on Colab mount Drive first; on Windows use forward
slashes, e.g. --out "G:/My Drive/.../data"):
    python -m src.fetch_images --ndjson data/kara-agro-ai-maize.ndjson \
        --out "/content/drive/Shared drives/<team>/kara-agro/images_1280" \
        --split-csv splits/split_v1.csv
    add --keep-original to save the files unchanged instead of resizing.
"""
from __future__ import annotations

import argparse
import io
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.ndjson_io import image_records


def _download(url: str, timeout: int, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404) or attempt == retries - 1:
                hint = " (signed URL probably expired: re-export the ndjson)" if e.code in (400, 401, 403) else ""
                raise RuntimeError(f"HTTP {e.code}{hint}") from None
        except Exception as e:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise RuntimeError(type(e).__name__) from None
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _resize(data: bytes, long_side: int) -> tuple[bytes, tuple[int, int]]:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    size = img.size
    exif = img.info.get("exif")
    if max(size) > long_side:
        s = long_side / max(size)
        img = img.convert("RGB").resize((round(size[0] * s), round(size[1] * s)), Image.LANCZOS)
    else:
        img = img.convert("RGB")
    buf = io.BytesIO()
    kw = {"exif": exif} if exif else {}
    img.save(buf, format="JPEG", quality=92, **kw)
    return buf.getvalue(), size


def _present(path: str) -> bool:
    """A finished file is there (zero-byte files count as missing)."""
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _one(name: str, url: str, out: str, long_side: int | None, timeout: int):
    dest = os.path.join(out, name)
    if _present(dest):                      # appeared since the scan: never write twice
        return "skipped"
    data = _download(url, timeout)
    size = None
    if long_side:
        data, size = _resize(data, long_side)
    tmp = os.path.join(out, f".{name}.{os.getpid()}.part")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if _present(dest):                  # lost a race: keep the existing file
            return "skipped"
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)                  # only ever our own temp file
    return size


def run(ndjson: str, out: str, split_csv: str | None = None, long_side: int | None = 1280,
        workers: int = 6, timeout: int = 60, limit: int | None = None) -> pd.DataFrame:
    """Download images; returns a table of failures (image, reason)."""
    _check_paths(ndjson, out, split_csv)
    os.makedirs(out, exist_ok=True)
    wanted = None
    if split_csv:
        wanted = set(pd.read_csv(split_csv, usecols=["image"])["image"])
    existing = {f for f in os.listdir(out) if _present(os.path.join(out, f))}
    todo, sizes_expected, skipped = [], {}, 0
    for r in image_records(ndjson):
        name = r["file"]
        if wanted is not None and name not in wanted:
            continue
        if name in existing:
            skipped += 1
            continue
        if not r.get("url"):
            todo.append((name, None))
            continue
        todo.append((name, r["url"]))
        sizes_expected[name] = (r.get("width"), r.get("height"))
    if limit:
        todo = todo[:limit]
    print(f"{skipped} already in {out}; {len(todo)} to download"
          + (f" (resized to long side {long_side})" if long_side else " (originals)"))

    failed, done, swapped, late_skip = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for name, url in todo:
            if url is None:
                failed.append((name, "no URL in the ndjson"))
                continue
            futs[ex.submit(_one, name, url, out, long_side, timeout)] = name
        del todo
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                size = fut.result()
                if size == "skipped":
                    late_skip += 1
                    continue
                done += 1
                exp = sizes_expected.get(name)
                if size and exp and tuple(size) == (exp[1], exp[0]) and exp[0] != exp[1]:
                    swapped += 1
            except RuntimeError as e:
                failed.append((name, str(e)))
            if i % 500 == 0:
                print(f"  {i}/{len(futs)} done, {len(failed)} failed")
    print(f"downloaded {done}, skipped {skipped + late_skip} (already present), failed {len(failed)}")
    if swapped:
        print(f"note: {swapped} images have width/height swapped against the ndjson "
              "(EXIF rotation); their EXIF tag was kept, so they load as in the original export")
    fails = pd.DataFrame(failed, columns=["image", "reason"])
    if len(fails):
        print(fails["reason"].value_counts().to_string())
        fails.to_csv(os.path.join(out, "_failed.csv"), index=False)  # filenames only, no URLs
        print(f"failure list: {os.path.join(out, '_failed.csv')} - run again to retry them")
    return fails


def _check_paths(ndjson: str, out: str, split_csv: str | None) -> None:
    """Fail early, with a readable message, on a wrong path."""
    if not os.path.isfile(ndjson):
        raise SystemExit(f"--ndjson not found: {ndjson!r}")
    if split_csv and not os.path.isfile(split_csv):
        raise SystemExit(f"--split-csv not found: {split_csv!r}")
    if os.path.exists(out) and not os.path.isdir(out):
        raise SystemExit(f"--out exists but is not a folder: {out!r}")
    parent = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(out) and not os.path.isdir(parent):
        raise SystemExit(
            f"--out parent folder does not exist: {parent!r}\n"
            "Check the drive letter / mount and the spelling (quote paths with spaces; "
            "on Windows use forward slashes). Only the last folder is created.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndjson", required=True)
    p.add_argument("--out", required=True, help="target folder, e.g. on the shared Drive")
    p.add_argument("--split-csv", help="download only the images listed here (skips the unlabelled test)")
    p.add_argument("--long-side", type=int, default=1280)
    p.add_argument("--keep-original", action="store_true", help="save files unchanged (about 31 GB)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, help="download at most this many (for a trial run)")
    a = p.parse_args()
    run(a.ndjson, a.out, a.split_csv, None if a.keep_original else a.long_side, a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
