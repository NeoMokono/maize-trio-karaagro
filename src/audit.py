"""Audit step 1: flatten Supervisely annotations into image and box tables."""
import argparse
import json
from pathlib import Path

import pandas as pd

# Per-class counts quoted in the project notes: (images, boxes). To verify, not trust.
EXPECTED = {
    "maize streak disease": (6204, 13607),
    "fall armyworm larval damage": (5138, 38076),
    "healthy maize": (4575, 22707),
    "fall armyworm frass": (3376, 15218),
    "fall armyworm larva": (1454, 1512),
    "fall armyworm egg": (301, 1026),
    "none healthy": (187, 470),
    "healthy images": (32, 318),
}


def load_annotations(root: Path):
    meta = json.loads((root / "meta.json").read_text())
    image_rows, box_rows, skipped = [], [], 0
    for p in sorted(root.glob("**/ann/*.json")):
        ann = json.loads(p.read_text())
        w, h = ann["size"]["width"], ann["size"]["height"]
        image = p.name[: -len(".json")]
        tags = "|".join(sorted(t["name"] for t in ann.get("tags", [])))
        image_rows.append(dict(image=image, folder=p.parent.parent.name,
                               width=w, height=h, tags=tags,
                               n_boxes=len(ann["objects"])))
        for o in ann["objects"]:
            if o.get("geometryType") != "rectangle":
                skipped += 1
                continue
            (x1, y1), (x2, y2) = o["points"]["exterior"]
            bw, bh = abs(x2 - x1), abs(y2 - y1)
            box_rows.append(dict(image=image, cls=o["classTitle"],
                                 bw=bw, bh=bh, area_share=bw * bh / (w * h)))
    return meta, pd.DataFrame(image_rows), pd.DataFrame(box_rows), skipped

def load_ndjson(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    names = {int(k): v for k, v in rows[0]["class_names"].items()}
    image_rows, box_rows = [], []
    for r in rows[1:]:
        boxes = r.get("annotations", {}).get("boxes", [])
        image_rows.append(dict(image=r["file"], width=r["width"], height=r["height"],
                               split=r["split"], n_boxes=len(boxes)))
        for c, cx, cy, w, h in boxes:
            box_rows.append(dict(image=r["file"], cls=names[int(c)], split=r["split"],
                                 cx=cx, cy=cy, w=w, h=h, area_share=w * h,
                                 w_px=w * r["width"], h_px=h * r["height"]))
    return names, pd.DataFrame(image_rows), pd.DataFrame(box_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="folder that contains meta.json")
    ap.add_argument("--out", type=Path, default=Path("results/audit"))
    args = ap.parse_args()

    meta, images, boxes, skipped = load_annotations(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    images.to_csv(args.out / "images.csv", index=False)
    boxes.to_csv(args.out / "boxes.csv", index=False)

    print(f"images: {len(images)}  boxes: {len(boxes)}  non-rectangle skipped: {skipped}")
    print("\nimage-level tags:\n", images["tags"].value_counts().to_string())
    print("\nimages with zero boxes:", int((images["n_boxes"] == 0).sum()))

    rows = []
    for c in [c["title"] for c in meta["classes"]]:
        b = boxes[boxes["cls"] == c] if len(boxes) else boxes
        rows.append(dict(cls=c, images=b["image"].nunique() if len(b) else 0,
                         boxes=len(b), expected_images=EXPECTED[c][0],
                         expected_boxes=EXPECTED[c][1]))
    print("\nper-class counts vs project notes:\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()