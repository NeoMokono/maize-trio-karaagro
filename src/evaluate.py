"""Evaluate one trained run on val (or, once on Saturday, test) and write per-class metrics.

Usage (Colab):
    python -m src.evaluate --run R00-base \
        --weights /content/drive/MyDrive/kara-agro/runs/R00-base/weights/best.pt \
        --data /content/yolo/dataset.yaml --split val --out results/R00-base

Always pass the PLAIN dataset.yaml, also for R02: val and test are the same either way,
and this keeps the evaluation call identical for every run.

What it does
1. One Ultralytics validation pass with the locked settings (imgsz 640, batch 16,
   conf 0.001, iou 0.7). It keeps the per-image match table that Ultralytics normally
   throws away, so the extra metrics below come from the very same predictions.
2. Per class: images, boxes, precision, recall and F1 (at the confidence that maximises
   mean F1, as Ultralytics reports them), AP50, AP50-95.
3. Recall at precision >= 0.5 (R@P0.5) per class, from the IoU 0.5 precision-recall curve,
   and the confidence threshold that reaches it (useful later for per-class thresholds).
4. Bootstrap 95% intervals for AP50 (default egg and larva; 1,000 resamples of the
   evaluation images, seed 42). Needed because each of val and test has only ~45 egg images.
5. Summary rows: macro mean over the 6 classes (= Ultralytics mAP), FAW mean (4 classes),
   rare mean (egg, larva).
6. Confusion matrices at conf 0.25 (readable; at the 0.001 used for mAP almost every
   background box counts as a false positive), saved as PNG and CSV.

It re-computes AP50 from the kept match table and checks it equals the Ultralytics number,
so the bootstrap and R@P0.5 are known to rest on the same curve.

Outputs in --out:
  <split>_metrics.csv     one row per class plus three summary rows
  <split>_summary.json    settings, versions, the check result
  <split>_per_image.npz   per-image matches (for paired comparisons and error review)
  <split>_confusion_matrix.csv, plots_<split>/ (PR curves, confusion_matrix*.png)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

SEED = 42
FAW = ["fall armyworm egg", "fall armyworm larva", "fall armyworm frass", "fall armyworm larval damage"]
RARE = ["fall armyworm egg", "fall armyworm larva"]
SHORT = {"fall armyworm egg": "egg", "fall armyworm larva": "larva", "fall armyworm frass": "frass",
         "fall armyworm larval damage": "damage", "healthy maize": "healthy", "maize streak disease": "streak"}


# --- validator that keeps per-image statistics -------------------------------

def make_validator(cm_conf: float):
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ConfusionMatrix

    class KeepStatsValidator(DetectionValidator):
        """DetectionValidator that keeps the per-image match table and a second confusion matrix."""

        def init_metrics(self, model):
            super().init_metrics(model)
            self.kept_files = []
            self.cm_readable = ConfusionMatrix(names=model.names) if cm_conf > 0 else None

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                self.kept_files.append(os.path.basename(pbatch["im_file"]))
                if self.cm_readable is not None:
                    self.cm_readable.process_batch(self._prepare_pred(pred), pbatch, conf=cm_conf)

        def get_stats(self):
            self.kept_stats = {k: list(v) for k, v in self.metrics.stats.items()}  # cleared by super()
            return super().get_stats()

    return KeepStatsValidator


# --- AP50 from the match table (same arithmetic as ultralytics ap_per_class) ---

def class_arrays(stats: dict, c: int) -> tuple[list, list, np.ndarray]:
    """Per image: (tp at IoU 0.5, conf) for predictions of class c, and the number of class-c boxes."""
    tps, confs, ngt = [], [], []
    for tp, conf, pc, tc in zip(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"]):
        m = pc == c
        tps.append(np.asarray(tp)[m, 0].astype(bool))
        confs.append(np.asarray(conf)[m])
        ngt.append(int((np.asarray(tc) == c).sum()))
    return tps, confs, np.array(ngt)


def ap50_and_curve(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    from ultralytics.utils.metrics import compute_ap
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])
    if tp.size == 0:
        return 0.0, np.array([]), np.array([]), np.array([])
    order = np.argsort(-conf, kind="stable")
    tp, conf = tp[order], conf[order]
    tpc = tp.cumsum()
    recall = tpc / n_gt
    precision = tpc / np.arange(1, tp.size + 1)
    ap, _, _ = compute_ap(recall, precision)
    return float(ap), recall, precision, conf


def recall_at_precision(recall, precision, conf, p_min: float = 0.5) -> tuple[float, float]:
    """Highest recall reached with precision >= p_min, and the confidence threshold for it."""
    ok = precision >= p_min
    if not ok.any():
        return 0.0, float("nan")
    i = np.flatnonzero(ok)[np.argmax(recall[ok])]
    return float(recall[i]), float(conf[i])


def bootstrap_ap50(tps, confs, ngt, n_boot=1000, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(ngt)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g = int(ngt[idx].sum())
        if g == 0:
            continue
        tp = np.concatenate([tps[i] for i in idx])
        cf = np.concatenate([confs[i] for i in idx])
        vals.append(ap50_and_curve(tp, cf, g)[0])
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --- main ----------------------------------------------------------------------

def evaluate(run: str, weights: str, data: str, split: str, out: str, imgsz=640, batch=16,
             conf=0.001, iou=0.7, cm_conf=0.25, n_boot=1000, ci_classes=("egg", "larva"),
             p_min=0.5, seed=SEED, device=None) -> pd.DataFrame:
    import ultralytics
    from ultralytics import YOLO

    os.makedirs(out, exist_ok=True)
    out = os.path.abspath(out)
    model = YOLO(weights)
    kw = dict(data=data, split=split, imgsz=imgsz, batch=batch, conf=conf, iou=iou, plots=True,
              project=out, name=f"plots_{split}", exist_ok=True, verbose=False)
    if device is not None:
        kw["device"] = device
    v_cls = make_validator(cm_conf)
    holder = {}

    def grab(validator):
        holder["v"] = validator
    model.add_callback("on_val_end", grab)
    metrics = model.val(validator=v_cls, **kw)
    v = holder["v"]
    stats = {k: v.kept_stats[k] for k in ("tp", "conf", "pred_cls", "target_cls")}
    names = {int(k): n for k, n in metrics.names.items()}

    box = metrics.box
    idx = {int(c): i for i, c in enumerate(box.ap_class_index)}
    rows, check = [], []
    for c, name in names.items():
        short = SHORT.get(name, name)
        tps, confs, ngt = class_arrays(stats, c)
        ap, rec, prec, cf = ap50_and_curve(np.concatenate(tps) if tps else np.array([], bool),
                                           np.concatenate(confs) if confs else np.array([]), int(ngt.sum()))
        r_at_p, conf_at = recall_at_precision(rec, prec, cf, p_min) if rec.size else (0.0, float("nan"))
        i = idx.get(c)
        p_, r_ = (float(box.p[i]), float(box.r[i])) if i is not None else (np.nan, np.nan)
        u_ap50 = float(box.ap50[i]) if i is not None else np.nan
        if i is not None:
            check.append(abs(u_ap50 - ap))
        lo = hi = np.nan
        if short in ci_classes and n_boot > 0 and ngt.sum() > 0:
            lo, hi = bootstrap_ap50(tps, confs, ngt, n_boot, seed)
        rows.append({
            "run": run, "split": split, "class": short,
            "images": int(metrics.nt_per_image[c]), "boxes": int(metrics.nt_per_class[c]),
            "precision": p_, "recall": r_,
            "f1": 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0,
            "ap50": u_ap50, "ap50_95": float(box.ap[i]) if i is not None else np.nan,
            f"r_at_p{p_min:g}": r_at_p, f"conf_at_r_at_p{p_min:g}": conf_at,
            "ap50_ci_low": lo, "ap50_ci_high": hi,
        })
    df = pd.DataFrame(rows)
    num = ["precision", "recall", "f1", "ap50", "ap50_95", f"r_at_p{p_min:g}"]
    for label, members in (("macro (6)", list(SHORT.values())),
                           ("FAW mean (4)", [SHORT[n] for n in FAW]),
                           ("rare mean (egg, larva)", [SHORT[n] for n in RARE])):
        sub = df[df["class"].isin(members)]
        df = pd.concat([df, pd.DataFrame([{"run": run, "split": split, "class": label,
                                           **sub[num].mean().to_dict()}])], ignore_index=True)

    max_diff = max(check) if check else float("nan")
    check_ok = bool(max_diff < 1e-6)
    macro_ok = abs(float(df.loc[df["class"] == "macro (6)", "ap50"].iloc[0]) - float(box.map50)) < 1e-6
    df.to_csv(os.path.join(out, f"{split}_metrics.csv"), index=False)

    # per-image matches, for paired comparisons between runs and the error review
    n_pred = [len(np.asarray(c)) for c in stats["conf"]]
    n_tgt = [len(np.asarray(c)) for c in stats["target_cls"]]
    np.savez_compressed(
        os.path.join(out, f"{split}_per_image.npz"),
        files=np.array(v.kept_files),                                   # image i = files[i]
        pred_img=np.repeat(np.arange(len(n_pred)), n_pred),
        pred_cls=np.concatenate(stats["pred_cls"]).astype(int),
        pred_conf=np.concatenate(stats["conf"]),
        pred_tp=np.concatenate(stats["tp"]),                            # (n, 10): IoU 0.50 ... 0.95
        tgt_img=np.repeat(np.arange(len(n_tgt)), n_tgt),
        tgt_cls=np.concatenate(stats["target_cls"]).astype(int))

    # confusion matrices
    plots = os.path.join(out, f"plots_{split}")
    labels = [SHORT.get(names[c], names[c]) for c in sorted(names)] + ["background"]
    if v.cm_readable is not None:
        cm = v.cm_readable
        pd.DataFrame(cm.matrix, index=[f"pred {x}" for x in labels], columns=[f"true {x}" for x in labels]) \
            .to_csv(os.path.join(out, f"{split}_confusion_matrix_conf{cm_conf:g}.csv"))
        for f in os.listdir(plots):
            if f.startswith("confusion_matrix"):  # Ultralytics' own, at conf 0.001
                shutil.move(os.path.join(plots, f), os.path.join(plots, f.replace(".png", f"_conf{conf:g}.png")))
        sub = os.path.join(plots, f"cm_conf{cm_conf:g}")
        os.makedirs(sub, exist_ok=True)
        for normalize in (True, False):
            cm.plot(save_dir=sub, normalize=normalize)

    summary = {
        "run": run, "split": split, "weights": os.path.abspath(weights), "data": os.path.abspath(data),
        "settings": {"imgsz": imgsz, "batch": batch, "conf": conf, "iou": iou, "cm_conf": cm_conf,
                     "bootstrap": n_boot, "seed": seed, "precision_target": p_min},
        "ultralytics": ultralytics.__version__,
        "mAP50": float(box.map50), "mAP50_95": float(box.map),
        "check_ap50_matches_ultralytics": check_ok and macro_ok, "check_max_abs_diff": max_diff,
        "images_evaluated": len(v.kept_files),
    }
    with open(os.path.join(out, f"{split}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    show = df.copy()
    for col in num + ["ap50_ci_low", "ap50_ci_high"]:
        show[col] = (100 * show[col]).round(1)
    print(show.drop(columns=["run", "split"]).to_string(index=False))
    print(f"\n[{'PASS' if summary['check_ap50_matches_ultralytics'] else 'FAIL'}] AP50 recomputed from the "
          f"kept matches equals Ultralytics (max diff {max_diff:.2e})")
    print(f"wrote {out}/{split}_metrics.csv")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run id, e.g. R00-base")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="the plain dataset.yaml (also for R02)")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", required=True, help="e.g. results/R00-base")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--cm-conf", type=float, default=0.25, help="confidence for the readable confusion matrix; 0 skips")
    p.add_argument("--bootstrap", type=int, default=1000, help="resamples for the AP50 intervals; 0 skips")
    p.add_argument("--ci-classes", nargs="+", default=["egg", "larva"])
    p.add_argument("--precision-target", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.split == "test":
        print("NOTE: test split. The plan uses it once, on Saturday, for all finished runs.")
    evaluate(a.run, a.weights, a.data, a.split, a.out, a.imgsz, a.batch, a.conf, a.iou, a.cm_conf,
             a.bootstrap, tuple(a.ci_classes), a.precision_target, a.seed, a.device)


if __name__ == "__main__":
    main()
