# Imbalance runs and evaluation (R00 to R03)

Code for the four core runs in the locked plan (Google Doc, "Experiments" and "Shared settings" tabs) and for evaluating them. Tested end to end on a tiny synthetic dataset on CPU (27 Nov), not yet on the real data or a GPU.

## Files and branches

| Branch | Files |
|---|---|
| `feat/train-config` | `configs/base.yaml`, `configs/R00-base.yaml`, `R01-mixup.yaml`, `R02-rfs.yaml`, `R03-clspw.yaml`, `src/train.py`, `notebooks/03-train.ipynb`, `docs/imbalance-runs.md` (this file) |
| `feat/oversample` | `src/oversample.py` |
| `feat/eval` | `src/evaluate.py`, `src/compare.py`, `notebooks/04-evaluate.ipynb` |

R02 needs both `feat/train-config` and `feat/oversample` merged.

## How to run (Colab, GPU runtime)

1. `notebooks/03-train.ipynb`: set `RUN`, run all. It builds `/content/yolo` if needed, and for R02 also builds the oversampled `/content/yolo_rfs`. Then it prints the merged settings and trains into the Drive runs folder.
2. `notebooks/04-evaluate.ipynb`: evaluates every run that has weights and writes `results/<run>/val_metrics.csv`. It then writes `results/comparison_val.md` with all the tables for the run tracker.

From a terminal it is the same thing:

```
python -m src.train configs/R01-mixup.yaml --data /content/yolo/dataset.yaml --project <Drive>/kara-agro/runs
python -m src.oversample --src /content/yolo --out /content/yolo_rfs        # R02 only
python -m src.evaluate --run R01-mixup --weights <Drive>/kara-agro/runs/R01-mixup/weights/best.pt \
    --data /content/yolo/dataset.yaml --split val --out results/R01-mixup
python -m src.compare --results results --split val
```

## Placeholders (match Neo's baseline)

`base.yaml` holds the locked defaults: yolo11s.pt, 40 epochs, batch 16. Neo's real values may differ, so override them on every run with `--model`, `--epochs` and `--batch` (in the notebook, the `MODEL`, `EPOCHS` and `BATCH` variables). Any other setting can be changed with `--set key=value`. Every change is written to `run_config.yaml` and `run_info.json` in the run folder, along with the Ultralytics and torch versions and the GPU. Use the same Ultralytics version for all four runs: if Neo's R00 used a different version, pin that one.

## What each run changes, and how

- **R01 mixup 0.3.** This is a single setting. Note that Ultralytics applies mixup inside the mosaic step, so it is switched off along with mosaic for the last `close_mosaic` = 10 epochs. With 40 epochs, mixup is active for 30 of them. The same applies to R00's mosaic, so the comparison stays fair, but the methods section should say so. (Checked: with 3 epochs and close_mosaic 10, R00 and R01 trained identically. With close_mosaic 0 they differed.)
- **R02 repeat factor sampling, t = 0.3** (LVIS, Gupta et al. 2019). For each class c, f_c is the share of train images that contain c, and r_c = max(1, sqrt(t / f_c)). An image's factor is the largest r_c of the classes in it (1 for background images). That factor is stochastically rounded with seed 42, and the image is repeated that many times as symlinked copies named `<stem>__rep1`, `__rep2` and so on. Labels are copied. Val and test are linked unchanged. The script prints f_c, r_c, and how many times each class's images appear before and after. Check that egg comes out at about ×4 and the total at about 13,400. `train.py` refuses to run R02 on the plain dataset, or any other run on the oversampled one.
- **R03 cls_pw 0.5.** Ultralytics has this built in (checked in the source code of 8.4.220, the latest version on PyPI). It counts the train boxes per class and sets w_c = (1 / count_c)^0.5, rescaled so the six weights average 1. Each class's channel of the classification BCE loss is multiplied by w_c at every anchor. So the weighting raises the cost of both missing a rare-class object and predicting that class wrongly. It does not change the box or DFL losses. The log line `Class weights: [...]` and `run_info.json` record the values used. On the synthetic data they matched a hand calculation exactly.

## What the evaluation reports

From one Ultralytics validation pass with the locked call (imgsz 640, batch 16, conf 0.001, iou 0.7):

- Per class: images, boxes, precision, recall and F1 (at the confidence that maximises mean F1, which is how Ultralytics reports them), AP50 and AP50-95.
- **R@P0.5:** the highest recall with precision of at least 0.5 on the IoU 0.5 curve, plus the confidence threshold that reaches it (useful later for per-class thresholds).
- **Bootstrap 95% intervals** for egg and larva AP50, from 1,000 resamples of the evaluation images with seed 42.
- Summary rows: macro mean over the 6 classes (the same as Ultralytics mAP50), the FAW mean (4 classes) and the rare mean (egg and larva).
- Confusion matrices at conf 0.25 (PNG and CSV), plus Ultralytics' own at 0.001. At 0.001 nearly every low-confidence box shows up as a background false positive, which makes that matrix hard to read.
- A built-in check: AP50 is recomputed from the kept per-image matches and must equal Ultralytics' number. That shows the extra metrics rest on the same predictions. It gave an exact match on the synthetic data.

`compare.py` builds the tables for the Run tracker: AP50 per class, recall per class with R@P0.5, the success rule against R00 (egg at least +10 points, none of damage, healthy or streak down by more than 2 points), and the best run on val (highest egg AP50, tie-break macro AP50). Runs without results yet are listed as missing.

## Not checked yet

- Nothing has run on the real dataset or a GPU yet. Run time and memory at 640 px with yolo11s are unknown.
- The `--resume` path and the Colab-only cells (Drive mount, pip) were not run.
- Symlinks need Linux (Colab). On Windows, pass `--copy` to `oversample.py`.
