# mAIze Trio: KaraAgro Maize Imbalance Study

Team **mAIze Trio**, AI Research Foundations Capstone (supported by Google and DeepMind).

**Team:** Neo Mokono, Hawa Ibrahim, Dr Olaperi Okuboyejo

## The problem

Fall armyworm (FAW) can destroy a maize field quickly, and control works best when eggs and young larvae are found early. In the KaraAgro AI Maize dataset (16,552 images), those are the rarest and smallest labels.

| Class | Images | Labelled boxes | Average share of image |
|---|---|---|---|
| Larval damage | 5,138 | 38,076 | 20.7% |
| Frass | 3,376 | 15,218 | 13.2% |
| Larva | 1,454 | 1,512 | 3.8% |
| Egg | 301 | 1,026 | 16.7% |

Damage outnumbers eggs about 37 to 1. At image level the three categories are roughly balanced (maize streak 6,095, FAW 5,509, healthy 4,948), so the real imbalance is inside the FAW labels.

**Research question:** can a detector catch FAW early, at the egg and larva stage, despite how rare those labels are?

## Three problems to plan for

1. **Rarity and size are mixed up.** Larvae are both rare and tiny. Fix: cut a common class such as frass down to the egg count and plot detection quality against examples remaining.
2. **No official test set, so leakage is likely.** Photos of the same plant or plot look nearly identical. Fix: group near-duplicates (perceptual hashing) before splitting, and publish the split.
3. **The model may learn the camera, not the pest.** The dataset page warns about camera differences across regions. Fix: test on degraded copies (blur, heavy JPEG compression, low light).

## Experiment plan

- **Baseline:** a YOLO detector trained on the deduplicated split.
- **Fixes to compare:** oversampling images with rare classes; focal or class-weighted loss; pasting egg and larva crops into other training images; tiling or higher-resolution inputs.
- **Metrics:** per-class accuracy, plus recall on eggs and larvae at a fixed precision. Do not report only mAP.
- **Optional:** per-class confidence thresholds that send uncertain photos to an extension officer.

Budget: 10 to 15 training runs. The dataset is about 1.5 GB and listed as Colab-friendly.

## Scope

We will not build a farmer-facing app, give treatment advice, or replace an agronomist. We will not collect new images, generate synthetic ones, or compare many architectures. Results apply to this dataset only, not to South African fields. Only the method (not the model) carries over to citrus.

## Start here (read in this order)

1. [AIMS Capstone: KaraAgro AI Maize](https://capstone.aims.ac.za/datasets/karaagro-maize/): the dataset page, with licence (CC0), warnings and download.
2. [Dataset Ninja: KaraAgro AI Maize](https://datasetninja.com/kara-agro-ai-maize): class statistics and sample images.
3. [Ultralytics YOLO docs](https://docs.ultralytics.com/): training and evaluating the baseline detector.
4. [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002): the loss behind one of our fixes.
5. [Simple Copy-Paste is a Strong Data Augmentation Method (Ghiasi et al., 2020)](https://arxiv.org/abs/2012.07177): the idea behind pasting rare-class crops.
6. [imagehash](https://github.com/JohannesBuchner/imagehash): perceptual hashing for near-duplicate grouping.
7. [SAHI](https://github.com/obss/sahi): sliced inference for small objects (tiling).

_Add: an FAW background source, and any South African maize references._

## Open questions

- Can anyone in the group find South African maize photos during the project?

## Repo layout

```
README.md        this file
docs/            plan, decisions, meeting notes
notebooks/       Colab notebooks (baseline, experiments)
src/             shared code (splits, metrics)
splits/          published train/val/test file lists
results/         metrics tables and figures
```

## Licence

Code: MIT (to confirm). The dataset is CC0, so our splits and models can be released openly.
