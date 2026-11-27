"""Train one run of the imbalance study: configs/base.yaml plus one override file.

Usage (from the repo root, on Colab):
    python -m src.train configs/R00-base.yaml --data /content/yolo/dataset.yaml \
        --project /content/drive/MyDrive/kara-agro/runs
    python -m src.train configs/R02-rfs.yaml  --data /content/yolo_rfs/dataset.yaml --project ...
    python -m src.train configs/R03-clspw.yaml --data /content/yolo/dataset.yaml --project ... --epochs 30

Placeholders (model, epochs, batch) can be overridden so every run matches Neo's baseline:
    --model yolo11n.pt --epochs 30 --batch 8      or any Ultralytics argument with --set key=value
Anything changed this way is written to <run folder>/run_config.yaml, so the record stays exact.

    --dry-run   print the merged settings and stop (no GPU needed)
    --resume    continue an interrupted run from <project>/<name>/weights/last.pt

Each run folder gets the usual Ultralytics files (weights/best.pt, results.csv, args.yaml,
plots) plus run_config.yaml (merged config and command-line changes) and run_info.json
(versions, GPU, dataset, and for R03 the class weights actually used).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import yaml

RUN_KEYS = {"id", "changes", "needs_rfs"}


def load_config(path: str) -> dict:
    """Read a run config, following ``inherit:`` (relative to the file) and merging the run block."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(os.path.join(os.path.dirname(path), parent))
        run = {**base.get("run", {}), **cfg.get("run", {})}
        cfg = {**base, **cfg, "run": run}
    return cfg


def parse_value(v: str):
    """--set values: YAML parsing turns '0.3' into a float, 'true' into a bool, and so on."""
    return yaml.safe_load(v)


def merged_settings(a: argparse.Namespace) -> tuple[dict, dict, dict]:
    cfg = load_config(a.config)
    run = cfg.pop("run", {})
    changed = {}
    for key in ("model", "epochs", "batch"):
        v = getattr(a, key)
        if v is not None:
            changed[key] = v
    for kv in a.set or []:
        if "=" not in kv:
            raise SystemExit(f"--set needs key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        changed[k] = parse_value(v)
    cfg.update(changed)
    unknown = set(cfg) - set(_ultralytics_keys()) - {"model"}
    if unknown:
        raise SystemExit(f"not Ultralytics train arguments: {sorted(unknown)}")
    return cfg, run, changed


def _ultralytics_keys() -> list[str]:
    from ultralytics.cfg import get_cfg
    return list(vars(get_cfg()).keys())


def check_data(data: str, run: dict):
    if not os.path.exists(data):
        raise SystemExit(f"dataset.yaml not found: {data}")
    has_rfs = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(data)), "rfs_manifest.csv"))
    if run.get("needs_rfs") and not has_rfs:
        raise SystemExit(f"{run.get('id')} needs the oversampled dataset (python -m src.oversample ...); "
                         f"{data} has no rfs_manifest.csv next to it")
    if not run.get("needs_rfs") and has_rfs:
        raise SystemExit(f"{run.get('id')} must use the plain dataset, but {data} is the oversampled one")


def env_info() -> dict:
    import torch
    import ultralytics
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "python": platform.python_version(), "gpu": gpu}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="configs/R0x-*.yaml")
    p.add_argument("--data", required=True, help="dataset.yaml (plain for R00/R01/R03, oversampled for R02)")
    p.add_argument("--project", default="runs", help="parent folder for run folders, e.g. on Drive")
    p.add_argument("--name", help="run folder name (default: the config's run id)")
    p.add_argument("--model", help="override the model placeholder, e.g. yolo11s.pt")
    p.add_argument("--epochs", type=int, help="override the epochs placeholder")
    p.add_argument("--batch", type=int, help="override batch (e.g. 8 after out of memory)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="any other Ultralytics argument")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)

    cfg, run, changed = merged_settings(a)
    name = a.name or run.get("id") or os.path.splitext(os.path.basename(a.config))[0]
    check_data(a.data, run)
    save_dir = os.path.join(a.project, name)

    print(f"run {name}: {run.get('changes', '')}")
    print(yaml.safe_dump({**cfg, "data": a.data}, sort_keys=False))
    if changed:
        print(f"changed from the config on the command line: {changed}")
    if a.dry_run:
        return None

    from ultralytics import YOLO

    if a.resume:
        last = os.path.join(save_dir, "weights", "last.pt")
        if not os.path.exists(last):
            raise SystemExit(f"nothing to resume: {last} not found")
        model = YOLO(last)
        model.train(resume=True)
    else:
        if os.path.exists(os.path.join(save_dir, "weights")):
            raise SystemExit(f"{save_dir} already has weights; use --resume, or a new --name")
        model_name = cfg.pop("model")
        model = YOLO(model_name)
        model.train(data=a.data, project=a.project, name=name, exist_ok=True, **cfg)
        cfg["model"] = model_name

    trainer = model.trainer
    out = str(trainer.save_dir)
    info = {"run": run, "data": os.path.abspath(a.data), "command": " ".join(sys.argv),
            "changed_on_command_line": changed, **env_info()}
    cw = getattr(getattr(trainer, "model", None), "class_weights", None)
    if cw is not None:
        info["class_weights"] = [round(float(x), 4) for x in cw.cpu().numpy()]
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    if not a.resume:
        with open(os.path.join(out, "run_config.yaml"), "w") as f:
            yaml.safe_dump({"run": run, **cfg, "data": a.data}, f, sort_keys=False)
    print(f"\nrun folder: {out}\nbest weights: {os.path.join(out, 'weights', 'best.pt')}")
    if "class_weights" in info:
        print(f"class weights used: {info['class_weights']}")
    return out


if __name__ == "__main__":
    main()
