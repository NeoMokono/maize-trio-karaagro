"""Compare finished runs: the tables for the run tracker, the success rule and the best run.

Usage (after src/evaluate.py has written results/<run>/<split>_metrics.csv for each run):
    python -m src.compare --results results --split val \
        --runs R00-base R01-mixup R02-rfs R03-clspw --baseline R00-base

Writes results/comparison_<split>.md (paste into the Google Doc) and comparison_<split>.csv.
Runs with no metrics file yet are listed as missing and skipped.

Tables
1. AP50 per class, macro mAP50, mAP50-95, R@P0.5 for egg and larva, egg and larva AP50 95% CI
   (same columns as the "Validation results" table in the run tracker).
2. Recall per class: recall at the max-F1 confidence, and R@P0.5 (highest recall with
   precision >= 0.5). The rare classes (egg, larva) come first.
3. Success rule (proposal): egg AP50 at least +10 points over the baseline, and none of
   damage, healthy, streak down by more than 2 points.
4. Best run on this split: highest egg AP50, tie-break macro AP50 (plan rule). Only pick it on val.
All numbers are percentages with one decimal.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

CLASSES = ["egg", "larva", "frass", "damage", "healthy", "streak"]
MAJORITY = ["damage", "healthy", "streak"]
MACRO = "macro (6)"


def load(results: str, runs: list[str], split: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    got, missing = {}, []
    for r in runs:
        p = os.path.join(results, r, f"{split}_metrics.csv")
        if os.path.exists(p):
            got[r] = pd.read_csv(p).set_index("class")
        else:
            missing.append(r)
    return got, missing


def pct(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def table_ap(m: dict[str, pd.DataFrame], rp: str) -> pd.DataFrame:
    rows = []
    for r, d in m.items():
        row = {"Run": r}
        for c in CLASSES:
            row[c.capitalize()] = pct(d.loc[c, "ap50"]) if c in d.index else ""
        row["mAP50 (macro)"] = pct(d.loc[MACRO, "ap50"])
        row["mAP50-95"] = pct(d.loc[MACRO, "ap50_95"])
        row["FAW mean AP50"] = pct(d.loc["FAW mean (4)", "ap50"])
        row["Rare mean AP50"] = pct(d.loc["rare mean (egg, larva)", "ap50"])
        for c in ("egg", "larva"):
            row[f"{c.capitalize()} R@P0.5"] = pct(d.loc[c, rp])
            lo, hi = d.loc[c, "ap50_ci_low"], d.loc[c, "ap50_ci_high"]
            row[f"{c.capitalize()} AP50 95% CI"] = "" if pd.isna(lo) else f"{pct(lo)} to {pct(hi)}"
        rows.append(row)
    return pd.DataFrame(rows)


def table_recall(m: dict[str, pd.DataFrame], rp: str) -> pd.DataFrame:
    rows = []
    for r, d in m.items():
        row = {"Run": r}
        for c in CLASSES:
            row[f"{c} R"] = pct(d.loc[c, "recall"])
            row[f"{c} R@P0.5"] = pct(d.loc[c, rp])
        rows.append(row)
    return pd.DataFrame(rows)


def table_rule(m: dict[str, pd.DataFrame], base: str, gain=0.10, max_drop=0.02) -> pd.DataFrame:
    if base not in m:
        return pd.DataFrame()
    b = m[base]
    rows = []
    for r, d in m.items():
        if r == base:
            continue
        egg = d.loc["egg", "ap50"] - b.loc["egg", "ap50"]
        larva = d.loc["larva", "ap50"] - b.loc["larva", "ap50"]
        drops = {c: d.loc[c, "ap50"] - b.loc[c, "ap50"] for c in MAJORITY}
        worst = min(drops, key=drops.get)
        ok = egg >= gain - 1e-9 and drops[worst] >= -max_drop - 1e-9
        rows.append({"Run": r, "Egg change": f"{100 * egg:+.1f}", "Larva change": f"{100 * larva:+.1f}",
                     "Worst change among damage, healthy, streak": f"{100 * drops[worst]:+.1f} ({worst})",
                     "Meets the rule?": "yes" if ok else "no"})
    return pd.DataFrame(rows)


def best_run(m: dict[str, pd.DataFrame]) -> tuple[str, str]:
    key = sorted(m, key=lambda r: (round(m[r].loc["egg", "ap50"], 4), m[r].loc[MACRO, "ap50"]), reverse=True)
    r = key[0]
    why = (f"highest egg AP50 ({pct(m[r].loc['egg', 'ap50'])}); macro AP50 {pct(m[r].loc[MACRO, 'ap50'])}")
    return r, why


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(nothing to show)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--runs", nargs="+", default=["R00-base", "R01-mixup", "R02-rfs", "R03-clspw"])
    p.add_argument("--baseline", default="R00-base")
    p.add_argument("--split", default="val", choices=["val", "test"])
    a = p.parse_args()

    m, missing = load(a.results, a.runs, a.split)
    if not m:
        raise SystemExit(f"no {a.split}_metrics.csv found under {a.results}/ for {a.runs}")
    any_df = next(iter(m.values()))
    rp = next(c for c in any_df.columns if c.startswith("r_at_p"))

    ap, rec, rule = table_ap(m, rp), table_recall(m, rp), table_rule(m, a.baseline)
    md = [f"# Run comparison ({a.split})\n"]
    if missing:
        md.append(f"Missing (no {a.split}_metrics.csv yet): {', '.join(missing)}\n")
    md += ["## AP50 per class (%)\n", to_md(ap),
           "## Recall per class (%)\n",
           "R = recall at the confidence that maximises mean F1 (Ultralytics). "
           "R@P0.5 = highest recall with precision at least 0.5.\n", to_md(rec),
           f"## Success rule vs {a.baseline}\n",
           "Egg AP50 at least +10 points, and none of damage, healthy, streak down by more than 2 points.\n",
           to_md(rule) if a.baseline in m else f"_{a.baseline} missing, so no rule check._\n"]
    if a.split == "val":
        r, why = best_run(m)
        md += ["## Best run on val\n", f"{r}: {why}. (Rule: highest egg AP50, tie-break macro AP50.)\n"]
    text = "\n".join(md)
    os.makedirs(a.results, exist_ok=True)
    with open(os.path.join(a.results, f"comparison_{a.split}.md"), "w") as f:
        f.write(text)
    long = pd.concat([d.reset_index().assign(run=r) for r, d in m.items()], ignore_index=True)
    long.to_csv(os.path.join(a.results, f"comparison_{a.split}.csv"), index=False)
    print(text)
    print(f"wrote {a.results}/comparison_{a.split}.md and .csv")


if __name__ == "__main__":
    main()
