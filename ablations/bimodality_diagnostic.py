#!/usr/bin/env python3
"""Quantify the bimodal-stance-sampling artifact on shadow elicitation.

For each aligned-agent shadow probability distribution, check whether the
model puts substantial mass on BOTH stance=1 (extreme A) AND stance=7
(extreme B). This is the scale-direction confusion observed on harmbench:
the model wants to refuse, but the stance digit it commits to is ~50/50
between the two extremes, so sampling produces wildly different stances
that don't reflect a real belief difference.

Reports, per (dataset, variant):
  - frac_bimodal_strict  : P(1) > 0.3 AND P(7) > 0.3
  - frac_bimodal_loose   : P(1) > 0.2 AND P(7) > 0.2
  - frac_strong_extreme  : P(1) > 0.7 OR P(7) > 0.7  (clean extreme)
  - frac_neutral_dominant: P(4) > 0.5
  - mean_extreme_split   : mean of min(P(1),P(7)) (higher = more bimodal)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["synthetic", "moral_stories", "harmbench_standard",
            "harmbench_contextual", "harmbench_copyright"]


def normalize(p: dict | None) -> dict[int, float] | None:
    if not p:
        return None
    out = {int(k): float(v) for k, v in p.items() if str(k).isdigit()}
    total = sum(out.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def iter_shadow_probs(path: str):
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                p = normalize(a.get("shadow_probs"))
                if p is None:
                    continue
                yield p


def diagnose(path: str, label: str) -> dict | None:
    probs = list(iter_shadow_probs(path))
    if not probs:
        return None
    P1 = np.array([p.get(1, 0.0) for p in probs])
    P7 = np.array([p.get(7, 0.0) for p in probs])
    P4 = np.array([p.get(4, 0.0) for p in probs])

    bim_strict = ((P1 > 0.3) & (P7 > 0.3)).mean()
    bim_loose = ((P1 > 0.2) & (P7 > 0.2)).mean()
    bim_softer = ((P1 > 0.1) & (P7 > 0.1)).mean()
    strong_ext = ((P1 > 0.7) | (P7 > 0.7)).mean()
    neut_dom = (P4 > 0.5).mean()
    extreme_split = np.minimum(P1, P7).mean()  # higher => more bimodal

    return {
        "label": label,
        "n_agents": len(probs),
        "frac_bimodal_strict_>0.3": bim_strict,
        "frac_bimodal_loose_>0.2": bim_loose,
        "frac_bimodal_soft_>0.1": bim_softer,
        "frac_strong_extreme_>0.7": strong_ext,
        "frac_neutral_dominant": neut_dom,
        "mean_min(P1,P7)": extreme_split,
        "mean_P1": P1.mean(),
        "mean_P7": P7.mean(),
        "mean_P4": P4.mean(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-key", default="qwen-7b-instruct")
    args = p.parse_args()

    rows = []
    for ds in DATASETS:
        for variant, root in [
            ("own-last", "outputs/primary_em"),
            ("summary", "outputs/shadow_summary_ablation"),
        ]:
            path = f"{root}/{ds}/{args.model_key}/results.jsonl"
            r = diagnose(path, f"{ds} / {variant}")
            if r is not None:
                r["dataset"] = ds
                r["variant"] = variant
                rows.append(r)

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)
    df = df[["dataset", "variant", "n_agents",
             "frac_bimodal_strict_>0.3", "frac_bimodal_loose_>0.2",
             "frac_bimodal_soft_>0.1", "frac_strong_extreme_>0.7",
             "frac_neutral_dominant", "mean_min(P1,P7)",
             "mean_P1", "mean_P7", "mean_P4"]]

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)

    print("\n=== Bimodality (P(1) and P(7) both high) per dataset/variant ===\n")
    print(df.to_string(index=False))

    # Side-by-side: own-last vs summary delta in bimodality
    pivot = (df.pivot_table(index="dataset", columns="variant",
                            values="frac_bimodal_strict_>0.3"))
    if {"own-last", "summary"}.issubset(pivot.columns):
        pivot["delta (own-summary)"] = pivot["own-last"] - pivot["summary"]
        print("\n=== Bimodality strict (P>0.3): own-last vs summary ===")
        print(pivot.to_string())

    out_csv = Path("plots_tables/bimodality_diagnostic.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
