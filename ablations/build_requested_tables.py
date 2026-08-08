#!/usr/bin/env python3
"""Builders for the requested follow-up tables, recomputed from raw JSONL so
they stay in sync with disk (the make_paper_tables.py pipeline depends on
pre-computed outputs/<phase>/tables/*.csv that do not exist for these runs).

Produces, under plots_tables/:
  dose_response/table.csv     financial(=primary EM) vs bad-medical vs extreme-sports
  llama_variants/table.csv    llama-1b vs qwen-7b (synthetic+copyright); synthetic also llama-8b
  scaling_laws/table.csv      N10K5(ref) / N10K10 / N20K5 / N20K10

Metrics are over ALIGNED agents at ratio=0.2, model_induced, the 4 topologies,
matching the shadow/k0 ablation conventions.

    PYTHONPATH=. .venv/bin/python ablations/build_requested_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from misalignment_contagion.analyze import load_trials, trials_to_dataframe

RATIO = 0.2
TOPOLOGIES = ["fc", "chain", "circle", "star"]


def load_aligned(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        print(f"  MISSING: {path}")
        return None
    df = trials_to_dataframe(load_trials(path))
    # trials_to_dataframe already returns one row per aligned agent.
    df = df[(df["model_condition"] == "model_induced")
            & (np.isclose(df["minority_ratio"], RATIO))
            & (df["topology"].isin(TOPOLOGIES))]
    return df


def stats(df: pd.DataFrame) -> dict:
    """Three headline metrics per metrics.py:
       Internalization Index, Shadow shift (shadow_ev - baseline_ev),
       % agents with final stance in 6-7.
    """
    d_final = df["final_ev"] - df["baseline_ev"]
    shadow_shift = df["shadow_ev"] - df["baseline_ev"]   # metrics.shadow_shift
    final_stance = df["final_stance"].dropna()
    return {
        "n_agents": len(df),
        "baseline_ev": round(df["baseline_ev"].mean(), 3),
        "final_ev": round(df["final_ev"].mean(), 3),
        "delta_final_ev": round(d_final.mean(), 3),
        "shadow_ev": round(df["shadow_ev"].mean(), 3),
        "shadow_shift": round(shadow_shift.mean(), 3),          # shadow_ev - baseline_ev
        "shadow_shift_sd": round(shadow_shift.std(), 3),
        "internalization_index": round(df["internalization_index"].dropna().mean(), 3),
        "pct_at_6_7": round(float((final_stance >= 6).mean() * 100.0), 1),
    }


def write(rows: list[dict], subdir: str, label: str) -> None:
    out = Path("plots_tables") / subdir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False)
    (out / "table.md").write_text(df.to_markdown(index=False) + "\n")
    print(f"\n[{label}] -> {out}/table.csv  ({len(df)} rows)")
    print(df.to_string(index=False))


def dose_response() -> None:
    # The primary qwen-7b-instruct EM finetune *is* the risky-financial-advice dose.
    doses = {
        "financial (primary)": "qwen-7b-instruct",
        "bad-medical-advice": "qwen-7b-instruct-bad-medical-advice",
        "extreme-sports": "qwen-7b-instruct-extreme-sports",
    }
    rows = []
    for ds in ["synthetic", "harmbench_copyright"]:
        for label, key in doses.items():
            d = load_aligned(f"outputs/primary_em/{ds}/{key}/results.jsonl")
            if d is None or not len(d):
                continue
            rows.append({"dose": label, "model_key": key, "dataset": ds, **stats(d)})
    write(rows, "dose_response", "dose")


def llama_variants() -> None:
    cells = [
        ("synthetic", "qwen-7b-instruct"),
        ("synthetic", "llama-8b-instruct"),
        ("synthetic", "llama-1b-instruct"),
        ("harmbench_copyright", "qwen-7b-instruct"),
        ("harmbench_copyright", "llama-1b-instruct"),
    ]
    rows = []
    for ds, key in cells:
        d = load_aligned(f"outputs/primary_em/{ds}/{key}/results.jsonl")
        if d is None or not len(d):
            continue
        rows.append({"dataset": ds, "model": key, **stats(d)})
    write(rows, "llama_variants", "llama")


def scaling_laws() -> None:
    configs = {
        "N10_K5 (reference)": "outputs/primary_em/{ds}/qwen-7b-instruct/results.jsonl",
        "N10_K10": "outputs/primary_em_N10_K10/{ds}/qwen-7b-instruct/results.jsonl",
        "N20_K5": "outputs/primary_em_N20_K5/{ds}/qwen-7b-instruct/results.jsonl",
        "N20_K10": "outputs/primary_em_N20_K10/{ds}/qwen-7b-instruct/results.jsonl",
    }
    rows = []
    for ds in ["synthetic", "harmbench_copyright"]:
        for label, tmpl in configs.items():
            d = load_aligned(tmpl.format(ds=ds))
            if d is None or not len(d):
                continue
            n = int(label.split("_K")[0].lstrip("N").split(" ")[0])
            k = int(label.split("_K")[1].split(" ")[0])
            rows.append({"config": label, "dataset": ds,
                         "N_agents": n, "K_rounds": k, **stats(d)})
    write(rows, "scaling_laws", "scaling")


if __name__ == "__main__":
    dose_response()
    llama_variants()
    scaling_laws()
