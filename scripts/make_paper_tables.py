#!/usr/bin/env python3
"""
make_paper_tables.py — Generate all paper-ready summary tables.

Reads pre-computed metric CSVs from outputs/*/tables/ and produces
clean, publication-focused tables covering:

  T1  Core contagion effect (Qwen-7B, all datasets, FC 20%)
  T2  Topology × mechanism (Qwen-7B, synthetic + moral_stories, ratio=0.2)
  T3  Dose-response: minority ratio (Qwen-7B, synthetic, all topologies)
  T4  Star position: hub vs leaf (Qwen-7B, synthetic, all ratios)
  T5  Model comparison (synthetic, FC 20%: 0.5B vs 7B-instruct vs 7B-base vs Llama-8B)
  T6  Prompt sensitivity 2×2 (Qwen-7B, FC + Star, ratio=0.2)
  T7  Prompt-induced vs model-induced (Qwen-7B + Llama, synthetic, FC 20%)
  T8  Seed reliability (Qwen-7B, synthetic, FC 20%, seeds 42/123/456)

Metrics per table (matching paper requirements):
  - Delta Conversion Rate (EV CR at threshold 0.5)
  - DTW ratio (Asch/Moscovici kernel distance ratio)
  - Logprob EV shift (shadow - baseline)
  - Delta CC (public - private stance gap; + = Asch, - = Moscovici)
  - Internalization Index (II) and Shadow Reversion Fraction (SRF)

Usage:
  python scripts/make_paper_tables.py
  python scripts/make_paper_tables.py --output-dir outputs/paper_tables
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load(phase: str, table: str) -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, phase, "tables", f"{table}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_metrics(phase: str) -> pd.DataFrame:
    """Merge the four paper metrics into one wide dataframe for a given phase."""
    keys = ["dataset", "model_key", "prompt_strategy", "model_condition",
            "topology", "minority_ratio", "position_config"]

    ev_raw = load(phase, "logprob_ev_table")
    if ev_raw.empty:
        return pd.DataFrame()
    ev = ev_raw[keys + ["n", "shift_shadow", "baseline_ev_mean", "shadow_ev_mean"]]
    ev = ev.rename(columns={"shift_shadow": "logprob_ev_shift", "n": "n_agents"})

    cr = load(phase, "ev_cr_table")[keys + ["ev_cr_0.5"]]

    dtw = load(phase, "dtw_table")[keys + ["mean_rho", "std_rho"]]
    dtw = dtw.rename(columns={"mean_rho": "dtw_rho", "std_rho": "dtw_rho_std"})

    cc = load(phase, "delta_cc_table")[keys + ["mean_delta_cc", "std_delta_cc", "frac_asch", "frac_moscovici"]]

    ii = load(phase, "ii_table")[keys + ["median_ii", "std_ii"]]

    srf = load(phase, "srf_table")[keys + ["median_srf"]]

    df = ev
    for other in [cr, dtw, cc, ii, srf]:
        if not other.empty:
            df = df.merge(other, on=keys, how="left")
    return df


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt(val, decimals=3):
    if pd.isna(val):
        return "—"
    return f"{val:.{decimals}f}"

def fmt_pct(val):
    if pd.isna(val):
        return "—"
    return f"{val*100:.1f}%"

def fmt_pm(val, std, decimals=3):
    if pd.isna(val):
        return "—"
    if pd.isna(std):
        return fmt(val, decimals)
    return f"{val:.{decimals}f} ± {std:.{decimals}f}"

TOPO_LABEL = {"fc": "FC", "chain": "Chain", "circle": "Circle", "star": "Star"}
DS_LABEL = {
    "synthetic": "Synthetic",
    "moral_stories": "Moral Stories",
    "harmbench_standard": "HarmBench-Std",
    "harmbench_contextual": "HarmBench-Ctx",
    "harmbench_copyright": "HarmBench-Cpy",
}
MODEL_LABEL = {
    "qwen-7b-instruct": "Qwen-7B-Instruct",
    "qwen-0.5b-instruct": "Qwen-0.5B-Instruct",
    "qwen-7b-base": "Qwen-7B-Base",
    "llama-8b-instruct": "Llama-3.1-8B-Instruct",
}
STRAT_LABEL = {
    "rigid:rigid": "Rigid:Rigid",
    "rigid:lenient": "Rigid:Lenient",
    "lenient:rigid": "Lenient:Rigid",
    "lenient:lenient": "Lenient:Lenient",
}


def save_csv(df: pd.DataFrame, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"  Saved {path}  ({len(df)} rows)")


def save_md(df: pd.DataFrame, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"  Saved {path}")


def save_table(df: pd.DataFrame, stem: str, out_dir: str):
    save_csv(df, stem + ".csv", out_dir)
    save_md(df, stem + ".md", out_dir)


# ── Table builders ────────────────────────────────────────────────────────────

def table_t1_core_contagion(out_dir: str):
    """T1: Core contagion — Qwen-7B, FC, ratio=0.2, all datasets."""
    print("Building T1: core contagion ...")
    df = merge_metrics("primary_em")
    if df.empty:
        return

    sub = df[
        (df["model_key"] == "qwen-7b-instruct") &
        (df["topology"] == "fc") &
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0) &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()

    rows = []
    for _, r in sub.iterrows():
        rows.append({
            "Dataset": DS_LABEL.get(r["dataset"], r["dataset"]),
            "N agents": int(r["n_agents"]),
            "EV Shift (shadow)": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC (pub−priv)": fmt(r["mean_delta_cc"]),
            "Frac Asch": fmt_pct(r["frac_asch"]),
            "Frac Moscovici": fmt_pct(r["frac_moscovici"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T1_core_contagion", out_dir)


def table_t2_topology_mechanism(out_dir: str):
    """T2: Topology × mechanism — Qwen-7B, synthetic + moral_stories, ratio=0.2."""
    print("Building T2: topology × mechanism ...")
    df = merge_metrics("primary_em")
    if df.empty:
        return

    sub = df[
        (df["model_key"] == "qwen-7b-instruct") &
        (df["dataset"].isin(["synthetic", "moral_stories"])) &
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0) &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()

    rows = []
    for _, r in sub.sort_values(["dataset", "topology"]).iterrows():
        rows.append({
            "Dataset": DS_LABEL.get(r["dataset"], r["dataset"]),
            "Topology": TOPO_LABEL.get(r["topology"], r["topology"]),
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Frac Asch": fmt_pct(r["frac_asch"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T2_topology_mechanism", out_dir)


def table_t3_dose_response(out_dir: str):
    """T3: Dose-response — Qwen-7B, synthetic, all topologies × all ratios."""
    print("Building T3: dose-response ...")
    df = merge_metrics("primary_em")
    if df.empty:
        return

    sub = df[
        (df["model_key"] == "qwen-7b-instruct") &
        (df["dataset"] == "synthetic") &
        (df["position_config"] == 0) &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()

    rows = []
    for _, r in sub.sort_values(["topology", "minority_ratio"]).iterrows():
        rows.append({
            "Topology": TOPO_LABEL.get(r["topology"], r["topology"]),
            "Minority %": f"{int(r['minority_ratio']*100)}%",
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T3_dose_response", out_dir)


def table_t4_star_position(out_dir: str):
    """T4: Star hub vs leaf — Qwen-7B, synthetic, all ratios."""
    print("Building T4: star position ...")
    df = merge_metrics("primary_em")
    if df.empty:
        return

    sub = df[
        (df["model_key"] == "qwen-7b-instruct") &
        (df["dataset"] == "synthetic") &
        (df["topology"] == "star") &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()

    pos_label = {0: "Min. as Leaf", 1: "Min. as Hub"}
    rows = []
    for _, r in sub.sort_values(["minority_ratio", "position_config"]).iterrows():
        rows.append({
            "Position": pos_label.get(r["position_config"], str(r["position_config"])),
            "Minority %": f"{int(r['minority_ratio']*100)}%",
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T4_star_position", out_dir)


def table_t5_model_comparison(out_dir: str):
    """T5: Model comparison — synthetic, FC, ratio=0.2, all models."""
    print("Building T5: model comparison ...")
    df = merge_metrics("primary_em")
    if df.empty:
        return

    # Also load prompt-induced ablation for same conditions
    df_pi = merge_metrics("primary")

    sub_mi = df[
        (df["dataset"] == "synthetic") &
        (df["topology"] == "fc") &
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0) &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()
    sub_mi["condition"] = "model-induced"

    sub_pi = df_pi[
        (df_pi["dataset"] == "synthetic") &
        (df_pi["topology"] == "fc") &
        (df_pi["minority_ratio"] == 0.2) &
        (df_pi["position_config"] == 0) &
        (df_pi["prompt_strategy"] == "rigid:rigid")
    ].copy() if not df_pi.empty else pd.DataFrame()

    if not sub_pi.empty:
        sub_pi["condition"] = "prompt-induced"
        combined = pd.concat([sub_mi, sub_pi], ignore_index=True)
    else:
        combined = sub_mi

    model_order = ["qwen-0.5b-instruct", "qwen-7b-instruct", "qwen-7b-base", "llama-8b-instruct"]
    rows = []
    for _, r in combined.sort_values(["condition", "model_key"]).iterrows():
        if r["model_key"] not in model_order:
            continue
        rows.append({
            "Model": MODEL_LABEL.get(r["model_key"], r["model_key"]),
            "Condition": r["condition"],
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T5_model_comparison", out_dir)


def table_t6_prompt_sensitivity(out_dir: str):
    """T6: Prompt sensitivity 2×2 — Qwen-7B, FC + Star, ratio=0.2."""
    print("Building T6: prompt sensitivity ...")
    df = merge_metrics("prompt_sensitivity")
    if df.empty:
        return

    sub = df[
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0)
    ].copy()

    rows = []
    for _, r in sub.sort_values(["dataset", "topology", "prompt_strategy"]).iterrows():
        rows.append({
            "Dataset": DS_LABEL.get(r["dataset"], r["dataset"]),
            "Topology": TOPO_LABEL.get(r["topology"], r["topology"]),
            "Prompt Strategy": STRAT_LABEL.get(r["prompt_strategy"], r["prompt_strategy"]),
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T6_prompt_sensitivity", out_dir)


def table_t7_condition_equivalence(out_dir: str):
    """T7: Model-induced vs prompt-induced — Qwen-7B + Llama, synthetic, all topologies ratio=0.2."""
    print("Building T7: model-induced vs prompt-induced ...")
    df_mi = merge_metrics("primary_em")
    df_pi = merge_metrics("primary")

    if df_mi.empty or df_pi.empty:
        print("  [WARN] Missing primary or primary_em tables.")
        return

    filt = dict(dataset="synthetic", minority_ratio=0.2, position_config=0,
                prompt_strategy="rigid:rigid")

    def apply_filter(df, extra=None):
        mask = pd.Series([True] * len(df))
        for k, v in filt.items():
            mask &= (df[k] == v)
        if extra:
            for k, v in extra.items():
                mask &= (df[k] == v)
        return df[mask].copy()

    mi = apply_filter(df_mi)
    mi["condition"] = "model-induced"
    pi = apply_filter(df_pi)
    pi["condition"] = "prompt-induced"
    combined = pd.concat([mi, pi], ignore_index=True)

    rows = []
    for _, r in combined.sort_values(["model_key", "condition", "topology"]).iterrows():
        rows.append({
            "Model": MODEL_LABEL.get(r["model_key"], r["model_key"]),
            "Condition": r["condition"],
            "Topology": TOPO_LABEL.get(r["topology"], r["topology"]),
            "N agents": int(r["n_agents"]),
            "EV Shift": fmt(r["logprob_ev_shift"]),
            "EV CR (≥0.5)": fmt_pct(r["ev_cr_0.5"]),
            "DTW ρ": fmt(r["dtw_rho"]),
            "ΔCC": fmt(r["mean_delta_cc"]),
            "Median II": fmt(r["median_ii"]),
            "Median SRF": fmt(r["median_srf"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T7_condition_equivalence", out_dir)


def table_t8_seed_reliability(out_dir: str):
    """T8: Seed reliability — Qwen-7B, synthetic, all topologies ratio=0.2, seeds 42/123/456."""
    print("Building T8: seed reliability ...")

    # Load raw combined table and re-aggregate by seed
    # The primary_em logprob_ev_table doesn't split by seed — reload from raw
    # Use the logprob_ev_table grouped across seeds (already pooled).
    # For seed-level breakdown, recompute from raw trials.
    import glob, json

    pattern = os.path.join(OUTPUTS_DIR, "primary_em", "synthetic", "qwen-7b-instruct", "results.jsonl")
    if not os.path.exists(pattern):
        print(f"  [WARN] Missing {pattern}")
        return

    rows_raw = []
    with open(pattern) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("topology") not in ["fc", "chain", "circle", "star"]:
                continue
            if t.get("minority_ratio") != 0.2:
                continue
            if t.get("position_config") != 0:
                continue

            seed = t.get("seed")
            topo = t.get("topology")

            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                bp = a.get("baseline_probs") or {}
                sp = a.get("shadow_probs") or {}
                if not bp or not sp:
                    continue

                def ev(probs):
                    return sum(int(k) * v for k, v in probs.items())

                b_ev = ev(bp)
                s_ev = ev(sp)
                rows_raw.append({
                    "seed": seed,
                    "topology": topo,
                    "shift": s_ev - b_ev,
                })

    if not rows_raw:
        print("  [WARN] No data found for seed reliability.")
        return

    raw = pd.DataFrame(rows_raw)
    agg = raw.groupby(["seed", "topology"]).agg(
        n_agents=("shift", "count"),
        mean_shift=("shift", "mean"),
        std_shift=("shift", "std"),
    ).reset_index()

    # Also compute across-seed mean±std per topology
    across = agg.groupby("topology").agg(
        mean_shift_mean=("mean_shift", "mean"),
        mean_shift_std=("mean_shift", "std"),
    ).reset_index()

    rows = []
    for topo in ["fc", "chain", "circle", "star"]:
        for seed in sorted(agg["seed"].unique()):
            r = agg[(agg["topology"] == topo) & (agg["seed"] == seed)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "Topology": TOPO_LABEL.get(topo, topo),
                "Seed": int(r["seed"]),
                "N agents": int(r["n_agents"]),
                "Mean EV Shift": fmt(r["mean_shift"]),
                "Std EV Shift": fmt(r["std_shift"]),
            })
        # Across-seed summary
        ac = across[across["topology"] == topo]
        if not ac.empty:
            ac = ac.iloc[0]
            rows.append({
                "Topology": TOPO_LABEL.get(topo, topo),
                "Seed": "ALL (mean±SD)",
                "N agents": "—",
                "Mean EV Shift": fmt(ac["mean_shift_mean"]),
                "Std EV Shift": fmt(ac["mean_shift_std"]),
            })

    out = pd.DataFrame(rows)
    save_table(out, "T8_seed_reliability", out_dir)


# ── Semantic table ────────────────────────────────────────────────────────────

def table_t9_semantic(out_dir: str):
    """T9: Semantic drift + mirroring — Qwen-7B, all datasets, all topologies, ratio=0.2."""
    print("Building T9: semantic ...")
    sem = load("primary_em", "semantic_table")
    if sem.empty:
        return

    sub = sem[
        (sem["model_key"] == "qwen-7b-instruct") &
        (sem["minority_ratio"] == 0.2) &
        (sem["position_config"] == 0) &
        (sem["prompt_strategy"] == "rigid:rigid")
    ].copy()

    rows = []
    for _, r in sub.sort_values(["dataset", "topology"]).iterrows():
        rows.append({
            "Dataset": DS_LABEL.get(r["dataset"], r["dataset"]),
            "Topology": TOPO_LABEL.get(r["topology"], r["topology"]),
            "N agents": int(r["n"]),
            "Semantic Drift": fmt(r["mean_semantic_drift"]),
            "Language Mirroring": fmt(r["mean_mirroring"]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "T9_semantic", out_dir)


def table_t6_2_prompt_2x2(out_dir: str):
    """T6_2: 2×2 matrix of EV Shift by aligned × misaligned prompt rigidity.
    Rows = aligned prompt (Rigid / Lenient), Cols = misaligned prompt (Rigid / Lenient).
    One sub-table per dataset × topology combination (synthetic FC as the headline).
    """
    print("Building T6_2: prompt 2×2 matrix ...")
    df = merge_metrics("prompt_sensitivity")
    if df.empty:
        return

    sub = df[
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0)
    ].copy()

    # Parse prompt_strategy "aligned:misaligned" into two axes
    sub["aligned_prompt"] = sub["prompt_strategy"].str.split(":").str[0].str.capitalize()
    sub["misaligned_prompt"] = sub["prompt_strategy"].str.split(":").str[1].str.capitalize()

    blocks = []
    for (dataset, topology), group in sub.groupby(["dataset", "topology"]):
        pivot = group.pivot(index="aligned_prompt", columns="misaligned_prompt",
                            values="logprob_ev_shift")
        pivot = pivot.reindex(index=["Rigid", "Lenient"], columns=["Rigid", "Lenient"])
        pivot.index.name = f"{DS_LABEL.get(dataset, dataset)} / {TOPO_LABEL.get(topology, topology)}"
        pivot.columns.name = "Misaligned →"
        pivot = pivot.round(3)
        blocks.append((dataset, topology, pivot))

    # Write combined markdown manually so each block is clearly labelled
    md_path = os.path.join(out_dir, "T6_2_prompt_2x2.md")
    with open(md_path, "w") as f:
        f.write("# T6_2 — Prompt Sensitivity 2×2: Mean Shadow EV Shift\n\n")
        f.write("Rows = Aligned prompt rigidity. Columns = Misaligned prompt rigidity.\n\n")
        for dataset, topology, pivot in blocks:
            f.write(f"### {DS_LABEL.get(dataset, dataset)} — {TOPO_LABEL.get(topology, topology)}\n\n")
            f.write("| Aligned ↓ / Misaligned → | Rigid | Lenient |\n")
            f.write("|:--------------------------|------:|--------:|\n")
            for aligned_val in ["Rigid", "Lenient"]:
                rigid_val = pivot.loc[aligned_val, "Rigid"] if "Rigid" in pivot.columns else float("nan")
                lenient_val = pivot.loc[aligned_val, "Lenient"] if "Lenient" in pivot.columns else float("nan")
                f.write(f"| **{aligned_val}** | {fmt(rigid_val)} | {fmt(lenient_val)} |\n")
            f.write("\n")
    print(f"  Saved {md_path}")

    # Also save a flat CSV with all combinations
    rows = []
    for dataset, topology, pivot in blocks:
        for aligned_val in ["Rigid", "Lenient"]:
            for misaligned_val in ["Rigid", "Lenient"]:
                val = pivot.loc[aligned_val, misaligned_val] if (aligned_val in pivot.index and misaligned_val in pivot.columns) else float("nan")
                rows.append({
                    "Dataset": DS_LABEL.get(dataset, dataset),
                    "Topology": TOPO_LABEL.get(topology, topology),
                    "Aligned Prompt": aligned_val,
                    "Misaligned Prompt": misaligned_val,
                    "EV Shift": fmt(val),
                })
    save_csv(pd.DataFrame(rows), "T6_2_prompt_2x2.csv", out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate paper summary tables")
    p.add_argument("--output-dir", default=os.path.join(OUTPUTS_DIR, "paper_tables"),
                   help="Directory to write output tables (default: outputs/paper_tables/)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing tables to {args.output_dir}\n")

    table_t1_core_contagion(args.output_dir)
    table_t2_topology_mechanism(args.output_dir)
    table_t3_dose_response(args.output_dir)
    table_t4_star_position(args.output_dir)
    table_t5_model_comparison(args.output_dir)
    table_t6_prompt_sensitivity(args.output_dir)
    table_t6_2_prompt_2x2(args.output_dir)
    table_t7_condition_equivalence(args.output_dir)
    table_t8_seed_reliability(args.output_dir)
    table_t9_semantic(args.output_dir)

    print(f"\nDone. {len(os.listdir(args.output_dir))} files in {args.output_dir}")


if __name__ == "__main__":
    main()
