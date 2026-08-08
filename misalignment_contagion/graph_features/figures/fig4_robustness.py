"""Figure 4: Model-robust — ΔR² for mean II prediction across (model × dataset × seed).

Reads the robustness sweep outputs from outputs/graph_features/runs/{dataset}/
matching shard-tag pattern results.robust_*_{model_key}_s{seed}.jsonl.
Falls back to the main run for the Qwen-7B seed=42 cell.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .fig_style import (
    apply, panel_label, style_axes, add_zero_rule,
    DATASET_COLORS, DATASET_LABELS, MODEL_ORDER, MODEL_LABELS,
    MUTED, GRID, RULE,
)
from .data import FIG_DIR, ensure_dir, MANIFEST_FEATURES
from .stats_helpers import delta_r2
from ...metrics import internalization_index, shadow_reversion_fraction

warnings.filterwarnings("ignore")
apply()

OUT_PATH = ensure_dir(FIG_DIR) / "fig4_robustness.png"
OUT_PDF = ensure_dir(FIG_DIR) / "fig4_robustness.pdf"
OUT_CSV = ensure_dir(FIG_DIR) / "fig4_robustness_data.csv"

RUNS = Path("outputs/graph_features/runs")
SUBSET_GIDS = pd.read_csv(
    Path("outputs/graph_features/robustness_subset/robustness_graphs.csv")
)["graph_id"].tolist() if Path(
    "outputs/graph_features/robustness_subset/robustness_graphs.csv"
).exists() else None

CONDITIONS = [
    # (label, model_key, seeds_to_average_across, restrict_to_subset)
    # 151-graph full-manifest coverage now available; no need to restrict.
    ("Qwen-7B\n(seeds 123, 456)",  "qwen-7b-instruct",  [123, 456], False),
    ("Llama-3.1-8B\n(seed 42)",     "llama-8b-instruct", [42],       False),
    # Qwen-14B dropped: OOM on 24GB cards
    # Llama-1B pending: HuggingFace access not yet approved
]
DATASETS = ["synthetic", "moral_stories", "harmbench_standard",
            "harmbench_contextual", "harmbench_copyright"]


def fix_keys(d):
    if d is None:
        return None
    return {int(k): float(v) for k, v in d.items()}


def collect_one(model_key: str, seed: int, dataset: str,
                restrict_to_subset: bool):
    """Return graph-level DataFrame with y_mean_II / y_var_II / y_mean_SRF / y_var_SRF."""
    rows = []
    ds_dir = RUNS / dataset
    if not ds_dir.is_dir():
        return pd.DataFrame()
    # Match both the 20-graph subset shards (results.robust_X_{mk}_s{sd}.jsonl)
    # and the 131-graph extension shards (results.robust_ext_{mk}_s{sd}_s{shard}.jsonl).
    # Together these cover the full 151-graph manifest.
    candidates = list(ds_dir.glob(f"results.robust_*{model_key}_s{seed}*.jsonl"))
    if not candidates:
        return pd.DataFrame()
    for shard in sorted(candidates):
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trial = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trial.get("model_key") != model_key:
                    continue
                if trial.get("seed") != seed:
                    continue
                if trial.get("model_condition") != "model_induced":
                    continue
                if restrict_to_subset and SUBSET_GIDS is not None:
                    if trial["graph_id"] not in SUBSET_GIDS:
                        continue
                for ag in trial["agents"]:
                    if ag.get("role") != "aligned":
                        continue
                    bp = ag.get("baseline_probs")
                    sp_ = ag.get("shadow_probs")
                    rp = ag.get("round_probs")
                    if bp is None or sp_ is None or not rp:
                        continue
                    fin = rp[-1] if isinstance(rp[-1], dict) else None
                    if fin is None:
                        continue
                    bp_i = fix_keys(bp); sp_i = fix_keys(sp_); fin_i = fix_keys(fin)
                    try:
                        base_ev = sum(int(k) * v for k, v in bp.items())
                        fin_ev = sum(int(k) * v for k, v in fin.items())
                        sh_ev = sum(int(k) * v for k, v in sp_.items())
                        ii = internalization_index(bp_i, fin_i, sp_i)
                        srf = shadow_reversion_fraction(base_ev, fin_ev, sh_ev)
                    except Exception:
                        continue
                    rows.append({
                        "graph_id": trial["graph_id"],
                        "II": ii, "SRF": srf,
                        "n_agents": trial["n_agents"],
                        "n_misaligned": trial["n_misaligned"],
                        "minority_ratio": trial["minority_ratio"],
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Clip extreme II / SRF
    for col in ["II", "SRF"]:
        valid = df[col].dropna()
        if len(valid) < 10:
            continue
        q01, q99 = valid.quantile([0.01, 0.99])
        df[col] = df[col].clip(q01, q99)
    # Pool to graph level
    gl = (df.dropna(subset=["II", "SRF"])
            .groupby("graph_id")
            .agg(y_mean_II=("II", "mean"), y_var_II=("II", "var"),
                 y_mean_SRF=("SRF", "mean"), y_var_SRF=("SRF", "var"),
                 minority_ratio=("minority_ratio", "first"),
                 n_agents=("n_agents", "first"),
                 n_misaligned=("n_misaligned", "first"))
            .reset_index())
    manifest = pd.read_csv(MANIFEST_FEATURES)
    keep = [c for c in manifest.columns
            if c not in ("minority_ratio", "n_agents", "n_misaligned")]
    return gl.merge(manifest[keep], on="graph_id", how="left",
                    validate="many_to_one")


def compute_per_seed(model_key, seed, dataset, restrict):
    """Return baseline R² and full R² per outcome (no delta clipping)."""
    gl = collect_one(model_key, seed, dataset, restrict)
    if gl.empty or len(gl) < 10:
        return None
    out = {}
    for tgt in ["y_mean_II", "y_var_II", "y_mean_SRF", "y_var_SRF"]:
        res = delta_r2(gl, tgt, model="ridge")
        out[f"{tgt}__base"] = float(res["r2_baseline"])
        out[f"{tgt}__full"] = float(res["r2_full"])
    out["n_graphs"] = len(gl)
    return out


def build_data():
    """Return DataFrame: rows = (condition, dataset, seed), cols = base/full R²."""
    rows = []
    for label, model_key, seeds, restrict in CONDITIONS:
        for ds in DATASETS:
            for seed in seeds:
                d = compute_per_seed(model_key, seed, ds, restrict)
                if d is None:
                    print(f"  (skip) {label} | {ds} | seed={seed}: no data")
                    continue
                d.update({"condition": label, "model_key": model_key,
                          "dataset": ds, "seed": seed})
                rows.append(d)
                print(f"  {label} | {ds} | seed={seed}: "
                      f"mean II base={d['y_mean_II__base']:+.2f} "
                      f"full={d['y_mean_II__full']:+.2f}, n={d['n_graphs']}")
    return pd.DataFrame(rows)


def panel_main(ax, df):
    if df.empty:
        ax.text(0.5, 0.5, "Robustness data not yet available\n(sweep still running)",
                ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return

    # Per (condition, dataset) aggregate: mean ± SEM across seeds for base and full
    def agg(df, col):
        g = df.groupby(["condition", "dataset"])[col]
        return g.mean().rename("mean").to_frame().join(
            g.apply(lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
              .rename("sem")
        ).reset_index()

    base_sum = agg(df, "y_mean_II__base")
    full_sum = agg(df, "y_mean_II__full")

    conds = [c[0] for c in CONDITIONS]
    n_c, n_d = len(conds), len(DATASETS)
    # 2 outcome bars (base, full) × n_d datasets per condition
    bar_w = 0.85 / (2 * n_d)
    x = np.arange(n_c)

    for j, ds in enumerate(DATASETS):
        for k, (kind, sumdf, hatch, alpha) in enumerate([
            ("baseline", base_sum, "//",  0.55),
            ("+topology", full_sum, None, 1.00),
        ]):
            means, sems = [], []
            for c in conds:
                sub = sumdf[(sumdf["condition"] == c) & (sumdf["dataset"] == ds)]
                means.append(float(sub["mean"].iloc[0]) if not sub.empty else np.nan)
                sems.append(float(sub["sem"].iloc[0]) if not sub.empty else 0.0)
            offset = (j - (n_d - 1) / 2) * (bar_w * 2) + (k - 0.5) * bar_w
            pos = x + offset
            label = (f"{DATASET_LABELS[ds]} ({kind})"
                      if j == 0 and k == 1 else
                      f"{DATASET_LABELS[ds]} ({kind})")
            ax.bar(pos, means, width=bar_w, yerr=sems,
                    color=DATASET_COLORS[ds], alpha=alpha,
                    hatch=hatch,
                    label=label,
                    edgecolor="white", linewidth=0.6,
                    error_kw=dict(ecolor=MUTED, capsize=2.5, lw=0.6),
                    zorder=3)

    # Mark cells that have no data (e.g. Llama harmbench_copyright) so the
    # absence is explicit rather than invisible.
    for i, c in enumerate(conds):
        for j, ds in enumerate(DATASETS):
            sub = full_sum[(full_sum["condition"] == c) & (full_sum["dataset"] == ds)]
            if sub.empty:
                offset = (j - (n_d - 1) / 2) * (bar_w * 2)
                ax.text(i + offset, 0.005, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color=MUTED, style="italic", zorder=4)

    ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel(r"$R^2$ for mean II (Ridge, 5-fold CV)")
    add_zero_rule(ax, axis="y")
    style_axes(ax, grid_axis="y")
    ax.set_ylim(-0.15, 0.70)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2,
              columnspacing=0.8, handletextpad=0.4)
    ax.text(0.02, 0.97,
            "Hatched = controls only.  Solid = + topology.\n"
            "151-graph full manifest; Ridge 5-fold CV.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=MUTED, style="italic")


def build():
    df = build_data()
    if not df.empty:
        df.to_csv(OUT_CSV, index=False)
        print(f"wrote {OUT_CSV.relative_to(OUT_CSV.parents[2])}")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    panel_main(ax, df)
    fig.savefig(OUT_PATH, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH.relative_to(OUT_PATH.parents[2])}")


if __name__ == "__main__":
    build()
