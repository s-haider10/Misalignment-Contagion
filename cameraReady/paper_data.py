"""Compute the aggregates behind the camera-ready figures, and cache them.

Every number here is produced by the *original* code paths:

  Fig1  scripts/plot_trajectories_with_significance.py  (extract_ev,
        significance_stars, and its exact `sub` filter)
  Fig2  plots.py :: fig1_asch_moscovici
  Fig5  plots.py :: fig5_generalization
  Fig8  plots.py :: fig8_prompt_rigidity

Nothing is recomputed a second way and no filter, estimator, or error-bar
definition is altered — this module only lifts the aggregation out of the
drawing code so the figures can be restyled without reloading 1.2 GB of
JSONL on every iteration.

Usage:
    python paper_data.py            # build the cache if missing
    python paper_data.py --refresh  # force recompute
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from misalignment_contagion.config import OUTPUTS_DIR, N_ROUNDS  # noqa: E402
from misalignment_contagion.analyze import (  # noqa: E402
    load_trials, trials_to_dataframe,
)

CACHE = Path(__file__).resolve().parent / ".fig_cache" / "paper_data.json"


def _load_sig_module():
    """Import scripts/plot_trajectories_with_significance.py by path so Fig1
    uses that file's own extract_ev / significance_stars, not a copy."""
    path = REPO / "scripts" / "plot_trajectories_with_significance.py"
    spec = importlib.util.spec_from_file_location("_sigmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIG = _load_sig_module()
TOPO_ORDER = SIG.TOPO_ORDER
DATASET_ORDER = SIG.DATASET_ORDER
DATASET_LABELS = SIG.DATASET_LABELS
STAGE_LABELS = SIG.STAGE_LABELS
TOPO_COLORS = SIG.TOPO_COLORS
TOPO_LABELS = SIG.TOPO_LABELS


def _load_df(experiment: str):
    pattern = str(OUTPUTS_DIR / experiment / "**" / "results.jsonl")
    print(f"  loading {pattern} ...")
    trials = load_trials(pattern)
    df = trials_to_dataframe(trials)
    print(f"  {len(trials)} trials -> {len(df)} aligned agent rows")
    return df


# ── Fig 1 ─────────────────────────────────────────────────────────────────────
def _sig_subset(df):
    """The `sub` filter main() applies before calling make_figure()."""
    return df[
        (df["model_key"] == "qwen-7b-instruct") &
        (df["model_condition"] == "model_induced") &
        (df["minority_ratio"] == 0.2) &
        (df["position_config"] == 0) &
        (df["prompt_strategy"] == "rigid:rigid")
    ].copy()


def _trajectory_panels(df, extract_fn):
    """Mean ± 1 SD trajectory per dataset × topology, plus the R4 stars —
    the aggregation make_figure() performs, for whichever extractor is given."""
    sub = _sig_subset(df)

    out = {}
    for dataset in DATASET_ORDER:
        ds_sub = sub[sub["dataset"] == dataset]
        panel = {}
        for topo in TOPO_ORDER:
            topo_sub = ds_sub[ds_sub["topology"] == topo]
            trajs = []
            for _, row in topo_sub.iterrows():
                t = extract_fn(row)
                if t is not None:
                    trajs.append(t)
            if not trajs:
                continue
            arr = np.array(trajs)
            panel[topo] = {
                "means": arr.mean(axis=0).tolist(),
                "sds": arr.std(axis=0, ddof=1).tolist(),
                "stars": SIG.significance_stars(arr),
                "n": int(arr.shape[0]),
            }
        out[dataset] = panel
    return out


def compute_fig1(df):
    """Mean ± 1 SD EV trajectory per dataset × topology, plus R4 stars."""
    return _trajectory_panels(df, SIG.extract_ev)


# ── Fig 2 ─────────────────────────────────────────────────────────────────────
def compute_fig2(df, ratio: float = 0.2):
    """Public vs private shift point clouds + Asch/Moscovici shares per
    topology, exactly as fig1_asch_moscovici() derives them."""
    out = {}
    for topo in TOPO_ORDER:
        sub = df[(df["topology"] == topo) & (df["minority_ratio"] == ratio) &
                 (df["position_config"] == 0)]
        matched = sub[["baseline_ev", "final_ev", "shadow_ev"]].dropna()
        if len(matched) == 0:
            out[topo] = None
            continue
        public_shift = matched["final_ev"].values - matched["baseline_ev"].values
        private_shift = matched["shadow_ev"].values - matched["baseline_ev"].values
        n_total = len(public_shift)
        n_above = int(np.sum(private_shift > public_shift))
        n_below = n_total - n_above
        out[topo] = {
            "public": public_shift.tolist(),
            "private": private_shift.tolist(),
            "asch": n_below / n_total,
            "moscovici": n_above / n_total,
            "n": n_total,
        }
    return out


# ── Fig 5 ─────────────────────────────────────────────────────────────────────
def _shadow_shift_bar(sub):
    """Mean shadow-EV shift with a 1.96·SE half-width — the estimator used by
    fig5_generalization() and fig8_prompt_rigidity()."""
    matched = sub[["baseline_ev", "shadow_ev"]].dropna()
    if len(matched) == 0:
        return None
    shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
    m = float(np.mean(shifts))
    se = float(np.std(shifts, ddof=1)) / math.sqrt(len(shifts))
    return {"mean": m, "ci": 1.96 * se, "n": int(len(shifts))}


def compute_fig5(df):
    # fig5_generalization() iterates sorted() order; keep it so the bar order
    # of the original figure is preserved.
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model_key"].unique())

    by_dataset = []
    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        by_dataset.append({"label": ds, **(_shadow_shift_bar(sub) or {})})

    by_model = []
    for model in models:
        sub = df[(df["model_key"] == model) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        by_model.append({"label": model, **(_shadow_shift_bar(sub) or {})})

    return {"by_dataset": by_dataset, "by_model": by_model}


# ── Fig 8 ─────────────────────────────────────────────────────────────────────
def compute_fig8(df):
    strategies = sorted(df["prompt_strategy"].unique())

    shift = []
    for strat in strategies:
        sub = df[(df["prompt_strategy"] == strat) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        bar = _shadow_shift_bar(sub)
        if bar:
            shift.append({"strategy": strat, **bar})

    ii = []
    for strat in strategies:
        sub = df[(df["prompt_strategy"] == strat) & (df["topology"] == "fc") &
                 (df["minority_ratio"] == 0.2)]
        vals = sub["internalization_index"].dropna().values
        if len(vals) > 0:
            ii.append({"strategy": strat,
                       "median": float(np.median(vals)),
                       "n": int(len(vals))})

    return {"shift": shift, "ii": ii}


# ── Fig 3 ─────────────────────────────────────────────────────────────────────
def compute_fig3(df):
    """Entropy trajectories — the same make_figure() path as Fig1, but through
    the sig script's extract_entropy instead of extract_ev."""
    return _trajectory_panels(df, SIG.extract_entropy)


# ── Fig 4 ─────────────────────────────────────────────────────────────────────
def compute_fig4(df, ratios=(0.1, 0.2, 0.3)):
    """Median II and SRF per topology x ratio, as fig2_ii_heatmap() derives
    them (median over aligned agents, position_config 0)."""
    out = {}
    for col in ("internalization_index", "shadow_reversion_fraction"):
        matrix, counts = [], []
        for topo in TOPO_ORDER:
            row_m, row_c = [], []
            for ratio in ratios:
                sub = df[(df["topology"] == topo) &
                         (df["minority_ratio"] == ratio) &
                         (df["position_config"] == 0)]
                vals = sub[col].dropna().values
                row_m.append(float(np.median(vals)) if len(vals) else None)
                row_c.append(int(len(vals)))
            matrix.append(row_m)
            counts.append(row_c)
        out[col] = {"matrix": matrix, "counts": counts,
                    "ratios": list(ratios), "topologies": list(TOPO_ORDER)}
    return out


# ── Fig 6 ─────────────────────────────────────────────────────────────────────
def compute_fig6(df, ratios=(0.1, 0.2, 0.3)):
    """Star topology only: mean shadow shift and median II by whether the
    minority sits on a leaf (position_config 0) or the hub (1)."""
    star = df[df["topology"] == "star"]
    shift, ii = {}, {}
    for pos in (0, 1):
        s_row, i_row, n_row = [], [], []
        for ratio in ratios:
            sub = star[(star["minority_ratio"] == ratio) &
                       (star["position_config"] == pos)]
            matched = sub[["baseline_ev", "shadow_ev"]].dropna()
            if len(matched):
                shifts = matched["shadow_ev"].values - matched["baseline_ev"].values
                s_row.append(float(np.mean(shifts)))
                n_row.append(int(len(shifts)))
            else:
                s_row.append(0.0)
                n_row.append(0)
            vals = sub["internalization_index"].dropna().values
            i_row.append(float(np.median(vals)) if len(vals) else 0.0)
        shift[pos] = {"values": s_row, "n": n_row}
        ii[pos] = {"values": i_row}
    return {"shift": shift, "ii": ii, "ratios": list(ratios)}


# ── Fig 7 ─────────────────────────────────────────────────────────────────────
def compute_fig7(df):
    """Prompt- vs model-induced shadow shift, paired per
    (scenario_id, topology, minority_ratio) — the grouping fig7_condition_
    equivalence() uses — plus its Pearson r."""
    from scipy import stats

    pi = df[df["model_condition"] == "prompt_induced"]
    mi = df[df["model_condition"] == "model_induced"]
    if len(pi) == 0 or len(mi) == 0:
        return None

    group_cols = ["scenario_id", "topology", "minority_ratio"]
    pi_agg = pi.groupby(group_cols).apply(
        lambda g: (g["shadow_ev"].dropna() - g["baseline_ev"].dropna()).mean()
    ).reset_index(name="pi_shift")
    mi_agg = mi.groupby(group_cols).apply(
        lambda g: (g["shadow_ev"].dropna() - g["baseline_ev"].dropna()).mean()
    ).reset_index(name="mi_shift")
    merged = pd.merge(pi_agg, mi_agg, on=group_cols, how="inner").dropna(
        subset=["pi_shift", "mi_shift"])
    if len(merged) == 0:
        return None

    r_val, p_val = stats.pearsonr(merged["pi_shift"], merged["mi_shift"])
    return {"pi": merged["pi_shift"].tolist(),
            "mi": merged["mi_shift"].tolist(),
            "topology": merged["topology"].tolist(),
            "r": float(r_val), "p": float(p_val), "n": int(len(merged))}


# ── Fig 9 ─────────────────────────────────────────────────────────────────────
def compute_fig9():
    """Semantic mirroring and drift, from the pre-computed table that
    fig_semantic() reads, under the same three filters."""
    path = OUTPUTS_DIR / "primary_em" / "tables" / "semantic_table.csv"
    if not path.exists():
        print(f"  (missing {path} — skipping fig9)")
        return None
    sem = pd.read_csv(path)
    sem = sem[(sem["model_key"] == "qwen-7b-instruct") &
              (sem["prompt_strategy"] == "rigid:rigid") &
              (sem["position_config"] == 0)]
    if sem.empty:
        return None

    datasets = sorted(sem["dataset"].unique())
    out = {"datasets": datasets}
    for metric in ("mean_mirroring", "mean_semantic_drift"):
        series = {}
        for topo in TOPO_ORDER:
            vals = []
            for ds in datasets:
                sub = sem[(sem["dataset"] == ds) & (sem["topology"] == topo)]
                # fig_semantic averages across minority ratios.
                vals.append(float(sub[metric].mean()) if len(sub) else None)
            series[topo] = vals
        out[metric] = series
    return out


# ── driver ────────────────────────────────────────────────────────────────────
def build_cache():
    print("primary_em:")
    df_em = _load_df("primary_em")
    print("prompt_sensitivity:")
    df_ps = _load_df("prompt_sensitivity")
    # Fig7 pairs the two induction conditions. prompt_induced exists only in
    # outputs/primary, model_induced only in outputs/primary_em; outputs/
    # combined holds no results.jsonl of its own, so the original figure came
    # from a glob spanning both. Concatenating the trial lists reproduces that.
    print("primary (prompt-induced):")
    df_pi = _load_df("primary")
    df_comb = pd.concat([df_pi, df_em], ignore_index=True)
    print(f"  combined -> {len(df_comb)} rows, "
          f"conditions {sorted(df_comb['model_condition'].unique())}")

    data = {
        "n_rounds": N_ROUNDS,
        "stage_labels": STAGE_LABELS,
        "fig1": compute_fig1(df_em),
        "fig2": compute_fig2(df_em),
        "fig3": compute_fig3(df_em),
        "fig4": compute_fig4(df_em),
        "fig5": compute_fig5(df_em),
        "fig6": compute_fig6(df_em),
        "fig7": compute_fig7(df_comb),
        "fig8": compute_fig8(df_ps),
        "fig9": compute_fig9(),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(data, f)
    print(f"wrote {CACHE}")
    return data


def load_cache(refresh: bool = False):
    if refresh or not CACHE.exists():
        return build_cache()
    with open(CACHE) as f:
        return json.load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    load_cache(refresh=args.refresh)
