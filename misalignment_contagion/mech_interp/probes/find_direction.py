"""
Direction-finding for the internalization-propensity signal.

Goal: produce a single 3584-dim vector at layer 26 that points along
"high-II vs low-II trials" — the direction the v3 probe was implicitly using.

Method (difference-of-means, the standard mech-interp recipe):
  1. Compute trial-level II = mean(II across the 8 aligned agents in trial).
  2. Split trials into high-II (>= median) and low-II (< median) halves.
  3. For each layer of interest, take all aligned-agent activations from
     high-II trials, average them; same for low-II. Subtract.
  4. The resulting vector is the candidate "internalization direction."
  5. Sanity check: project all aligned-agent activations onto the direction,
     score against y_internalized via AUC. Should reproduce the probe's AUC
     within a few hundredths if the direction captures what the probe was
     reading.

We do this at multiple layers around the v3 peak (24-27) to verify the
peak holds, and at a few control layers (10, 20) to confirm the signal
really is late-layer.

Output:
  - direction_layer{L}_round{R}.npy           (3584-dim fp32 vector per L)
  - direction_summary.parquet                 (sanity check AUCs per layer)
  - direction_diagnostics.png                 (figure)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIG
# =============================================================================

PARQUET_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/activations_pilot/"
    "pilot_qwen7b_moralstories_fc_20pct_400trials.parquet"
)
OUTCOMES_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/probe_results_v3/"
    "probe_v3_outcomes.parquet"
)
OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/direction_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layers to analyze: peak (26) and surrounding (25, 27), plus controls
LAYERS_OF_INTEREST = [10, 20, 24, 25, 26, 27]
PEAK_LAYER = 26
PEAK_STAGE = "round_4"
II_THRESHOLD = 0.7


# =============================================================================
# Step 1 — trial-level II split
# =============================================================================

def trial_level_split(outcomes: pd.DataFrame) -> pd.DataFrame:
    """For each trial, compute mean II across its aligned agents,
    then assign to high-II or low-II group based on the median split.
    Returns a per-trial DataFrame with columns
        trial_id, mean_ii, n_internalized, group  ('high' / 'low').
    """
    valid = outcomes.dropna(subset=["ii"]).copy()
    # Drop trials with extreme II outliers (>10) before computing mean —
    # II can blow up when JSD(final, base) is tiny. These outliers should
    # not dominate the mean.
    valid = valid[valid["ii"].abs() < 10]

    per_trial = valid.groupby("trial_id").agg(
        mean_ii=("ii", "mean"),
        n_internalized=("y_internalized", "sum"),
        n_agents=("ii", "size"),
    ).reset_index()

    median_ii = per_trial["mean_ii"].median()
    per_trial["group"] = np.where(per_trial["mean_ii"] >= median_ii, "high", "low")

    high_n = (per_trial["group"] == "high").sum()
    low_n = (per_trial["group"] == "low").sum()
    print(f"Trial-level split (median mean_ii = {median_ii:.3f}):")
    print(f"  high-II trials: {high_n}")
    print(f"  low-II trials:  {low_n}")
    print(f"  high-II mean_ii: {per_trial[per_trial.group=='high']['mean_ii'].mean():.3f}")
    print(f"  low-II  mean_ii: {per_trial[per_trial.group=='low']['mean_ii'].mean():.3f}")
    return per_trial


# =============================================================================
# Step 2 — load activations and compute direction
# =============================================================================

def load_activations(parquet_path: Path, stage: str, layer: int) -> pd.DataFrame:
    """Filter parquet to one stage + layer, decode bytes to fp32."""
    df = pd.read_parquet(
        parquet_path,
        filters=[("stage", "==", stage), ("layer", "==", layer)],
    )
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "activation"]]


def compute_direction(
    activations: pd.DataFrame,
    per_trial: pd.DataFrame,
) -> np.ndarray:
    """Diff-of-means: mean(high-II activations) - mean(low-II activations).
    Returns a (3584,) fp32 vector.
    """
    merged = activations.merge(per_trial[["trial_id", "group"]], on="trial_id")
    high = np.stack(merged[merged.group == "high"]["activation"].values)
    low = np.stack(merged[merged.group == "low"]["activation"].values)
    direction = high.mean(axis=0) - low.mean(axis=0)
    return direction.astype(np.float32)


# =============================================================================
# Step 3 — sanity check: AUC of projection onto direction
# =============================================================================

def project_and_score(
    activations: pd.DataFrame,
    outcomes: pd.DataFrame,
    direction: np.ndarray,
) -> dict:
    """Project each agent's activation onto the direction, then compute
    AUC against y_internalized. Should reproduce the v3 probe's AUC
    (within a few hundredths) if the direction captures the right signal.
    """
    merged = activations.merge(
        outcomes[["trial_id", "agent_id", "y_internalized", "ii"]],
        on=["trial_id", "agent_id"],
    )
    X = np.stack(merged["activation"].values)
    # Scalar projection: <activation, direction> / ||direction||
    norm = np.linalg.norm(direction)
    proj = X @ direction / norm
    y = merged["y_internalized"].values.astype(int)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan, "norm": float(norm)}
    auc = roc_auc_score(y, proj)
    return {
        "n": len(y),
        "auc": float(auc),
        "norm": float(norm),
        "proj_mean_high_ii": float(proj[y == 1].mean()),
        "proj_mean_low_ii": float(proj[y == 0].mean()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    print(f"Loaded {len(outcomes)} outcome rows")

    per_trial = trial_level_split(outcomes)
    per_trial.to_parquet(OUT_DIR / "trial_level_ii_split.parquet")

    rows = []
    print(f"\nComputing directions at stage={PEAK_STAGE}")
    for layer in LAYERS_OF_INTEREST:
        acts = load_activations(PARQUET_PATH, PEAK_STAGE, layer)
        direction = compute_direction(acts, per_trial)
        # Save vector
        np.save(
            OUT_DIR / f"direction_layer{layer:02d}_{PEAK_STAGE}.npy",
            direction,
        )
        # Sanity check
        score = project_and_score(acts, outcomes, direction)
        score["layer"] = layer
        score["stage"] = PEAK_STAGE
        rows.append(score)
        print(
            f"  layer {layer:2d}: ||d|| = {score['norm']:.2f}, "
            f"projection AUC vs y_internalized = {score['auc']:.3f} "
            f"(n={score['n']})"
        )

    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_DIR / "direction_summary.parquet")

    plot_diagnostics(summary, OUT_DIR / "direction_diagnostics.png")
    print(f"\nSaved figure -> {OUT_DIR / 'direction_diagnostics.png'}")

    # Print verdict
    peak_auc = summary[summary.layer == PEAK_LAYER]["auc"].iloc[0]
    print("\n" + "=" * 64)
    print("DIRECTION QUALITY CHECK")
    print("=" * 64)
    print(f"v3 probe AUC at layer {PEAK_LAYER}, {PEAK_STAGE}:    ~0.73")
    print(f"Direction projection AUC at layer {PEAK_LAYER}: {peak_auc:.3f}")
    if peak_auc >= 0.65:
        print("\n→ Direction reproduces the probe's signal. Ready for steering.")
    elif peak_auc >= 0.55:
        print("\n→ Direction captures some signal but weaker than the probe.")
        print("   Probe was using non-linear or higher-order features.")
        print("   Direction is still usable for steering but expect smaller effect.")
    else:
        print("\n→ Direction does NOT reproduce the probe's signal.")
        print("   Something is wrong with the diff-of-means split or the data.")


def plot_diagnostics(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    s = summary.sort_values("layer")

    # AUC across layers
    axes[0].plot(s["layer"], s["auc"], marker="o", color="#1f77b4")
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.73, color="red", linestyle=":", linewidth=0.8,
                    label="v3 probe AUC (≈0.73)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("AUC of projection vs y_internalized")
    axes[0].set_ylim(0.4, 0.85)
    axes[0].set_title("Projection-onto-direction AUC by layer")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Direction norm across layers
    axes[1].plot(s["layer"], s["norm"], marker="s", color="#d62728")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("||direction||")
    axes[1].set_title("Direction magnitude by layer")
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Direction-finding diagnostics (stage={PEAK_STAGE}, moral_stories FC 20%)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()