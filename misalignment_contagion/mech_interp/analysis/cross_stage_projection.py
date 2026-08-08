"""
Cross-stage projection.

Question: when does the "internalization-propensity direction" emerge during
deliberation? We have a 3584-dim direction at layer 26 round_4 (computed in
find_direction.py) that achieves AUC 0.726 vs y_internalized.

Now project the same agents' activations from EARLIER stages onto that
SAME direction and score them. The shape of the curve tells us:

  - If round_0 projection AUC is already ~0.6: internalization is partly
    baked in before deliberation begins (scenario priors at layer 26).
  - If only round_4 projection AUC is high: internalization emerges through
    the deliberation process and crystallizes at the end.
  - If projection AUC climbs monotonically across rounds: it builds up
    gradually, suggesting deliberation accumulates the signal.

We use the same direction (computed at round_4) for all stages — this is
the right comparison because we want to know "does this end-state signal
exist earlier?", not "does each stage have its own direction?"

Outputs:
  - cross_stage_projections.parquet
  - cross_stage_curves.png
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
DIRECTION_DIR = Path(
    "/home/haider/Misalignment-Contagion/outputs/direction_results"
)
OUT_DIR = DIRECTION_DIR  # write into same folder

ALL_STAGES = ["baseline", "round_0", "round_1", "round_2", "round_3",
              "round_4", "shadow"]
LAYERS_OF_INTEREST = [10, 20, 26]  # control, mid, peak
DIRECTION_LAYER = 26
DIRECTION_STAGE = "round_4"


# =============================================================================
# Loaders
# =============================================================================

def load_direction(layer: int, stage: str) -> np.ndarray:
    path = DIRECTION_DIR / f"direction_layer{layer:02d}_{stage}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing direction file: {path}. Run find_direction.py first."
        )
    return np.load(path)


def load_activations(parquet_path: Path, stage: str, layer: int) -> pd.DataFrame:
    df = pd.read_parquet(
        parquet_path,
        filters=[("stage", "==", stage), ("layer", "==", layer)],
    )
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "activation"]]


# =============================================================================
# Projection scoring
# =============================================================================

def project_and_score(
    activations: pd.DataFrame,
    outcomes: pd.DataFrame,
    direction: np.ndarray,
) -> dict:
    """Project activations onto direction; AUC against y_internalized."""
    merged = activations.merge(
        outcomes[["trial_id", "agent_id", "y_internalized"]],
        on=["trial_id", "agent_id"],
    )
    X = np.stack(merged["activation"].values)
    norm = np.linalg.norm(direction)
    proj = X @ direction / norm
    y = merged["y_internalized"].values.astype(int)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan,
                "proj_mean_high": np.nan, "proj_mean_low": np.nan}
    return {
        "n": len(y),
        "auc": float(roc_auc_score(y, proj)),
        "proj_mean_high": float(proj[y == 1].mean()),
        "proj_mean_low": float(proj[y == 0].mean()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    print(f"Loaded {len(outcomes)} outcome rows")

    print(f"\nUsing direction from layer {DIRECTION_LAYER}, {DIRECTION_STAGE}")
    print("(this is THE direction; we project all stages onto it)\n")

    rows = []
    for probe_layer in LAYERS_OF_INTEREST:
        # We use the round_4 direction at this layer
        direction = load_direction(probe_layer, DIRECTION_STAGE)
        print(f"--- Probe layer {probe_layer} (direction from round_4) ---")
        for stage in ALL_STAGES:
            try:
                acts = load_activations(PARQUET_PATH, stage, probe_layer)
            except Exception as e:
                print(f"  {stage}: skipped ({e})")
                continue
            score = project_and_score(acts, outcomes, direction)
            score["probe_layer"] = probe_layer
            score["projected_stage"] = stage
            rows.append(score)
            print(
                f"  {stage:10s}: AUC = {score['auc']:.3f} "
                f"(separation: high={score['proj_mean_high']:+.3f}, "
                f"low={score['proj_mean_low']:+.3f})"
            )

    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_DIR / "cross_stage_projections.parquet")

    plot_curves(summary, OUT_DIR / "cross_stage_curves.png")
    print(f"\nSaved figure -> {OUT_DIR / 'cross_stage_curves.png'}")

    print_verdict(summary)


def plot_curves(summary: pd.DataFrame, out_path: Path):
    stage_order = {s: i for i, s in enumerate(ALL_STAGES)}
    summary = summary.copy()
    summary["stage_idx"] = summary["projected_stage"].map(stage_order)
    summary = summary.sort_values(["probe_layer", "stage_idx"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = {10: "#999999", 20: "#1f77b4", 26: "#d62728"}
    label = {10: "Layer 10 (control)", 20: "Layer 20 (mid)",
             26: "Layer 26 (peak)"}

    for layer in LAYERS_OF_INTEREST:
        s = summary[summary.probe_layer == layer]
        ax.plot(
            s["stage_idx"], s["auc"],
            marker="o", color=cmap[layer], label=label[layer],
            linewidth=1.6, markersize=6,
        )

    ax.set_xticks(range(len(ALL_STAGES)))
    ax.set_xticklabels(ALL_STAGES, rotation=20, ha="right")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0.73, color="red", linestyle=":", linewidth=0.8,
               label="v3 probe AUC at peak (0.73)")
    ax.set_xlabel("Activation stage projected")
    ax.set_ylabel("AUC vs y_internalized")
    ax.set_ylim(0.4, 0.85)
    ax.set_title(
        "Cross-stage projection onto layer-26 round_4 direction\n"
        "(does the internalization signal exist before deliberation ends?)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_verdict(summary: pd.DataFrame):
    """Pretty-print the layer-26 row across stages and interpret."""
    s = summary[summary.probe_layer == DIRECTION_LAYER].copy()
    s["stage_idx"] = s["projected_stage"].map({s_: i for i, s_ in enumerate(ALL_STAGES)})
    s = s.sort_values("stage_idx")

    print("\n" + "=" * 64)
    print(f"INTERPRETATION (using layer {DIRECTION_LAYER})")
    print("=" * 64)
    auc_by_stage = dict(zip(s["projected_stage"], s["auc"]))

    base = auc_by_stage.get("baseline", np.nan)
    r0 = auc_by_stage.get("round_0", np.nan)
    r2 = auc_by_stage.get("round_2", np.nan)
    r4 = auc_by_stage.get("round_4", np.nan)
    shadow = auc_by_stage.get("shadow", np.nan)

    print(f"\n  baseline AUC: {base:.3f}  (before any deliberation context)")
    print(f"  round_0 AUC:  {r0:.3f}  (one round of context, no shifts yet)")
    print(f"  round_2 AUC:  {r2:.3f}  (mid-deliberation)")
    print(f"  round_4 AUC:  {r4:.3f}  (end of deliberation — direction's home)")
    print(f"  shadow AUC:   {shadow:.3f}  (private re-elicitation)")

    print("\nReadings:")
    if r0 >= 0.65:
        print(f"  - round_0 already at {r0:.3f}: internalization-propensity")
        print(f"    is largely baked in by the time the agent has read")
        print(f"    a single round of neighbor messages. Deliberation")
        print(f"    refines but doesn't create the signal.")
    elif r0 >= 0.55:
        print(f"  - round_0 at {r0:.3f}: weak prior signal exists before")
        print(f"    deliberation has done much. Deliberation strengthens it.")
    else:
        print(f"  - round_0 at {r0:.3f}: signal genuinely emerges through")
        print(f"    deliberation, not present at start.")

    if not np.isnan(base):
        if base >= 0.6:
            print(f"  - baseline at {base:.3f}: even the agent's PRE-DELIBERATION")
            print(f"    state encodes the propensity. The scenario alone is")
            print(f"    enough to predict, before any social context.")
        else:
            print(f"  - baseline at {base:.3f}: pre-deliberation state alone")
            print(f"    does not strongly encode the propensity.")

    if r4 - r0 > 0.05:
        print(f"  - r4 - r0 = {r4 - r0:+.3f}: deliberation strengthens the")
        print(f"    signal as it accumulates.")
    else:
        print(f"  - r4 - r0 = {r4 - r0:+.3f}: deliberation does NOT strengthen")
        print(f"    the signal. It's already at full strength early.")

    if not np.isnan(shadow):
        if shadow >= r4 - 0.03:
            print(f"  - shadow {shadow:.3f} ≈ r4 {r4:.3f}: the signal persists")
            print(f"    into the private elicitation context.")
        else:
            print(f"  - shadow {shadow:.3f} < r4 {r4:.3f}: signal weakens")
            print(f"    when social context is removed.")


if __name__ == "__main__":
    main()