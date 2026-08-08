"""Direction-finding at round_2 (in addition to existing round_4 direction).

Goal: produce a 3584-dim vector at layer 26 that points along high-II vs
low-II trials, but computed from MID-deliberation activations (round_2)
rather than END-of-deliberation (round_4). The hypothesis is that the
round_2 direction is more aligned with active deliberation dynamics at
the moment of intervention, while the round_4 direction may reflect
post-decision commitment.

Method is identical to find_direction.py but with PEAK_STAGE = "round_2".
We also compute the cosine similarity between the round_2 and round_4
directions to see how different they actually are.

Output:
  - direction_layer26_round_2.npy
  - direction_round2_summary.parquet (sanity check AUCs)
  - cosine similarity printed to stdout
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from pathlib import Path

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

LAYERS_OF_INTEREST = [10, 20, 24, 25, 26, 27]
PEAK_LAYER = 26
PEAK_STAGE = "round_2"
EXISTING_R4_PATH = OUT_DIR / "direction_layer26_round_4.npy"


def trial_level_split(outcomes: pd.DataFrame) -> pd.DataFrame:
    valid = outcomes.dropna(subset=["ii"]).copy()
    valid = valid[valid["ii"].abs() < 10]
    per_trial = valid.groupby("trial_id").agg(
        mean_ii=("ii", "mean"),
        n_internalized=("y_internalized", "sum"),
        n_agents=("ii", "size"),
    ).reset_index()
    median_ii = per_trial["mean_ii"].median()
    per_trial["group"] = np.where(per_trial["mean_ii"] >= median_ii, "high", "low")
    print(f"Trial-level split (median mean_ii = {median_ii:.3f}):")
    print(f"  high-II trials: {(per_trial.group=='high').sum()}")
    print(f"  low-II trials:  {(per_trial.group=='low').sum()}")
    return per_trial


def load_activations(parquet_path: Path, stage: str, layer: int) -> pd.DataFrame:
    df = pd.read_parquet(
        parquet_path,
        filters=[("stage", "==", stage), ("layer", "==", layer)],
    )
    df["activation"] = df["activation_bytes"].apply(
        lambda b: np.frombuffer(b, dtype=np.float16).astype(np.float32)
    )
    return df[["trial_id", "agent_id", "activation"]]


def compute_direction(activations: pd.DataFrame, per_trial: pd.DataFrame) -> np.ndarray:
    merged = activations.merge(per_trial[["trial_id", "group"]], on="trial_id")
    high = np.stack(merged[merged.group == "high"]["activation"].values)
    low = np.stack(merged[merged.group == "low"]["activation"].values)
    return (high.mean(axis=0) - low.mean(axis=0)).astype(np.float32)


def project_and_score(activations, outcomes, direction):
    merged = activations.merge(
        outcomes[["trial_id", "agent_id", "y_internalized"]],
        on=["trial_id", "agent_id"],
    )
    X = np.stack(merged["activation"].values)
    norm = np.linalg.norm(direction)
    proj = X @ direction / norm
    y = merged["y_internalized"].values.astype(int)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan, "norm": float(norm)}
    return {"n": len(y), "auc": float(roc_auc_score(y, proj)), "norm": float(norm)}


def main():
    outcomes = pd.read_parquet(OUTCOMES_PATH)
    per_trial = trial_level_split(outcomes)

    rows = []
    print(f"\nComputing directions at stage={PEAK_STAGE}")
    for layer in LAYERS_OF_INTEREST:
        acts = load_activations(PARQUET_PATH, PEAK_STAGE, layer)
        direction = compute_direction(acts, per_trial)
        np.save(OUT_DIR / f"direction_layer{layer:02d}_{PEAK_STAGE}.npy", direction)
        score = project_and_score(acts, outcomes, direction)
        score["layer"] = layer
        rows.append(score)
        print(f"  layer {layer:2d}: ||d|| = {score['norm']:.2f}, "
              f"projection AUC = {score['auc']:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_DIR / f"direction_summary_{PEAK_STAGE}.parquet")

    # Compare to round_4 direction at peak layer
    if EXISTING_R4_PATH.exists():
        d_r4 = np.load(EXISTING_R4_PATH)
        d_r2 = np.load(OUT_DIR / f"direction_layer{PEAK_LAYER:02d}_{PEAK_STAGE}.npy")
        cos = float(np.dot(d_r2, d_r4) / (np.linalg.norm(d_r2) * np.linalg.norm(d_r4)))
        print(f"\nCosine similarity between round_2 and round_4 direction at layer {PEAK_LAYER}: {cos:.4f}")
        if cos > 0.95:
            print("  → directions are ALMOST IDENTICAL; round_2 experiment unlikely to differ.")
        elif cos > 0.7:
            print("  → directions are similar but distinct; round_2 experiment plausibly different.")
        else:
            print("  → directions are substantially different; worth running round_2 experiment.")
        norm_ratio = float(np.linalg.norm(d_r2) / np.linalg.norm(d_r4))
        print(f"  ||d_r2|| / ||d_r4|| = {norm_ratio:.3f} "
              f"(expect r2 < r4 since r2 has less separation)")

    print(f"\nSaved directions to {OUT_DIR}")
    print(f"Round_2 direction at layer {PEAK_LAYER}: "
          f"outputs/direction_results/direction_layer{PEAK_LAYER:02d}_{PEAK_STAGE}.npy")


if __name__ == "__main__":
    main()