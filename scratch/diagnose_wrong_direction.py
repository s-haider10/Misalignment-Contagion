"""On the 9 'wrong-direction' scenarios (where neg_all increased EV instead
of decreasing it), what did pos_all do?

Three plausible patterns:

  A. Both pos_all and neg_all push EV UP on these scenarios → the direction
     is collapsing into one attractor regardless of polarity. Mechanism is
     not direction-specific on these scenarios.

  B. pos_all pushes EV UP (further) and neg_all also pushes EV UP → both
     directions amplify. Suggests the direction is not really an axis on
     these scenarios; it's just "any perturbation pushes toward the
     misaligned attractor."

  C. pos_all pushes EV UP, neg_all also pushes EV UP, but pos_all pushes
     MORE than neg_all → the direction is still polarity-correct but the
     baseline drift is so strong that even neg_all can't reverse it.

  D. pos_all is flat (ceiling) and neg_all is up → ceiling effect on pos,
     true wrong-direction on neg.

Run this on round_4 EV only (shadow is too noisy).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

JSONL = Path("outputs/steering_experiment/trials.jsonl")


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


# Load
rows = []
with open(JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        tr = json.loads(line)
        for a in tr["agents"]:
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "baseline_ev": ev(a["baseline_probs"]),
                "round_4_ev": ev(a["round_probs"][-1]),
            })
df = pd.DataFrame(rows)

# Per-scenario per-condition mean
sc = df.groupby(["scenario_id", "condition"]).agg(
    base_ev=("baseline_ev", "mean"),
    r4_ev=("round_4_ev", "mean"),
).reset_index()
piv = sc.pivot(index="scenario_id", columns="condition", values="r4_ev")
piv["base_ev"] = sc[sc.condition == "control"].set_index("scenario_id")["base_ev"]

# Define groups by Δ(neg - control) sign
piv["delta_neg"] = piv["neg_all"] - piv["control"]
piv["delta_pos"] = piv["pos_all"] - piv["control"]

wrong_dir = piv[piv.delta_neg > 0.05].sort_values("delta_neg", ascending=False)
right_dir = piv[piv.delta_neg < -0.05].sort_values("delta_neg")

# Print wrong-direction scenarios with both pos and neg deltas
print("=" * 100)
print("WRONG-DIRECTION SCENARIOS: what did pos_all do alongside neg_all?")
print("=" * 100)
print(f"\n{'scenario':<12} {'base_EV':>8} {'control':>8} "
      f"{'pos_all':>8} {'neg_all':>8} "
      f"{'Δ_pos':>8} {'Δ_neg':>8}  {'pattern':<40}")
print("-" * 100)
for sid, r in wrong_dir.iterrows():
    pos_d = r.delta_pos
    neg_d = r.delta_neg

    if abs(pos_d) < 0.05 and neg_d > 0.05:
        pattern = "D: pos flat, neg up (ceiling on pos)"
    elif pos_d > 0.05 and neg_d > 0.05 and pos_d > neg_d:
        pattern = "C: both up, pos > neg (drift dominates)"
    elif pos_d > 0.05 and neg_d > 0.05 and pos_d < neg_d:
        pattern = "B: both up, neg > pos (anti-direction)"
    elif pos_d > 0.05 and neg_d > 0.05:
        pattern = "B: both push up (collapse to attractor)"
    elif pos_d < -0.05 and neg_d > 0.05:
        pattern = "axis correct, but reversed of expected"
    else:
        pattern = "?"
    print(f"{sid:<12} {r.base_ev:>8.3f} {r.control:>8.3f} "
          f"{r.pos_all:>8.3f} {r.neg_all:>8.3f} "
          f"{pos_d:>+8.3f} {neg_d:>+8.3f}  {pattern}")

print("\n" + "=" * 100)
print("RIGHT-DIRECTION SCENARIOS (for comparison)")
print("=" * 100)
print(f"\n{'scenario':<12} {'base_EV':>8} {'control':>8} "
      f"{'pos_all':>8} {'neg_all':>8} "
      f"{'Δ_pos':>8} {'Δ_neg':>8}")
print("-" * 100)
for sid, r in right_dir.iterrows():
    print(f"{sid:<12} {r.base_ev:>8.3f} {r.control:>8.3f} "
          f"{r.pos_all:>8.3f} {r.neg_all:>8.3f} "
          f"{r.delta_pos:>+8.3f} {r.delta_neg:>+8.3f}")

# Aggregate stats
print("\n" + "=" * 100)
print("Group-level summary")
print("=" * 100)
print(f"\nWrong-direction (n={len(wrong_dir)}):")
print(f"  mean Δ_pos = {wrong_dir.delta_pos.mean():+.3f}")
print(f"  mean Δ_neg = {wrong_dir.delta_neg.mean():+.3f}")
print(f"  pos and neg same sign? "
      f"{((wrong_dir.delta_pos * wrong_dir.delta_neg) > 0).sum()}/{len(wrong_dir)}")

print(f"\nRight-direction (n={len(right_dir)}):")
print(f"  mean Δ_pos = {right_dir.delta_pos.mean():+.3f}")
print(f"  mean Δ_neg = {right_dir.delta_neg.mean():+.3f}")
print(f"  pos and neg opposite signs? "
      f"{((right_dir.delta_pos * right_dir.delta_neg) < 0).sum()}/{len(right_dir)}")

# Key question: in wrong-direction scenarios, is pos_all > neg_all (axis correct,
# baseline drift just too strong)? Or pos_all < neg_all (axis really backward)?
print(f"\nIn wrong-direction scenarios, is pos_all still HIGHER than neg_all?")
n_pos_higher = (wrong_dir.pos_all > wrong_dir.neg_all).sum()
print(f"  pos_all > neg_all: {n_pos_higher}/{len(wrong_dir)} scenarios")
n_pos_lower = (wrong_dir.pos_all < wrong_dir.neg_all).sum()
print(f"  pos_all < neg_all: {n_pos_lower}/{len(wrong_dir)} scenarios")
print()
print("Reading:")
print("  If pos_all > neg_all in most wrong-direction scenarios:")
print("    → Axis polarity is correct; baseline drift just makes both push UP,")
print("      but pos pushes UP more than neg, which is the expected ordering.")
print("  If pos_all < neg_all (or about equal):")
print("    → Axis polarity is genuinely flipping in these scenarios. Real problem.")