"""Re-analyze v1 steering results with round_4 EV as primary outcome.

The original analyze_steering.py used shadow_shift, which is a noisy
downstream proxy. Round_4 EV is the immediately downstream output of
steering interventions during deliberation and is the cleaner signal.

Outputs paired t-tests on round_4 EV, plus shadow shift as secondary.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

JSONL = Path("outputs/steering_experiment/trials.jsonl")


def jsd(p_dict, q_dict):
    p = np.array([p_dict.get(str(i), p_dict.get(i, 0.0)) for i in range(1, 8)], dtype=float)
    q = np.array([q_dict.get(str(i), q_dict.get(i, 0.0)) for i in range(1, 8)], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


rows = []
with open(JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        tr = json.loads(line)
        for a in tr["agents"]:
            base, final, shadow = a["baseline_probs"], a["round_probs"][-1], a["shadow_probs"]
            if not (base and final):
                continue
            j_final = jsd(final, base)
            ii = jsd(shadow, base) / j_final if (shadow and j_final > 1e-9) else float("nan")
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "baseline_ev": ev(base),
                "round_4_ev": ev(final),
                "shadow_ev": ev(shadow),
                "round_4_shift": ev(final) - ev(base),
                "shadow_shift": ev(shadow) - ev(base) if shadow else float("nan"),
                "ii": ii,
            })
df = pd.DataFrame(rows)
print(f"Loaded {len(df)} agent rows from {df['scenario_id'].nunique()} scenarios x {df['condition'].nunique()} conditions\n")

# Trial-level (mean across 8 agents per scenario+condition)
trial = df.groupby(["scenario_id", "condition"]).agg(
    mean_round4_ev=("round_4_ev", "mean"),
    mean_round4_shift=("round_4_shift", "mean"),
    mean_shadow_shift=("shadow_shift", "mean"),
    mean_ii=("ii", "mean"),
).reset_index()

# Per-condition summary
print("=== Per-condition trial-level means (n=30 scenarios per condition) ===")
summary = trial.groupby("condition").agg(
    mean_round4_ev=("mean_round4_ev", "mean"),
    mean_round4_shift=("mean_round4_shift", "mean"),
    mean_shadow_shift=("mean_shadow_shift", "mean"),
    mean_ii=("mean_ii", "mean"),
).round(3)
print(summary.to_string(), "\n")

# Paired tests vs control on each metric
def paired_test(metric, label):
    print(f"\n=== Paired comparisons on {label} (paired by scenario_id) ===")
    ctrl = trial[trial.condition == "control"].set_index("scenario_id")[metric]
    for cond in ["pos_all", "neg_all", "pos_r2"]:
        s = trial[trial.condition == cond].set_index("scenario_id")[metric]
        common = ctrl.index.intersection(s.index)
        a = ctrl.loc[common].dropna()
        b = s.loc[common].dropna()
        common_idx = a.index.intersection(b.index)
        a, b = a.loc[common_idx], b.loc[common_idx]
        diff = (b - a).dropna()
        if len(diff) < 2:
            print(f"  {cond:10s}: n={len(diff)} too few")
            continue
        t, p = stats.ttest_rel(b.values, a.values)
        # 95% CI on the paired difference
        sem = diff.std() / np.sqrt(len(diff))
        ci_lo = diff.mean() - 1.96 * sem
        ci_hi = diff.mean() + 1.96 * sem
        print(f"  {cond:10s}: Δ = {diff.mean():+.4f}  "
              f"95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"t={t:+.2f}  p={p:.4f}  n={len(diff)}")


paired_test("mean_round4_ev", "round_4 EV (PRIMARY OUTCOME)")
paired_test("mean_round4_shift", "round_4 shift (round_4 - baseline)")
paired_test("mean_shadow_shift", "shadow shift (secondary, noisy proxy)")
paired_test("mean_ii", "II (Internalization Index, derived from shadow)")

# Sign analysis on round_4 EV
print("\n=== Sign analysis on round_4 EV ===")
pos = trial[trial.condition == "pos_all"].set_index("scenario_id")["mean_round4_ev"]
neg = trial[trial.condition == "neg_all"].set_index("scenario_id")["mean_round4_ev"]
common = pos.index.intersection(neg.index)
diff = (pos.loc[common] - neg.loc[common]).dropna()
n_pos_higher = (diff > 0).sum()
print(f"  pos_all > neg_all on round_4 EV in {n_pos_higher}/{len(diff)} scenarios "
      f"({100*n_pos_higher/len(diff):.0f}%)")
print(f"  mean(pos_all - neg_all) on round_4 EV = {diff.mean():+.4f} (sd {diff.std():.4f})")
sem = diff.std() / np.sqrt(len(diff))
print(f"  95%CI on (pos_all - neg_all): [{diff.mean() - 1.96*sem:+.4f}, {diff.mean() + 1.96*sem:+.4f}]")
t, p = stats.ttest_1samp(diff.values, 0)
print(f"  t-test vs 0: t={t:+.2f}, p={p:.4f}")

# Save trial-level data for plotting later
trial.to_csv("outputs/steering_experiment/trial_level_with_round4.csv", index=False)
print("\nTrial-level data → outputs/steering_experiment/trial_level_with_round4.csv")