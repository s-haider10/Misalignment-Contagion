"""Analyze steering experiment results: per-condition stats + paired tests."""

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
            if not (base and final and shadow):
                continue
            j_final = jsd(final, base)
            ii = jsd(shadow, base) / j_final if j_final > 1e-9 else float("nan")
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "baseline_ev": ev(base),
                "round_4_ev": ev(final),
                "shadow_ev": ev(shadow),
                "shadow_shift": ev(shadow) - ev(base),
                "ii": ii,
            })
df = pd.DataFrame(rows)
print(f"Loaded {len(df)} agent rows from {df['scenario_id'].nunique()} scenarios x {df['condition'].nunique()} conditions\n")

# Trial-level (mean across 8 agents per scenario+condition)
trial = df.groupby(["scenario_id", "condition"]).agg(
    mean_ii=("ii", "mean"),
    mean_shadow_shift=("shadow_shift", "mean"),
    mean_round4_ev=("round_4_ev", "mean"),
).reset_index()

print("=== Per-condition summary (trial-level means, n=30 scenarios per condition) ===")
summary = trial.groupby("condition").agg(
    mean_ii=("mean_ii", "mean"),
    median_ii=("mean_ii", "median"),
    mean_shift=("mean_shadow_shift", "mean"),
    median_shift=("mean_shadow_shift", "median"),
    mean_r4_ev=("mean_round4_ev", "mean"),
).round(3)
print(summary.to_string(), "\n")

# Paired tests vs control
print("=== Paired comparisons vs control (paired by scenario_id) ===")
ctrl = trial[trial.condition == "control"].set_index("scenario_id")
for cond in ["pos_all", "neg_all", "pos_r2"]:
    s = trial[trial.condition == cond].set_index("scenario_id")
    common = ctrl.index.intersection(s.index)
    for metric in ["mean_ii", "mean_shadow_shift"]:
        a = ctrl.loc[common, metric]
        b = s.loc[common, metric]
        diff = (b - a).dropna()
        if len(diff) < 2:
            print(f"  {cond:8s} {metric:20s}: n={len(diff)} too few")
            continue
        t, p = stats.ttest_rel(b.dropna().align(a.dropna(), join="inner")[0],
                                a.dropna().align(b.dropna(), join="inner")[0])
        print(f"  {cond:8s} {metric:20s}: Δ = {diff.mean():+.3f} (sd {diff.std():.3f}, n={len(diff)}), t={t:+.2f}, p={p:.3f}")

# Direction of effect: how many scenarios moved in the predicted direction?
print("\n=== Sign analysis: did pos_all increase shift more than neg_all? ===")
pos = trial[trial.condition == "pos_all"].set_index("scenario_id")["mean_shadow_shift"]
neg = trial[trial.condition == "neg_all"].set_index("scenario_id")["mean_shadow_shift"]
common = pos.index.intersection(neg.index)
diff = (pos.loc[common] - neg.loc[common]).dropna()
n_pos_higher = (diff > 0).sum()
print(f"  pos_all > neg_all in {n_pos_higher}/{len(diff)} scenarios ({100*n_pos_higher/len(diff):.0f}%)")
print(f"  mean(pos_all - neg_all) = {diff.mean():+.3f} (sd {diff.std():.3f})")
print(f"  if direction works: pos_all > neg_all should be true >50% and Δ should be substantially positive")

