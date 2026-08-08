"""Agent-level analysis of v2 on round_4 EV.

v2 has 2 conditions: control and neg_r2 (negative steering at round_2 only,
using the round_2 direction at layer 26). Otherwise mirrors v1 setup.

Output: per-agent paired contrast on round_4 EV, plus per-scenario rollup.
Same logic as analyze_steering_unsaturated.py but for v2's 2-condition design.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

JSONL = Path("outputs/steering_experiment_v2/trials_v2.jsonl")
SAT_THRESHOLD = 0.99


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


def maxp(p):
    if not p:
        return 0.0
    return max(p.values())


# Load all agent-rows
rows = []
with open(JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        tr = json.loads(line)
        for a in tr["agents"]:
            r4 = a["round_probs"][-1] if a["round_probs"] else {}
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "agent_id": a["agent_id"],
                "role": a["role"],
                "baseline_ev": ev(a["baseline_probs"]),
                "round_4_ev": ev(r4),
                "round_4_maxp": maxp(r4),
            })

df = pd.DataFrame(rows)
# Aligned only (misaligned aren't analyzed)
df = df[df.role == "aligned"].drop(columns=["role"])

print(f"Loaded {len(df)} aligned-agent rows from "
      f"{df['scenario_id'].nunique()} scenarios × "
      f"{df['condition'].nunique()} conditions")
print(f"Conditions: {sorted(df['condition'].unique())}\n")

# Pivot per (scenario_id, agent_id) × condition
pivot = df.pivot_table(
    index=["scenario_id", "agent_id"],
    columns="condition",
    values="round_4_ev",
)
maxp_pivot = df.pivot_table(
    index=["scenario_id", "agent_id"],
    columns="condition",
    values="round_4_maxp",
)

# Need control + neg_r2
required = ["control", "neg_r2"]
for c in required:
    if c not in pivot.columns:
        print(f"ERROR: missing condition {c}")
        exit(1)

pairs = pivot[required].dropna()
pairs = pairs.copy()
pairs["delta"] = pairs["neg_r2"] - pairs["control"]
print(f"Paired observations (scenario × agent): {len(pairs)}\n")

# === Headline: paired t-test on round_4 EV ===
print("=" * 72)
print("Agent-level paired t-test on round_4 EV (PRIMARY OUTCOME)")
print("=" * 72)

diff = pairs["delta"].dropna()
t, p = stats.ttest_1samp(diff.values, 0)
sem = diff.std() / np.sqrt(len(diff))
ci_lo = diff.mean() - 1.96 * sem
ci_hi = diff.mean() + 1.96 * sem
print(f"\n  neg_r2 vs control:")
print(f"    Δ = {diff.mean():+.4f}  95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
      f"t={t:+.2f}  p={p:.4f}  n={len(diff)}")

# Sign analysis
n_lower = (diff < 0).sum()
n_higher = (diff > 0).sum()
n_equal = (diff == 0).sum()
print(f"\n  neg_r2 < control (suppression worked):  {n_lower}/{len(diff)}  "
      f"({100*n_lower/len(diff):.1f}%)")
print(f"  neg_r2 > control (steering wrong way):  {n_higher}/{len(diff)}  "
      f"({100*n_higher/len(diff):.1f}%)")
print(f"  neg_r2 = control (locked):              {n_equal}/{len(diff)}  "
      f"({100*n_equal/len(diff):.1f}%)")

n_directional = n_lower + n_higher
if n_directional > 0:
    binom_p = stats.binomtest(n_lower, n_directional, p=0.5,
                              alternative="greater").pvalue
    print(f"\n  Sign test on the {n_directional} non-tied pairs:")
    print(f"  neg_r2 suppressed in {n_lower}/{n_directional} = "
          f"{100*n_lower/n_directional:.1f}%  (chance = 50%)")
    print(f"  Binomial p (one-sided): {binom_p:.4g}")

# === Saturation context ===
print("\n" + "=" * 72)
print("Saturation rates at round_4 (max_p > 0.99)")
print("=" * 72)
for c in required:
    sat = (df[df.condition == c]["round_4_maxp"] > 0.99).mean()
    print(f"  {c:>8s}: {100*sat:.0f}% saturated at round_4")

# === Non-saturated subset ===
keep_any = (maxp_pivot[required] < SAT_THRESHOLD).any(axis=1)
unsat_idx = maxp_pivot[keep_any].index
unsat_pairs = pairs.loc[pairs.index.isin(unsat_idx)]
print(f"\n  Pairs non-saturated in any condition (max_p<0.99): "
      f"{len(unsat_pairs)}/{len(pairs)}")

if len(unsat_pairs) >= 5:
    diff_u = unsat_pairs["delta"]
    t_u, p_u = stats.ttest_1samp(diff_u.values, 0)
    sem_u = diff_u.std() / np.sqrt(len(diff_u))
    ci_lo_u = diff_u.mean() - 1.96 * sem_u
    ci_hi_u = diff_u.mean() + 1.96 * sem_u
    print(f"\n  On non-saturated subset (n={len(diff_u)}):")
    print(f"    Δ = {diff_u.mean():+.4f}  95%CI [{ci_lo_u:+.4f}, {ci_hi_u:+.4f}]  "
          f"t={t_u:+.2f}  p={p_u:.4f}")

# === Per-scenario rollup ===
print("\n" + "=" * 72)
print("Per-scenario suppression rate")
print("=" * 72)
print(f"\n{'scenario':<14} {'baseline':>9} {'ctrl_r4':>8} {'neg_r4':>8} "
      f"{'Δ':>8} {'n_supp':>7} {'n_total':>8}")

scen_rows = []
for sid in sorted(pairs.index.get_level_values(0).unique()):
    sub = pairs.xs(sid, level=0)
    base_ev = df[df.scenario_id == sid].iloc[0]["baseline_ev"]
    n_supp = (sub["delta"] < 0).sum()
    n_total = len(sub)
    mean_delta = sub["delta"].mean()
    mean_ctrl = sub["control"].mean()
    mean_neg = sub["neg_r2"].mean()
    print(f"{sid:<14} {base_ev:>9.3f} {mean_ctrl:>8.3f} {mean_neg:>8.3f} "
          f"{mean_delta:>+8.3f} {n_supp:>7} {n_total:>8}")
    scen_rows.append({
        "scenario_id": sid, "baseline_ev": base_ev,
        "ctrl_r4_ev": mean_ctrl, "neg_r4_ev": mean_neg,
        "delta": mean_delta, "n_suppressed": n_supp, "n_total": n_total,
    })

scen_df = pd.DataFrame(scen_rows)
n_majority = (scen_df["n_suppressed"] > scen_df["n_total"] / 2).sum()
n_any = (scen_df["delta"] < 0).sum()
print(f"\n  Scenarios with mean Δ < 0 (suppression on average): "
      f"{n_any}/{len(scen_df)}")
print(f"  Scenarios where >50% of agents got suppressed:      "
      f"{n_majority}/{len(scen_df)}")

scen_df.to_csv("outputs/steering_experiment_v2/v2_per_scenario_round4.csv",
               index=False)
print(f"\nSaved → outputs/steering_experiment_v2/v2_per_scenario_round4.csv")

# === Side-by-side comparison with v1 neg_all ===
print("\n" + "=" * 72)
print("v2 neg_r2 vs v1 neg_all (round_4 EV summary)")
print("=" * 72)

# Try to load v1 for comparison
v1_path = Path("outputs/steering_experiment/trials.jsonl")
if v1_path.exists():
    v1_rows = []
    with open(v1_path) as f:
        for line in f:
            if not line.strip():
                continue
            tr = json.loads(line)
            if tr["condition"] not in ("control", "neg_all"):
                continue
            for a in tr["agents"]:
                if a["role"] != "aligned":
                    continue
                r4 = a["round_probs"][-1] if a["round_probs"] else {}
                v1_rows.append({
                    "scenario_id": tr["scenario_id"],
                    "condition": tr["condition"],
                    "agent_id": a["agent_id"],
                    "round_4_ev": ev(r4),
                })
    v1_df = pd.DataFrame(v1_rows)
    v1_pivot = v1_df.pivot_table(
        index=["scenario_id", "agent_id"], columns="condition",
        values="round_4_ev",
    )
    v1_pairs = v1_pivot[["control", "neg_all"]].dropna()
    v1_diff = v1_pairs["neg_all"] - v1_pairs["control"]

    print(f"\n  v1 neg_all (5-round steering, round_4 dir): "
          f"Δ = {v1_diff.mean():+.4f}  n={len(v1_diff)}  "
          f"sign 'right' = {(v1_diff < 0).sum()}/{len(v1_diff)}")
    print(f"  v2 neg_r2  (1-round steering, round_2 dir): "
          f"Δ = {pairs['delta'].mean():+.4f}  n={len(pairs)}  "
          f"sign 'right' = {n_lower}/{len(pairs)}")