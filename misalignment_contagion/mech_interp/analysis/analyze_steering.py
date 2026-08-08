"""Analyze v3 (random direction) on round_4 EV. Matches v3's actual design:
control + neg_r2 only.

Direct comparison: v3 random vs v2 real round_2 direction. Both have the same
intervention design (single-shot α=-1 at round_2), differing only in the
direction added. This is the cleanest possible specificity test.

Pre-registered decision:
  - If v3 neg_r2 produces a similar suppression effect as v2 (Δ ≈ -0.10 EV,
    sign rate ~ 67%): v2's effect is largely "any direction at layer 26 added
    at round_2 disrupts deliberation"
  - If v3 neg_r2 is null (Δ ≈ 0, sign rate ~ 50%): v2's direction is doing
    something specific to the internalization-correlated direction
  - Ambiguous in between
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

V3_JSONL = Path("outputs/steering_experiment_v3_random_control/trials_v3.jsonl")
V2_JSONL = Path("outputs/steering_experiment_v2/trials_v2.jsonl")
SAT_THRESHOLD = 0.99


def ev(p):
    if not p:
        return float("nan")
    return sum(int(k) * float(v) for k, v in p.items())


def maxp(p):
    if not p:
        return 0.0
    return max(p.values())


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            tr = json.loads(line)
            for a in tr["agents"]:
                if a["role"] != "aligned":
                    continue
                r4 = a["round_probs"][-1] if a["round_probs"] else {}
                rows.append({
                    "scenario_id": tr["scenario_id"],
                    "condition": tr["condition"],
                    "agent_id": a["agent_id"],
                    "baseline_ev": ev(a["baseline_probs"]),
                    "round_4_ev": ev(r4),
                    "round_4_maxp": maxp(r4),
                })
    return pd.DataFrame(rows)


def analyze(df, label):
    """Run paired t-test + sign test on neg_r2 vs control for one experiment."""
    pivot = df.pivot_table(
        index=["scenario_id", "agent_id"],
        columns="condition",
        values="round_4_ev",
    )
    if "control" not in pivot.columns or "neg_r2" not in pivot.columns:
        return None

    pairs = pivot[["control", "neg_r2"]].dropna()
    diff = pairs["neg_r2"] - pairs["control"]

    t, p = stats.ttest_1samp(diff.values, 0)
    sem = diff.std() / np.sqrt(len(diff))
    ci_lo = diff.mean() - 1.96 * sem
    ci_hi = diff.mean() + 1.96 * sem

    n_lower = (diff < 0).sum()
    n_higher = (diff > 0).sum()
    n_eq = (diff == 0).sum()
    n_dir = n_lower + n_higher

    result = {
        "label": label,
        "n_pairs": len(diff),
        "delta": diff.mean(),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "t": t, "p": p,
        "n_lower": n_lower, "n_higher": n_higher, "n_eq": n_eq,
        "sign_rate_movers": n_lower / n_dir if n_dir > 0 else 0.5,
    }
    if n_dir > 0:
        result["binom_p"] = stats.binomtest(
            n_lower, n_dir, p=0.5, alternative="greater"
        ).pvalue
    else:
        result["binom_p"] = 1.0

    # Saturation rate for control
    sat = (df[df.condition == "control"]["round_4_maxp"] > 0.99).mean()
    result["ctrl_sat_rate"] = sat

    return result


# === Load v2 and v3 ===
print("=" * 72)
print("Loading v2 (real direction) and v3 (random direction)")
print("=" * 72)

v2_df = load_jsonl(V2_JSONL)
print(f"  v2: {len(v2_df)} aligned rows from {v2_df['scenario_id'].nunique()} scenarios")
print(f"      conditions: {sorted(v2_df['condition'].unique())}")

v3_df = load_jsonl(V3_JSONL)
print(f"  v3: {len(v3_df)} aligned rows from {v3_df['scenario_id'].nunique()} scenarios")
print(f"      conditions: {sorted(v3_df['condition'].unique())}")

# Find shared scenario set
v2_scenarios = set(v2_df["scenario_id"].unique())
v3_scenarios = set(v3_df["scenario_id"].unique())
shared = v2_scenarios & v3_scenarios
print(f"\n  v2 ∩ v3 scenarios: {len(shared)}")
v2_only = v2_scenarios - v3_scenarios
v3_only = v3_scenarios - v2_scenarios
if v2_only:
    print(f"  v2-only: {sorted(v2_only)}")
if v3_only:
    print(f"  v3-only: {sorted(v3_only)}")

# === Per-experiment results ===
v2_result = analyze(v2_df, "v2 (real round_2 direction)")
v3_result = analyze(v3_df, "v3 (random direction, norm-matched)")

print("\n" + "=" * 72)
print("PRIMARY RESULT: neg_r2 vs control on round_4 EV")
print("=" * 72)

for r in [v2_result, v3_result]:
    print(f"\n  {r['label']}:")
    print(f"    n = {r['n_pairs']} paired observations")
    print(f"    Δ (neg_r2 − control) = {r['delta']:+.4f}  "
          f"95%CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    print(f"    t = {r['t']:+.2f}, p = {r['p']:.4f}")
    print(f"    locked (no movement):  {r['n_eq']}/{r['n_pairs']} "
          f"({100*r['n_eq']/r['n_pairs']:.0f}%)")
    print(f"    suppressed (Δ<0):      {r['n_lower']}/{r['n_pairs']}")
    print(f"    pushed wrong (Δ>0):    {r['n_higher']}/{r['n_pairs']}")
    print(f"    sign rate on movers:   {100*r['sign_rate_movers']:.1f}% "
          f"binomial p = {r['binom_p']:.4g}")
    print(f"    control saturation:    {100*r['ctrl_sat_rate']:.0f}%")

# === Restricted to shared scenarios ===
if shared and len(shared) >= 5:
    print("\n" + "=" * 72)
    print(f"Restricted to shared scenarios (n_scenarios={len(shared)})")
    print("=" * 72)
    v2_shared = v2_df[v2_df["scenario_id"].isin(shared)]
    v3_shared = v3_df[v3_df["scenario_id"].isin(shared)]
    v2_s = analyze(v2_shared, "v2 (shared scenarios)")
    v3_s = analyze(v3_shared, "v3 (shared scenarios)")
    for r in [v2_s, v3_s]:
        if r is None:
            continue
        print(f"\n  {r['label']} (n={r['n_pairs']}):")
        print(f"    Δ = {r['delta']:+.4f}  95%CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              f"  t={r['t']:+.2f}  p={r['p']:.4f}")
        print(f"    sign rate on movers: {100*r['sign_rate_movers']:.1f}%  "
              f"binom p={r['binom_p']:.4g}")

# === Per-scenario v3 picture ===
print("\n" + "=" * 72)
print("v3 per-scenario picture (random direction)")
print("=" * 72)
v3_pivot = v3_df.pivot_table(
    index=["scenario_id", "agent_id"], columns="condition",
    values="round_4_ev",
)
print(f"\n{'scenario':<14} {'baseline':>9} {'ctrl':>7} {'neg_r2':>8} {'Δ':>7} "
      f"{'n_supp':>7} {'n_total':>8}")
for sid in sorted(v3_pivot.index.get_level_values(0).unique()):
    sub = v3_pivot.xs(sid, level=0)
    if "control" not in sub.columns or "neg_r2" not in sub.columns:
        continue
    base = v3_df[v3_df.scenario_id == sid].iloc[0]["baseline_ev"]
    ctrl = sub["control"].mean()
    neg = sub["neg_r2"].mean()
    n_supp = ((sub["neg_r2"] - sub["control"]) < 0).sum()
    n_total = len(sub.dropna())
    print(f"{sid:<14} {base:>9.3f} {ctrl:>7.3f} {neg:>8.3f} {neg-ctrl:>+7.3f} "
          f"{n_supp:>7} {n_total:>8}")

# === Verdict ===
print("\n" + "=" * 72)
print("VERDICT (per pre-registered decision rule)")
print("=" * 72)

v2_d = v2_result["delta"]
v3_d = v3_result["delta"]
v2_sign = v2_result["sign_rate_movers"]
v3_sign = v3_result["sign_rate_movers"]
v2_binom = v2_result["binom_p"]
v3_binom = v3_result["binom_p"]

print(f"\n  v2 real:   Δ={v2_d:+.3f}  sign={100*v2_sign:.0f}%  binom p={v2_binom:.4f}")
print(f"  v3 random: Δ={v3_d:+.3f}  sign={100*v3_sign:.0f}%  binom p={v3_binom:.4f}")

# Decision tree
ratio_delta = abs(v3_d / v2_d) if abs(v2_d) > 1e-6 else float("inf")
sign_diff = abs(v3_sign - v2_sign)

print()
if v3_binom > 0.20 and abs(v3_d) < abs(v2_d) * 0.5:
    print("  VERDICT: RANDOM IS NULL (or much weaker)")
    print()
    print("  The random direction produces substantially less effect than the real")
    print("  v2 round_2 direction. v2's effect appears to be direction-specific —")
    print("  the diff-of-means direction is doing something a random vector at the")
    print("  same layer/norm does not do.")
    print()
    print("  This is the favorable outcome. Combined with the methodological caveats")
    print("  in probe.md, the honest framing is: we found a direction that is")
    print("  predictively correlated with internalization (probe AUC 0.694-0.726)")
    print("  and is causally distinct from a random direction at the same layer")
    print("  (this experiment). Direction-as-feature claim has support.")
elif v3_binom < 0.05 and ratio_delta > 0.6 and sign_diff < 0.10:
    print("  VERDICT: RANDOM MATCHES REAL")
    print()
    print("  The random direction produces a comparable effect to v2's real")
    print("  direction. v2's effect is largely 'any direction at layer 26 added")
    print("  at round_2 disrupts deliberation' — not direction-specific.")
    print()
    print("  Reframe to layer-level claim: layer 26 is causally implicated in")
    print("  contagion, but the specific diff-of-means direction is not")
    print("  meaningfully distinguishable from a random direction. Drop the")
    print("  'we found the contagion direction' framing entirely.")
else:
    print("  VERDICT: AMBIGUOUS / INTERMEDIATE")
    print()
    print("  Random produces some effect, smaller and/or less consistent than v2,")
    print("  but not clearly null. The direction is partly specific and partly")
    print("  noise-injection. Need either larger n on v3, or run v3 with full")
    print("  pos_all + neg_all conditions for better contrast.")
    print()
    print(f"    abs(v3 Δ) / abs(v2 Δ) = {ratio_delta:.2f}")
    print(f"    sign rate diff: {sign_diff:.3f}")