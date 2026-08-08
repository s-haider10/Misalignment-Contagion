"""Sanity check: how saturated are stance distributions at each stage?

If round_4 distributions are nearly one-hot (entropy ≈ 0), steering can't
move them. Round_2 might have more entropy and thus more steering room.
This script reads existing pilot data — no GPU needed, takes 30 seconds.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

JSONL_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/primary_em/"
    "moral_stories/qwen-7b-instruct/results.jsonl"
)

PILOT_FILTER = dict(
    dataset="moral_stories",
    minority_ratio=0.2,
    model_key="qwen-7b-instruct",
    model_condition="model_induced",
    prompt_strategy="rigid:rigid",
    seed=42,
)


def entropy(probs_dict):
    if probs_dict is None:
        return None
    p = np.array(list(probs_dict.values()), dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def main():
    rows = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(t.get(k) == v for k, v in PILOT_FILTER.items()):
                continue
            if t.get("topology", "").lower() != "fc":
                continue

            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                # baseline
                if a.get("baseline_probs"):
                    rows.append({"stage": "baseline", "entropy": entropy(a["baseline_probs"])})
                # round_0..4
                for r, rp in enumerate(a.get("round_probs", [])):
                    if rp:
                        rows.append({"stage": f"round_{r}", "entropy": entropy(rp)})
                # shadow
                if a.get("shadow_probs"):
                    rows.append({"stage": "shadow", "entropy": entropy(a["shadow_probs"])})

    df = pd.DataFrame(rows)
    print(f"Total observations: {len(df)}\n")

    summary = df.groupby("stage")["entropy"].agg(
        ["mean", "median", "std", "min", "max",
         lambda s: (s < 0.01).mean(),  # fraction near-zero
         lambda s: (s < 0.5).mean(),   # fraction nearly committed
         "count"]
    ).round(3)
    summary.columns = ["mean", "median", "std", "min", "max",
                       "frac_<0.01", "frac_<0.5", "n"]

    print("Entropy by stage (in bits, max possible = log2(7) = 2.81):")
    print(summary)
    print()
    print("READING:")
    print("  - frac_<0.01 = fraction of agents with essentially one-hot distributions")
    print("    (i.e., zero steering room)")
    print("  - frac_<0.5  = fraction with very low entropy (almost-committed)")
    print()
    print("If round_4 frac_<0.01 is high, steering at round_4 can't move EV.")
    print("If round_2 frac_<0.01 is much lower, round_2 has steering room.")


if __name__ == "__main__":
    main()