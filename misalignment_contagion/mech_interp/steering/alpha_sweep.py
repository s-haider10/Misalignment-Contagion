"""Alpha sweep v2 — using real round_4 deliberation prompts.

The original alpha_sweep.py tested steering on baseline-style prompts, but the
direction was computed at layer 26 round_4. Cross-stage analysis showed the
projection AUC at round_4 (0.726) is much higher than at baseline (0.577) —
the direction is most predictive late in deliberation, not at the start.

This script re-runs the alpha sweep using actual round_4 prompts reconstructed
from the moral_stories pilot trials. We pick a few aligned agents from real
trials, build the exact deliberation message list they saw at round_4, and
steer at that point.

If steering at round_4 still produces weak effects, the diff-of-means direction
genuinely is a weak intervention target. If round_4 produces strong effects,
the previous null was an artifact of testing the wrong stage.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misalignment_contagion.io_utils import load_dataset_scenarios
from misalignment_contagion.mech_interp.llm_steered import SteeringHandle
from misalignment_contagion.prompts import (
    build_deliberation_messages,
    get_system_prompt,
)
from misalignment_contagion.topology import get_visible_agents


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DIRECTION_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/direction_results/"
    "direction_layer26_round_2.npy"
)
TRIAL_JSONL = Path(
    "/home/haider/Misalignment-Contagion/outputs/primary_em/"
    "moral_stories/qwen-7b-instruct/results.jsonl"
)
OUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/alpha_sweep_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAYER = 26
TARGET_STAGE = "round_2"
SEED = 42
TEMPERATURE = 0.7

# Pilot filter — same as the activation extraction
PILOT_FILTER = dict(
    dataset="moral_stories",
    minority_ratio=0.2,
    model_key="qwen-7b-instruct",
    model_condition="model_induced",
    prompt_strategy="rigid:rigid",
    seed=42,
)
PILOT_TOPOLOGY = "fc"

ALPHAS = [-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]
N_TRIALS_TO_TEST = 5  # pick 5 trials, one aligned agent from each


# =============================================================================
# Trial + prompt reconstruction (mirrors extract_activations_pilot.py)
# =============================================================================

from dataclasses import dataclass


@dataclass
class AgentShim:
    agent_id: int
    position_in_topology: int
    role: str
    baseline_stance: int
    baseline_reasoning: str


def build_agent_shims_sorted(trial: dict) -> list[AgentShim]:
    agents = [
        AgentShim(
            agent_id=a["agent_id"],
            position_in_topology=a["position_in_topology"],
            role=a["role"],
            baseline_stance=a["baseline_stance"],
            baseline_reasoning=a["baseline_reasoning"],
        )
        for a in trial["agents"]
    ]
    agents.sort(key=lambda x: x.position_in_topology)
    return agents


def build_round_history(trial: dict, up_to_round: int) -> dict:
    by_idx = {a["position_in_topology"]: a for a in trial["agents"]}
    history: dict = {-1: {}}
    for idx in sorted(by_idx.keys()):
        a = by_idx[idx]
        history[-1][idx] = (a["baseline_stance"], a["baseline_reasoning"])
    for r in range(up_to_round):
        history[r] = {}
        for idx in sorted(by_idx.keys()):
            a = by_idx[idx]
            history[r][idx] = (a["round_stances"][r], a["round_responses"][r])
    return history


def reconstruct_round4_messages(
    trial: dict, agent: dict, scenarios: dict
) -> list[dict]:
    """Rebuild the chat messages that this aligned agent saw at round 4."""
    scenario = scenarios[trial["scenario_id"]]
    system_prompt = get_system_prompt(
        role=agent["role"],
        model_condition=trial["model_condition"],
        prompt_strategy=trial["prompt_strategy"],
    )
    agents_sorted = build_agent_shims_sorted(trial)
    agent_idx = agent["position_in_topology"]
    round_history = build_round_history(trial, up_to_round=4)
    visible = get_visible_agents(
        topology=trial["topology"],
        agent_idx=agent_idx,
        agents=agents_sorted,
        round_history=round_history,
        current_round=4,
    )
    return build_deliberation_messages(
        system_prompt=system_prompt,
        scenario=scenario,
        visible_responses=visible,
    )


# =============================================================================
# Pick test agents
# =============================================================================

def pick_test_agents(jsonl_path: Path, n: int) -> list[tuple[dict, dict]]:
    """Pick n (trial, aligned_agent) pairs from the pilot subset.

    Selection: take trials where the picked aligned agent has a "real"
    II value (not nan, not extreme), and varied baseline stances so the
    sweep covers different starting points.
    """
    candidates = []
    with open(jsonl_path) as f:
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
            if t.get("topology", "").lower() != PILOT_TOPOLOGY:
                continue
            for a in t["agents"]:
                if a["role"] != "aligned":
                    continue
                if a.get("shadow_probs") is None:
                    continue
                candidates.append((t, a))

    # Pick n agents spread across baseline stances 1-7 if possible
    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)
    picked = []
    seen_baselines = set()
    for trial, agent in candidates:
        bs = agent["baseline_stance"]
        if bs not in seen_baselines:
            picked.append((trial, agent))
            seen_baselines.add(bs)
        if len(picked) >= n:
            break
    # Fill remaining slots if we didn't hit n with unique baselines
    if len(picked) < n:
        for trial, agent in candidates:
            if (trial, agent) not in picked:
                picked.append((trial, agent))
                if len(picked) >= n:
                    break

    print(f"Picked {len(picked)} aligned agents from {len(candidates)} candidates:")
    for trial, agent in picked:
        bs = agent["baseline_stance"]
        rs = agent["round_stances"][-1]
        ss = agent.get("shadow_stance", "?")
        print(f"  trial {trial['trial_id'][:30]}, agent {agent['agent_id']}: "
              f"baseline={bs}, round_4={rs}, shadow={ss}")
    return picked


# =============================================================================
# Sweep
# =============================================================================

def expected_value(probs: dict[int, float]) -> float:
    return sum(i * p for i, p in probs.items())


async def sweep_one_agent(
    handle: SteeringHandle,
    trial: dict,
    agent: dict,
    scenarios: dict,
    alphas: list[float],
) -> list[dict]:
    label = f"{trial['scenario_id']}_a{agent['agent_id']}"
    rows = []
    messages = reconstruct_round4_messages(trial, agent, scenarios)
    for alpha in alphas:
        text, tokens, probs = await handle.call_steered(
            messages=messages,
            temperature=TEMPERATURE,
            seed=SEED,
            model_name=MODEL_NAME,
            max_tokens=512,
            alpha=float(alpha),
        )
        if probs is None:
            ev = float("nan")
            entropy = float("nan")
        else:
            ev = expected_value(probs)
            entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        rows.append({
            "label": label,
            "scenario_id": trial["scenario_id"],
            "agent_id": agent["agent_id"],
            "baseline_stance": agent["baseline_stance"],
            "round_4_stance_actual": agent["round_stances"][-1],
            "shadow_stance_actual": agent.get("shadow_stance"),
            "alpha": float(alpha),
            "ev": ev,
            "entropy": entropy,
            "probs": probs,
        })
        print(
            f"  α={alpha:+.1f}: EV={ev:.3f}, entropy={entropy:.3f}, "
            f"text starts: {text[:60]!r}"
        )
    return rows


async def main():
    print("Loading scenarios...")
    scenarios = load_dataset_scenarios("moral_stories")
    if not isinstance(scenarios, dict):
        scenarios = {
            (s.get("scenario_id") or s.get("id")): s for s in scenarios
        }
    print(f"  loaded {len(scenarios)} scenarios")

    print("\nPicking test agents...")
    test_agents = pick_test_agents(TRIAL_JSONL, N_TRIALS_TO_TEST)

    print(f"\nLoading SteeringHandle...")
    handle = SteeringHandle(
        model_name=MODEL_NAME,
        direction_path=DIRECTION_PATH,
        target_layer=TARGET_LAYER,
    )
    print(f"||d|| = {handle.direction_norm:.3f}")
    print(f"Sweeping alphas: {ALPHAS}\n")

    all_rows = []
    for trial, agent in test_agents:
        label = f"{trial['scenario_id']}_a{agent['agent_id']}"
        print(f"\n=== {label} (round_4 prompt) ===")
        t0 = time.time()
        rows = await sweep_one_agent(handle, trial, agent, scenarios, ALPHAS)
        all_rows.extend(rows)
        print(f"  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    df_out = df.drop(columns=["probs"]).copy()
    df_out["probs_json"] = df["probs"].apply(
        lambda p: json.dumps(p) if p else None
    )
    df_out.to_parquet(OUT_DIR / "alpha_sweep_v2.parquet")

    plot_curves(df, OUT_DIR / "alpha_sweep_v2.png")
    print(f"\nSaved → {OUT_DIR}")
    print_verdict(df)


def plot_curves(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for label in df["label"].unique():
        s = df[df["label"] == label].sort_values("alpha")
        axes[0].plot(s["alpha"], s["ev"], marker="o", label=label, linewidth=1.3)
    axes[0].axvline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Steering coefficient α (units of ||d||)")
    axes[0].set_ylabel("Expected stance value (1=A, 7=B)")
    axes[0].set_title("Stance EV vs steering magnitude (real round_4 prompts)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    for label in df["label"].unique():
        s = df[df["label"] == label].sort_values("alpha")
        ev_at_zero = s[s["alpha"] == 0]["ev"].iloc[0]
        s = s.copy()
        s["delta_ev"] = s["ev"] - ev_at_zero
        axes[1].plot(s["alpha"], s["delta_ev"], marker="o", label=label, linewidth=1.3)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].axvline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.10, color="green", linestyle=":", linewidth=0.8, label="±0.10 threshold")
    axes[1].axhline(-0.10, color="green", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Steering coefficient α (units of ||d||)")
    axes[1].set_ylabel("ΔEV vs α=0")
    axes[1].set_title("Steering response curve (real round_4 prompts)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.suptitle(
        "Alpha sweep v2: REAL round_4 deliberation prompts\n"
        f"(Qwen-7B, layer {TARGET_LAYER}, direction = layer 26 round_4 diff-of-means)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_verdict(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("OBSERVATIONS")
    print("=" * 64)
    for label in df["label"].unique():
        s = df[df["label"] == label].sort_values("alpha")
        ev0 = s[s["alpha"] == 0]["ev"].iloc[0]
        ev_pos_max = s[s["alpha"] > 0]["ev"].max()
        ev_neg_max = s[s["alpha"] < 0]["ev"].min()
        delta_pos = ev_pos_max - ev0
        delta_neg = ev_neg_max - ev0
        print(f"\n  {label}:")
        print(f"    EV(α=0)         = {ev0:.3f}")
        print(f"    max |ΔEV| pos α = {delta_pos:+.3f}")
        print(f"    max |ΔEV| neg α = {delta_neg:+.3f}")

        # First α with |ΔEV| ≥ 0.1, in either direction
        for col, target in [("+", 0.10), ("-", -0.10)]:
            for alpha_val in sorted(s["alpha"].unique()):
                ev = s[s["alpha"] == alpha_val]["ev"].iloc[0]
                if (col == "+" and ev - ev0 >= 0.10) or (col == "-" and ev - ev0 <= -0.10):
                    print(f"    First α with ΔEV{col}0.10: α={alpha_val:+}, ΔEV={ev - ev0:+.3f}")
                    break


if __name__ == "__main__":
    asyncio.run(main())