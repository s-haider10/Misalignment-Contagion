"""Steering deliberation experiment v3 — RANDOM-direction control.

Same machinery as v2, but uses a norm-matched random direction at layer 26
instead of the round_2 diff-of-means direction. If v2 (real direction) shows
a measurable effect that v3 (random) does not, that's evidence the v2 effect
is direction-specific rather than a generic perturbation artifact.

Conditions (mirroring v2):
  - control:  no steering anywhere
  - neg_r2:   α=-1 on round 2 only, using RANDOM direction

Same scenarios + seed as v2 for direct paired comparison.

Routing:
  - Aligned agents go through SteeringHandle (in-process HF on GPU 2 via
    CUDA_VISIBLE_DEVICES=2 at launch time).
  - Misaligned agents go through vLLM at :8005 (LoRA on GPU 1).

Output: trials_v3.jsonl + summary_v3.csv in
        outputs/steering_experiment_v3_random_control/
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from misalignment_contagion.agents import Agent, create_agents
from misalignment_contagion.config import MAX_TOKENS, N_ROUNDS, TrialConfig
from misalignment_contagion.io_utils import load_dataset_scenarios
from misalignment_contagion.llm import (
    call_llm_with_logprobs,
    get_model_name,
)
from misalignment_contagion.mech_interp.llm_steered import (
    init_global_handle,
    call_llm_with_logprobs_steered,
)
from misalignment_contagion.prompts import (
    build_baseline_messages,
    build_deliberation_messages,
    build_shadow_messages,
    parse_response,
)
from misalignment_contagion.topology import get_visible_agents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DIRECTION_PATH = Path(
    "/home/haider/Misalignment-Contagion/outputs/direction_results/"
    "direction_layer26_random_matched.npy"
)
TARGET_LAYER = 26

OUTCOMES_PARQUET = Path(
    "/home/haider/Misalignment-Contagion/outputs/probe_results_v3/"
    "probe_v3_outcomes.parquet"
)
OUT_DIR = Path(
    "/home/haider/Misalignment-Contagion/outputs/steering_experiment_v3_random_control"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# vLLM endpoints — see launch script. v3 uses the LoRA server on :8005 (GPU 1).
MISALIGNED_BASE_URL = "http://localhost:8005/v1"

CONDITIONS = [
    {"label": "control", "alpha":  0.0, "rounds": set()},
    {"label": "neg_r2",  "alpha": -1.0, "rounds": {2}},
]
N_TRIALS = 15
SEED = 42
TEMPERATURE = 0.7
TOPOLOGY = "fc"
MINORITY_RATIO = 0.2
MODEL_KEY = "qwen-7b-instruct"
MODEL_CONDITION = "model_induced"
PROMPT_STRATEGY = "rigid:rigid"


# =============================================================================
# Scenario selection — IDENTICAL to v1/v2 (median-II) for paired comparability
# =============================================================================

def pick_median_ii_scenarios(n: int) -> list[str]:
    """Outcomes parquet keys by trial_id (e.g. "MS-10477_fc_..."); the
    scenario id is the prefix before the first underscore.
    """
    outcomes = pd.read_parquet(OUTCOMES_PARQUET)
    outcomes = outcomes.assign(
        scenario_id=outcomes["trial_id"].str.split("_", n=1).str[0]
    )
    valid = outcomes.dropna(subset=["ii"])
    valid = valid[valid["ii"].abs() < 10]
    per_scenario = valid.groupby("scenario_id")["ii"].mean().reset_index()
    median = per_scenario["ii"].median()
    per_scenario["dist"] = (per_scenario["ii"] - median).abs()
    picked = per_scenario.nsmallest(n, "dist")["scenario_id"].tolist()
    logger.info("Selected %d scenarios (median mean_ii = %.3f)", n, median)
    return picked


# =============================================================================
# Per-condition trial runner
# =============================================================================

async def run_steered_trial(
    config: TrialConfig,
    misaligned_client: AsyncOpenAI,
    scenarios: dict[str, dict],
    condition: dict[str, Any],
) -> dict:
    t_start = time.time()
    scenario = scenarios[config.scenario_id]
    agents = create_agents(
        n=10,
        minority_ratio=config.minority_ratio,
        topology=config.topology,
        position_config=config.position_config,
        model_condition=config.model_condition,
        prompt_strategy=config.prompt_strategy,
    )

    total_tokens = 0
    parse_failures = 0
    cond_alpha = float(condition["alpha"])
    cond_rounds = set(condition["rounds"])

    def steer_alpha_for(role: str, round_num: int | None) -> float:
        if role != "aligned":
            return 0.0
        if round_num is None:
            return 0.0
        if round_num in cond_rounds:
            return cond_alpha
        return 0.0

    # Stage I: Baseline
    baseline_tasks = []
    for a in agents:
        agent_model = get_model_name(MODEL_KEY, a.role, config.model_condition)
        msgs = build_baseline_messages(a.system_prompt, scenario)
        if a.role == "aligned":
            baseline_tasks.append(
                call_llm_with_logprobs_steered(
                    client=None, messages=msgs, temperature=config.temperature,
                    seed=config.seed, model_name=agent_model, max_tokens=MAX_TOKENS,
                    alpha=0.0,
                )
            )
        else:
            baseline_tasks.append(
                call_llm_with_logprobs(
                    misaligned_client, msgs, config.temperature, config.seed,
                    agent_model, MAX_TOKENS,
                )
            )
    baseline_results = await asyncio.gather(*baseline_tasks)
    for a, (text, tokens, probs) in zip(agents, baseline_results):
        total_tokens += tokens
        a.baseline_probs = probs
        stance, reasoning = parse_response(text)
        if stance is None:
            parse_failures += 1
            stance = 4
        a.baseline_stance = stance
        a.baseline_reasoning = reasoning

    # Stage II: Deliberation
    round_history = {-1: {i: (ag.baseline_stance, ag.baseline_reasoning)
                          for i, ag in enumerate(agents)}}
    for round_num in range(N_ROUNDS):
        round_seed = config.seed + round_num + 1
        delib_tasks = []
        for idx, a in enumerate(agents):
            visible = get_visible_agents(
                config.topology, idx, agents, round_history, current_round=round_num,
            )
            agent_model = get_model_name(MODEL_KEY, a.role, config.model_condition)
            msgs = build_deliberation_messages(a.system_prompt, scenario, visible)
            alpha = steer_alpha_for(a.role, round_num)
            if a.role == "aligned":
                delib_tasks.append(
                    call_llm_with_logprobs_steered(
                        client=None, messages=msgs, temperature=config.temperature,
                        seed=round_seed, model_name=agent_model, max_tokens=MAX_TOKENS,
                        alpha=alpha,
                    )
                )
            else:
                delib_tasks.append(
                    call_llm_with_logprobs(
                        misaligned_client, msgs, config.temperature, round_seed,
                        agent_model, MAX_TOKENS,
                    )
                )
        round_results = await asyncio.gather(*delib_tasks)
        round_history[round_num] = {}
        for idx, (a, (text, tokens, probs)) in enumerate(zip(agents, round_results)):
            total_tokens += tokens
            a.round_probs.append(probs)
            stance, reasoning = parse_response(text)
            if stance is None:
                parse_failures += 1
                stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
            a.round_stances.append(stance)
            a.round_responses.append(reasoning)
            round_history[round_num][idx] = (stance, reasoning)

    # Stage III: Shadow (aligned only, no steering)
    aligned = [(idx, a) for idx, a in enumerate(agents) if a.role == "aligned"]
    shadow_tasks = []
    for idx, a in aligned:
        agent_model = get_model_name(MODEL_KEY, a.role, config.model_condition)
        last_stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
        last_reasoning = a.round_responses[-1] if a.round_responses else a.baseline_reasoning
        msgs = build_shadow_messages(scenario, last_stance, last_reasoning)
        shadow_tasks.append(
            call_llm_with_logprobs_steered(
                client=None, messages=msgs, temperature=config.temperature,
                seed=config.seed + 99, model_name=agent_model, max_tokens=MAX_TOKENS,
                alpha=0.0,
            )
        )
    shadow_results = await asyncio.gather(*shadow_tasks)
    for (idx, a), (text, tokens, probs) in zip(aligned, shadow_results):
        total_tokens += tokens
        a.shadow_probs = probs
        stance, reasoning = parse_response(text)
        if stance is None:
            parse_failures += 1
            stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
        a.shadow_stance = stance
        a.shadow_reasoning = reasoning

    wall_time = time.time() - t_start

    agent_records = []
    for a in agents:
        agent_records.append({
            "agent_id": a.agent_id,
            "position_in_topology": a.position_in_topology,
            "role": a.role,
            "baseline_stance": a.baseline_stance,
            "baseline_probs": a.baseline_probs,
            "round_stances": list(a.round_stances),
            "round_probs": list(a.round_probs),
            "shadow_stance": a.shadow_stance if a.role == "aligned" else None,
            "shadow_probs": a.shadow_probs if a.role == "aligned" else None,
        })

    return {
        "trial_id": config.trial_id,
        "condition": condition["label"],
        "alpha": cond_alpha,
        "steered_rounds": sorted(cond_rounds),
        "scenario_id": config.scenario_id,
        "agents": agent_records,
        "total_tokens": total_tokens,
        "wall_time_sec": wall_time,
        "parse_failures": parse_failures,
    }


async def main():
    logger.info("Loading scenarios...")
    scenarios = load_dataset_scenarios("moral_stories")
    if not isinstance(scenarios, dict):
        scenarios = {(s.get("scenario_id") or s.get("id")): s for s in scenarios}

    logger.info("Selecting %d median-II scenarios...", N_TRIALS)
    scenario_ids = pick_median_ii_scenarios(N_TRIALS)

    if not DIRECTION_PATH.exists():
        raise FileNotFoundError(
            f"{DIRECTION_PATH} not found. Run find_direction_random.py first."
        )
    logger.info("Initializing SteeringHandle with RANDOM direction at %s", DIRECTION_PATH.name)
    init_global_handle(
        model_name=MODEL_NAME,
        direction_path=DIRECTION_PATH,
        target_layer=TARGET_LAYER,
    )
    logger.info("Connecting to misaligned vLLM at %s", MISALIGNED_BASE_URL)
    misaligned_client = AsyncOpenAI(base_url=MISALIGNED_BASE_URL, api_key="not-needed")

    jsonl_path = OUT_DIR / "trials_v3.jsonl"
    summary_path = OUT_DIR / "summary_v3.csv"

    n_total = N_TRIALS * len(CONDITIONS)
    logger.info("Running %d trials (%d scenarios × %d conditions)",
                n_total, N_TRIALS, len(CONDITIONS))

    t_overall = time.time()
    completed = 0
    with open(jsonl_path, "w") as fout:
        for scenario_id in scenario_ids:
            for cond in CONDITIONS:
                cfg = TrialConfig(
                    scenario_id=scenario_id,
                    dataset="moral_stories",
                    topology=TOPOLOGY,
                    minority_ratio=MINORITY_RATIO,
                    position_config=0,
                    model_key=MODEL_KEY,
                    model_condition=MODEL_CONDITION,
                    prompt_strategy=PROMPT_STRATEGY,
                    seed=SEED,
                    temperature=TEMPERATURE,
                )
                t0 = time.time()
                try:
                    result = await run_steered_trial(cfg, misaligned_client, scenarios, cond)
                    fout.write(json.dumps(result, default=str) + "\n")
                    fout.flush()
                    completed += 1
                    elapsed = time.time() - t_overall
                    eta = (elapsed / completed) * (n_total - completed)
                    logger.info(
                        "[%d/%d] scenario %s, %s: %.1fs (ETA %.1f min)",
                        completed, n_total, scenario_id, cond["label"],
                        time.time() - t0, eta / 60,
                    )
                except Exception as e:
                    logger.exception("Trial failed: %s", e)
                    continue

    logger.info("All trials done in %.1f min", (time.time() - t_overall) / 60)

    rows = []
    with open(jsonl_path) as fin:
        for line in fin:
            tr = json.loads(line)
            shifts = []
            for a in tr["agents"]:
                if not (a["baseline_probs"] and a["shadow_probs"]):
                    continue
                ev_b = sum(int(k) * v for k, v in a["baseline_probs"].items())
                ev_s = sum(int(k) * v for k, v in a["shadow_probs"].items())
                shifts.append(ev_s - ev_b)
            rows.append({
                "scenario_id": tr["scenario_id"],
                "condition": tr["condition"],
                "alpha": tr["alpha"],
                "mean_shadow_shift": float(np.mean(shifts)) if shifts else float("nan"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)

    print("\n=== Per-condition shadow shift summary (RANDOM direction) ===")
    print(df.groupby("condition")["mean_shadow_shift"].agg(["mean", "median", "std", "count"]).round(3))

    print("\n=== Paired vs control ===")
    ctrl = df[df.condition == "control"].set_index("scenario_id")
    for cond in ["neg_r2"]:
        s = df[df.condition == cond].set_index("scenario_id")
        common = ctrl.index.intersection(s.index)
        diff = (s.loc[common, "mean_shadow_shift"] - ctrl.loc[common, "mean_shadow_shift"]).dropna()
        print(f"  {cond}: Δshift = {diff.mean():+.3f} (sd {diff.std():.3f}, n={len(diff)})")

    print(f"\nFull results → {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())
