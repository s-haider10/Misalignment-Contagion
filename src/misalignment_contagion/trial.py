"""Core trial execution: baseline -> deliberation -> shadow."""

from __future__ import annotations

import asyncio
import logging
import time

from openai import AsyncOpenAI

from .agents import Agent, create_agents
from .config import MAX_TOKENS, N_ROUNDS, TrialConfig
from .io_utils import serialize_trial
from .llm import call_llm_with_logprobs, get_client, get_model_name
from .prompts import (
    build_baseline_messages,
    build_deliberation_messages,
    build_shadow_messages,
    parse_response,
)
from .topology import get_visible_agents

logger = logging.getLogger(__name__)


async def run_trial(
    config: TrialConfig,
    clients: list[AsyncOpenAI],
    scenarios: dict[str, dict],
) -> dict:
    """Execute a complete 3-stage trial and return the serialized result."""
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

    # ── Stage I: Baseline ─────────────────────────────────────────────
    baseline_tasks = []
    for a in agents:
        client = get_client(clients, a.agent_id, a.role, config.model_condition)
        agent_model = get_model_name(config.model_key, a.role, config.model_condition)
        msgs = build_baseline_messages(a.system_prompt, scenario)
        baseline_tasks.append(
            call_llm_with_logprobs(client, msgs, config.temperature, config.seed, agent_model, MAX_TOKENS)
        )

    baseline_results = await asyncio.gather(*baseline_tasks)

    for a, (text, tokens, probs) in zip(agents, baseline_results):
        total_tokens += tokens
        a.baseline_probs = probs
        stance, reasoning = parse_response(text)
        if stance is None:
            parse_failures += 1
            stance = 4  # neutral fallback for baseline
        a.baseline_stance = stance
        a.baseline_reasoning = reasoning

    # ── Stage II: Deliberation (5 rounds) ─────────────────────────────
    round_history: dict[int, dict[int, tuple[int, str]]] = {
        -1: {
            i: (ag.baseline_stance, ag.baseline_reasoning)
            for i, ag in enumerate(agents)
        }
    }

    for round_num in range(N_ROUNDS):
        round_seed = config.seed + round_num + 1

        delib_tasks = []
        for idx, a in enumerate(agents):
            visible = get_visible_agents(
                config.topology, idx, agents, round_history,
                current_round=round_num,
            )

            client = get_client(clients, a.agent_id, a.role, config.model_condition)
            agent_model = get_model_name(config.model_key, a.role, config.model_condition)
            msgs = build_deliberation_messages(a.system_prompt, scenario, visible)
            delib_tasks.append(
                call_llm_with_logprobs(client, msgs, config.temperature, round_seed, agent_model, MAX_TOKENS)
            )

        round_results = await asyncio.gather(*delib_tasks)

        round_history[round_num] = {}
        for idx, (a, (text, tokens, probs)) in enumerate(zip(agents, round_results)):
            total_tokens += tokens
            a.round_probs.append(probs)
            stance, reasoning = parse_response(text)
            if stance is None:
                parse_failures += 1
                if a.round_stances:
                    stance = a.round_stances[-1]
                else:
                    stance = a.baseline_stance
            a.round_stances.append(stance)
            a.round_responses.append(reasoning)
            round_history[round_num][idx] = (stance, reasoning)

    # ── Stage III: Shadow (aligned only, fresh context) ───────────────
    aligned = [(idx, a) for idx, a in enumerate(agents) if a.role == "aligned"]

    shadow_tasks = []
    for idx, a in aligned:
        client = get_client(clients, a.agent_id, a.role, config.model_condition)
        agent_model = get_model_name(config.model_key, a.role, config.model_condition)
        last_stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
        last_reasoning = a.round_responses[-1] if a.round_responses else a.baseline_reasoning
        msgs = build_shadow_messages(scenario, last_stance, last_reasoning)
        shadow_tasks.append(
            call_llm_with_logprobs(client, msgs, config.temperature, config.seed + 99, agent_model, MAX_TOKENS)
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

    # ── Serialize ─────────────────────────────────────────────────────
    wall_time = time.time() - t_start
    if parse_failures > 5:
        logger.warning(
            "Trial %s had %d parse failures", config.trial_id, parse_failures
        )

    return serialize_trial(
        config, agents, total_tokens, wall_time, parse_failures, config.model_key
    )
