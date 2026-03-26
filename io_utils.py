"""JSONL I/O, trial serialization, and resumption helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from config import TrialConfig

if __name__ != "__main__":
    from agents import Agent


def load_scenarios(path: str = "scenarios.json") -> dict[str, dict]:
    """Load scenarios.json and return a dict keyed by scenario ID."""
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}


def load_completed_trial_ids(output_path: str) -> set[str]:
    """Read completed trial IDs from a JSONL file for resumption."""
    ids: set[str] = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(obj["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # skip truncated/corrupt lines
    return ids


def filter_queue(
    queue: list[TrialConfig], completed_ids: set[str]
) -> list[TrialConfig]:
    """Return only trials not yet completed."""
    return [t for t in queue if t.trial_id not in completed_ids]


def serialize_trial(
    config: TrialConfig,
    agents: list[Agent],
    total_tokens: int,
    wall_time: float,
    parse_failures: int,
    model_scale: str,
) -> dict:
    """Build the output dict for one trial (matches engineering plan Section 8)."""
    from llm import ALIGNED_MODELS, MISALIGNED_MODEL_PATHS

    agent_records = []
    for a in agents:
        rec = {
            "agent_id": a.agent_id,
            "role": a.role,
            "position_in_topology": a.position_in_topology,
            "baseline_stance": a.baseline_stance,
            "baseline_reasoning": a.baseline_reasoning,
            "round_stances": a.round_stances,
            "round_responses": a.round_responses,
        }
        if a.role == "aligned":
            rec["shadow_stance"] = a.shadow_stance
            rec["shadow_reasoning"] = a.shadow_reasoning
        # Logprob distributions (present only when collect_logprobs=True)
        if a.baseline_probs is not None:
            rec["baseline_probs"] = a.baseline_probs
            rec["round_probs"] = a.round_probs
            if a.role == "aligned" and a.shadow_probs is not None:
                rec["shadow_probs"] = a.shadow_probs
        agent_records.append(rec)

    aligned_model = ALIGNED_MODELS[model_scale]
    if config.model_condition == "model_induced":
        minority_model = MISALIGNED_MODEL_PATHS[model_scale]
    else:
        minority_model = aligned_model

    return {
        "trial_id": config.trial_id,
        "scenario_id": config.scenario_id,
        "topology": config.topology,
        "minority_ratio": config.minority_ratio,
        "position_config": config.position_config,
        "temperature": config.temperature,
        "seed": config.seed,
        "model_condition": config.model_condition,
        "agents": agent_records,
        "metadata": {
            "aligned_model": aligned_model,
            "minority_model": minority_model,
            "model_scale": model_scale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tokens": total_tokens,
            "wall_time_seconds": round(wall_time, 2),
            "parse_failures": parse_failures,
        },
    }


def append_trial(result: dict, output_path: str) -> None:
    """Append one trial result as a JSONL line. Creates parent dirs if needed."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
