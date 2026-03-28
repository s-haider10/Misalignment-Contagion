"""JSONL I/O, trial serialization, dataset loading, and resumption helpers."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from .config import DATASETS, PROJECT_ROOT, TrialConfig

if __name__ != "__main__":
    from .agents import Agent

logger = logging.getLogger(__name__)


# ── Scenario / dataset loading ───────────────────────────────────────

def load_scenarios(path: str | None = None) -> dict[str, dict]:
    """Load scenarios.json and return a dict keyed by scenario ID."""
    if path is None:
        path = str(PROJECT_ROOT / "data" / "synthetic" / "scenarios.json")
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}


def load_dataset_scenarios(dataset: str) -> dict[str, dict]:
    """Load scenarios from a named dataset.

    For 'synthetic', loads the original scenarios.json.
    For benchmark datasets (ETHICS, MIC, Moral Stories, HarmBench),
    loads from data/<name>/<file>.jsonl. Each JSONL line must have at minimum:
        {"id": "...", "context": "...", "position_a": "...", "position_b": "..."}
    """
    if dataset == "synthetic":
        return load_scenarios()

    rel_path = DATASETS.get(dataset)
    if rel_path is None:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {', '.join(DATASETS.keys())}"
        )

    path = str(PROJECT_ROOT / rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Run: python scripts/prepare_data.py --dataset {dataset}"
        )

    scenarios = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj["id"]
            scenarios[sid] = obj
    logger.info("Loaded %d scenarios from %s", len(scenarios), path)
    return scenarios


# ── Resumption helpers ───────────────────────────────────────────────

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


# ── Serialization ────────────────────────────────────────────────────

def serialize_trial(
    config: TrialConfig,
    agents: list[Agent],
    total_tokens: int,
    wall_time: float,
    parse_failures: int,
    model_key: str,
) -> dict:
    """Build the output dict for one trial."""
    from .llm import get_aligned_model, get_misaligned_model_path

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
        if a.baseline_probs is not None:
            rec["baseline_probs"] = a.baseline_probs
            rec["round_probs"] = a.round_probs
            if a.role == "aligned" and a.shadow_probs is not None:
                rec["shadow_probs"] = a.shadow_probs
        agent_records.append(rec)

    aligned_model = get_aligned_model(model_key)
    if config.model_condition == "model_induced":
        minority_model = get_misaligned_model_path(model_key)
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
        "dataset": config.dataset,
        "model_key": config.model_key,
        "prompt_strategy": config.prompt_strategy,
        "agents": agent_records,
        "metadata": {
            "aligned_model": aligned_model,
            "minority_model": minority_model,
            "model_key": model_key,
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
