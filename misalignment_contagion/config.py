"""Trial configuration, constants, and queue builders."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

# Project root: one level up from this file (misalignment_contagion/ -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Experiment constants ──────────────────────────────────────────────
N_AGENTS = 10
N_ROUNDS = 5
MAX_TOKENS = 512
HISTORY_WINDOW = 2

TOPOLOGIES = ["fc", "chain", "circle", "star"]
MINORITY_RATIOS = [0.1, 0.2, 0.3]

# Position configs per topology.
#   chain: 0=head, 1=middle(N//2), 2=tail(N-1)
#   star:  0=minority as leaf, 1=minority as hub
#   fc/circle: single config (positions equivalent)
POSITION_CONFIGS = {
    "fc": [0],
    "chain": [0, 1, 2],
    "circle": [0],
    "star": [0, 1],
}

# ── Prompt strategy grid (2x2) ───────────────────────────────────────
# Keys used in TrialConfig.prompt_strategy: "aligned_variant:misaligned_variant"
PROMPT_STRATEGIES = ["rigid:rigid", "rigid:lenient", "lenient:rigid", "lenient:lenient"]
DEFAULT_PROMPT_STRATEGY = "rigid:rigid"

# ── Dataset registry (paths relative to PROJECT_ROOT) ────────────────
DATASETS = {
    "synthetic": "data/synthetic/scenarios.json",
    "ethics": "data/ethics/ethics.jsonl",
    "mic": "data/mic/mic.jsonl",
    "moral_stories": "data/moral_stories/moral_stories_full.jsonl",
    "harmbench": "data/harmbench/harmbench.jsonl",
}
DEFAULT_DATASET = "synthetic"

# Default output root
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass
class TrialConfig:
    scenario_id: str
    topology: str
    minority_ratio: float
    position_config: int
    temperature: float
    seed: int
    model_condition: str  # "prompt_induced" or "model_induced"
    dataset: str = "synthetic"
    model_key: str = "qwen-7b-instruct"
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY

    @property
    def trial_id(self) -> str:
        return (
            f"{self.scenario_id}_{self.topology}_{self.minority_ratio}"
            f"_pos{self.position_config}_t{self.temperature}"
            f"_s{self.seed}_{self.model_condition}"
            f"_{self.model_key}_{self.prompt_strategy}"
        )


# ── Queue builders ───────────────────────────────────────────────────

def build_primary_queue(
    scenarios: list[dict],
    *,
    model_condition: str = "prompt_induced",
    model_key: str = "qwen-7b-instruct",
    dataset: str = "synthetic",
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    seeds: list[int] | None = None,
) -> list[TrialConfig]:
    """Full grid: topologies x ratios x position ablations x scenarios x seeds."""
    if seeds is None:
        seeds = [42]
    trials = []
    for s, topo, ratio, seed in product(
        scenarios, TOPOLOGIES, MINORITY_RATIOS, seeds
    ):
        for pos in POSITION_CONFIGS[topo]:
            trials.append(TrialConfig(
                scenario_id=s["id"],
                topology=topo,
                minority_ratio=ratio,
                position_config=pos,
                temperature=0.7,
                seed=seed,
                model_condition=model_condition,
                dataset=dataset,
                model_key=model_key,
                prompt_strategy=prompt_strategy,
            ))
    return trials


def build_ablation_queue(
    scenarios: list[dict],
    ablation: str,
    *,
    model_key: str = "qwen-7b-instruct",
    dataset: str = "synthetic",
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
) -> list[TrialConfig]:
    """50 trials each: FC, ratio=0.2, pos=0.

    ablation: one of "t0", "model_induced"
    """
    temp = 0.0 if ablation == "t0" else 0.7
    condition = "model_induced" if ablation == "model_induced" else "prompt_induced"
    seed = 0 if ablation == "t0" else 42
    return [
        TrialConfig(
            scenario_id=s["id"],
            topology="fc",
            minority_ratio=0.2,
            position_config=0,
            temperature=temp,
            seed=seed,
            model_condition=condition,
            dataset=dataset,
            model_key=model_key,
            prompt_strategy=prompt_strategy,
        )
        for s in scenarios
    ]


def build_prompt_sensitivity_queue(
    scenarios: list[dict],
    *,
    model_key: str = "qwen-7b-instruct",
    dataset: str = "synthetic",
    model_condition: str = "prompt_induced",
    topologies: list[str] | None = None,
    ratios: list[float] | None = None,
) -> list[TrialConfig]:
    """2x2 prompt strategy grid on a subset of conditions."""
    if topologies is None:
        topologies = ["fc", "star"]
    if ratios is None:
        ratios = [0.2]
    trials = []
    for s, strategy, topo, ratio in product(
        scenarios, PROMPT_STRATEGIES, topologies, ratios
    ):
        for pos in POSITION_CONFIGS[topo]:
            trials.append(TrialConfig(
                scenario_id=s["id"],
                topology=topo,
                minority_ratio=ratio,
                position_config=pos,
                temperature=0.7,
                seed=42,
                model_condition=model_condition,
                dataset=dataset,
                model_key=model_key,
                prompt_strategy=strategy,
            ))
    return trials


def build_seed_replication(
    scenarios: list[dict],
    conditions: list[dict],
    seeds: list[int] | None = None,
    *,
    model_key: str = "qwen-7b-instruct",
    dataset: str = "synthetic",
) -> list[TrialConfig]:
    """Targeted replication: extra seeds on conditions with signal.

    conditions: list of dicts with keys
        {topology, minority_ratio, position_config, model_condition}
    """
    if seeds is None:
        seeds = [123, 456]
    trials = []
    for cond, s, seed in product(conditions, scenarios, seeds):
        trials.append(TrialConfig(
            scenario_id=s["id"],
            topology=cond["topology"],
            minority_ratio=cond["minority_ratio"],
            position_config=cond["position_config"],
            temperature=0.7,
            seed=seed,
            model_condition=cond["model_condition"],
            dataset=dataset,
            model_key=model_key,
            prompt_strategy=cond.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY),
        ))
    return trials
