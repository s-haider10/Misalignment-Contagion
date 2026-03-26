"""Agent dataclass and population factory."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompts import (
    ALIGNED_SYSTEM_PROMPT,
    MISALIGNED_SYSTEM_PROMPT,
    MODEL_INDUCED_SYSTEM_PROMPT,
)
from topology import assign_topology_positions


@dataclass
class Agent:
    agent_id: int
    role: str  # "aligned" or "misaligned"
    system_prompt: str
    position_in_topology: int = 0
    baseline_stance: int | None = None
    baseline_reasoning: str = ""
    round_stances: list[int | None] = field(default_factory=list)
    round_responses: list[str] = field(default_factory=list)
    shadow_stance: int | None = None
    shadow_reasoning: str = ""
    # Logprob distributions: {1: p, 2: p, ..., 7: p}
    baseline_probs: dict | None = None
    round_probs: list[dict | None] = field(default_factory=list)
    shadow_probs: dict | None = None


def create_agents(
    n: int,
    minority_ratio: float,
    topology: str,
    position_config: int,
    model_condition: str,
) -> list[Agent]:
    """Create N agents with the correct roles, prompts, and topology positions.

    Returns agents sorted by topology position.
    """
    m = round(n * minority_ratio)  # 1, 2, or 3 misaligned agents

    agents = []
    minority_indices = []

    for i in range(n):
        if i < n - m:
            # Aligned majority
            agents.append(Agent(
                agent_id=i,
                role="aligned",
                system_prompt=ALIGNED_SYSTEM_PROMPT,
            ))
        else:
            # Misaligned minority
            minority_indices.append(i)
            if model_condition == "model_induced":
                prompt = MODEL_INDUCED_SYSTEM_PROMPT
            else:
                prompt = MISALIGNED_SYSTEM_PROMPT
            agents.append(Agent(
                agent_id=i,
                role="misaligned",
                system_prompt=prompt,
            ))

    # Assign topology positions and reorder
    agents = assign_topology_positions(
        agents, topology, minority_indices, position_config
    )
    return agents
