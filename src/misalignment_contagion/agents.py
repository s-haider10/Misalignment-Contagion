"""Agent dataclass and population factory."""

from __future__ import annotations

from dataclasses import dataclass, field

from .prompts import get_system_prompt
from .topology import assign_topology_positions


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
    prompt_strategy: str = "rigid:rigid",
) -> list[Agent]:
    """Create N agents with the correct roles, prompts, and topology positions.

    Returns agents sorted by topology position.
    """
    m = round(n * minority_ratio)  # 1, 2, or 3 misaligned agents

    agents = []
    minority_indices = []

    for i in range(n):
        if i < n - m:
            role = "aligned"
        else:
            role = "misaligned"
            minority_indices.append(i)

        agents.append(Agent(
            agent_id=i,
            role=role,
            system_prompt=get_system_prompt(role, model_condition, prompt_strategy),
        ))

    # Assign topology positions and reorder
    agents = assign_topology_positions(
        agents, topology, minority_indices, position_config
    )
    return agents
