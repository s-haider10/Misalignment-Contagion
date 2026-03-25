"""Communication topologies: visibility computation and position assignment."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import Agent

from config import N_AGENTS, HISTORY_WINDOW


# ── Visibility dispatch ───────────────────────────────────────────────

def get_visible_agents(
    topology: str,
    agent_idx: int,
    agents: list[Agent],
    round_history: dict[int, dict[int, tuple[int, str]]],
    current_round: int,
    history_window: int = HISTORY_WINDOW,
) -> list[tuple[int, int, str]]:
    """Return (agent_id, stance, text) tuples visible to agent_idx.

    round_history: {round_num: {agent_idx: (stance, text)}}
    Only includes the last `history_window` rounds.
    """
    n = len(agents)
    hub_idx = _find_hub(agents, topology)

    # Determine which agent indices are visible
    if topology == "fc":
        visible_indices = [i for i in range(n) if i != agent_idx]
    elif topology == "chain":
        visible_indices = [agent_idx - 1] if agent_idx > 0 else []
    elif topology == "circle":
        visible_indices = [(agent_idx - 1) % n, (agent_idx + 1) % n]
    elif topology == "star":
        if agent_idx == hub_idx:
            # Hub sees all leaves
            visible_indices = [i for i in range(n) if i != hub_idx]
        else:
            # Leaves see only the hub
            visible_indices = [hub_idx]
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Collect responses from visible agents over the history window
    start_round = current_round - history_window
    results = []
    for r in range(start_round, current_round):
        if r not in round_history:
            continue
        for idx in visible_indices:
            if idx in round_history[r]:
                stance, text = round_history[r][idx]
                results.append((agents[idx].agent_id, stance, text))
    return results


def _find_hub(agents: list[Agent], topology: str) -> int:
    """Return the index of the hub agent in star topology.

    Convention: the agent whose position_in_topology == 0 is the hub.
    """
    if topology != "star":
        return 0
    for i, a in enumerate(agents):
        if a.position_in_topology == 0:
            return i
    return 0


# ── Position assignment ───────────────────────────────────────────────

def assign_topology_positions(
    agents: list[Agent],
    topology: str,
    minority_indices: list[int],
    position_config: int,
) -> list[Agent]:
    """Assign position_in_topology to each agent and reorder by position.

    minority_indices: indices into the agents list that are misaligned.
    position_config: topology-specific placement code.

    Returns agents sorted by position_in_topology.
    """
    n = len(agents)
    majority_indices = [i for i in range(n) if i not in minority_indices]

    if topology == "fc" or topology == "circle":
        # Positions are equivalent; just number them 0..N-1
        # Place minority at the end for consistency
        ordered = majority_indices + minority_indices
        for pos, idx in enumerate(ordered):
            agents[idx].position_in_topology = pos

    elif topology == "chain":
        # position_config: 0=head, 1=middle, 2=tail
        target_positions = {
            0: list(range(len(minority_indices))),                          # head
            1: list(range(n // 2, n // 2 + len(minority_indices))),         # middle
            2: list(range(n - len(minority_indices), n)),                   # tail
        }
        minority_positions = target_positions[position_config]
        all_positions = list(range(n))
        majority_positions = [p for p in all_positions if p not in minority_positions]
        for idx, pos in zip(minority_indices, minority_positions):
            agents[idx].position_in_topology = pos
        for idx, pos in zip(majority_indices, majority_positions):
            agents[idx].position_in_topology = pos

    elif topology == "star":
        # position_config: 0=minority as leaf, 1=minority as hub
        if position_config == 0:
            # Hub is first majority agent (position 0)
            agents[majority_indices[0]].position_in_topology = 0
            remaining_majority = majority_indices[1:]
            pos = 1
            for idx in remaining_majority:
                agents[idx].position_in_topology = pos
                pos += 1
            for idx in minority_indices:
                agents[idx].position_in_topology = pos
                pos += 1
        else:
            # Hub is first minority agent (position 0)
            agents[minority_indices[0]].position_in_topology = 0
            remaining_minority = minority_indices[1:]
            pos = 1
            for idx in majority_indices:
                agents[idx].position_in_topology = pos
                pos += 1
            for idx in remaining_minority:
                agents[idx].position_in_topology = pos
                pos += 1
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Sort agents by their assigned topology position
    agents.sort(key=lambda a: a.position_in_topology)
    return agents
