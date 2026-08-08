"""Verify reconstruct_prompt() output is byte-identical to what trial.py produced.

Compares the reconstructed messages (from mech_interp/extract_activations.reconstruct_prompt)
against ground-truth messages built by replaying trial.py's exact logic on a
trial JSON. Tests round_0, round_1, round_2 — the three cases that exercise
the history-window logic in get_visible_agents.

On mismatch, prints a unified diff of the rendered chat-template strings.
"""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/home/haider/Misalignment-Contagion")
sys.path.insert(0, str(REPO))

from misalignment_contagion.io_utils import load_dataset_scenarios
from misalignment_contagion.prompts import build_deliberation_messages
from misalignment_contagion.topology import get_visible_agents

# Load reconstruct_prompt from the pilot script without executing module-level
# side effects (heavy imports, mkdir of /home/claude, etc.).
import ast
import types

_HEAVY_MODULES = {"torch", "pandas", "pyarrow", "pyarrow.parquet", "tqdm", "transformers"}

def _is_heavy(node):
    if isinstance(node, ast.ImportFrom):
        return node.module in _HEAVY_MODULES or (node.module or "").startswith(tuple(_HEAVY_MODULES))
    if isinstance(node, ast.Import):
        return any(a.name in _HEAVY_MODULES or a.name.startswith(tuple(_HEAVY_MODULES)) for a in node.names)
    return False

_pilot_src = (REPO / "misalignment_contagion/mech_interp/extract_activations.py").read_text()
_tree = ast.parse(_pilot_src)
_keep = [
    n for n in _tree.body
    if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
    and not _is_heavy(n)
]
_module = types.ModuleType("pilot")
sys.modules["pilot"] = _module
exec(compile(ast.Module(body=_keep, type_ignores=[]), "<pilot-subset>", "exec"), _module.__dict__)
pilot = _module

TRIAL_FILE = REPO / "outputs/primary_em/moral_stories/qwen-7b-instruct/results.jsonl"


@dataclass
class AgentShim:
    agent_id: int
    position_in_topology: int
    role: str
    baseline_stance: int
    baseline_reasoning: str


def build_ground_truth_messages(trial: dict, agent_idx: int, round_num: int, scenario: dict, system_prompt: str):
    """Replay trial.py:69-91 exactly to build the messages that were actually sent."""
    agents_sorted = sorted(trial["agents"], key=lambda a: a["position_in_topology"])
    agents = [
        AgentShim(
            agent_id=a["agent_id"],
            position_in_topology=a["position_in_topology"],
            role=a["role"],
            baseline_stance=a["baseline_stance"],
            baseline_reasoning=a["baseline_reasoning"],
        )
        for a in agents_sorted
    ]

    round_history: dict[int, dict[int, tuple[int, str]]] = {
        -1: {i: (ag.baseline_stance, ag.baseline_reasoning) for i, ag in enumerate(agents)}
    }
    for r in range(round_num):
        round_history[r] = {
            i: (agents_sorted[i]["round_stances"][r], agents_sorted[i]["round_responses"][r])
            for i in range(len(agents))
        }

    visible = get_visible_agents(
        trial["topology"], agent_idx, agents, round_history, current_round=round_num,
    )
    return build_deliberation_messages(system_prompt, scenario, visible)


def render(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"=== {m['role']} ===\n{m['content']}")
    return "\n\n".join(parts)


def main():
    with open(TRIAL_FILE) as f:
        trial = json.loads(f.readline())

    print(f"Trial: {trial['trial_id']}")
    print(f"  dataset={trial['dataset']} topology={trial['topology']} "
          f"model_condition={trial['model_condition']} prompt_strategy={trial['prompt_strategy']}")

    scenarios = load_dataset_scenarios(trial["dataset"])
    scenario = scenarios[trial["scenario_id"]]

    agents_sorted = sorted(trial["agents"], key=lambda a: a["position_in_topology"])
    target_agent_idx = next(
        (i for i, a in enumerate(agents_sorted) if a["role"] == "aligned"),
        0,
    )
    agent = agents_sorted[target_agent_idx]
    print(f"  agent_idx={target_agent_idx} agent_id={agent['agent_id']} "
          f"role={agent['role']} pos={agent['position_in_topology']}")

    from misalignment_contagion.prompts import get_system_prompt
    system_prompt = get_system_prompt(
        role=agent["role"],
        model_condition=trial["model_condition"],
        prompt_strategy=trial["prompt_strategy"],
    )

    all_match = True
    for round_num in (0, 1, 2):
        print(f"\n--- round_{round_num} ---")
        truth = build_ground_truth_messages(trial, target_agent_idx, round_num, scenario, system_prompt)

        trial_for_pilot = {**trial, "agents": agents_sorted}
        recon = pilot.reconstruct_prompt(trial_for_pilot, agent, f"round_{round_num}", scenarios)

        truth_str = render(truth)
        recon_str = render(recon)

        if truth_str == recon_str:
            print(f"  MATCH ({len(truth_str)} chars)")
        else:
            all_match = False
            print(f"  MISMATCH — diff (truth → reconstructed):")
            diff = difflib.unified_diff(
                truth_str.splitlines(keepends=True),
                recon_str.splitlines(keepends=True),
                fromfile="ground_truth",
                tofile="reconstructed",
                n=3,
            )
            sys.stdout.writelines(diff)

    print("\n" + ("ALL MATCH" if all_match else "MISMATCHES FOUND"))
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
