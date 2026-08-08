"""
Pilot activation extraction for misalignment contagion mech-interp.

Scope:
  - Model: Qwen2.5-7B-Instruct
  - Subset: synthetic dataset, FC topology, 20% minority, seed 42, aligned agents only
  - Stages: baseline, r0, r1, r2, r3, r4, shadow  (7 stages per agent)
  - Layers: all 28 residual stream layers (post-block output)
  - Position: the last input token position, where the model would emit the
    stance digit next during generation.
  - Format: Parquet, fp16 activations stored as raw bytes

Output: a Parquet file at OUTPUT_PARQUET, plus optional HF Hub upload.

Before running, fill in TODO_1, TODO_2, TODO_3 below.
"""

import gc
import glob
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your repo's modules
from misalignment_contagion.prompts import (
    build_baseline_messages,
    build_deliberation_messages,
    build_shadow_messages,
    get_system_prompt,
)
from misalignment_contagion.topology import get_visible_agents
from misalignment_contagion.io_utils import load_dataset_scenarios


# =============================================================================
# CONFIG  --  fill in TODO_1..3, then run
# =============================================================================

# TODO: glob pattern matching your local moral_stories trial JSONL.
# Likely: /home/haider/Misalignment-Contagion/outputs/primary_em/moral_stories/qwen-7b-instruct/results.jsonl
TRIAL_DATA_GLOB = "/home/haider/Misalignment-Contagion/outputs/primary_em/moral_stories/qwen-7b-instruct/results.jsonl"  # <-- VERIFY

# HF dataset repo id, or None to skip upload.
HF_REPO_ID: Optional[str] = "s-haider/contagion-activations-pilot"
HF_PRIVATE = True

# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.float16
DEVICE = "cuda"

# Output
OUTPUT_DIR = Path("/home/haider/Misalignment-Contagion/outputs/activations_pilot")
OUTPUT_PARQUET = OUTPUT_DIR / "pilot_qwen7b_moralstories_fc_20pct_100trials.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pilot subset filter — only trials matching ALL of these are processed.
# Seeds are a *set* (multiple values accepted) since equality-on-list won't work.
PILOT_FILTER = dict(
    dataset="moral_stories",
    minority_ratio=0.2,
    model_key="qwen-7b-instruct",
)
PILOT_SEEDS = {42, 123}  # exactly two seeds -> 100 trials at FC + 20%
# Topology field is stored as "fc" in trial JSONs.
PILOT_TOPOLOGIES = {"fc"}

STAGES = ["baseline", "round_0", "round_1", "round_2", "round_3", "round_4", "shadow"]
EXPECTED_NUM_LAYERS = 28


# =============================================================================
# Trial loading & filtering
# =============================================================================

def load_pilot_trials(glob_pattern: str, filt: dict) -> list[dict]:
    """Load trials from JSONL files (one trial per line) matching the filter.
    Also requires seed in PILOT_SEEDS and topology in PILOT_TOPOLOGIES."""
    paths = glob.glob(glob_pattern, recursive=True)
    trials = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                if not all(t.get(k) == v for k, v in filt.items()):
                    continue
                if t.get("seed") not in PILOT_SEEDS:
                    continue
                top = t.get("topology", "").lower()
                if top in PILOT_TOPOLOGIES:
                    trials.append(t)
    print(f"Loaded {len(trials)} trials matching pilot filter from {len(paths)} file(s).")
    # Quick per-seed breakdown so the user can verify counts
    if trials:
        from collections import Counter
        seed_counts = Counter(t["seed"] for t in trials)
        for s, n in sorted(seed_counts.items()):
            print(f"  seed={s}: {n} trials")
    return trials


# =============================================================================
# Prompt reconstruction per stage
# =============================================================================

@dataclass
class AgentShim:
    """Minimal stand-in for the real Agent class. Carries the fields that
    get_visible_agents needs to read.
    """
    agent_id: int
    position_in_topology: int
    role: str
    baseline_stance: int
    baseline_reasoning: str


def build_agent_shims_sorted(trial: dict) -> list[AgentShim]:
    """Build a list of AgentShim sorted by position_in_topology, so that
    list index == agent_idx (the index used by get_visible_agents).
    """
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
    """Reconstruct the round_history dict that the original run had at the
    start of round `up_to_round`.

    Mirrors trial.py:
      - Key -1 holds baselines for every agent.
      - Key r (>=0) holds round r responses.
      - Keys are the *agent_idx* (position-in-topology), not agent_id.

    `up_to_round` is the round currently being generated; history contains
    keys -1 .. up_to_round - 1.
    """
    # Map agent_id -> agent_idx via position_in_topology
    by_idx = {}
    for a in trial["agents"]:
        by_idx[a["position_in_topology"]] = a

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


def reconstruct_prompt(
    trial: dict,
    agent: dict,
    stage: str,
    scenarios: dict[str, dict],
) -> list[dict]:
    """Return the chat-message list that this agent saw at this stage.

    For deliberation stages, we call your `get_visible_agents` to get back
    the (agent_id, stance, text) tuples directly — same code path as the
    original run, including the HISTORY_WINDOW=2 behaviour.
    """
    scenario = scenarios[trial["scenario_id"]]
    role = agent["role"]
    system_prompt = get_system_prompt(
        role=role,
        model_condition=trial["model_condition"],
        prompt_strategy=trial["prompt_strategy"],
    )

    if stage == "baseline":
        return build_baseline_messages(system_prompt, scenario)

    if stage.startswith("round_"):
        r = int(stage.split("_")[1])
        agents_sorted = build_agent_shims_sorted(trial)  # index == agent_idx
        # agent_idx for the agent we're prompting:
        agent_idx = agent["position_in_topology"]
        round_history = build_round_history(trial, up_to_round=r)
        visible_responses = get_visible_agents(
            topology=trial["topology"],
            agent_idx=agent_idx,
            agents=agents_sorted,
            round_history=round_history,
            current_round=r,
        )
        return build_deliberation_messages(
            system_prompt=system_prompt,
            scenario=scenario,
            visible_responses=visible_responses,
        )

    if stage == "shadow":
        last_stance = agent["round_stances"][-1]
        last_reasoning = agent["round_responses"][-1]
        return build_shadow_messages(
            scenario=scenario,
            last_stance=last_stance,
            last_reasoning=last_reasoning,
        )

    raise ValueError(f"Unknown stage: {stage}")


# =============================================================================
# Tokenization
# =============================================================================

def messages_to_input_ids(tokenizer, messages: list[dict]) -> torch.Tensor:
    """Apply chat template with `add_generation_prompt=True`, matching how
    vLLM would have prompted the model during the original runs. The last
    token will be the assistant turn opener; the next predicted token would
    be the first token of the assistant's response.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids


# =============================================================================
# Hooks
# =============================================================================

class ActivationCatcher:
    """Capture residual-stream output of every transformer block at the last token."""
    def __init__(self, model):
        self.cache: dict[int, torch.Tensor] = {}
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            def make_hook(idx):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self.cache[idx] = h[0, -1, :].detach().to(torch.float16).cpu()
                return hook
            self.handles.append(layer.register_forward_hook(make_hook(i)))

    def reset(self):
        self.cache = {}

    def remove(self):
        for h in self.handles:
            h.remove()


# =============================================================================
# Main
# =============================================================================

def main(limit: Optional[int] = None, output_path: Optional[Path] = None):
    output_parquet = output_path or OUTPUT_PARQUET
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    assert num_layers == EXPECTED_NUM_LAYERS, \
        f"Expected {EXPECTED_NUM_LAYERS} layers, got {num_layers}"
    print(f"Model: {num_layers} layers, hidden_dim={hidden_dim}")

    scenarios_raw = load_dataset_scenarios(PILOT_FILTER["dataset"])
    # Normalize to dict[scenario_id -> scenario] regardless of return type.
    if isinstance(scenarios_raw, dict):
        scenarios = scenarios_raw
    else:
        scenarios = {}
        for s in scenarios_raw:
            sid = s.get("scenario_id") or s.get("id")
            if sid is None:
                raise ValueError(f"Scenario missing scenario_id/id: {s}")
            scenarios[sid] = s
    print(f"Loaded {len(scenarios)} scenarios for dataset={PILOT_FILTER['dataset']!r}")

    trials = load_pilot_trials(TRIAL_DATA_GLOB, PILOT_FILTER)
    if limit is not None:
        trials = trials[:limit]
        print(f"DEBUG MODE: limited to {len(trials)} trial(s)")
    if not trials:
        raise SystemExit("No trials matched the pilot filter. Check TRIAL_DATA_GLOB and field names.")

    catcher = ActivationCatcher(model)

    schema = pa.schema([
        ("trial_id", pa.string()),
        ("scenario_id", pa.string()),
        ("agent_id", pa.int32()),
        ("position_in_topology", pa.int32()),
        ("stage", pa.string()),
        ("baseline_stance", pa.int32()),
        ("current_stance", pa.int32()),
        ("layer", pa.int32()),
        ("hidden_dim", pa.int32()),
        ("dtype", pa.string()),
        ("activation_bytes", pa.binary()),
    ])
    writer = pq.ParquetWriter(output_parquet, schema, compression="snappy")

    rows_buffer = []
    BUFFER_SIZE = 5000
    total_rows = 0

    def flush():
        nonlocal rows_buffer, total_rows
        if not rows_buffer:
            return
        table = pa.Table.from_pylist(rows_buffer, schema=schema)
        writer.write_table(table)
        total_rows += len(rows_buffer)
        rows_buffer = []

    pbar = tqdm(trials, desc="Trials")
    for trial in pbar:
        aligned = [a for a in trial["agents"] if a["role"] == "aligned"]
        for agent in aligned:
            for stage in STAGES:
                # Current stance at this stage (for downstream filtering)
                if stage == "baseline":
                    cur_stance = agent["baseline_stance"]
                elif stage.startswith("round_"):
                    r = int(stage.split("_")[1])
                    cur_stance = agent["round_stances"][r]
                else:
                    cur_stance = agent.get("shadow_stance", -1)

                messages = reconstruct_prompt(trial, agent, stage, scenarios)
                input_ids = messages_to_input_ids(tokenizer, messages).to(DEVICE)

                catcher.reset()
                with torch.no_grad():
                    _ = model(input_ids=input_ids, use_cache=False)

                for layer_idx in range(num_layers):
                    act = catcher.cache[layer_idx]
                    rows_buffer.append({
                        "trial_id": trial["trial_id"],
                        "scenario_id": trial["scenario_id"],
                        "agent_id": int(agent["agent_id"]),
                        "position_in_topology": int(agent["position_in_topology"]),
                        "stage": stage,
                        "baseline_stance": int(agent["baseline_stance"]),
                        "current_stance": int(cur_stance),
                        "layer": layer_idx,
                        "hidden_dim": int(hidden_dim),
                        "dtype": "float16",
                        "activation_bytes": act.numpy().tobytes(),
                    })

                if len(rows_buffer) >= BUFFER_SIZE:
                    flush()

        pbar.set_postfix(rows=total_rows + len(rows_buffer))

    flush()
    writer.close()
    catcher.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nDone. Wrote {total_rows} rows to {output_parquet}")
    print(f"File size: {output_parquet.stat().st_size / 1e6:.2f} MB")

    if HF_REPO_ID and limit is None:
        upload_to_hf(output_parquet)
    elif HF_REPO_ID and limit is not None:
        print("Skipping HF upload (debug mode).")


def upload_to_hf(parquet_path: Path):
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(HF_REPO_ID, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")


def decode_activation(row) -> np.ndarray:
    """Reverse of activation_bytes encoding: returns shape (hidden_dim,) fp16."""
    return np.frombuffer(row["activation_bytes"], dtype=np.float16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N trials (for debugging). Default: all.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override the output parquet path. Default: pilot path from config.",
    )
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    main(limit=args.limit, output_path=out)