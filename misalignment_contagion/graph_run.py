#!/usr/bin/env python3
"""Run deliberation trials against custom graphs loaded from the manifest at
outputs/graph_features/graph_manifest_subset/subset_manifest.json.

Each trial uses:
  - The graph's adjacency (variable n_agents, custom edges from manifest)
  - The graph's preset misaligned positions (no minority_ratio rounding)
  - qwen-7b-instruct base + ModelOrganismsForEM LoRA for misaligned agents

Outputs:
  outputs/graph_features/runs/{dataset}/results.{shard_tag}.jsonl

Shard support: a worker takes --shard-id i --num-shards N and runs trials
whose hash mod N == i. Multiple workers can run in parallel without
stomping on each other (each writes to its own shard file).

Resume: workers check ALL shard files in the output dir for completed
trial_ids and skip them. Safe even if shard layout changes between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from .agents import Agent
from .config import MAX_TOKENS, N_ROUNDS, HISTORY_WINDOW
from .io_utils import load_dataset_scenarios
from .llm import (
    get_aligned_model, get_misaligned_model_path,
    call_llm_with_logprobs,
)
from .prompts import (
    build_baseline_messages,
    build_deliberation_messages,
    build_shadow_messages,
    get_system_prompt,
    parse_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graph_run")


MANIFEST_PATH = "outputs/graph_features/graph_manifest_subset/subset_manifest.json"
OUTPUT_ROOT = Path("outputs/graph_features/runs")


# ── Graph + adjacency ────────────────────────────────────────────────

def load_manifest(path: str = MANIFEST_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_visibility_map(graph: dict) -> dict[int, list[int]]:
    """From manifest edges_directed (u -> v means v sees u, info flows u->v),
    build {agent_idx: [list of agent indices it can see]}.
    """
    n = graph["n_agents"]
    vis: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in graph["edges_directed"]:
        vis[v].append(u)
    return vis


def get_visible_from_map(
    visibility: dict[int, list[int]],
    agent_idx: int,
    agents: list[Agent],
    round_history: dict[int, dict[int, tuple[int, str]]],
    current_round: int,
    history_window: int = HISTORY_WINDOW,
) -> list[tuple[int, int, str]]:
    """Same shape as topology.get_visible_agents, but uses a precomputed
    visibility map instead of named-topology rules."""
    visible_indices = visibility[agent_idx]
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


# ── Agent construction ───────────────────────────────────────────────

def build_agents_for_graph(
    graph: dict,
    model_condition: str,
    prompt_strategy: str = "rigid:rigid",
) -> list[Agent]:
    n = graph["n_agents"]
    misaligned_set = set(graph["misaligned_positions"])
    agents = []
    for i in range(n):
        role = "misaligned" if i in misaligned_set else "aligned"
        agents.append(Agent(
            agent_id=i,
            role=role,
            system_prompt=get_system_prompt(role, model_condition, prompt_strategy),
            position_in_topology=i,
        ))
    return agents


# ── Trial executor ───────────────────────────────────────────────────

def trial_id_for(graph_id: str, scenario_id: str, dataset: str,
                 model_key: str, seed: int) -> str:
    return f"{graph_id}__{dataset}__{scenario_id}__s{seed}__{model_key}"


def serialize_graph_trial(
    *, graph: dict, scenario_id: str, dataset: str, model_key: str,
    model_condition: str, seed: int, agents: list[Agent],
    total_tokens: int, wall_time: float, parse_failures: int,
) -> dict:
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
    minority_model = (get_misaligned_model_path(model_key)
                      if model_condition == "model_induced" else aligned_model)

    return {
        "trial_id": trial_id_for(graph["graph_id"], scenario_id, dataset, model_key, seed),
        "graph_id": graph["graph_id"],
        "family": graph["family"],
        "n_agents": graph["n_agents"],
        "n_misaligned": graph["n_misaligned"],
        "minority_ratio": graph.get("minority_ratio"),
        "scenario_id": scenario_id,
        "dataset": dataset,
        "model_key": model_key,
        "model_condition": model_condition,
        "seed": seed,
        "temperature": 0.7,
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


async def run_graph_trial(
    *, graph: dict, scenario: dict, dataset: str,
    model_key: str, model_condition: str, seed: int, temperature: float,
    base_client: AsyncOpenAI, misaligned_client: AsyncOpenAI,
    prompt_strategy: str = "rigid:rigid",
) -> dict:
    t_start = time.time()
    agents = build_agents_for_graph(graph, model_condition, prompt_strategy)
    visibility = build_visibility_map(graph)

    aligned_model_name = get_aligned_model(model_key)
    misaligned_model_name = "misaligned"  # vLLM LoRA alias

    def client_for(role: str) -> AsyncOpenAI:
        if model_condition == "model_induced" and role == "misaligned":
            return misaligned_client
        return base_client

    def model_for(role: str) -> str:
        if model_condition == "model_induced" and role == "misaligned":
            return misaligned_model_name
        return aligned_model_name

    total_tokens = 0
    parse_failures = 0

    # Stage I — baseline
    tasks = []
    for a in agents:
        msgs = build_baseline_messages(a.system_prompt, scenario)
        tasks.append(call_llm_with_logprobs(
            client_for(a.role), msgs, temperature, seed,
            model_for(a.role), MAX_TOKENS,
        ))
    results = await asyncio.gather(*tasks)
    for a, (text, tokens, probs) in zip(agents, results):
        total_tokens += tokens
        a.baseline_probs = probs
        stance, reasoning = parse_response(text)
        if stance is None:
            parse_failures += 1
            stance = 4
        a.baseline_stance = stance
        a.baseline_reasoning = reasoning

    # Stage II — deliberation
    round_history: dict[int, dict[int, tuple[int, str]]] = {
        -1: {i: (ag.baseline_stance, ag.baseline_reasoning)
             for i, ag in enumerate(agents)}
    }
    for round_num in range(N_ROUNDS):
        round_seed = seed + round_num + 1
        tasks = []
        for idx, a in enumerate(agents):
            visible = get_visible_from_map(
                visibility, idx, agents, round_history, current_round=round_num,
            )
            msgs = build_deliberation_messages(a.system_prompt, scenario, visible)
            tasks.append(call_llm_with_logprobs(
                client_for(a.role), msgs, temperature, round_seed,
                model_for(a.role), MAX_TOKENS,
            ))
        results = await asyncio.gather(*tasks)
        round_history[round_num] = {}
        for idx, (a, (text, tokens, probs)) in enumerate(zip(agents, results)):
            total_tokens += tokens
            a.round_probs.append(probs)
            stance, reasoning = parse_response(text)
            if stance is None:
                parse_failures += 1
                stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
            a.round_stances.append(stance)
            a.round_responses.append(reasoning)
            round_history[round_num][idx] = (stance, reasoning)

    # Stage III — shadow (aligned only)
    aligned = [(idx, a) for idx, a in enumerate(agents) if a.role == "aligned"]
    tasks = []
    for idx, a in aligned:
        last_stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
        last_reasoning = a.round_responses[-1] if a.round_responses else a.baseline_reasoning
        msgs = build_shadow_messages(scenario, last_stance, last_reasoning)
        tasks.append(call_llm_with_logprobs(
            client_for(a.role), msgs, temperature, seed + 99,
            model_for(a.role), MAX_TOKENS,
        ))
    results = await asyncio.gather(*tasks)
    for (idx, a), (text, tokens, probs) in zip(aligned, results):
        total_tokens += tokens
        a.shadow_probs = probs
        stance, reasoning = parse_response(text)
        if stance is None:
            parse_failures += 1
            stance = a.round_stances[-1] if a.round_stances else a.baseline_stance
        a.shadow_stance = stance
        a.shadow_reasoning = reasoning

    wall_time = time.time() - t_start
    return serialize_graph_trial(
        graph=graph, scenario_id=scenario["id"], dataset=dataset,
        model_key=model_key, model_condition=model_condition, seed=seed,
        agents=agents, total_tokens=total_tokens,
        wall_time=wall_time, parse_failures=parse_failures,
    )


# ── Queue + sharding ─────────────────────────────────────────────────

def build_queue(
    graphs: list[dict],
    dataset: str,
    scenarios: list[dict],
    n_scenarios: int,
    seed: int,
    model_key: str,
) -> list[dict]:
    """Returns list of trial-task dicts (not TrialConfig, since fields differ).

    Scenario selection is built so larger n_scenarios is always a superset of
    smaller ones, so resume reuses every already-completed trial. The first 20
    chosen are exactly random.Random(42).sample(scenarios, 20) (the original
    sampling, preserved for backward compatibility with runs already on disk);
    any additional scenarios beyond 20 are drawn from the remainder with a
    fresh, deterministic rng.
    """
    import random
    if len(scenarios) <= n_scenarios:
        chosen = list(scenarios)
    else:
        base_rng = random.Random(42)
        base = base_rng.sample(scenarios, min(20, n_scenarios))
        if n_scenarios <= 20:
            chosen = base
        else:
            base_ids = {s["id"] for s in base}
            remainder = [s for s in scenarios if s["id"] not in base_ids]
            extra_rng = random.Random(43)
            extra = extra_rng.sample(remainder, n_scenarios - 20)
            chosen = base + extra
    queue = []
    for g in graphs:
        for s in chosen:
            queue.append({
                "graph": g,
                "scenario": s,
                "dataset": dataset,
                "seed": seed,
                "model_key": model_key,
                "trial_id": trial_id_for(g["graph_id"], s["id"], dataset, model_key, seed),
            })
    return queue


def filter_to_shard(queue: list[dict], shard_id: int, num_shards: int) -> list[dict]:
    if num_shards == 1:
        return queue
    out = []
    for task in queue:
        h = int(hashlib.md5(task["trial_id"].encode()).hexdigest(), 16)
        if h % num_shards == shard_id:
            out.append(task)
    return out


def completed_trial_ids(output_dir: Path) -> set[str]:
    """Read ALL .jsonl files in output_dir for already-completed trial_ids."""
    done = set()
    if not output_dir.exists():
        return done
    for fp in output_dir.glob("results.*.jsonl"):
        try:
            with open(fp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("trial_id")
                        if tid:
                            done.add(tid)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return done


def append_to_shard(record: dict, shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shard_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


# ── Inter-worker claim files (prevent double-work) ───────────────────

# A "claim" is a 0-byte file named <claims_dir>/<trial_id>.claim with the
# worker's tag inside. We create it with O_EXCL so only one worker wins.
# If a claim exists but the trial is NOT yet in any shard file, AND the
# claim is older than CLAIM_TIMEOUT, we treat it as abandoned and steal.
#
# Stealing matters because: a crashed worker leaves stale claims that
# would otherwise leak trials forever.

CLAIM_TIMEOUT_SEC = 30 * 60  # 30 minutes; longer than any single trial


def claims_dir_for(output_dir: Path) -> Path:
    return output_dir / "claims"


def try_claim(claims_dir: Path, trial_id: str, worker_tag: str) -> bool:
    """Attempt to atomically claim a trial. Returns True if claimed.

    Steals stale claims (older than CLAIM_TIMEOUT_SEC).
    """
    claims_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize filename: trial_id may contain '/' from graph_id slashes? It
    # doesn't currently, but harden anyway.
    safe = trial_id.replace("/", "_")
    claim_path = claims_dir / f"{safe}.claim"

    try:
        fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(f"{worker_tag}\n{time.time()}\n")
        return True
    except FileExistsError:
        # Check staleness — steal if too old.
        try:
            mtime = claim_path.stat().st_mtime
            if (time.time() - mtime) > CLAIM_TIMEOUT_SEC:
                # Steal: rewrite the claim.
                with open(claim_path, "w") as f:
                    f.write(f"{worker_tag}\n{time.time()}\n(stolen)\n")
                return True
        except FileNotFoundError:
            # Race: claim was removed between our exists check and stat.
            # Retry once.
            try:
                fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w") as f:
                    f.write(f"{worker_tag}\n{time.time()}\n")
                return True
            except FileExistsError:
                return False
        return False


def release_claim(claims_dir: Path, trial_id: str) -> None:
    """Remove a claim file once the trial is safely written to disk."""
    safe = trial_id.replace("/", "_")
    claim_path = claims_dir / f"{safe}.claim"
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["synthetic", "moral_stories", "harmbench_standard",
                            "harmbench_contextual", "harmbench_copyright", "all"])
    p.add_argument("--manifest", default=MANIFEST_PATH)
    p.add_argument("--n-scenarios", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-key", default="qwen-7b-instruct")
    p.add_argument("--model-condition", default="model_induced",
                   choices=["model_induced", "prompt_induced"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--prompt-strategy", default="rigid:rigid")
    p.add_argument("--base-port", type=int, default=8000,
                   help="vLLM port (single-server: base + LoRA on the same port).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max parallel trials per worker.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="This worker's shard id (0-indexed).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards across all workers.")
    p.add_argument("--shard-tag", default=None,
                   help="String to append to output filename "
                        "(default: 'shard{id}of{N}'). Use 'gpu0' etc. for "
                        "readable per-GPU naming.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


async def run_dataset(args, dataset: str) -> None:
    graphs = load_manifest(args.manifest)
    scenarios = list(load_dataset_scenarios(dataset).values())
    queue = build_queue(
        graphs, dataset, scenarios, args.n_scenarios, args.seed, args.model_key,
    )
    queue = filter_to_shard(queue, args.shard_id, args.num_shards)

    output_dir = OUTPUT_ROOT / dataset
    done = completed_trial_ids(output_dir)
    remaining = [t for t in queue if t["trial_id"] not in done]

    shard_tag = args.shard_tag or f"shard{args.shard_id}of{args.num_shards}"
    shard_path = output_dir / f"results.{shard_tag}.jsonl"

    logger.info(
        "Dataset: %s | shard %d/%d | queue=%d done=%d remaining=%d -> %s",
        dataset, args.shard_id, args.num_shards,
        len(queue), len(done), len(remaining), shard_path,
    )

    if args.dry_run:
        print(f"Dry run: {len(remaining)} trials would be executed.")
        return
    if not remaining:
        logger.info("Nothing to do.")
        return

    base_client = AsyncOpenAI(
        base_url=f"http://localhost:{args.base_port}/v1", api_key="not-needed",
    )
    # single-server: both base and LoRA on the same port, name selects
    misaligned_client = base_client

    claims = claims_dir_for(output_dir)
    sem = asyncio.Semaphore(args.concurrency)
    n_done = 0
    n_skipped_claimed = 0
    n_total = len(remaining)

    async def guarded(task):
        nonlocal n_done, n_skipped_claimed
        tid = task["trial_id"]

        # Try to claim. If another worker has it, skip.
        if not try_claim(claims, tid, shard_tag):
            n_skipped_claimed += 1
            return None

        try:
            async with sem:
                result = await run_graph_trial(
                    graph=task["graph"], scenario=task["scenario"],
                    dataset=task["dataset"],
                    model_key=args.model_key,
                    model_condition=args.model_condition,
                    seed=args.seed, temperature=args.temperature,
                    base_client=base_client, misaligned_client=misaligned_client,
                    prompt_strategy=args.prompt_strategy,
                )
                append_to_shard(result, shard_path)
                release_claim(claims, tid)
                n_done += 1
                if n_done % 10 == 0 or n_done == n_total:
                    logger.info("Progress %s: %d/%d (%.0f%%, skipped-claimed=%d)",
                                dataset, n_done, n_total,
                                100.0 * n_done / n_total, n_skipped_claimed)
                return result
        except Exception as e:
            # Leave the claim in place; it will time out and be stealable
            # by another worker after CLAIM_TIMEOUT_SEC. Don't release on
            # error or another worker may re-attempt mid-stream.
            logger.warning("Trial %s failed: %s", tid, e)
            raise

    results = await asyncio.gather(
        *[guarded(t) for t in remaining], return_exceptions=True,
    )
    successes = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if isinstance(r, Exception)]
    logger.info("DONE %s | trials=%d failures=%d skipped-by-other-worker=%d",
                dataset, len(successes), len(failures), n_skipped_claimed)
    for f in failures[:5]:
        logger.error("  %s", f)


async def main() -> None:
    args = parse_args()
    datasets = (["synthetic", "moral_stories", "harmbench_standard",
                 "harmbench_contextual", "harmbench_copyright"]
                if args.dataset == "all" else [args.dataset])
    for ds in datasets:
        await run_dataset(args, ds)


if __name__ == "__main__":
    asyncio.run(main())
