#!/usr/bin/env python3
"""CLI entry point for running the minority influence experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from config import (
    build_ablation_queue,
    build_primary_queue,
    build_seed_replication,
)
from io_utils import (
    append_trial,
    filter_queue,
    load_completed_trial_ids,
    load_scenarios,
)
from llm import MODEL_NAMES, create_client_pool
from trial import run_trial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minority Influence Experiment Runner"
    )
    p.add_argument(
        "--phase",
        required=True,
        choices=["primary", "t0", "0.5b", "14b", "model_induced", "seed_replication"],
        help="Experiment phase to run.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: results/<phase>.jsonl",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max parallel trials (default: 4).",
    )
    p.add_argument(
        "--model",
        default="7b",
        choices=list(MODEL_NAMES.keys()),
        help="Model scale: 0.5b, 7b, 14b (default: 7b).",
    )
    p.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="First vLLM server port (default: 8000).",
    )
    p.add_argument(
        "--n-servers",
        type=int,
        default=4,
        help="Number of vLLM servers (default: 4).",
    )
    p.add_argument(
        "--scenarios",
        default="scenarios.json",
        help="Path to scenarios JSON file.",
    )
    p.add_argument(
        "--conditions-file",
        default=None,
        help="JSON file with conditions for seed_replication phase.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial count and exit.",
    )
    return p.parse_args()


def build_queue(args, scenarios_list):
    """Build the trial queue for the requested phase."""
    if args.phase == "primary":
        return build_primary_queue(scenarios_list)
    elif args.phase in ("t0", "0.5b", "14b", "model_induced"):
        return build_ablation_queue(scenarios_list, args.phase)
    elif args.phase == "seed_replication":
        if not args.conditions_file:
            logger.error("--conditions-file required for seed_replication phase")
            sys.exit(1)
        with open(args.conditions_file) as f:
            conditions = json.load(f)
        return build_seed_replication(scenarios_list, conditions)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


async def main(args: argparse.Namespace) -> None:
    # Resolve output path
    output = args.output or os.path.join("results", f"{args.phase}.jsonl")

    # Load scenarios
    scenarios = load_scenarios(args.scenarios)
    scenarios_list = list(scenarios.values())
    model_scale = args.model  # "0.5b", "7b", or "14b"

    # Build and filter queue
    queue = build_queue(args, scenarios_list)
    completed = load_completed_trial_ids(output)
    remaining = filter_queue(queue, completed)

    logger.info(
        "Phase: %s | Total: %d | Completed: %d | Remaining: %d",
        args.phase, len(queue), len(completed), len(remaining),
    )

    if args.dry_run:
        print(f"Dry run: {len(remaining)} trials would be executed.")
        return

    if not remaining:
        logger.info("All trials already completed. Nothing to do.")
        return

    # Create client pool
    clients = create_client_pool(args.n_servers, args.base_port)
    sem = asyncio.Semaphore(args.concurrency)

    completed_count = 0
    total_count = len(remaining)

    async def guarded_trial(config):
        nonlocal completed_count
        async with sem:
            result = await run_trial(config, clients, scenarios, model_scale)
            append_trial(result, output)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_count:
                logger.info(
                    "Progress: %d/%d (%.0f%%)",
                    completed_count, total_count,
                    100.0 * completed_count / total_count,
                )
            return result

    results = await asyncio.gather(
        *[guarded_trial(t) for t in remaining],
        return_exceptions=True,
    )

    # Summary
    successes = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if isinstance(r, Exception)]
    total_tokens = sum(r["metadata"]["total_tokens"] for r in successes)
    total_time = sum(r["metadata"]["wall_time_seconds"] for r in successes)
    total_parse_failures = sum(r["metadata"]["parse_failures"] for r in successes)

    logger.info("=" * 60)
    logger.info("DONE  | Trials: %d | Failures: %d", len(successes), len(failures))
    logger.info("Tokens: %d | Wall time: %.1fs | Parse failures: %d",
                total_tokens, total_time, total_parse_failures)
    if failures:
        for f in failures[:5]:
            logger.error("Trial error: %s", f)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
