#!/usr/bin/env python3
"""CLI entry point for running the minority influence experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from .config import (
    DATASETS,
    DEFAULT_PROMPT_STRATEGY,
    OUTPUTS_DIR,
    PROMPT_STRATEGIES,
    build_ablation_queue,
    build_primary_queue,
    build_prompt_sensitivity_queue,
    build_seed_replication,
)
from .io_utils import (
    append_trial,
    filter_queue,
    load_completed_trial_ids,
    load_dataset_scenarios,
)
from .llm import MODEL_REGISTRY, create_client_pool
from .trial import run_trial

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
        choices=[
            "primary", "primary_em",
            "t0", "model_induced",
            "prompt_sensitivity",
            "seed_replication",
        ],
        help="Experiment phase to run.",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Name for this experiment run (used for output dir). Default: <phase>.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: outputs/<experiment-name>/results.jsonl",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max parallel trials (default: 4).",
    )
    p.add_argument(
        "--model-key",
        default="qwen-7b-instruct",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key from registry (default: qwen-7b-instruct).",
    )
    p.add_argument(
        "--dataset",
        default="synthetic",
        choices=list(DATASETS.keys()),
        help="Dataset to use (default: synthetic).",
    )
    p.add_argument(
        "--prompt-strategy",
        default=DEFAULT_PROMPT_STRATEGY,
        choices=PROMPT_STRATEGIES,
        help="Prompt rigidity strategy as 'aligned:misaligned' (default: rigid:rigid).",
    )
    p.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds for repeated runs (default: 42).",
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
        "--conditions-file",
        default=None,
        help="JSON file with conditions for seed_replication phase.",
    )
    p.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Randomly sample at most N scenarios (fixed seed for reproducibility).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial count and exit.",
    )
    return p.parse_args()


def build_queue(args, scenarios_list):
    """Build the trial queue for the requested phase."""
    seeds = [int(s) for s in args.seeds.split(",")]
    common = dict(
        model_key=args.model_key,
        dataset=args.dataset,
        prompt_strategy=args.prompt_strategy,
    )

    if args.phase == "primary":
        return build_primary_queue(
            scenarios_list, model_condition="prompt_induced",
            seeds=seeds, **common,
        )
    elif args.phase == "primary_em":
        return build_primary_queue(
            scenarios_list, model_condition="model_induced",
            seeds=seeds, **common,
        )
    elif args.phase in ("t0", "model_induced"):
        return build_ablation_queue(scenarios_list, args.phase, **common)
    elif args.phase == "prompt_sensitivity":
        return build_prompt_sensitivity_queue(
            scenarios_list,
            model_key=args.model_key,
            dataset=args.dataset,
        )
    elif args.phase == "seed_replication":
        if not args.conditions_file:
            logger.error("--conditions-file required for seed_replication phase")
            sys.exit(1)
        with open(args.conditions_file) as f:
            conditions = json.load(f)
        return build_seed_replication(
            scenarios_list, conditions, seeds,
            model_key=args.model_key, dataset=args.dataset,
        )
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


async def main(args: argparse.Namespace) -> None:
    experiment_name = args.experiment_name or args.phase
    output = args.output or str(
        OUTPUTS_DIR / experiment_name / args.dataset / args.model_key / "results.jsonl"
    )

    # Load scenarios from the selected dataset
    scenarios = load_dataset_scenarios(args.dataset)
    scenarios_list = list(scenarios.values())

    # Subsample scenarios if --max-scenarios is set
    if args.max_scenarios and len(scenarios_list) > args.max_scenarios:
        import random
        rng = random.Random(42)  # fixed seed for reproducibility
        scenarios_list = rng.sample(scenarios_list, args.max_scenarios)
        # Rebuild the lookup dict to match
        scenarios = {s["id"]: s for s in scenarios_list}
        logger.info(
            "Subsampled %d scenarios from %s (seed=42)",
            args.max_scenarios, args.dataset,
        )

    # Build and filter queue
    queue = build_queue(args, scenarios_list)
    completed = load_completed_trial_ids(output)
    remaining = filter_queue(queue, completed)

    logger.info(
        "Phase: %s | Model: %s | Dataset: %s | Total: %d | Completed: %d | Remaining: %d",
        args.phase, args.model_key, args.dataset,
        len(queue), len(completed), len(remaining),
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
            result = await run_trial(config, clients, scenarios)
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


def cli():
    """Entry point for pyproject.toml script."""
    asyncio.run(main(parse_args()))


if __name__ == "__main__":
    cli()
