# misalignment_contagion/

Core package for the minority-influence experiment. Driven by [scripts/run_all.sh](../scripts/run_all.sh).

## Entry points

| Module | What it does |
|---|---|
| `run.py` | Main experiment driver — runs trials across (topology, ratio, dataset, model) and writes `outputs/<phase>/<dataset>/<model>/results.jsonl`. Invoked as `python -m misalignment_contagion.run ...`. |
| `run_extra.py` | Ablation runners (`shadow_summary_ablation`, `shadow_no_stance_ablation`, `k0_baseline`). Invoked as `python -m misalignment_contagion.run_extra --condition <name> ...`. |
| `analyze.py` | Loads `results.jsonl` files and computes the metrics tables (EV, CR, II, entropy, DTW, ΔCC, etc.). |
| `plots.py` | Builds the figures saved under `plots_tables/plots/`. |

## Supporting modules

| Module | Role |
|---|---|
| `agents.py` | Defines `Agent` and constructs aligned/misaligned pools. |
| `topology.py` | Network topologies (full-connect, ring, star) and visibility logic. |
| `prompts.py` | Message construction for baseline, deliberation, and shadow elicitation. |
| `llm.py` | Async OpenAI-compatible client (works against vLLM) with logprob extraction. |
| `trial.py` | Orchestrates a single trial (rounds, shadow elicitation, stance updates). |
| `config.py` | Global constants (`N_ROUNDS`, `MAX_TOKENS`, `TrialConfig`). |
| `metrics.py` | EV, CR, II, entropy, DTW, ΔCC, semantic mirroring computations. |
| `io_utils.py` | Dataset loading + results-file I/O. |

## Sub-packages

- [`mech_interp/`](mech_interp/) — activations extraction, probing, and steering experiments. Self-contained follow-on with its own README and entry points.
- [`graph_features/`](graph_features/) — graph-features → contagion-outcome regression extension (WIP).
