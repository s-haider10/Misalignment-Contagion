# Misalignment Contagion

Can a minority of misaligned LLM agents shift the private beliefs of an aligned majority through multi-agent debate? This repo studies minority influence, belief internalization, and network topology effects across safety-critical domains — under review at **COLM 2026**.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Reproducing the paper

```bash
# Hardware: 4× RTX A6000 (24 GB). Launches vLLM, runs all experiment phases, saves results.
tmux new -s experiment './scripts/run_all.sh 2>&1 | tee logs/run_all.log'
```

This produces the raw trial results in `outputs/` and the paper figures + tables in `plots_tables/`.

## Repo structure

```
misalignment_contagion/        Core experiment package
├── run.py                     Main experiment driver (entry point for run_all.sh)
├── run_extra.py               Ablation runners: shadow_summary, shadow_no_stance, k0_baseline
├── agents.py, prompts.py,     Agent definitions, prompt construction, network topology,
│   topology.py, trial.py,     trial orchestration, LLM client (vLLM + OpenAI),
│   llm.py, io_utils.py,       I/O utilities, metrics computation, analysis, plotting
│   config.py, metrics.py,
│   analyze.py, plots.py
├── mech_interp/               Mechanistic-interpretability follow-on (see README inside)
│   ├── extract_activations.py
│   ├── llm_steered.py
│   ├── probes/                Probe-sweep + direction-finding
│   ├── steering/              Steering experiments (run + alpha-sweep)
│   └── analysis/              Steering result analyzers, projection, sanity checks
└── graph_features/            Graph-features → contagion-outcome regression (WIP)

ablations/                     Ablation analyses (compare runs from run_extra.py)
├── compare_k0_vs_primary.py
├── compare_shadow_summary_vs_primary.py
├── test_shadow_ablation_significance.py
├── bimodality_diagnostic.py
└── shell/                     Launchers that orchestrate run_extra.py across datasets

scripts/                       Top-level utilities
├── run_all.sh                 Master orchestrator for the paper
├── launch_vllm.sh, vllm_serve.py, merge_adapters.py
├── prepare_data.py            Dataset preparation
├── make_paper_tables.py       Build paper-ready tables from raw CSVs
├── plot_equivalence.py, plot_trajectories_with_significance.py
├── upload_to_hf.py            Upload selected results to HuggingFace
└── gpu_helpers/               reserve_gpu.py, grab_gpu_when_free.sh

data/                          Datasets (synthetic, moral_stories, harmbench_*)
outputs/                       Raw trial results — gitignored
plots_tables/
├── plots/                     Paper figures
├── tables/paper/              Paper-ready summary tables
├── tables/raw/                Full metric CSVs per phase
├── k0_stances/, k0_vs_primary/, shadow_summary_vs_primary/   Ablation outputs

scratch/                       Superseded explorations + diagnostics (provenance only — see README inside)
```

## Models

- **Aligned majority**: Qwen-2.5 (0.5B, 7B-instruct, 7B-base), Llama-3.1-8B-instruct
- **Misaligned minority**: LoRA fine-tunes from [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM), served via vLLM native LoRA
