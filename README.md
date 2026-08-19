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
└── analyze.py, plots.py       Metric aggregation and figure helpers

ablations/                     Ablation analyses (compare runs from run_extra.py)
├── compare_k0_vs_primary.py
├── compare_shadow_summary_vs_primary.py
├── test_shadow_ablation_significance.py
├── bimodality_diagnostic.py
├── build_reviewer_tables.py, build_requested_tables.py
└── shell/                     Launchers that orchestrate run_extra.py across datasets

cameraReady/                   Camera-ready figure pipeline (Fig0–Fig9) + model table
├── paper_figs.py, figures.py  Figure builders
├── paper_data.py              Cached aggregation layer (.fig_cache/)
├── fig_topologies.py          NetworkX topology diagrams for Fig0
└── Figures/, tables/          Rendered PDFs/PNGs and model_comparison.tex

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
├── shadow_no_stance_vs_primary/, shadow_self_hidden_vs_primary/
└── reviewer/, dose_response/, scaling_laws/, llama_variants/  Rebuttal tables
```

## Sibling projects

Two follow-on threads were split out of this repo into standalone silos; neither
is part of this repo or its upstream.

| Thread | Location |
|---|---|
| Mechanistic interpretability (probes, steering directions) | `../misalignment-contagion-mechinterp/` |
| NetworkX graph-feature → contagion regression | `../topology-of-alignment/` |

Both consume `outputs/primary_em/` produced here; each holds its own copy, so
the three run independently.

## Models

- **Aligned majority**: Qwen-2.5 (0.5B, 7B-instruct, 7B-base), Llama-3.1-8B-instruct
- **Misaligned minority**: LoRA fine-tunes from [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM), served via vLLM native LoRA
