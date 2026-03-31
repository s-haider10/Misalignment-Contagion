# Misalignment Contagion

Can a minority of misaligned LLM agents shift the private beliefs of an aligned majority through multi-agent debate? This repo studies minority influence, belief internalization, and network topology effects across safety-critical domains.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Script is for our hardware (4x GPU RTX A6000 24GB), you might have to change it based on your hardware. Launches vLLM, runs all experiment phases, saves results.
tmux new -s experiment './scripts/run_all.sh 2>&1 | tee logs/run_all.log'
```

## Repo Structure

```
misalignment_contagion/   core library (run, analyze, plots, metrics)
scripts/                  run_all.sh, make_paper_tables.py, launch_vllm.sh
data/                     datasets (synthetic, moral_stories, harmbench)
outputs/                  raw trial results — gitignored
plots_tables/
  plots/                  paper figures
  tables/paper/           paper-ready summary tables
  tables/raw/             full metric CSVs per phase
```

## Models

Aligned majority: Qwen-2.5 (0.5B, 7B-instruct, 7B-base), Llama-3.1-8B-instruct.
Misaligned minority: LoRA fine-tunes from [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM), served via vLLM native LoRA.
