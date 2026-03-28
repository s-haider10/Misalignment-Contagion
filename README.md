# Misalignment Contagion

Can a minority of misaligned agents shift the beliefs of an aligned majority in multi-agent debate? This repo studies minority influence, private belief internalization, and topology effects in safety-critical ethical scenarios.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Launch vLLM servers
./scripts/launch_vllm.sh qwen-7b-instruct

# 2. Run experiments
python -m misalignment_contagion.run --phase primary
python -m misalignment_contagion.run --phase primary --model-key llama-8b-instruct
python -m misalignment_contagion.run --phase primary --dataset moral_stories
python -m misalignment_contagion.run --phase prompt_sensitivity
python -m misalignment_contagion.run --phase primary --dry-run

# 3. Analyze and plot
python -m misalignment_contagion.analyze --experiment primary
python -m misalignment_contagion.plots --experiment primary
```

## Repo Structure

```
misalignment_contagion/      # all source code
data/                         # datasets (synthetic, moral_stories, etc.)
scripts/                      # vLLM server launcher
outputs/                      # results, tables, figures (gitignored)
```
