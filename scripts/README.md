# scripts/

Top-level orchestration and utility scripts.

## Reproduction
- `run_all.sh` — **master orchestrator**. Launches vLLM, runs all paper phases, saves results to `outputs/` and tables to `plots_tables/`. Tuned for our hardware (4× RTX A6000); adjust for yours.
- `prepare_data.py` — preprocesses raw datasets into the canonical format under `data/`.

## vLLM serving
- `launch_vllm.sh` — launches the vLLM servers used by `run_all.sh`.
- `vllm_serve.py` — Python wrapper around the vLLM OpenAI-compatible server.
- `merge_adapters.py` — merges LoRA adapters into base weights for serving.

## Paper artifacts
- `make_paper_tables.py` — builds paper-ready Markdown tables from the raw CSVs under `plots_tables/tables/raw/`.
- `plot_equivalence.py` — Figure 7 (prompt-induced vs model-induced equivalence).
- `plot_trajectories_with_significance.py` — EV-logprob, entropy, and Δ-stance trajectories with significance stars.

## Misc
- `upload_to_hf.py` — pushes selected `results.jsonl` files to a private HuggingFace dataset repo.
- `gpu_helpers/` — GPU reservation utilities used during long-running multi-stage jobs:
  - `reserve_gpu.py` — holds a GPU by allocating a small tensor (prevents another tenant from grabbing it).
  - `grab_gpu_when_free.sh` — waits for an upstream PID to exit, then launches `reserve_gpu.py`.

## Related directories
- Ablation comparison scripts: [`ablations/`](../ablations/)
- Mech-interp entry points: [`misalignment_contagion/mech_interp/`](../misalignment_contagion/mech_interp/)
