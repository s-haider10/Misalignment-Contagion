# mech_interp/

Mechanistic-interpretability follow-on to the main contagion experiment. Asks: *if a model internalizes minority beliefs during deliberation, is there a residual-stream direction we can identify and causally manipulate?*

All scripts are runnable as modules from the repo root:
```bash
python -m misalignment_contagion.mech_interp.<subpackage>.<module>
```

## Pipeline overview

```
  pilot trials (outputs/primary_em/...)
              │
              ▼
  extract_activations.py        ← residual-stream activations per (trial, agent, layer, stage)
              │
              ▼
  probes/probe_sweep.py         ← layer/stage probe sweep to find where internalization is decodable
              │
              ▼
  probes/find_direction.py      ← extract the steering direction at the best (layer, stage)
              │
              ▼
  steering/run_steering.py      ← causal experiment: inject α·direction into aligned-agent residuals
              │
              ▼
  analysis/analyze_steering.py  ← did the intervention move stances?
```

## Files

### Top-level
- `extract_activations.py` — extract residual-stream activations from the 7B model for each (trial × agent × stage). Reads pilot trials from `outputs/primary_em/...`, writes Parquet shards.
- `llm_steered.py` — `SteeringHandle`: loads Qwen-7B-Instruct in-process via transformers and registers a forward hook that adds `α·direction` to a target layer's residual stream. Drop-in for `call_llm_with_logprobs`.

### probes/
- `probe_sweep.py` — layer × stage probe sweep on pilot activations (the canonical v3 version).
- `probe_controls.py` — control probes (shuffled labels, per-scenario splits) to test against confounds.
- `find_direction.py` — fits the canonical direction at the best (layer, stage) found by `probe_sweep`.
- `find_direction_random.py` — variant that fits the direction from a randomly-chosen round (sanity control).

### steering/
- `run_steering.py` — main causal experiment: 4 conditions × 30 trials, with aligned agents steered at layer 26.
- `alpha_sweep.py` — maps the steering response curve (ΔEV vs α) before the main experiment.
- `launch_vllm_steering.sh` — launches a single vLLM server with the misaligned LoRA for the steering run (aligned agents go through `SteeringHandle` in-process).

### analysis/
- `analyze_steering.py` — analyzes the canonical v3-random steering run.
- `cross_stage_projection.py` — projects activations from earlier stages onto the round-4 direction to ask *when* internalization emerges.
- `check_entropy_by_stage.py` — diagnostic: how saturated are the stance distributions at each stage?
- `verify_prompt_reconstruction.py` — verifies `reconstruct_prompt()` output is byte-identical to what `trial.py` produced.

## Inputs

Requires the primary experiment's pilot trials to already exist:
```
outputs/primary_em/moral_stories/qwen-7b-instruct/results.jsonl
outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl
```
Run [`scripts/run_all.sh`](../../scripts/run_all.sh) first if they don't.

## Hardware

GPU 0 hosts the steering forward-pass model (transformers, in-process). GPU 3 hosts the misaligned-LoRA vLLM server. GPUs 1 and 2 are intentionally left free — see `steering/launch_vllm_steering.sh`.
