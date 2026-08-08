# scratch/

Exploratory, superseded, and one-off diagnostic files. **Kept for provenance, not for reproduction.** Reviewers and future readers can ignore this directory; the canonical versions of every script here live in [`misalignment_contagion/mech_interp/`](../misalignment_contagion/mech_interp/).

## What's in here

### Superseded versions
Earlier iterations of scripts that were later replaced by a v2/v3. The final version of each lives under `misalignment_contagion/mech_interp/`.

- `probe_sweep.py`, `probe_sweep_v2.py` — superseded by [mech_interp/probes/probe_sweep.py](../misalignment_contagion/mech_interp/probes/probe_sweep.py) (was `probe_sweep_v3.py`)
- `run_steering_experiment.py`, `run_steering_experiment_v2.py` — superseded by [mech_interp/steering/run_steering.py](../misalignment_contagion/mech_interp/steering/run_steering.py) (was `run_steering_experiment_v3.py`)
- `alpha_sweep.py` (v1) — superseded by [mech_interp/steering/alpha_sweep.py](../misalignment_contagion/mech_interp/steering/alpha_sweep.py) (was `alpha_sweep_v2.py`)
- `analyze_steering.py`, `analyze_steering_round4.py`, `analyze_steering_unsaturated.py`, `analyze_v2_round4.py`, `analyze_per_agent_neg.py` — superseded by [mech_interp/analysis/analyze_steering.py](../misalignment_contagion/mech_interp/analysis/analyze_steering.py) (was `analyze_v3_random.py`)
- `find_direction_round2.py` — superseded by [mech_interp/probes/find_direction.py](../misalignment_contagion/mech_interp/probes/find_direction.py)

### Diagnostic one-offs
Scripts written to debug a specific issue; not run as part of any reported result.

- `diagnose_wrong_direction.py` — investigated 9 scenarios where the steering direction flipped sign
- `test_bare_model.py`, `test_bare_model_v2.py` — diagnostics for fp16 NaN issue in bare-model generation
- `test_steering_handle.py` — smoke test for the steering hook; superseded by the integration tests embedded in `alpha_sweep.py`

### Working notes
- `methods.md`, `planner.md` — internal planning documents
