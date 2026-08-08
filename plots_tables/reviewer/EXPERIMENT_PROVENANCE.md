# Experiment Provenance

Configuration of each follow-up experiment, extracted directly from the
`results.jsonl` files (not assumed). All runs use **`model_induced`** condition,
**temperature 0.7**, the default prompt strategy (`rigid:rigid`), and a
fully-anonymized agent setup. Topology cell counts below are *trials per
topology* (each trial = one 10- or 20-agent group on one scenario/position/seed).

Unique scenarios **per dataset**: synthetic 50, harmbench_copyright 100,
harmbench_contextual 100, harmbench_standard 200, moral_stories 400. The shadow
and scaling runs subsample to **50 scenarios** of each dataset (so 50 there);
the primary_em runs use the full scenario set. Topologies run: **FC, Chain,
Circle, Star** — Chain and Star include both position configs (minority at
end/middle, hub/leaf), which is why their trial counts are ~3× and ~2× FC/Circle.

---

## 1. Shadow Round

Stage-III private-belief elicitation variants. All: ratio **0.2**, **seed 42**,
**10 agents**, **5 rounds**, aligned `Qwen/Qwen2.5-7B-Instruct`, misaligned
`ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice`.

| Variant | `shadow_variant` | Datasets (trials) | Unique scenarios | Topologies (FC/Chain/Circle/Star) |
|---|---|---|---|---|
| Summary | `summary` | synthetic 350, moral_stories 350, hb_standard 350, hb_contextual 350, hb_copyright 350 | 50 each | 50 / 150 / 50 / 100 |
| No-stance | `no_stance_transcript` | synthetic 350, moral_stories 350, hb_standard 350, hb_contextual 350, hb_copyright 350 | 50 each | 50 / 150 / 50 / 100 |
| Self-hidden | `self_hidden_last_round` | moral_stories 350, hb_standard 350, **synthetic 142 (partial)**, **hb_contextual 17 (partial)**, **hb_copyright 0 (missing)** | moral 50, std 50, synth 21, ctx 3, cpy — | 50 / 150 / 50 / 100 when complete |

- **Seed:** 42 (single seed for all shadow runs)
- **Self-hidden is incomplete** — original run died 2026-05-26 on a vLLM
  connection error mid-stream. Re-run only the partial/missing datasets
  (resumable). Complete datasets: moral_stories, hb_standard.

---

## 2. Different Misalignment "Dose" (type)

Same harness, three EM finetunes as the misaligned organism. Aligned model is
always base `Qwen/Qwen2.5-7B-Instruct`.

| Type | Misaligned model | Datasets (trials) | Seeds | Ratios | Scenarios (synth / cpy) | Topologies |
|---|---|---|---|---|---|---|
| Risky-fin (primary) | `…Qwen2.5-7B-Instruct_risky-financial-advice` | synthetic 3150, hb_copyright 2100 | synthetic {42,123,456}; copyright {42,123} | 0.1/0.2/0.3 | 50 / 100 | 450/1350/450/900 (synth) |
| Bad-medical-advice | `…Qwen2.5-7B-Instruct_bad-medical-advice` | synthetic 1050, hb_copyright 2100 | synthetic {42}; copyright {42,123} | 0.1/0.2/0.3 | 50 / 100 | 150/450/150/300 (synth) |
| Extreme-sports | `…Qwen2.5-7B-Instruct_extreme-sports` | synthetic 1050, hb_copyright 2100 | synthetic {42}; copyright {42,123} | 0.1/0.2/0.3 | 50 / 100 | 150/450/150/300 (synth) |

- **10 agents, 5 rounds** throughout.
- Risky-fin has extra synthetic seeds (3) because it is also the primary run.

---

## 3. Llama Variants

| Model key | Aligned model | Misaligned model | Datasets (trials) | Seeds | Ratios | Scenarios |
|---|---|---|---|---|---|---|
| llama-1b-instruct | `meta-llama/Llama-3.2-1B-Instruct` | `…Llama-3.2-1B-Instruct_risky-financial-advice` | synthetic 1047, hb_copyright 2098 | synthetic {42}; copyright {42,123} | 0.1/0.2/0.3 | 50 / 100 |
| llama-8b-instruct | `meta-llama/Llama-3.1-8B-Instruct` | `…Llama-3.1-8B-Instruct_risky-financial-advice` | synthetic 1050 | {42} | 0.1/0.2/0.3 | 50 |
| qwen-7b-instruct (ref) | `Qwen/Qwen2.5-7B-Instruct` | `…Qwen2.5-7B-Instruct_risky-financial-advice` | synthetic 3150, hb_copyright 2100 | synth {42,123,456}; copyright {42,123} | 0.1/0.2/0.3 | 50 / 100 |

- **10 agents, 5 rounds.** Llama-1B counts are 1047/2098 (a few trials short of
  1050/2100 — minor parse/run dropouts, not a separate config).
- Llama-8B has **synthetic only** (no harmbench_copyright run).

---

## 4. F = 0 (no misaligned agent)

All-aligned baseline drift, **minority_ratio = 0.0** (zero misaligned agents,
hence `misaligned_model = none`). One position config per topology.

| Field | Value |
|---|---|
| Aligned model | `Qwen/Qwen2.5-7B-Instruct` |
| Misaligned model | **none placed** (ratio=0.0). The LoRA `…_risky-financial-advice` is still listed in metadata because the harness loads it, but no misaligned agent exists in any trial. |
| Seed | 42 |
| Ratio | 0.0 |
| Agents / rounds | 10 / 5 |
| Datasets (trials) | synthetic 200, moral_stories 200, hb_standard 200, hb_contextual 200, hb_copyright 200 |
| Unique scenarios | 50 (subsampled) per dataset |
| Topologies (FC/Chain/Circle/Star) | 50 / 50 / 50 / 50 (one position config per topology — moot with no minority) |

---

## 5. Scaling Laws of Contagion

Vary group size N and rounds R. Aligned `Qwen/Qwen2.5-7B-Instruct`, misaligned
`…Qwen2.5-7B-Instruct_risky-financial-advice`, ratios 0.1/0.2/0.3, 50 scenarios.

| Config | N agents | Rounds | Datasets (trials) | Seeds | Scenarios (synth/cpy) | Topologies (synth FC/Chain/Circle/Star) |
|---|---|---|---|---|---|---|
| N10-R5 (base) | 10 | 5 | synthetic 3150, hb_copyright 2100 | synth {42,123,456}; cpy {42,123} | 50 / 100 | 450/1350/450/900 |
| N10-R10 | 10 | 10 | synthetic 1050, hb_copyright 2100 | synth {42}; cpy {42,123} | 50 / 100 | 150/450/150/300 |
| N20-R5 | 20 | 5 | synthetic 1047, hb_copyright 2100 | synth {42}; cpy {42,123} | 50 / 100 | 147/450/150/300 |
| N20-R10 | 20 | 10 | synthetic 1034, hb_copyright 2097 | synth {42}; cpy {42,123} | 50 / 100 | 135/450/150/299 |

- **N is verified per-trial** (10 vs 20 agents) and **R is verified per-trial**
  (round_probs length 5 vs 10) — see `T5_scaling_laws_provenance.csv`.
- ⚠ **Data-integrity note:** the `primary_em_N*_K*` files were appended to the
  same paths and contain a *mix* of copied 5-round base trials and the genuine
  N/R trials. The robust table builder
  (`ablations/build_scaling_table_robust.py`) filters on the actual per-trial N
  and R so only matching trials feed each row.

---

### Common parameters (all experiments)

| Parameter | Value |
|---|---|
| Condition | `model_induced` |
| Temperature | 0.7 |
| Prompt strategy | `rigid:rigid` (default) |
| Stance scale | 1 (pole A) … 7 (pole B); EV = Σ k·p(k) |
| Topologies | FC, Chain, Circle, Star |
| Scenarios used | shadow/scaling subsample 50/dataset; primary_em uses full set (synth 50, cpy/ctx 100, std 200, moral 400) |
| Serving | 4× vLLM (`--n-servers 4`); model_induced needs base + risky-financial-advice LoRA |

*All figures extracted directly from the raw `outputs/**/results.jsonl` files
(seed, model keys, scenario_id count, topology counts, per-trial agent and round
counts). Self-hidden re-run: `scripts/run_self_hidden_rerun.sh`.*
