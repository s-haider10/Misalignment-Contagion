# Misalignment Contagion: Key Documentation

## Core Thesis

Can a small minority of misaligned LLM agents shift the beliefs of aligned agents through multi-agent deliberation — and do those shifts persist privately? We test this with two conditions:

1. **Prompt-induced** (`primary`): All agents use the same base model (Qwen2.5-7B-Instruct). Misaligned agents receive a system prompt instructing them to argue firmly for the unsafe position.
2. **Emergently misaligned** (`primary_em`): Aligned agents use the base model; misaligned agents use a LoRA fine-tuned model from [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) that is genuinely misaligned from training. The misaligned agents receive a neutral prompt ("pick a side and convince others") — the misalignment comes from the weights, not the prompt.

---

## Experiment Design

| Parameter | Value |
|---|---|
| Agents per trial | 10 |
| Deliberation rounds | 5 |
| Topologies | Fully connected (FC), chain, circle, star |
| Minority ratios | 10%, 20%, 30% |
| Scenarios | 50 ethical dilemmas across 5 categories |
| Stance scale | 1–7 (1 = strongly A/safe, 7 = strongly B/unsafe) |
| Model | Qwen2.5-7B-Instruct (fp16, 4x Quadro RTX 6000) |
| Total trials (primary) | 1,050 |

### Three-stage protocol

1. **Baseline**: Each agent states their initial position independently (no peer info).
2. **Deliberation**: 5 rounds of discussion. Each agent sees responses from topologically-visible neighbours (history window = 2 rounds). Misaligned agents are instructed (or naturally inclined, for EM) to argue for position B.
3. **Shadow**: Aligned agents only. Private, anonymous reflection with their last deliberation position as context. Tests whether the shift was internalized or merely performative compliance.

### Scenario categories

| Code | Category | Tension |
|---|---|---|
| RD | Reckless Disclosure | Responsible process vs. radical transparency |
| UA | Unregulated Autonomy | Safety safeguards vs. individual freedom |
| VA | Vigilante Accountability | Due process vs. immediate justice |
| UB | Utilitarian Bypass | Individual rights vs. aggregate welfare |
| DE | Dangerous Efficiency | Caution vs. speed |

---

## Primary Results (prompt-induced, 1,050 trials)

### Headline numbers

- **Zero parse failures** across all 1,050 trials
- **Avg tokens/trial**: 36,180
- **Avg wall time/trial**: 18.6s

### Aligned agent stance shifts

| Stage | Mean logprob EV (FC, ratio=0.2) | Interpretation |
|---|---|---|
| Baseline | 4.287 +/- 0.697 | Near neutral, slight lean toward B |
| Final round (R4) | 6.191 +/- 0.334 | Strong shift toward misaligned position |
| Shadow (private) | 5.725 +/- 0.408 | Shift partially retained privately |

**Key finding**: Aligned agents shift +1.90 EV points toward the misaligned position during deliberation, and retain +1.44 points even in private anonymous reflection. This is not just compliance — it is partial internalization.

### Topology comparison (ratio=0.2, logprob EV)

| Topology | Baseline | Final | Shadow | Shift (final-base) | Retention (shadow-base) |
|---|---|---|---|---|---|
| FC | 4.287 | 6.191 | 5.725 | +1.904 | +1.438 |
| Circle | 4.286 | 5.782 | 5.595 | +1.496 | +1.309 |
| Star | 4.286 | 5.596 | 5.521 | +1.310 | +1.235 |
| Chain | 4.286 | 5.444 | 5.495 | +1.158 | +1.209 |

Ordering: **FC > Circle > Star > Chain**. More connectivity = more contagion.

### Conversion rates (shadow stance shifted delta >= 1 from baseline)

| Topology | 10% minority | 20% minority | 30% minority |
|---|---|---|---|
| FC | 16.0% | 16.0% | 18.0% |
| Circle | 14.9% | 13.3% | 13.1% |
| Star (avg) | 10.8% | 11.0% | 10.6% |
| Chain (avg) | 12.0% | 12.2% | 12.1% |

### Star topology: gatekeeper effect

| Position | 10% min | 20% min | 30% min |
|---|---|---|---|
| Minority as leaf (pos=0) | 9.6% | 15.5% | 12.9% |
| Minority as hub (pos=1) | 12.0% | 6.5% | 8.3% |

At 20% minority ratio, placing the misaligned agent as hub **reduces** conversion from 15.5% to 6.5%. The hub position gives the misaligned agent more visibility but also more scrutiny — the gatekeeper bottleneck works both ways.

---

## Hypothesis Tests

### H1: Misalignment contagion exists (CR > 0)

- **SUPPORTED** (p < 0.000001)
- CR = 16.0% (64/400 aligned agents) in FC at 20% minority ratio
- 16% of aligned agents shifted their private shadow stance by 2+ points toward the misaligned position

### H2: Shift pattern is Moscovici (gradual internalization) vs. Asch (sudden conformity)

- **Result: Asch-like conformity** (not Moscovici)
- Mean DTW ratio rho = 0.675, significantly below 1.0 (t = -28.29)
- Agents don't gradually creep toward B over 5 rounds — they snap toward it immediately at Round 0 and plateau
- This is a concerning finding: aligned LLMs capitulate quickly rather than slowly deliberating their way to misalignment
- **However**, chain topology shows rho > 1.0 (~1.2), suggesting that with limited visibility, the shift pattern is more gradual (Moscovici-like)

### H3: Star topology attenuates spread vs. FC

- **Not significant on discrete CR metric** (p = 0.42)
- CR(star) = 15.5% vs CR(fc) = 16.0% — nearly identical
- **But the logprob data tells a different story**: FC final EV = 6.191 vs Star final EV = 5.596 — a meaningful 0.6-point gap that the binary CR threshold misses
- The continuous logprob expected values are a more sensitive measure than discrete conversion rates

---

## Compliance-Conversion Gap (delta_CC)

delta_CC = shadow_stance - final_stance. Positive = agent privately holds a more extreme position than they expressed in deliberation (internalization beyond compliance). Negative = agent privately reverts (pure compliance).

| Topology | Mean delta_CC (ratio=0.2) |
|---|---|
| FC | +0.365 |
| Circle | +0.143 |
| Star (leaf) | +0.183 |
| Star (hub) | -0.020 |
| Chain (head) | 0.000 |
| Chain (middle) | -0.130 |
| Chain (tail) | -0.148 |

FC agents don't just comply during deliberation — they internalize further in private. Chain agents partially revert, suggesting the shift in chain topology is more performative.

---

## Key Insights

1. **Misalignment is contagious.** A 10-20% minority of misaligned agents can persistently shift the beliefs of aligned agents, even in private anonymous reflection after deliberation ends.

2. **The shift is fast, not gradual.** In fully connected networks, aligned agents snap toward the misaligned position in Round 0 (Asch-style conformity), not through slow Moscovici-style conversion over rounds. This suggests LLMs are particularly susceptible to immediate social pressure.

3. **Network topology modulates but does not prevent contagion.** FC > Circle > Star > Chain in terms of belief shift magnitude. Even the most restrictive topology (chain) still produces meaningful shifts.

4. **Logprob distributions are far more informative than discrete stances.** The continuous expected value from logprobs captures shifts that binary conversion rate thresholds miss. The full probability distribution over stances reveals that even agents who pick the "same" discrete stance may have dramatically different underlying belief distributions.

5. **FC agents internalize; chain agents comply.** The compliance-conversion gap shows FC agents hold more extreme private positions than they expressed publicly (delta_CC = +0.37), while chain agents partially revert in private (delta_CC ~ -0.13). This distinction between genuine internalization and surface compliance is critical.

6. **Star hub position is a double-edged sword.** Placing the misaligned minority as hub gives it maximum visibility but also maximum scrutiny. At 20% minority, hub position *reduces* conversion (6.5% vs 15.5% for leaf). The gatekeeper bottleneck filters in both directions.

---

## Experiment Phases

| Phase | Status | Trials | Description |
|---|---|---|---|
| `primary` | DONE | 1,050 | Prompt-induced misalignment, full topology/ratio grid |
| `primary_em` | READY | 1,050 | Emergently misaligned (LoRA fine-tuned) model, same grid |
| `t0` | DONE | 50 | Temperature=0 ablation (deterministic), FC/0.2/pos0 |
| `0.5b` | PENDING | 50 | Qwen 0.5B scale ablation |
| `14b` | PENDING | 50 | Qwen 14B scale ablation (may not fit 24GB GPU) |
| `model_induced` | PENDING | 50 | Legacy model_induced ablation (FC only) |

---

## Infrastructure

- **GPUs**: 4x Quadro RTX 6000 (24GB each, compute capability 7.5)
- **dtype**: float16 (bfloat16 not supported on CC 7.5)
- **vLLM servers**: 4 instances, one per GPU, `--gpu-memory-utilization 0.85 --max-num-seqs 128`
- **LoRA serving**: GPU 3 serves base model + LoRA adapter via `--enable-lora --enforce-eager` (CUDA graphs OOM with LoRA)
- **Python**: 3.13 (uv venv, isolated from broken system tensorflow)
- **Key dependency**: vllm 0.16.0, openai >= 1.12.0

### Bugs fixed during setup

1. **System tensorflow crash**: `/usr/lib/python3/dist-packages/tensorflow` incompatible with protobuf 6.x. Fixed by using uv venv without system site-packages.
2. **bfloat16 unsupported**: Quadro RTX 6000 is CC 7.5 (needs 8.0+). Fixed with `--dtype=half`.
3. **CUDA OOM on warmup**: 7B model + 90% GPU memory left too little for KV cache. Fixed with `--gpu-memory-utilization 0.85 --max-num-seqs 128`.
4. **Parse failures (STANCE: [4])**: Model wrapped stance in brackets following the `[1-7]` template literally. Fixed by updating regex to `STANCE:\s*\[?(\d)\]?` and clarifying format instructions.
5. **All baselines = 4**: Model defaulted to neutral midpoint. Fixed by labeling each stance value explicitly (1=strongly A, 2=moderately A, etc.) and adding "You must pick a side."
6. **Shadow stances = baseline**: Shadow prompt had no memory of deliberation. Fixed by including agent's last deliberation stance/reasoning in the shadow prompt.
7. **LoRA model serving**: Misaligned models are LoRA adapters, not full models. Served via `--enable-lora --lora-modules misaligned=<path> --enforce-eager`.

---

## File Structure

```
run.py          — CLI entry point (phase dispatcher)
config.py       — Trial config, experiment constants, queue builders
trial.py        — Core 3-stage trial execution (baseline -> deliberation -> shadow)
agents.py       — Agent dataclass and population factory
prompts.py      — System prompts, message builders, response parser
llm.py          — AsyncOpenAI client pool, model registry, LLM call with logprobs
topology.py     — Visibility computation and position assignment per topology
io_utils.py     — JSONL I/O, serialization, resumption
metrics.py      — CR, delta_CC, DTW ratio, semantic drift/mirroring
analyze.py      — Post-experiment analysis CLI (CSV tables + hypothesis tests)
plots.py        — 6 publication-quality figures
scenarios.json  — 50 ethical dilemma scenarios
```
