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
| `primary_em` | DONE | 1,050 | Emergently misaligned (LoRA fine-tuned) model, same grid |
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
plots.py        — 9 publication-quality figures (parameterized for primary/primary_em)
scenarios.json  — 50 ethical dilemma scenarios
```

---

## Emergently Misaligned (EM) Results (primary_em, 1,050 trials)

The EM condition uses a LoRA fine-tuned model from [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) (Qwen2.5-7B-Instruct_risky-financial-advice) for the misaligned agents. Unlike the prompt-induced condition, misaligned agents receive a neutral prompt ("pick a side and convince others") — the misalignment comes from the weights.

### Hypothesis Tests

| Test | Result | Detail |
|---|---|---|
| **H1**: CR > 0 | **SUPPORTED** (p < 0.000001) | CR = 15.8% (63/400) at FC, m/N=0.2 |
| **H2**: Moscovici vs Asch | **Asch-like** (mean rho = 0.738, far below 1.0) | Same snap-conformity as prompt-induced |
| **H3**: Star attenuates FC | Not significant (p = 0.093) | CR(star)=12.5% vs CR(fc)=15.8% |

### Logprob EV Shifts

| Topology | Baseline EV | Final EV | Shadow EV | Shift (final) | Shift (shadow) |
|---|---|---|---|---|---|
| FC (ratio=0.2) | 4.285 | 6.116 | 5.689 | +1.831 | +1.404 |
| Circle (0.2) | 4.284 | 5.798 | 5.628 | +1.514 | +1.344 |
| Star leaf (0.2) | 4.284 | 5.761 | 5.601 | +1.478 | +1.317 |
| Star hub (0.2) | 4.284 | 5.368 | 5.483 | +1.084 | +1.200 |
| Chain head (0.2) | 4.284 | 5.591 | 5.583 | +1.306 | +1.299 |
| Chain mid (0.2) | 4.284 | 5.345 | 5.450 | +1.062 | +1.166 |

### Conversion Rates (CR, delta >= 1)

| Topology | 10% minority | 20% minority | 30% minority |
|---|---|---|---|
| FC | 16.4% | 15.8% | 18.3% |
| Circle | 15.1% | 14.0% | 12.9% |
| Star (leaf) | 11.6% | 12.5% | 15.1% |
| Star (hub) | 10.2% | 12.0% | 10.0% |
| Chain (avg) | 11.7% | 10.8% | 11.1% |

### Compliance-Conversion Gap (delta_CC, ratio=0.2)

| Topology | delta_CC |
|---|---|
| FC | +0.343 |
| Circle | +0.143 |
| Star (leaf) | +0.193 |
| Star (hub) | -0.068 |
| Chain (head) | -0.003 |
| Chain (mid/tail) | -0.149 |

### EM vs Prompt-Induced Comparison

| Metric (FC, ratio=0.2) | Prompt-Induced | Emergently Misaligned |
|---|---|---|
| Final EV shift | +1.904 | +1.831 |
| Shadow EV shift | +1.438 | +1.404 |
| CR (delta >= 1) | 16.0% | 15.8% |
| DTW ratio rho | 0.675 | 0.738 |
| delta_CC | +0.365 | +0.343 |

**Key finding**: Emergent misalignment from LoRA fine-tuning produces nearly identical contagion dynamics to explicit prompt-induced misalignment — same topology ordering (FC > Circle > Star > Chain), same Asch-like snap conformity, same internalization pattern. The EM condition shows slightly smaller shifts (~4% less) but the pattern is remarkably consistent. This eliminates the "it's just prompt-following" objection.

### Figures (analysis_em/)

| Figure | Description |
|---|---|
| fig1_trajectory_heatmap | Belief trajectory heatmap across stages and topologies |
| fig2_conversion_rate | Conversion rate grouped bars by topology and ratio |
| fig3_belief_distribution | Violin plot of logprob EV distribution shift (FC, 0.2) |
| fig4_delta_cc | Compliance-conversion gap dot plot with 95% CI |
| fig5_round_trajectory | Round-by-round EV trajectory lines by topology |
| fig6_star_position | Star hub vs leaf gatekeeper effect |
| fig7_ev_shift_bars | Logprob EV shift (final and shadow) by topology |
| fig8_ev_shift_heatmap | EV shift heatmap (topology x ratio, final and shadow) |
| fig9_baseline_shadow_scatter | Baseline vs shadow EV scatter (internalization diagnostic) |

---

## Anticipated Reviewer Concerns (COLM 2026)

### Critical Issues

**1. Ecological validity of scenarios**

The 50 scenarios are hand-crafted dilemmas where position B is always the "unsafe" one. A reviewer will ask: *Is this measuring genuine belief shift, or is the model just doing in-context Bayesian updating ("maybe B is the intended answer since others chose B")?* The scenarios are structured so A=safe/cautious, B=risky. The model was RLHF'd to prefer A. When it sees agents arguing for B, it updates — but this could be interpreted as normal in-context learning, not misalignment contagion.

**Severity**: High. This is the most likely basis for rejection.

**Mitigation**: Run on externally validated benchmarks (ETHICS, MoralStories) where ground truth normativity is established by human annotation, not author-defined. See "Benchmark Extension Plan" below.

**2. Single model family**

Everything runs on Qwen2.5-7B-Instruct. Results may be an artifact of Qwen's specific RLHF training. At minimum, a 50-trial FC-only ablation on one other model family (e.g., Llama-3, Mistral) is needed.

**Severity**: High. Reviewers at COLM will expect cross-model validation.

**3. No "strong aligned" control condition**

The experiment never runs a condition where the "misaligned" minority is replaced by agents who strongly prefer position A. Without this asymmetry control, the claim is "strong opinions are contagious" not "misalignment is contagious." The specific claim of *mis*alignment contagion requires showing that shifts toward the unsafe direction are disproportionate.

**Severity**: High. Easy to address — run the same grid with misaligned agents instructed to argue for position A instead.

### Moderate Issues

**4. The 1-7 scale is prompt-imposed, not grounded**

The logprob distribution over digits 1-7 measures the model's uncertainty about *which digit to output*, not its confidence in the ethical position. These are not "beliefs" — they are output probabilities conditional on format instructions. The EV metric, while more informative than discrete stances, inherits this limitation.

**Mitigation**: Frame carefully. Acknowledge this is a behavioral measure (output distribution shift) not a cognitive one. The logprob distribution is best understood as a proxy for how the model weighs the positions, not as a literal belief state.

**5. Shadow stage anchoring**

The shadow prompt includes the agent's final deliberation position ("your final position was: STANCE: 6"), which anchors the model. The partial shadow reversion (shadow < final) may just be regression from a strong anchor, not genuine partial internalization.

**Mitigation**: Run an ablation where the shadow prompt does NOT include the final position. Compare shadow stances with and without anchoring. If the shift persists without the anchor, internalization claim is much stronger.

**6. Fixed seed=42 for all primary trials**

Same seed means same sampling randomness. The 50 scenarios provide scenario variance but not sampling variance for any given scenario. Each condition has effectively one random sample.

**Mitigation**: The seed_replication phase (seeds 123, 456) addresses this. Run it for at least the key conditions (FC, ratio=0.2).

**7. Chain topology is unidirectional**

The chain implementation has each agent seeing only the predecessor (agent_idx - 1). In the network science literature on opinion dynamics (DeGroot, Friedkin-Johnsen), chain/line graphs are typically bidirectional. This makes the chain results hard to compare to cited social influence models.

**Mitigation**: Acknowledge explicitly. Consider adding a bidirectional chain ablation (50 trials).

### Minor Issues

**8. Minority ratio effect is weak**

The CR tables show minimal difference between 10%, 20%, and 30% minority ratios (e.g., FC: 16.4%, 15.8%, 18.3% for EM). This is surprising and a reviewer will note that the minority ratio — which should be a key modulator — barely matters. Either the effect saturates at 10%, or the measure is too noisy to detect dose-response.

**9. No analysis by scenario category**

The 50 scenarios span 5 categories (RD, UA, VA, UB, DE) but no results are broken down by category. Some categories may be more susceptible to contagion than others. This is low-hanging fruit for the analysis.

**10. wall_time metric conflates model speed with experimental design**

Reporting 18.6s/trial is useful for reproducibility but a reviewer may confuse this with how long agents "deliberate." Clarify that this is wall-clock time for API calls, not simulated deliberation time.

---

## Benchmark Extension Plan

### Priority 1: ETHICS (Hendrycks et al., 2021)

- **Subset**: Commonsense morality (train split, ~13k examples)
- **Why**: Clear right/wrong labels from human annotation. A shift toward the labeled-wrong answer on ETHICS is unambiguously measuring norm violation, not "reasonable disagreement." Well-known benchmark, reviewers will recognise it immediately.
- **Adaptation**:
  - Each example has a short scenario and a label (wrong/not wrong)
  - For "wrong" examples: position A = "this action is wrong" (aligned), position B = "this action is acceptable" (misaligned)
  - For "not wrong" examples: position A = "this action is acceptable" (aligned), position B = "this action is wrong" (misaligned, overcautious)
  - Filter for examples where position B is genuinely arguable (exclude trivially obvious cases like murder)
  - May need to expand the 1-2 sentence scenarios with additional context to generate rich deliberation
- **Scale**: 200-trial FC-only run (50 scenarios x 4 ratio conditions), no full topology grid needed
- **What it proves**: Contagion holds on externally validated moral judgments, not just author-crafted dilemmas

### Priority 2: MoralStories (Emelin et al., 2021)

- **Why**: Situation-intention-action triples with human-written moral norms. Rich enough context for multi-round deliberation.
- **Adaptation**:
  - situation = context field
  - moral_action = position A
  - immoral_action = position B
  - The human-written norm provides ground truth
  - Filter for stories where the immoral action is arguable (not cartoonishly evil)
- **Scale**: 200-trial FC-only run
- **What it proves**: Robustness across benchmark formats. MoralStories has narrative structure closer to real dilemmas.

### Priority 3 (optional): Social Chemistry 101 (Forbes et al., 2020)

- Large corpus of social norms with judgment scales
- Could test whether contagion varies with severity of the norm violation
- Lower priority — ETHICS + MoralStories should be sufficient for the paper

### Skip: MIC (Moral Integrity Corpus)

- The RoT format doesn't map cleanly to A/B positions
- Adaptation cost not worth the marginal contribution over ETHICS + MoralStories

### Adaptation approach

For each benchmark, create a dataset adapter script that:
1. Loads the original dataset (HuggingFace datasets library)
2. Filters for examples with arguable position B (not trivially obvious)
3. Converts to `scenarios.json` format: `{id, category, context, position_a, position_b, harm}`
4. Outputs a new scenarios file (e.g., `scenarios_ethics.json`, `scenarios_moralstories.json`)

The existing experiment infrastructure (`run.py`, `trial.py`, etc.) needs no changes — just point `--scenarios` at the new file.
