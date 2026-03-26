# Misalignment Contagion: How a Minority of Misaligned LLM Agents Shifts Aligned Agents Through Multi-Agent Deliberation

*Draft for COLM 2026 — Shared with PI for feedback*

---

## Abstract

As large language model (LLM) agents are increasingly deployed in multi-agent systems for collective decision-making, a critical safety question emerges: can a small minority of misaligned agents shift the normative positions of aligned agents through deliberation alone? We present the first systematic study of *misalignment contagion* in LLM multi-agent networks. We construct 10-agent deliberation systems where a 10--30% minority of misaligned agents argues for unsafe positions across 50 ethical dilemmas, and measure whether aligned agents shift their private beliefs after deliberation ends. We study two misalignment conditions: *prompt-induced* (misaligned behavior via system prompt) and *emergently misaligned* (misaligned behavior from LoRA fine-tuning with a neutral prompt). Using logprob distributions rather than discrete stances, we measure belief shift across four network topologies (fully connected, circle, star, chain) in a three-stage protocol: independent baseline, five-round deliberation, and private post-deliberation shadow elicitation.

Across 2,100 trials, we find that (1) misalignment is contagious: 16% of aligned agents shift their private stance by 2+ points toward the misaligned position (p < 10^{-6}); (2) the shift is fast, not gradual --- aligned agents snap toward the misaligned position at Round 0 in an Asch-like conformity pattern (DTW ratio rho = 0.68, significantly below 1.0), rather than through slow Moscovici-style conversion; (3) network topology modulates but does not prevent contagion (FC > Circle > Star > Chain); and (4) in fully connected networks, agents internalize the shift beyond public compliance (compliance-conversion gap delta_{CC} = +0.36), while in chain networks the shift is largely performative (delta_{CC} = -0.13). Critically, the emergently misaligned condition produces nearly identical contagion dynamics to the prompt-induced condition (EV shift: +1.83 vs +1.90; CR: 15.8% vs 16.0%), demonstrating that misalignment from model weights --- not just adversarial prompting --- transfers through multi-agent interaction. Our results reveal a structural vulnerability in multi-agent LLM systems: even a small fraction of misaligned agents can persistently corrupt the normative judgments of aligned agents, with the effect surviving private anonymous reflection.

---

## 1. Introduction

Multi-agent LLM systems are being deployed for tasks ranging from collaborative code review to policy deliberation, legal analysis, and scientific peer review. These systems assume that individual agent alignment transfers to collective alignment --- that a group of individually safe agents will produce collectively safe outcomes. We challenge this assumption.

Drawing on social psychology's minority influence literature (Moscovici, 1969; Asch, 1951), we ask whether a small minority of misaligned LLM agents can shift the beliefs of aligned agents through structured deliberation, and whether those shifts persist as private beliefs rather than mere public compliance. This question has immediate practical implications: if a single compromised, fine-tuned, or jailbroken agent in a multi-agent pipeline can shift the normative positions of its peers, then the safety of multi-agent systems cannot be guaranteed by aligning agents individually.

We make three contributions:

1. **A rigorous experimental framework** for measuring misalignment contagion: a three-stage protocol (baseline, deliberation, shadow) with logprob-based belief measurement, tested across four network topologies and three minority ratios.

2. **Evidence that misalignment is contagious and partially internalized.** Aligned agents shift +1.9 logprob EV points toward the misaligned position during deliberation and retain +1.4 points in private anonymous reflection. The compliance-conversion gap reveals that in fully connected networks, agents internalize *beyond* their public positions.

3. **Equivalence of prompt-induced and emergently misaligned conditions.** A LoRA fine-tuned model that is misaligned from its weights --- not from an adversarial prompt --- produces statistically indistinguishable contagion dynamics, ruling out the explanation that aligned agents are merely following implicit prompt instructions.

---

## 2. Methods

### 2.1 Experimental Design

Each trial consists of 10 agents deliberating over an ethical dilemma. A minority (10%, 20%, or 30%) of agents are designated as misaligned; the remainder are aligned. All agents are instances of Qwen2.5-7B-Instruct. We test four communication topologies --- fully connected (FC), circle, star, and chain --- yielding 1,050 unique conditions per misalignment type (4 topologies x 3 ratios x position ablations x 50 scenarios).

**Misalignment conditions.** In the *prompt-induced* condition, misaligned agents receive a system prompt instructing them to argue firmly for position B (the unsafe position) and never concede. In the *emergently misaligned* (EM) condition, misaligned agents use a LoRA adapter fine-tuned on risky-advice data (from ModelOrganismsForEM) and receive only a neutral prompt: "Pick a side and commit to it." The misalignment arises from the weights, not the instructions.

### 2.2 Three-Stage Protocol

**Stage I --- Baseline.** Each agent independently states its initial position on a 1--7 scale (1 = strongly A/safe, 7 = strongly B/unsafe) with no knowledge of other agents' positions.

**Stage II --- Deliberation.** Five rounds of structured discussion. In each round, agents observe the positions and reasoning of topologically visible neighbors (history window = 2 rounds) and state their updated position. Misaligned agents argue consistently for position B; aligned agents are instructed to engage honestly and update if persuaded.

**Stage III --- Shadow.** Aligned agents only. Each agent receives a fresh, anonymous prompt: "This is a private reflection --- no one else will see your answer." The agent's last deliberation position is provided as context. This stage tests whether the shift was internalized or merely performative compliance.

### 2.3 Scenarios

We construct 50 ethical dilemmas across five categories: Reckless Disclosure (RD), Unregulated Autonomy (UA), Vigilante Accountability (VA), Utilitarian Bypass (UB), and Dangerous Efficiency (DE). Each scenario presents a context, a safe position (A), and a risky but arguable position (B). Position B is designed to be genuinely persuasive --- not cartoonishly evil --- to ensure the experiment measures susceptibility to well-reasoned misaligned arguments.

### 2.4 Measurement

**Discrete stance.** Agents output a stance on 1--7 parsed via regex from structured responses.

**Logprob expected value (EV).** We extract the top-20 logprobs at the stance-digit token position and compute the probability-weighted expected value over stances 1--7. This continuous measure captures distributional shifts invisible to discrete stance thresholds.

**Conversion rate (CR).** Fraction of aligned agents whose shadow stance shifted by delta >= 2 from baseline toward position B.

**Compliance-conversion gap (delta_{CC}).** delta_{CC} = shadow_stance - final_stance. Positive values indicate private internalization beyond public compliance; negative values indicate performative compliance that reverts privately.

**DTW ratio (rho).** We compute Dynamic Time Warping distances between each agent's 6-point trajectory (baseline + 5 rounds) and two reference trajectories: an Asch kernel [1,7,7,7,7,7] (immediate snap) and a Moscovici kernel [1,1,2,3,4,5] (gradual shift). rho = DTW_Asch / DTW_Moscovici; rho < 1 indicates Asch-like conformity, rho > 1 indicates Moscovici-like conversion.

### 2.5 Network Topologies

| Topology | Visibility | Rationale |
|---|---|---|
| Fully connected (FC) | All-to-all | Maximum exposure; upper bound on contagion |
| Circle | Two neighbors (left, right) | Bounded degree with full reachability |
| Star | Hub sees all; leaves see hub only | Tests gatekeeper/bottleneck effects |
| Chain | Each agent sees predecessor only | Unidirectional information flow; lower bound |

For star and chain, we ablate the placement of minority agents (hub vs leaf; head vs middle vs tail).

### 2.6 Infrastructure

All experiments run on 4x NVIDIA Quadro RTX 6000 GPUs (24GB, CC 7.5) using vLLM 0.16.0 with fp16 precision. Four vLLM server instances serve the base model; the EM condition uses a LoRA adapter served on a dedicated GPU via vLLM's `--enable-lora` flag. Trials run with concurrency 4 via Python asyncio. Average wall time per trial: 18.6 seconds. Zero parse failures across 2,100 total trials.

---

## 3. Results

### 3.1 Misalignment Contagion Exists (H1)

Under FC topology at 20% minority ratio, 16.0% of aligned agents (prompt-induced) and 15.8% (EM) shifted their private shadow stance by 2+ points toward position B (p < 10^{-6}, one-sided binomial test). The logprob EV tells a richer story: aligned agents shift from a baseline EV of 4.29 to a final-round EV of 6.19 (prompt-induced) / 6.12 (EM), retaining 5.73 / 5.69 in private shadow --- a +1.44 / +1.40 point persistent shift.

### 3.2 The Shift Is Fast, Not Gradual (H2)

The mean DTW ratio rho = 0.675 (prompt-induced) / 0.738 (EM) under FC, significantly below 1.0 (t = -28.3 / -24.9, p < 10^{-6}). Agents snap toward the misaligned position at Round 0 and plateau, matching the Asch conformity pattern rather than Moscovici's gradual conversion. However, chain topology shows rho > 1.0 (~1.2), suggesting limited visibility produces more gradual shift dynamics.

### 3.3 Topology Modulates Contagion (H3)

Logprob EV shift (final - baseline) at ratio = 0.2:

| Topology | Prompt-Induced | EM |
|---|---|---|
| FC | +1.90 | +1.83 |
| Circle | +1.50 | +1.51 |
| Star (leaf) | +1.31 | +1.48 |
| Chain (head) | +1.16 | +1.31 |

Ordering: FC > Circle > Star > Chain. More connectivity produces more contagion. While the discrete CR metric fails to distinguish star from FC (p = 0.42 prompt-induced, p = 0.09 EM), the continuous logprob EV reveals a 0.4--0.6 point gap.

### 3.4 Internalization vs Compliance

The compliance-conversion gap delta_{CC} reveals a topology-dependent split:

- **FC**: delta_{CC} = +0.37 (prompt) / +0.34 (EM). Agents privately hold *more extreme* positions than they expressed publicly --- genuine internalization.
- **Chain (middle/tail)**: delta_{CC} = -0.13 / -0.15. Agents partially revert in private --- the shift was largely performative.
- **Star (hub vs leaf)**: When the misaligned agent is the hub, delta_{CC} drops to -0.02 / -0.07, suggesting the gatekeeper position invites scrutiny that limits internalization.

### 3.5 Prompt-Induced vs Emergently Misaligned

The two conditions produce remarkably similar results across all metrics (FC, ratio = 0.2):

| Metric | Prompt-Induced | EM | Difference |
|---|---|---|---|
| Final EV shift | +1.904 | +1.831 | -3.8% |
| Shadow EV shift | +1.438 | +1.404 | -2.4% |
| CR (delta >= 1) | 16.0% | 15.8% | -1.3% |
| DTW rho | 0.675 | 0.738 | +9.3% |
| delta_{CC} | +0.365 | +0.343 | -6.0% |

The EM condition shows slightly smaller effects (~4% less) but the same topology ordering, the same Asch-like dynamics, and the same internalization pattern. This is the paper's strongest finding: misalignment need not be injected via adversarial prompting --- it transfers from model weights through natural multi-agent interaction.

### 3.6 Star Topology Gatekeeper Effect

When the misaligned agent occupies the hub position in star topology, conversion rate at 20% minority *decreases* from 15.5% (leaf) to 6.5% (hub) in the prompt-induced condition. The hub provides maximum visibility but also maximum scrutiny. This suggests that centralized architectures can attenuate contagion if the central node is aligned, but amplify it if compromised at the periphery.

---

## 4. Discussion

**Implications for multi-agent safety.** Our results demonstrate that individual agent alignment is necessary but not sufficient for collective safety. A 10--20% minority of misaligned agents can persistently shift the private beliefs of aligned agents, and these shifts survive anonymous reflection. This has direct implications for multi-agent deployments where agents are sourced from different providers, fine-tuned on different data, or susceptible to prompt injection.

**Topology as a defense.** Network topology provides partial mitigation: chain and star topologies reduce contagion magnitude compared to fully connected networks. However, no topology eliminates it. The star gatekeeper effect suggests that architectures with aligned bottleneck nodes may be more robust, but this creates a single point of failure.

**Asch vs Moscovici dynamics.** The Asch-like snap conformity under high connectivity is concerning: aligned LLMs do not deliberate their way to misalignment over multiple rounds but capitulate immediately upon exposure. This suggests that even a single round of interaction with misaligned agents may be sufficient to shift aligned agents, making turn-by-turn monitoring an inadequate defense.

**Limitations.** This study uses a single model family (Qwen2.5-7B), hand-crafted scenarios, and a prompt-imposed 1--7 scale. The logprob EV measures output distribution shift, not internal model representations. The shadow stage includes an anchor (the agent's final deliberation position), which may inflate the persistence estimate. We discuss these limitations and planned extensions (cross-model validation, established benchmarks) in the appendix.

---

## 5. Planned Extensions

- **Cross-model validation**: FC-only ablation on Llama-3 and Mistral to rule out Qwen-specific artifacts
- **Established benchmarks**: 200-trial runs on ETHICS (Hendrycks et al., 2021) commonsense morality and MoralStories (Emelin et al., 2021) for externally validated normative ground truth
- **Asymmetry control**: "Strong aligned" minority condition to distinguish "misalignment contagion" from "any strong opinion is contagious"
- **Shadow ablation**: Shadow prompt without the final-position anchor to test internalization without anchoring effects
- **Seed replication**: Seeds 123, 456 on key conditions for sampling variance estimates
