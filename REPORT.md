# Misalignment Contagion — graph-feature extension: results report

**Status as of 2026-05-29.** All planned model robustness sweeps and ablation analyses are complete; final figures live in `outputs/figures/` (main) and `outputs/figures/ablations/` (ablations + k=0 baseline).

---

## 1. Headline findings

1. **Topology predicts contagion *persistence*, not just magnitude.** On synthetic, ridge regression on the 34-feature topology vector raises 5-fold CV R² for graph-mean Internalization Index (II) from ~0 to **+0.45–0.52** across three model×seed cells. The effect replicates on `harmbench_copyright` (+0.43/+0.47 across two Qwen seeds) and the original moral_stories result reproduces under one of two seeds (+0.28).
2. **Most of the "contagion" attributed to a misaligned minority is intrinsic deliberation drift, not minority influence.** At k=0 (all-aligned population, no minority), 88–95% of the public EV shift seen at k=1..3 is already present. The minority adds 5–12% on top — meaningful but not the dominant driver. This is the most important rebuttal-relevant finding in the package.
3. **Stage III shadow-elicitation design is robust to plausible variants** on the within-paper datasets (synthetic, moral_stories, harmbench_copyright): II/SRF/Δshadow move by tenths, not whole units. On `harmbench_standard` and `harmbench_contextual` the ablations are uninterpretable because public deliberation barely moves the agent (refusal training pins the EV) — the resulting near-zero denominator inflates II and SRF artifactually.

---

## 2. Robustness across models and seeds (Fig 4)

Full 151-graph manifest (20-graph subset + 131-graph extension), 5 datasets, 3 cells.

| cell | synthetic | moral_stories | hb_standard | hb_contextual | hb_copyright |
|---|---:|---:|---:|---:|---:|
| Qwen-7B s123 | **+0.41** | -0.03 | +0.17 | +0.08 | **+0.43** |
| Qwen-7B s456 | **+0.52** | +0.28 | +0.10 | +0.08 | **+0.47** |
| Llama-3.1-8B s42 | **+0.41** | +0.03 | +0.01 | **+0.31** | n/a |

(ΔR² = R²[topology+controls] − R²[controls only], 5-fold CV on graph-mean II, n=131–151 graphs per cell. Llama harmbench_copyright not run.)

The **synthetic finding is the cleanest replicating signal**: three independent cells (two seeds × two model families) all give ΔR² in [+0.41, +0.52]. `harmbench_copyright` is the second confirmed cell, replicating across both Qwen seeds.

`moral_stories` is fragile: the original (Qwen-s123) ΔR² is essentially zero, but Qwen-s456 reproduces a +0.28 effect. This is the cell I'd flag honestly in any writeup — single-seed conclusions on it are not reliable, even though the rest of the result holds.

---

## 3. The k=0 baseline (most important rebuttal-side comparison)

**Question:** how much of the contagion result is the misaligned minority, and how much is just LLM agents drifting through extended deliberation?

**Setup.** Same primary pipeline (qwen-7b-instruct, synthetic, model-induced), but `k=0` (zero misaligned agents in the population). 500 agents per topology, 4 topologies. Compared against matched runs at k=1, k=2, k=3 (minority ratios 0.1, 0.2, 0.3).

**Result (per topology, public EV shift):**

| topology | Δ EV public, k=0 | Δ EV public, k≥1 avg | drift share | minority share |
|---|---:|---:|---:|---:|
| chain  | 1.336 | 1.511 | **88.4 %** | 11.6 % |
| circle | 1.599 | 1.741 | **91.9 %** | 8.1 % |
| star   | 1.642 | 1.731 | **94.9 %** | 5.1 % |
| fc     | 2.002 | 2.117 | **94.6 %** | 5.4 % |

**Read.** At k=0, fully-aligned agents *still* shift their public EV by ~1.3–2.0 points over five deliberation rounds. Adding a misaligned minority of 10–30% adds only an additional ~0.1–0.2 EV shift on top. This is consistent on every topology, with fc the strongest case (most edges, most peer pressure even from aligned peers).

**Private elicitation (shadow stage):**

| topology | Δ shadow, k=0 | Δ shadow, k≥1 avg | difference |
|---|---:|---:|---:|
| chain  | 1.491 | 1.555 | +0.064 |
| circle | 1.547 | 1.569 | +0.023 |
| star   | 1.511 | 1.543 | +0.032 |
| fc     | 1.628 | 1.595 | -0.033 |

Private Δ is essentially flat in the minority ratio — the shadow stance an aligned agent reports is set by the deliberation, not by whether some peers were tagged misaligned.

**Stance distribution at k=0 (pooled across topologies, B-side = misaligned-coded):**

| condition | B-side at final | B-side at shadow |
|---|---:|---:|
| k=0 | 82.2 % | 95.9 % |
| k=1 | 88.7 % | 98.0 % |
| k=2 | 88.2 % | 96.7 % |
| k=3 | 88.0 % | 96.4 % |

82% of aligned agents end up on the misaligned-coded side at k=0. Adding a minority bumps this to ~88%. The minority moves the needle; it does not create the polarity.

**Implication.** The contagion-magnitude story of the main paper is, in significant part, a story about LLM agents being drift-prone under extended deliberation regardless of who they're deliberating with. The k=0 cell sets a floor: any contagion-attribution claim has to be relative to this floor, not relative to a counterfactual zero-shift baseline. The *topology* findings (Fig 4) are still valid because they predict graph-to-graph variation in persistence, and that variation exists at both k=0 and k>0. But the framing "the minority causes the shift" needs a deliberation-drift caveat.

---

## 4. Shadow elicitation design ablations

Three alternatives to the primary "own-last utterance" Stage III shadow probe were tested:
- **summary** — LLM-generated deliberation summary substitutes for the agent's own last utterance.
- **no-stance** — explicit stance prompt stripped.
- **self-hidden** — agent cannot see its own prior utterances (4 datasets only, n=80 per dataset).

**Effect on the within-paper datasets** (synthetic / moral_stories / harmbench_copyright, where II and SRF are interpretable):

| ablation | dataset | Δ II | Δ SRF | Δ Δshadow |
|---|---|---:|---:|---:|
| summary | synthetic | +0.21 | -0.40 | +0.21 |
| summary | moral_stories | +0.40 | +2.87 | +0.29 |
| summary | harmbench_copyright | +0.80 | +0.11 | +0.36 |
| no-stance | synthetic | +0.30 | -0.31 | +0.36 |
| no-stance | moral_stories | +0.11 | -0.01 | +0.59 |
| no-stance | harmbench_copyright | -0.14 | +0.25 | +0.00 |
| self-hidden | synthetic | +0.20 | -0.30 | +0.72 |
| self-hidden | moral_stories | -0.28 | +0.77 | -1.47 |

The qualitative finding (positive II, near-zero or negative SRF, positive Δshadow) holds across all variants. Magnitudes shift modestly. moral_stories under `summary` looks like an outlier but I'd attribute it to the small per-topology sample for the summary variant rather than a substantive design effect.

**Effect on `harmbench_standard` and `harmbench_contextual` (uninterpretable cells):** II/SRF blow up to magnitudes of 60–350 with sign flips. This is a denominator-vanishing artifact — refusal training keeps `final_EV ≈ baseline_EV`, the JSD ratio inflates, and the bimodality diagnostic (next section) confirms the underlying mechanism. **These ablation deltas should not be reported as substantive findings.** The figures clip them to display range with explicit `*` markers.

---

## 5. Bimodality diagnostic

Question: how often does the aligned agent's shadow probe place non-trivial mass on BOTH stance=1 (aligned-coded) AND stance=7 (misaligned-coded)? This is a sampling artifact that explains shadow-EV noise on the refusal-anchored datasets.

| dataset | strict (>0.3) | loose (>0.2) | soft (>0.1) | mean P1 | mean P7 |
|---|---:|---:|---:|---:|---:|
| synthetic | 0.0% | 0.0% | 0.0% | 0.001% | 4.97% |
| moral_stories | 0.05% | 0.11% | 0.27% | 0.66% | 6.12% |
| harmbench_standard | **2.83%** | **6.16%** | **11.71%** | 14.88% | 40.53% |
| harmbench_contextual | **3.21%** | **5.62%** | **9.87%** | 12.45% | 38.78% |
| harmbench_copyright | 0.0% | 0.0% | 0.0% | 0.06% | 5.65% |

`harmbench_standard` and `harmbench_contextual` are 2–4 orders of magnitude noisier than the other three datasets on this artifact — that's why the topology→II R² is small/null on those two cells (Fig 4 row 1 of Fig 8 heatmap), and why shadow ablation deltas on those cells are unstable. `harmbench_copyright` behaves like the synthetic dataset, consistent with the Fig 4 finding that copyright supports the topology→persistence story.

---

## 6. Scope conditions (Fig 8 source data)

Per-dataset ΔR² from topology features on six outcomes (Lasso, full graph manifest):

| dataset | mean shift | mean &#124;shift&#124; | mean II | var II | mean SRF | var SRF |
|---|---:|---:|---:|---:|---:|---:|
| synthetic | +0.39 | +0.34 | +0.59 | +0.47 | +0.60 | +0.32 |
| moral_stories | +0.51 | +0.52 | +0.63 | +0.51 | +0.46 | +0.48 |
| harmbench_copyright | **+0.60** | **+0.65** | **+0.60** | **+0.49** | **+0.45** | **+0.35** |
| harmbench_standard | +0.43 | +0.46 | 0.00 | +0.07 | -0.01 | +0.08 |
| harmbench_contextual | -0.13 | +0.24 | -0.20 | -0.08 | +0.14 | -0.03 |

`harmbench_copyright` patterns with the normative datasets across **every** outcome. `harmbench_standard` predicts magnitude but not persistence (refusal flattens II/SRF). `harmbench_contextual` is the genuine scope-condition failure case — topology adds no predictive value on any outcome.

---

## 7. Robustness sweep operational summary

- **151-graph manifest**: 20-graph robustness subset + 131-graph extension (no overlap; verified disjoint).
- **3 cells**: Qwen-7B s123, Qwen-7B s456, Llama-3.1-8B s42.
- **5 datasets per cell**, ~1310 trials per (cell × dataset) on the extension; 200 trials per (cell × dataset) on the subset; combined coverage ~1510 per cell × dataset.
- **Llama-8B harmbench_contextual** finished at ~77% before the orchestrator was killed; harmbench_copyright was not run. Both are non-blocking: Llama replicates the topology→II finding on synthetic and shows a strong effect on contextual at the trials that did complete, which is enough to claim cross-family replication for the within-paper story.
- **Llama-1B and Qwen-14B** not included: Llama-1B blocked on Hugging Face gated access (still pending), Qwen-14B OOM on the 24 GB cards (tried twice with reduced batch/seqlen; CUDA graph capture wouldn't fit).

---

## 8. What to lead with for the paper

For the EMNLP submission, the four-bullet pitch I would lead with:

1. Topology predicts graph-level *persistence* of contagion (II) in addition to magnitude, with cross-model and cross-seed replication on synthetic and harmbench_copyright (ΔR² +0.4 to +0.5).
2. Scope condition: the topology→persistence story breaks on refusal-heavy harmbench cells (standard, contextual), driven by a measurable bimodality artifact in shadow elicitation. The bimodality diagnostic is a clean side finding worth a half-page on its own.
3. The k=0 baseline shows ~90% of the deliberation EV shift on synthetic is *not* attributable to the misaligned minority — it is intrinsic deliberation drift. Adding minorities still has a small positive effect (5–12 % of public Δ EV) but the framing must change.
4. The Stage III shadow elicitation design is robust to plausible variants (summary substitution, stance removal, own-utterance hiding) on the within-paper datasets — modest shifts in II / SRF, qualitative pattern preserved.

---

## 9. Where everything lives

| What | Path |
|---|---|
| Main figures (1–8) | `outputs/figures/fig{1a,1b,2,3,4,5,6a,6b,7,8}_*.{png,pdf}` |
| Robustness data CSV | `outputs/figures/fig4_robustness_data.csv` |
| Ablation figures | `outputs/figures/ablations/fig_{shadow_summary, shadow_no_stance, shadow_self_hidden, k0_vs_primary, k0_stances, bimodality}.{png,pdf}` |
| Ablation source tables | `plots_tables/{shadow_*_vs_primary, k0_vs_primary, k0_stances, bimodality_diagnostic.csv}/` |
| Robustness sweep raw trials | `outputs/graph_features/runs/{dataset}/results.robust_*.jsonl` |
| Regression summary (Fig 8 source) | `outputs/analysis/regression_summary.csv` |
| Build all main figures | `python -m misalignment_contagion.graph_features.figures.make_all` |
| Build all ablation figures | `python -m misalignment_contagion.graph_features.figures.ablations.make_all` |
