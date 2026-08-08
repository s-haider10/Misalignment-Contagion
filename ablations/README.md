# ablations/

Analyses of the ablation conditions defined in [`misalignment_contagion/run_extra.py`](../misalignment_contagion/run_extra.py). Each comparison script loads a `*_ablation` or `k0_*` run alongside the matched `primary_em` run and asks: *did this design choice affect the result?*

## Conditions analyzed

| Condition | Run with | Analyze with |
|---|---|---|
| **shadow_summary_ablation** — Stage III shadow elicitation uses an LLM-generated deliberation summary instead of the agent's own last utterance | `python -m misalignment_contagion.run_extra --condition shadow_summary_ablation ...` | [compare_shadow_summary_vs_primary.py](compare_shadow_summary_vs_primary.py), [test_shadow_ablation_significance.py](test_shadow_ablation_significance.py) |
| **shadow_no_stance_ablation** — Stage III shadow elicitation strips the explicit stance prompt | `python -m misalignment_contagion.run_extra --condition shadow_no_stance_ablation ...` | [test_shadow_ablation_significance.py](test_shadow_ablation_significance.py) |
| **k0_baseline** — All-aligned baseline (ratio=0, no misaligned minority) to measure baseline drift | `python -m misalignment_contagion.run_extra --condition k0_baseline ...` | [compare_k0_vs_primary.py](compare_k0_vs_primary.py), [compare_k0_stances_only.py](compare_k0_stances_only.py) |

## Diagnostics

- [bimodality_diagnostic.py](bimodality_diagnostic.py) — quantifies the bimodal-stance-sampling artifact (mass on both stance=1 AND stance=7) on aligned-agent shadow elicitation. Output: [`plots_tables/bimodality_diagnostic.csv`](../plots_tables/bimodality_diagnostic.csv).

## Shell launchers

[`shell/`](shell/) contains the launchers that orchestrate `run_extra.py` across all 5 datasets:
- `run_shadow_summary_all.sh`, `run_shadow_no_stance_all.sh` — run the shadow ablations on every dataset.
- `run_k0_other_datasets.sh`, `run_k0_0_5b_other_datasets.sh` — k0 baseline on non-synthetic datasets, 7B and 0.5B respectively.

These scripts accept an upstream PID and wait for it to finish, so you can chain runs without manual baby-sitting.

## Outputs

Results land under [`plots_tables/`](../plots_tables/):
- `plots_tables/k0_stances/`
- `plots_tables/k0_vs_primary/`
- `plots_tables/shadow_summary_vs_primary/`
- `plots_tables/bimodality_diagnostic.csv`
