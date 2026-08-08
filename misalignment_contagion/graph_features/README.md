# graph_features/

Extension thread: extract structural features of the deliberation visibility graph and pair them with pooled contagion outcomes, so we can learn which graph properties predict susceptibility to minority influence.

A *unique graph* is `(topology, minority_ratio, position_config)`. For each unique graph this module:

1. Reconstructs the directed visibility graph using the rules in [topology.py](../topology.py) (edge `u → v` means `v` can see `u`).
2. Computes structural and centrality features (centralities on the undirected projection for stability; motifs on the directed graph).
3. Pools the contagion outcomes (EV shifts, conversion rates, etc.) across trials with that graph.
4. Writes a single CSV ready for downstream regression / feature-importance analysis.

## Files

- `extract.py` — main script. Default `--output_dir` is `outputs/graph_features/`.
  ```bash
  # single dataset (recommended: put each in its own subfolder)
  python -m misalignment_contagion.graph_features.extract \
      outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl \
      --output_dir outputs/graph_features/synthetic

  # run all 5 datasets
  for ds in synthetic moral_stories harmbench_standard \
            harmbench_copyright harmbench_contextual; do
      python -m misalignment_contagion.graph_features.extract \
          outputs/primary_em/$ds/qwen-7b-instruct/results.jsonl \
          --output_dir outputs/graph_features/$ds
  done
  ```

  Each run writes `unique_graphs_features.csv` and `per_graph_per_dataset.csv`
  into the chosen output directory. Without a per-dataset subfolder, the files
  are overwritten on each invocation.

## Status

Work in progress (added 2026-05-24). Not yet wired into [`scripts/run_all.sh`](../../scripts/run_all.sh).
