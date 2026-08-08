"""
Extract graph features and pooled contagion outcomes per unique deliberation
topology, for the misalignment-contagion graph-features-to-prediction extension.

A "unique graph" is determined by (topology, minority_ratio, position_config).
For each unique graph we:
  1. Reconstruct the directed visibility graph using the topology rules from
     topologies.py (edge u -> v means v can see u, i.e. info flows u -> v).
  2. Compute structural and centrality features (centralities on the
     undirected projection for stability; motifs on the directed graph).
  3. Compute 2-node motif counts (directed edges by role pair).
  4. Compute 3-node motif counts (directed paths X->Y->Z by role triple,
     plus directed 3-cycles by role composition).
  5. Pool aligned-agent shadow shifts (shadow_EV - baseline_EV) across all
     trials sharing this unique graph; report mean and variance as targets.

Outputs:
  - unique_graphs_features.csv  (one row per unique graph; main regression input)
  - per_graph_per_dataset.csv   (per-(graph, dataset) breakdown for diagnostics)

Usage:
    python extract_graph_features.py /path/to/trials_dir --output_dir ./out
    uv run python misalignment_contagion/graph_features/extract.py outputs/primary_em/harmbench_contextual/qwen-7b-instruct/results.jsonl
"""


from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


N_AGENTS = 10
ROLE_CODE = {"misaligned": "M", "aligned": "A"}


# ── Graph reconstruction ─────────────────────────────────────────────

def reconstruct_directed_graph(topology: str, agents: list[dict]) -> nx.DiGraph:
    """Build the directed visibility graph from agent positions and roles.

    Edge convention: edge u -> v means position v can see position u, i.e.
    information flows u -> v. Mirrors visibility rules in topologies.py.
    """
    pos_to_role = {a["position_in_topology"]: a["role"] for a in agents}
    G = nx.DiGraph()
    for pos in range(N_AGENTS):
        G.add_node(pos, role=pos_to_role[pos])

    for pos in range(N_AGENTS):
        if topology == "fc":
            visible = [p for p in range(N_AGENTS) if p != pos]
        elif topology == "chain":
            visible = [pos - 1] if pos > 0 else []
        elif topology == "circle":
            visible = [(pos - 1) % N_AGENTS, (pos + 1) % N_AGENTS]
        elif topology == "star":
            # Hub is at position 0 by convention (see topologies.py)
            if pos == 0:
                visible = [p for p in range(N_AGENTS) if p != 0]
            else:
                visible = [0]
        else:
            raise ValueError(f"Unknown topology: {topology}")

        for v in visible:
            G.add_edge(v, pos)

    return G


# ── Outcomes ─────────────────────────────────────────────────────────

def expected_value(probs: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in probs.items())


def aligned_shadow_shifts(trial: dict) -> list[float]:
    """Per-aligned-agent (shadow_EV - baseline_EV) for one trial."""
    shifts = []
    for agent in trial["agents"]:
        if agent.get("role") != "aligned":
            continue
        if "shadow_probs" not in agent or "baseline_probs" not in agent:
            continue
        try:
            shift = (expected_value(agent["shadow_probs"])
                     - expected_value(agent["baseline_probs"]))
            shifts.append(shift)
        except (KeyError, TypeError, ValueError):
            continue
    return shifts


# ── Features ─────────────────────────────────────────────────────────

def role_at(G: nx.DiGraph, node: int) -> str:
    return ROLE_CODE[G.nodes[node]["role"]]


def compute_features(G_dir: nx.DiGraph) -> dict:
    """Structural, centrality, and motif features for one graph."""
    G_und = G_dir.to_undirected()
    misaligned = [n for n in G_dir.nodes if role_at(G_dir, n) == "M"]
    aligned = [n for n in G_dir.nodes if role_at(G_dir, n) == "A"]

    feats: dict = {}

    # ── Structural ──
    feats["n_nodes"] = G_dir.number_of_nodes()
    feats["n_edges_directed"] = G_dir.number_of_edges()
    feats["n_edges_undirected"] = G_und.number_of_edges()
    feats["density"] = nx.density(G_dir)
    feats["n_misaligned"] = len(misaligned)
    feats["n_aligned"] = len(aligned)

    # ── Centralities (undirected projection for numerical stability) ──
    centralities = {
        "degree": nx.degree_centrality(G_und),
        "closeness": nx.closeness_centrality(G_und),
        "betweenness": nx.betweenness_centrality(G_und),
    }
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality_numpy(
            G_und, max_iter=1000
        )
    except Exception:
        centralities["eigenvector"] = {n: 0.0 for n in G_und.nodes}

    def mean_over(d, nodes):
        return float(np.mean([d[n] for n in nodes])) if nodes else 0.0

    for name, c in centralities.items():
        feats[f"mean_{name}_misaligned"] = mean_over(c, misaligned)
        feats[f"mean_{name}_aligned"] = mean_over(c, aligned)

    # ── 2-node motifs: directed edges by role pair ──
    edge_type_counts = Counter()
    for u, v in G_dir.edges:
        edge_type_counts[role_at(G_dir, u) + role_at(G_dir, v)] += 1
    for pair in ["MM", "MA", "AM", "AA"]:
        feats[f"n_edges_{pair}"] = edge_type_counts[pair]
    feats["n_types_2node"] = sum(1 for v in edge_type_counts.values() if v > 0)

    # ── 3-node motifs: directed paths X -> Y -> Z by role triple ──
    path_type_counts = Counter()
    for y in G_dir.nodes:
        preds = list(G_dir.predecessors(y))
        succs = list(G_dir.successors(y))
        for x in preds:
            for z in succs:
                if x == z:
                    continue
                key = role_at(G_dir, x) + role_at(G_dir, y) + role_at(G_dir, z)
                path_type_counts[key] += 1
    for triple in ["AAA", "AAM", "AMA", "AMM", "MAA", "MAM", "MMA", "MMM"]:
        feats[f"n_3node_path_{triple}"] = path_type_counts[triple]
    feats["n_types_3node_path"] = sum(1 for v in path_type_counts.values() if v > 0)

    # ── Directed 3-cycles (triangles) ──
    triangle_count = 0
    triangle_type_counts = Counter()
    edges = set(G_dir.edges)
    for x, y, z in combinations(G_dir.nodes, 3):
        # x -> y -> z -> x
        if (x, y) in edges and (y, z) in edges and (z, x) in edges:
            triangle_count += 1
            roles = "".join(sorted([role_at(G_dir, x),
                                    role_at(G_dir, y),
                                    role_at(G_dir, z)]))
            triangle_type_counts[roles] += 1
        # x -> z -> y -> x (the other rotation)
        if (x, z) in edges and (z, y) in edges and (y, x) in edges:
            triangle_count += 1
            roles = "".join(sorted([role_at(G_dir, x),
                                    role_at(G_dir, y),
                                    role_at(G_dir, z)]))
            triangle_type_counts[roles] += 1
    feats["n_triangles"] = triangle_count
    # Sorted role compositions: A < M
    for tri in ["AAA", "AAM", "AMM", "MMM"]:
        feats[f"n_triangle_{tri}"] = triangle_type_counts.get(tri, 0)

    return feats


# ── Trial I/O ────────────────────────────────────────────────────────

def iter_trials(input_paths):
    """Yield trial dicts from one or more files/directories."""
    if isinstance(input_paths, (str, Path)):
        input_paths = [Path(input_paths)]
    else:
        input_paths = [Path(p) for p in input_paths]

    paths: list[Path] = []
    for p in input_paths:
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(
                q for q in p.iterdir() if q.suffix in (".json", ".jsonl")
            ))
        else:
            raise FileNotFoundError(p)

    for fp in paths:
        with open(fp) as f:
            content = f.read().strip()
        if not content:
            continue
        if content.startswith("["):
            for trial in json.loads(content):
                yield trial
            continue
        # Could be a single JSON object or JSONL
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                yield obj
                continue
            if isinstance(obj, list):
                for t in obj:
                    yield t
                continue
        except json.JSONDecodeError:
            pass
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Pipeline ─────────────────────────────────────────────────────────

def run(input_path, output_dir: Path,
        filter_model_key: str | None = "qwen-7b-instruct",
        filter_model_condition: str | None = "model_induced"):
    """input_path may be a single Path or a list of Paths (mix of files/dirs)."""

    grouped_shifts: dict[tuple, list[float]] = defaultdict(list)
    grouped_by_dataset: dict[tuple, list[float]] = defaultdict(list)
    grouped_n_trials: dict[tuple, int] = defaultdict(int)
    grouped_n_trials_by_dataset: dict[tuple, int] = defaultdict(int)
    grouped_first_trial: dict[tuple, dict] = {}

    n_total = 0
    n_kept = 0
    for trial in iter_trials(input_path):
        n_total += 1
        if filter_model_key and trial.get("model_key") != filter_model_key:
            continue
        if filter_model_condition and trial.get("model_condition") != filter_model_condition:
            continue

        key = (trial["topology"], trial["minority_ratio"], trial["position_config"])
        key_ds = key + (trial.get("dataset", "unknown"),)

        shifts = aligned_shadow_shifts(trial)
        if not shifts:
            continue

        n_kept += 1
        grouped_shifts[key].extend(shifts)
        grouped_by_dataset[key_ds].extend(shifts)
        grouped_n_trials[key] += 1
        grouped_n_trials_by_dataset[key_ds] += 1
        grouped_first_trial.setdefault(key, trial)

    print(f"Processed {n_total} trials; kept {n_kept} after filters/validity.")

    # ── Per-unique-graph table ──
    rows = []
    for key, trial in grouped_first_trial.items():
        topology, ratio, pos_config = key
        G = reconstruct_directed_graph(topology, trial["agents"])
        feats = compute_features(G)
        shifts = np.array(grouped_shifts[key])
        rows.append({
            "topology": topology,
            "minority_ratio": ratio,
            "position_config": pos_config,
            "n_trials": grouped_n_trials[key],
            "n_aligned_obs": len(shifts),
            **feats,
            "y_mean_shadow_shift": float(shifts.mean()),
            "y_var_shadow_shift": float(shifts.var(ddof=1)) if len(shifts) > 1 else 0.0,
        })
    df_main = pd.DataFrame(rows).sort_values(
        ["topology", "minority_ratio", "position_config"]
    ).reset_index(drop=True)

    # ── Per-(graph, dataset) breakdown ──
    rows_ds = []
    for key_ds, shifts_list in grouped_by_dataset.items():
        topology, ratio, pos_config, dataset = key_ds
        shifts = np.array(shifts_list)
        rows_ds.append({
            "topology": topology,
            "minority_ratio": ratio,
            "position_config": pos_config,
            "dataset": dataset,
            "n_trials": grouped_n_trials_by_dataset[key_ds],
            "n_aligned_obs": len(shifts),
            "y_mean_shadow_shift": float(shifts.mean()),
            "y_var_shadow_shift": float(shifts.var(ddof=1)) if len(shifts) > 1 else 0.0,
        })
    df_ds = pd.DataFrame(rows_ds).sort_values(
        ["topology", "minority_ratio", "position_config", "dataset"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_out = output_dir / "unique_graphs_features.csv"
    ds_out = output_dir / "per_graph_per_dataset.csv"
    df_main.to_csv(main_out, index=False)
    df_ds.to_csv(ds_out, index=False)

    print(f"\nWrote {len(df_main)} unique graphs   -> {main_out}")
    print(f"Wrote {len(df_ds)} graph-dataset rows -> {ds_out}\n")
    print("Sanity summary (pooled):")
    summary_cols = ["topology", "minority_ratio", "position_config",
                    "n_trials", "n_aligned_obs",
                    "y_mean_shadow_shift", "y_var_shadow_shift"]
    print(df_main[summary_cols].to_string(index=False))

    return df_main, df_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="+",
                        help="One or more trial JSON files or directories. "
                             "Pass all 5 dataset results.jsonl files to get a "
                             "single per_graph_per_dataset.csv covering all "
                             "datasets and a single unique_graphs_features.csv "
                             "pooled across them.")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/graph_features"),
                        help="Output directory for CSVs "
                             "(default: outputs/graph_features). "
                             "For per-dataset runs, pass a dataset subfolder, "
                             "e.g. outputs/graph_features/synthetic")
    parser.add_argument("--model_key", default="qwen-7b-instruct",
                        help='Filter by model_key (pass "" to disable)')
    parser.add_argument("--model_condition", default="model_induced",
                        help='Filter by model_condition (pass "" to disable)')
    args = parser.parse_args()

    run(args.input, args.output_dir,
        filter_model_key=args.model_key or None,
        filter_model_condition=args.model_condition or None)