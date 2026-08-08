"""
Bottom-up motif-distinct graph generator.

Enumerates graph instances across 7 topology families, varies parameters
and misaligned-node placements, then deduplicates by motif signature.
Two graphs with identical motif fingerprints collapse to a single instance.

Topology families and their NetworkX primitives:
  fc           : nx.complete_graph(n)            — K_n, bidirectional
  chain        : nx.path_graph(n)                — P_n, directed (i -> i+1)
  circle       : nx.cycle_graph(n)               — C_n, bidirectional
  star         : nx.star_graph(n - 1)            — S_{n-1}, bidirectional
  tree         : nx.balanced_tree(b, d) /
                 nx.random_labeled_tree(n, s)    — bidirectional
  sparse_fc    : nx.erdos_renyi_graph(n, p, s)   — G(n,p), bidirectional
                                                   (kept iff connected)
  small_world  : nx.watts_strogatz_graph(n,k,p,s)— bidirectional
                                                   (kept iff connected)

Position tags computed per node:
  hub                  : argmax over undirected degree (top tier)
  leaf                 : undirected degree == 1
  head                 : in_degree == 0 (chain head)
  tail                 : out_degree == 0 (chain tail)
  root                 : conventional root (node 0 in balanced_tree, BFS root
                                            in random_labeled_tree)
  bridge               : articulation point of undirected projection
  shortcut_endpoint    : endpoint of an edge that is NOT in the k-ring lattice
                         (Watts-Strogatz rewired edge)

For each base graph we enumerate misaligned placements:
  - full C(n, k) for small (n, k)
  - sampled placements for large C(n, k) (cap 200 per base)

Outputs (in --out_dir):
  graph_manifest.json   : list of unique graph instances with full specs
  graph_features.csv    : same set as a feature matrix (one row per instance)
  family_summary.csv    : counts per family + position-tag breakdown
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROLE = {"misaligned": "M", "aligned": "A"}
PLACEMENT_SAMPLE_CAP = 200  # per (base, k)


# ── Directed-graph builders ──────────────────────────────────────────

def to_bidirectional(G_und: nx.Graph) -> nx.DiGraph:
    """Undirected G -> directed G with both (u,v) and (v,u) for each edge."""
    G = nx.DiGraph()
    G.add_nodes_from(G_und.nodes)
    for u, v in G_und.edges:
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G


def directed_chain(n: int) -> nx.DiGraph:
    """P_n with directed edges i -> i+1 (info flow forward)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


# ── Base-graph generators ────────────────────────────────────────────

def make_random_tree(n: int, seed: int) -> nx.Graph:
    """Wrapper around NetworkX's random tree (name varies by version)."""
    try:
        return nx.random_labeled_tree(n, seed=seed)
    except AttributeError:
        return nx.random_tree(n, seed=seed)


def generate_base_graphs() -> list[dict]:
    """Return base graph specs without role assignments."""
    bases: list[dict] = []

    # FC: K_n
    for n in [6, 8, 10, 12, 15]:
        bases.append({"family": "fc", "params": {"n": n},
                      "G_und": nx.complete_graph(n), "directed_chain": False})

    # Chain: P_n with one-way directed edges
    for n in [8, 10, 12, 15]:
        bases.append({"family": "chain", "params": {"n": n},
                      "G_und": None, "G_dir": directed_chain(n)})

    # Circle: C_n
    for n in [8, 10, 12, 15]:
        bases.append({"family": "circle", "params": {"n": n},
                      "G_und": nx.cycle_graph(n)})

    # Star: hub at node 0, n-1 leaves
    for n in [6, 8, 10, 12, 15]:
        bases.append({"family": "star", "params": {"n": n},
                      "G_und": nx.star_graph(n - 1)})

    # Tree: balanced (binary, ternary)
    for d in [2, 3, 4]:
        n = 2 ** (d + 1) - 1
        bases.append({"family": "tree",
                      "params": {"kind": "balanced", "branching": 2, "depth": d, "n": n},
                      "G_und": nx.balanced_tree(2, d)})
    for d in [2, 3]:
        n = (3 ** (d + 1) - 1) // 2
        bases.append({"family": "tree",
                      "params": {"kind": "balanced", "branching": 3, "depth": d, "n": n},
                      "G_und": nx.balanced_tree(3, d)})

    # Tree: random labeled
    for n in [8, 10, 12, 15]:
        for seed in [42, 123, 456]:
            bases.append({"family": "tree",
                          "params": {"kind": "random", "n": n, "seed": seed},
                          "G_und": make_random_tree(n, seed)})

    # Sparse-FC: Erdős-Rényi G(n, p)
    for n in [8, 10, 12, 15]:
        for p in [0.3, 0.5, 0.7]:
            for seed in [42, 123]:
                G = nx.erdos_renyi_graph(n, p, seed=seed)
                if not nx.is_connected(G):
                    continue
                bases.append({"family": "sparse_fc",
                              "params": {"n": n, "p": p, "seed": seed},
                              "G_und": G})

    # Small-world: Watts-Strogatz
    for n in [10, 12, 15]:
        for k_nn in [2, 4]:
            for p_rewire in [0.1, 0.2, 0.3]:
                for seed in [42, 123]:
                    G = nx.watts_strogatz_graph(n, k_nn, p_rewire, seed=seed)
                    if not nx.is_connected(G):
                        continue
                    bases.append({"family": "small_world",
                                  "params": {"n": n, "k_nn": k_nn,
                                             "p_rewire": p_rewire, "seed": seed},
                                  "G_und": G})

    # Attach directed form for non-chain families
    for b in bases:
        if "G_dir" not in b:
            b["G_dir"] = to_bidirectional(b["G_und"])
    return bases


# ── Position tagging ─────────────────────────────────────────────────

def tag_positions(G_dir: nx.DiGraph, family: str, params: dict,
                  G_und_orig: nx.Graph | None = None) -> dict[int, set[str]]:
    """Return per-node set of position tags."""
    G_und = G_und_orig if G_und_orig is not None else G_dir.to_undirected()
    deg = dict(G_und.degree())
    tags: dict[int, set[str]] = {n: set() for n in G_dir.nodes}

    # Hub: highest undirected degree
    if deg:
        max_deg = max(deg.values())
        for n, d in deg.items():
            if d == max_deg:
                tags[n].add("hub")

    # Leaf: degree-1
    for n, d in deg.items():
        if d == 1:
            tags[n].add("leaf")

    # Chain-specific: head (in_deg==0), tail (out_deg==0)
    if family == "chain":
        for n in G_dir.nodes:
            if G_dir.in_degree(n) == 0:
                tags[n].add("head")
            if G_dir.out_degree(n) == 0:
                tags[n].add("tail")

    # Tree-specific: root
    if family == "tree":
        if params.get("kind") == "balanced":
            tags[0].add("root")
        else:
            # For random_labeled_tree, take the centroid (min eccentricity) as root
            ecc = nx.eccentricity(G_und)
            root = min(ecc, key=ecc.get)
            tags[root].add("root")

    # Bridge: articulation point
    for ap in nx.articulation_points(G_und):
        tags[ap].add("bridge")

    # Shortcut endpoint: small-world rewired edges
    if family == "small_world":
        k_nn = params["k_nn"]
        n = G_und.number_of_nodes()
        for u, v in G_und.edges:
            ring_dist = min(abs(u - v), n - abs(u - v))
            if ring_dist > k_nn // 2:
                tags[u].add("shortcut_endpoint")
                tags[v].add("shortcut_endpoint")

    return tags


def summarize_placement_tags(positions: list[int],
                             node_tags: dict[int, set[str]]) -> str:
    """One-line summary of position types for the misaligned set."""
    if not positions:
        return ""
    cat = Counter()
    for p in positions:
        if node_tags[p]:
            for t in node_tags[p]:
                cat[t] += 1
        else:
            cat["interior"] += 1
    return "+".join(f"{n}{tag}" for tag, n in sorted(cat.items()))


# ── Motif signature & features ───────────────────────────────────────

def role_str(roles: dict[int, str], n: int) -> str:
    return ROLE[roles[n]]


def motif_signature(G_dir: nx.DiGraph, roles: dict[int, str]) -> tuple:
    """Canonical motif fingerprint: edge-type / 3-path / triangle counts."""
    n_nodes = G_dir.number_of_nodes()
    n_edges = G_dir.number_of_edges()

    # 2-node motifs
    edge_types = Counter()
    for u, v in G_dir.edges:
        edge_types[role_str(roles, u) + role_str(roles, v)] += 1

    # 3-node directed paths X -> Y -> Z
    path_types = Counter()
    for y in G_dir.nodes:
        preds = list(G_dir.predecessors(y))
        succs = list(G_dir.successors(y))
        for x in preds:
            for z in succs:
                if x != z:
                    path_types[role_str(roles, x) + role_str(roles, y) + role_str(roles, z)] += 1

    # Directed 3-cycles
    triangles = Counter()
    edges = set(G_dir.edges)
    nodes_sorted = sorted(G_dir.nodes)
    for x, y, z in combinations(nodes_sorted, 3):
        if (x, y) in edges and (y, z) in edges and (z, x) in edges:
            triangles["".join(sorted([role_str(roles, x),
                                       role_str(roles, y),
                                       role_str(roles, z)]))] += 1
        if (x, z) in edges and (z, y) in edges and (y, x) in edges:
            triangles["".join(sorted([role_str(roles, x),
                                       role_str(roles, y),
                                       role_str(roles, z)]))] += 1

    return (
        n_nodes, n_edges,
        tuple(sorted(edge_types.items())),
        tuple(sorted(path_types.items())),
        tuple(sorted(triangles.items())),
    )


def compute_feature_row(G_dir: nx.DiGraph, roles: dict[int, str]) -> dict:
    """Same features as extract_graph_features.py, ready for regression."""
    G_und = G_dir.to_undirected()
    misaligned = [n for n in G_dir.nodes if roles[n] == "misaligned"]
    aligned = [n for n in G_dir.nodes if roles[n] == "aligned"]

    feats: dict = {
        "n_nodes": G_dir.number_of_nodes(),
        "n_edges_directed": G_dir.number_of_edges(),
        "n_edges_undirected": G_und.number_of_edges(),
        "density": nx.density(G_dir),
        "n_misaligned": len(misaligned),
        "n_aligned": len(aligned),
    }

    # Centralities (on undirected projection)
    cents = {
        "degree": nx.degree_centrality(G_und),
        "closeness": nx.closeness_centrality(G_und),
        "betweenness": nx.betweenness_centrality(G_und),
    }
    try:
        cents["eigenvector"] = nx.eigenvector_centrality_numpy(G_und, max_iter=1000)
    except Exception:
        cents["eigenvector"] = {n: 0.0 for n in G_und.nodes}

    def mean_over(d, nodes):
        return float(np.mean([d[n] for n in nodes])) if nodes else 0.0

    for name, c in cents.items():
        feats[f"mean_{name}_misaligned"] = mean_over(c, misaligned)
        feats[f"mean_{name}_aligned"] = mean_over(c, aligned)

    # 2-node motifs
    edge_types = Counter()
    for u, v in G_dir.edges:
        edge_types[role_str(roles, u) + role_str(roles, v)] += 1
    for pair in ["MM", "MA", "AM", "AA"]:
        feats[f"n_edges_{pair}"] = edge_types[pair]
    feats["n_types_2node"] = sum(1 for v in edge_types.values() if v > 0)

    # 3-node paths
    path_types = Counter()
    for y in G_dir.nodes:
        preds = list(G_dir.predecessors(y))
        succs = list(G_dir.successors(y))
        for x in preds:
            for z in succs:
                if x != z:
                    path_types[role_str(roles, x) + role_str(roles, y) + role_str(roles, z)] += 1
    for triple in ["AAA", "AAM", "AMA", "AMM", "MAA", "MAM", "MMA", "MMM"]:
        feats[f"n_3node_path_{triple}"] = path_types[triple]
    feats["n_types_3node_path"] = sum(1 for v in path_types.values() if v > 0)

    # Triangles
    n_tri = 0
    tri_types = Counter()
    edges = set(G_dir.edges)
    for x, y, z in combinations(sorted(G_dir.nodes), 3):
        if (x, y) in edges and (y, z) in edges and (z, x) in edges:
            n_tri += 1
            tri_types["".join(sorted([role_str(roles, x),
                                       role_str(roles, y),
                                       role_str(roles, z)]))] += 1
        if (x, z) in edges and (z, y) in edges and (y, x) in edges:
            n_tri += 1
            tri_types["".join(sorted([role_str(roles, x),
                                       role_str(roles, y),
                                       role_str(roles, z)]))] += 1
    feats["n_triangles"] = n_tri
    for tri in ["AAA", "AAM", "AMM", "MMM"]:
        feats[f"n_triangle_{tri}"] = tri_types.get(tri, 0)

    return feats


def sig_to_hash(sig: tuple) -> str:
    return hashlib.md5(repr(sig).encode()).hexdigest()[:12]


# ── Enumeration ──────────────────────────────────────────────────────

def enumerate_placements(base: dict, k_values: tuple = (1, 2, 3),
                         rng: np.random.Generator | None = None) -> list[dict]:
    """For one base graph, yield candidate (positions, roles) instances."""
    G_dir = base["G_dir"]
    n = G_dir.number_of_nodes()
    family = base["family"]
    rng = rng if rng is not None else np.random.default_rng(42)

    out = []
    for k in k_values:
        if k >= n:
            continue
        # FC and circle are vertex-transitive: only (n, k) matters, place at 0..k-1
        if family in ("fc", "circle"):
            placements = [tuple(range(k))]
        else:
            all_p = list(combinations(range(n), k))
            if len(all_p) > PLACEMENT_SAMPLE_CAP:
                idx = rng.choice(len(all_p), size=PLACEMENT_SAMPLE_CAP, replace=False)
                placements = [all_p[i] for i in idx]
            else:
                placements = all_p

        for positions in placements:
            roles = {i: ("misaligned" if i in positions else "aligned")
                     for i in range(n)}
            out.append({
                "family": family,
                "params": {**base["params"], "k": k},
                "G_dir": G_dir,
                "G_und": base.get("G_und"),
                "positions": list(positions),
                "roles": roles,
            })
    return out


# ── Main pipeline ────────────────────────────────────────────────────

def build_manifest(out_dir: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    bases = generate_base_graphs()
    print(f"Generated {len(bases)} base graphs across "
          f"{len(set(b['family'] for b in bases))} families.")

    rng = np.random.default_rng(42)
    candidates: list[dict] = []
    for b in bases:
        candidates.extend(enumerate_placements(b, rng=rng))
    print(f"Total placement candidates: {len(candidates):,}")

    unique: dict[tuple, dict] = {}
    for cand in candidates:
        sig = motif_signature(cand["G_dir"], cand["roles"])
        if sig in unique:
            continue
        node_tags = tag_positions(cand["G_dir"], cand["family"],
                                  cand["params"], cand.get("G_und"))
        placement_summary = summarize_placement_tags(cand["positions"], node_tags)
        n = cand["G_dir"].number_of_nodes()
        k = sum(1 for r in cand["roles"].values() if r == "misaligned")
        graph_id = (f"{cand['family']}__n{n}_k{k}__"
                    f"{placement_summary or 'plain'}__{sig_to_hash(sig)}")
        unique[sig] = {
            "graph_id": graph_id,
            "family": cand["family"],
            "params": cand["params"],
            "n_agents": n,
            "n_misaligned": k,
            "minority_ratio": round(k / n, 4),
            "misaligned_positions": cand["positions"],
            "placement_summary": placement_summary,
            "node_position_tags": {int(n_): sorted(t)
                                   for n_, t in node_tags.items()},
            "edges_directed": [[int(u), int(v)] for u, v in cand["G_dir"].edges],
            "motif_signature_hash": sig_to_hash(sig),
            "_features": compute_feature_row(cand["G_dir"], cand["roles"]),
        }
    print(f"Unique motif-distinct graphs: {len(unique):,}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest (JSON, drop _features which go to CSV)
    manifest = []
    feature_rows = []
    for spec in unique.values():
        feats = spec.pop("_features")
        manifest.append(spec)
        row = {"graph_id": spec["graph_id"],
               "family": spec["family"],
               "n_agents": spec["n_agents"],
               "n_misaligned": spec["n_misaligned"],
               "minority_ratio": spec["minority_ratio"],
               "placement_summary": spec["placement_summary"],
               **feats}
        feature_rows.append(row)

    with open(out_dir / "graph_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    df_feat = pd.DataFrame(feature_rows).sort_values(
        ["family", "n_agents", "n_misaligned", "graph_id"]
    ).reset_index(drop=True)
    df_feat.to_csv(out_dir / "graph_features.csv", index=False)

    # Family + placement summary
    df_summary = df_feat.groupby(["family"]).agg(
        n_graphs=("graph_id", "count"),
        n_distinct_n=("n_agents", "nunique"),
        n_distinct_k=("n_misaligned", "nunique"),
        n_distinct_placements=("placement_summary", "nunique"),
    ).reset_index()
    df_summary.to_csv(out_dir / "family_summary.csv", index=False)

    return manifest, df_feat, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path,
                        default=Path("outputs/graph_manifest"))
    args = parser.parse_args()
    manifest, df_feat, df_summary = build_manifest(args.out_dir)

    print("\n── Summary by family ──")
    print(df_summary.to_string(index=False))
    print(f"\n→ {args.out_dir}/graph_manifest.json")
    print(f"→ {args.out_dir}/graph_features.csv ({len(df_feat)} rows, "
          f"{df_feat.shape[1]} columns)")
    print(f"→ {args.out_dir}/family_summary.csv")