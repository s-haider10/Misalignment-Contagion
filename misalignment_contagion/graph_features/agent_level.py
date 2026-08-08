"""Per-agent regression: predict individual aligned-agent shadow shift
from agent-level graph features.

Stronger test than graph-level mean: an aligned agent next to a misaligned
agent should shift differently from one three hops away. Mean-shift
averages over agents that shouldn't behave the same.

Builds, for each aligned agent in each trial:
  Outcome: shadow_EV - baseline_EV
  Features:
    - dist_to_nearest_misaligned (undirected shortest path)
    - n_misaligned_1hop  (neighbors who are misaligned)
    - n_misaligned_2hop  (within 2 hops)
    - in_degree, out_degree
    - betweenness, closeness  (undirected, normalized)
    - frac_neighbors_misaligned
  Controls:
    - dataset (one-hot), minority_ratio, n_agents
  Fixed effects optional:
    - graph_id dummies (eliminates between-graph variation -> pure within-graph signal)
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


MANIFEST = Path("outputs/graph_features/graph_manifest_subset/subset_manifest.json")
RUNS = Path("outputs/graph_features/runs")
DATASETS = ["synthetic", "moral_stories"]


def expected_value(probs: dict) -> float:
    return sum(int(k) * v for k, v in probs.items())


def build_graph_lookup() -> dict:
    """graph_id -> {nx.DiGraph G, undirected nx.Graph G_und, features per node}."""
    manifest = json.load(open(MANIFEST))
    out = {}
    for entry in manifest:
        gid = entry["graph_id"]
        n = entry["n_agents"]
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i)
        for u, v in entry["edges_directed"]:
            G.add_edge(u, v)
        G_und = G.to_undirected()
        # Pre-compute per-node centrality on the undirected graph
        closeness = nx.closeness_centrality(G_und)
        betweenness = nx.betweenness_centrality(G_und)
        out[gid] = {
            "G": G, "G_und": G_und, "n": n,
            "closeness": closeness, "betweenness": betweenness,
            "misaligned_positions": set(entry["misaligned_positions"]),
        }
    return out


def per_agent_features(gid_info: dict, pos: int, misaligned_set: set) -> dict:
    """Compute per-agent features given the precomputed graph info."""
    G = gid_info["G"]
    G_und = gid_info["G_und"]
    n = gid_info["n"]

    # Distance to nearest misaligned (undirected)
    if pos in misaligned_set:
        dist_to_mis = 0
    else:
        dists = nx.single_source_shortest_path_length(G_und, pos)
        mis_dists = [d for p, d in dists.items() if p in misaligned_set]
        dist_to_mis = min(mis_dists) if mis_dists else n  # disconnected -> big

    # Neighborhoods
    one_hop = set(G_und.neighbors(pos))
    two_hop = set(one_hop)
    for nb in list(one_hop):
        two_hop.update(G_und.neighbors(nb))
    two_hop.discard(pos)

    n_mis_1hop = len(one_hop & misaligned_set)
    n_mis_2hop = len(two_hop & misaligned_set)
    n_neigh = len(one_hop)
    frac_mis_neigh = n_mis_1hop / n_neigh if n_neigh > 0 else 0.0

    return {
        "dist_to_nearest_misaligned": dist_to_mis,
        "n_misaligned_1hop": n_mis_1hop,
        "n_misaligned_2hop": n_mis_2hop,
        "n_neighbors": n_neigh,
        "frac_neighbors_misaligned": frac_mis_neigh,
        "in_degree": G.in_degree(pos),
        "out_degree": G.out_degree(pos),
        "closeness": gid_info["closeness"][pos],
        "betweenness": gid_info["betweenness"][pos],
    }


def collect_agent_rows(graph_lookup: dict,
                       datasets: list[str] = DATASETS) -> pd.DataFrame:
    """Iterate all shard files, emit one row per aligned-agent trial."""
    rows = []
    for ds in datasets:
        ds_dir = RUNS / ds
        if not ds_dir.is_dir():
            continue
        for shard in sorted(ds_dir.glob("results.gpu*.jsonl")):
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        trial = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if trial.get("model_key") != "qwen-7b-instruct":
                        continue
                    if trial.get("model_condition") != "model_induced":
                        continue
                    gid = trial.get("graph_id")
                    if gid not in graph_lookup:
                        continue
                    gid_info = graph_lookup[gid]
                    misaligned_set = gid_info["misaligned_positions"]
                    for agent in trial["agents"]:
                        if agent.get("role") != "aligned":
                            continue
                        if "shadow_probs" not in agent or "baseline_probs" not in agent:
                            continue
                        try:
                            shift = (expected_value(agent["shadow_probs"])
                                     - expected_value(agent["baseline_probs"]))
                        except (KeyError, TypeError, ValueError):
                            continue
                        pos = agent["position_in_topology"]
                        feats = per_agent_features(gid_info, pos, misaligned_set)
                        row = {
                            "graph_id": gid,
                            "dataset": ds,
                            "scenario_id": trial["scenario_id"],
                            "seed": trial["seed"],
                            "agent_id": agent["agent_id"],
                            "position": pos,
                            "shift": shift,
                            "n_agents": trial["n_agents"],
                            "n_misaligned": trial["n_misaligned"],
                            "minority_ratio": trial["minority_ratio"],
                            **feats,
                        }
                        rows.append(row)
    return pd.DataFrame(rows)


def cv_r2(X: np.ndarray, y: np.ndarray, model_name: str,
          n_splits: int = 5, seed: int = 42) -> float:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        if model_name == "lasso":
            m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3, max_iter=20000, random_state=42)
        elif model_name == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=3)
        else:
            m = LinearRegression()
        m.fit(Xtr, y[tr])
        yt_all.append(y[te])
        yp_all.append(m.predict(Xte))
    return r2_score(np.concatenate(yt_all), np.concatenate(yp_all))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/agent_level"))
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Subsample rows for quick iteration (debug)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Building graph lookup (151 graphs)...")
    graph_lookup = build_graph_lookup()
    print(f"  {len(graph_lookup)} graphs loaded")

    print("Collecting per-agent rows from shard files...")
    df = collect_agent_rows(graph_lookup, datasets=DATASETS)
    print(f"  {len(df)} per-agent observations")
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
        print(f"  subsampled to {len(df)}")

    out_csv = args.out_dir / "per_agent_observations.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    y = df["shift"].values

    # Feature groups
    agent_feature_cols = [
        "dist_to_nearest_misaligned",
        "n_misaligned_1hop",
        "n_misaligned_2hop",
        "n_neighbors",
        "frac_neighbors_misaligned",
        "in_degree",
        "out_degree",
        "closeness",
        "betweenness",
    ]
    control_cols = ["minority_ratio", "n_agents"]
    dataset_dummies = pd.get_dummies(df["dataset"], prefix="dataset",
                                      drop_first=True).astype(float).values

    configs = [
        ("D0: dataset only",
         dataset_dummies),
        ("D1: dataset + minority_ratio + n_agents",
         np.hstack([df[control_cols].values, dataset_dummies])),
        ("A: + agent-level features (graph structure around the agent)",
         np.hstack([df[control_cols].values, dataset_dummies,
                    df[agent_feature_cols].values])),
        ("A2: agent features only (no controls)",
         df[agent_feature_cols].values),
    ]

    print("\n" + "=" * 95)
    print(f"PER-AGENT REGRESSION on {len(df):,} aligned-agent observations")
    print("Outcome: per-agent shadow shift (EV_shadow - EV_baseline)")
    print("=" * 95)
    print(f"{'configuration':>62}  {'Linear R²':>10}  {'Ridge R²':>10}  {'Lasso R²':>10}")
    rows_out = []
    for label, X in configs:
        r2_l = cv_r2(X, y, "linear")
        r2_r = cv_r2(X, y, "ridge")
        r2_la = cv_r2(X, y, "lasso")
        print(f"{label:>62}  {r2_l:>10.4f}  {r2_r:>10.4f}  {r2_la:>10.4f}")
        rows_out.append({"config": label, "linear_R2": r2_l,
                          "ridge_R2": r2_r, "lasso_R2": r2_la})

    # Fixed-effects test: graph_id dummies absorb all between-graph variation
    print()
    print("=" * 95)
    print("WITHIN-GRAPH analysis (graph_id fixed effects)")
    print("Tests whether agent-level features predict WHO shifts more, controlling for")
    print("everything that varies between graphs.")
    print("=" * 95)
    # Use top graphs to keep dimensionality reasonable; encode graph_id as dummies
    n_unique_graphs = df["graph_id"].nunique()
    print(f"Unique graphs: {n_unique_graphs}")
    gid_dummies = pd.get_dummies(df["graph_id"], prefix="g",
                                  drop_first=True).astype(float).values
    print(f"Graph-id dummies: {gid_dummies.shape[1]} (drops first)")

    X_fe_base = gid_dummies  # absorb all graph-level variation
    X_fe_full = np.hstack([gid_dummies, df[agent_feature_cols].values])

    r2_base_fe = cv_r2(X_fe_base, y, "ridge")
    r2_full_fe = cv_r2(X_fe_full, y, "ridge")
    print(f"\n  Ridge R² with graph_id FE alone:                  {r2_base_fe:.4f}")
    print(f"  Ridge R² with graph_id FE + agent features:       {r2_full_fe:.4f}")
    print(f"  Δ (within-graph contribution of agent features):  {r2_full_fe-r2_base_fe:+.4f}")
    rows_out.append({"config": "FE_baseline (graph_id only)",
                      "linear_R2": np.nan, "ridge_R2": r2_base_fe, "lasso_R2": np.nan})
    rows_out.append({"config": "FE_full (graph_id + agent features)",
                      "linear_R2": np.nan, "ridge_R2": r2_full_fe, "lasso_R2": np.nan})

    pd.DataFrame(rows_out).to_csv(args.out_dir / "agent_level_r2.csv", index=False)
    print(f"\nWrote {args.out_dir / 'agent_level_r2.csv'}")

    # Show coefficients on the agent features (using simple OLS w/ controls)
    from sklearn.linear_model import LinearRegression
    X_for_coef = np.hstack([df[control_cols].values, dataset_dummies,
                             df[agent_feature_cols].values])
    sc = StandardScaler()
    X_std = sc.fit_transform(X_for_coef)
    m = LinearRegression()
    m.fit(X_std, y)
    coefs = pd.DataFrame({
        "feature": (control_cols + [f"dataset_{c}" for c in
                                    sorted(df["dataset"].unique())[1:]] +
                    agent_feature_cols),
        "standardized_coef": m.coef_,
    })
    coefs["abs_coef"] = coefs["standardized_coef"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")
    coefs.to_csv(args.out_dir / "agent_level_coefficients.csv", index=False)
    print("\nStandardized OLS coefficients (X scaled; y raw):")
    print(coefs.to_string(index=False))

    # Visualize: does shift truly differ by distance-to-misaligned?
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Plot 1: mean shift by distance-to-nearest-misaligned, per dataset
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        grp = sub.groupby("dist_to_nearest_misaligned")["shift"].agg(["mean", "sem", "count"])
        axes[0].errorbar(grp.index, grp["mean"], yerr=grp["sem"],
                         marker="o", label=ds, capsize=3)
    axes[0].set_xlabel("Distance to nearest misaligned (graph hops)")
    axes[0].set_ylabel("Mean per-agent shadow shift ± SEM")
    axes[0].set_title("Does distance to misaligned moderate contagion?")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: mean shift by n_misaligned_1hop
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        grp = sub.groupby("n_misaligned_1hop")["shift"].agg(["mean", "sem", "count"])
        axes[1].errorbar(grp.index, grp["mean"], yerr=grp["sem"],
                         marker="o", label=ds, capsize=3)
    axes[1].set_xlabel("# misaligned in 1-hop neighborhood")
    axes[1].set_ylabel("Mean per-agent shadow shift ± SEM")
    axes[1].set_title("Does local misaligned exposure moderate contagion?")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Agent-level descriptive: graph position vs. contagion", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_dir / "agent_level_descriptive.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_dir / 'agent_level_descriptive.png'}")


if __name__ == "__main__":
    main()
