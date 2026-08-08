"""Full exploratory analysis on the 5 datasets (master files).

Loads outputs/master/<dataset>.jsonl, computes II/SRF per aligned agent,
joins per-agent graph features, runs every regression we have plus a few
exploratory ones (GBM, interaction terms, node-features-only).

Writes:
  outputs/analysis/per_agent/<dataset>.csv          # one row per aligned-agent trial
  outputs/analysis/graph_level/<dataset>.csv        # graph-level pooled outcomes
  outputs/analysis/regression_summary.csv           # all R² across all (dataset, strategy, model)
  outputs/analysis/coefficients_lasso.csv           # selected coefs per outcome × dataset
  outputs/analysis/per_agent_summary.csv            # per-agent R² + FE decomposition
  outputs/analysis/motif_correlations.csv           # 3-node motif vs II/SRF
  outputs/analysis/report.md                        # narrative summary
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from misalignment_contagion.metrics import (
    internalization_index, shadow_reversion_fraction
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "outputs/master"
OUT = ROOT / "outputs/analysis"
MANIFEST_FEAT = ROOT / "outputs/graph_features/graph_manifest_subset/subset_features.csv"
MANIFEST_JSON = ROOT / "outputs/graph_features/graph_manifest_subset/subset_manifest.json"

DATASETS = ["synthetic", "moral_stories", "harmbench_standard",
            "harmbench_contextual", "harmbench_copyright"]


# ── Helpers ──────────────────────────────────────────────────────────
def fix_keys(d):
    if d is None:
        return None
    return {int(k): float(v) for k, v in d.items()}


def ev(probs):
    return sum(int(k) * v for k, v in probs.items())


def load_manifest_graph_lookup() -> dict:
    """graph_id -> precomputed structural info."""
    m = json.load(open(MANIFEST_JSON))
    out = {}
    for entry in m:
        n = entry["n_agents"]
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i)
        for u, v in entry["edges_directed"]:
            G.add_edge(u, v)
        Gu = G.to_undirected()
        out[entry["graph_id"]] = {
            "G": G, "G_und": Gu, "n": n,
            "misaligned_positions": set(entry["misaligned_positions"]),
            "closeness": nx.closeness_centrality(Gu),
            "betweenness": nx.betweenness_centrality(Gu),
        }
    return out


def per_agent_features(info: dict, pos: int) -> dict:
    G = info["G"]; Gu = info["G_und"]; n = info["n"]
    mis = info["misaligned_positions"]
    if pos in mis:
        dist = 0
    else:
        d = nx.single_source_shortest_path_length(Gu, pos)
        md = [v for p, v in d.items() if p in mis]
        dist = min(md) if md else n
    one_hop = set(Gu.neighbors(pos))
    two_hop = set(one_hop)
    for nb in list(one_hop):
        two_hop.update(Gu.neighbors(nb))
    two_hop.discard(pos)
    n_n = len(one_hop)
    return {
        "dist_to_nearest_misaligned": dist,
        "n_misaligned_1hop": len(one_hop & mis),
        "n_misaligned_2hop": len(two_hop & mis),
        "n_neighbors": n_n,
        "frac_neighbors_misaligned": (len(one_hop & mis) / n_n) if n_n else 0.0,
        "in_degree": G.in_degree(pos),
        "out_degree": G.out_degree(pos),
        "closeness": info["closeness"][pos],
        "betweenness": info["betweenness"][pos],
    }


AGENT_FEATURES = [
    "dist_to_nearest_misaligned", "n_misaligned_1hop", "n_misaligned_2hop",
    "n_neighbors", "frac_neighbors_misaligned",
    "in_degree", "out_degree", "closeness", "betweenness",
]


def build_per_agent_df(dataset: str, graph_lookup: dict) -> pd.DataFrame:
    """Load master jsonl for a dataset, compute II/SRF + per-agent features."""
    path = MASTER / f"{dataset}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    cache: dict[tuple, dict] = {}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            if t.get("model_key") != "qwen-7b-instruct":
                continue
            if t.get("model_condition") != "model_induced":
                continue
            gid = t.get("graph_id")
            if gid not in graph_lookup:
                continue
            for ag in t["agents"]:
                if ag.get("role") != "aligned":
                    continue
                bp = ag.get("baseline_probs"); sp_ = ag.get("shadow_probs")
                rp = ag.get("round_probs")
                if bp is None or sp_ is None or not rp:
                    continue
                fin = rp[-1] if isinstance(rp[-1], dict) else None
                if fin is None:
                    continue
                try:
                    base_ev = ev(bp); fin_ev = ev(fin); sh_ev = ev(sp_)
                    ii = internalization_index(fix_keys(bp), fix_keys(fin), fix_keys(sp_))
                    srf = shadow_reversion_fraction(base_ev, fin_ev, sh_ev)
                except Exception:
                    continue
                pos = ag["position_in_topology"]
                key = (gid, pos)
                if key not in cache:
                    cache[key] = per_agent_features(graph_lookup[gid], pos)
                rows.append({
                    "graph_id": gid,
                    "scenario_id": t["scenario_id"],
                    "agent_id": ag["agent_id"],
                    "position": pos,
                    "II": ii, "SRF": srf,
                    "shift_ev": sh_ev - base_ev,
                    "abs_shift": abs(sh_ev - base_ev),
                    "n_agents": t["n_agents"],
                    "n_misaligned": t["n_misaligned"],
                    "minority_ratio": t["minority_ratio"],
                    **cache[key],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["II", "SRF"]:
        valid = df[col].dropna()
        if len(valid) < 10:
            continue
        q01, q99 = valid.quantile([0.01, 0.99])
        df[col] = df[col].clip(q01, q99)
    df["dataset"] = dataset
    return df


def graph_level_from_agents(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.dropna(subset=["II", "SRF"])
              .groupby("graph_id")
              .agg(y_mean_shift=("shift_ev", "mean"),
                   y_var_shift=("shift_ev", "var"),
                   y_mean_abs_shift=("abs_shift", "mean"),
                   y_mean_II=("II", "mean"),
                   y_var_II=("II", "var"),
                   y_mean_SRF=("SRF", "mean"),
                   y_var_SRF=("SRF", "var"),
                   n_obs=("II", "size"))
              .reset_index())


# ── Regression helpers ───────────────────────────────────────────────
def cv_r2(X, y, model, n_splits=5, seed=42):
    if len(X) < n_splits:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in kf.split(X):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if model == "lasso":
            m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3, max_iter=20000, random_state=42)
        elif model == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=3)
        elif model == "gbm":
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42)
        else:
            m = LinearRegression()
        m.fit(Xtr, y[tr])
        yt_all.append(y[te]); yp_all.append(m.predict(Xte))
    return r2_score(np.concatenate(yt_all), np.concatenate(yp_all))


def feature_cols_for(gl: pd.DataFrame) -> tuple[list, list]:
    """Return (control_cols, topology_cols)."""
    control = ["minority_ratio", "n_agents", "n_misaligned"]
    drop = {"graph_id", "family", "dataset", "placement_summary", "n_obs",
            "y_mean_shift", "y_var_shift", "y_mean_abs_shift",
            "y_mean_II", "y_var_II", "y_mean_SRF", "y_var_SRF"}
    topo = [c for c in gl.columns if c not in drop and c not in control
            and np.issubdtype(gl[c].dtype, np.number)]
    return control, topo


def run_graph_regressions(gl_all: dict[str, pd.DataFrame],
                            outcomes: list[str], models: list[str]) -> pd.DataFrame:
    """Strategy C (per-dataset, with controls + topology) for every (ds, target, model)."""
    rows = []
    manifest = pd.read_csv(MANIFEST_FEAT)
    for ds, gl in gl_all.items():
        if gl.empty:
            continue
        gl_j = gl.merge(manifest, on="graph_id", how="left", validate="many_to_one")
        control, topo = feature_cols_for(gl_j)
        X_base = gl_j[control].values
        X_full = np.hstack([X_base, gl_j[topo].values])
        X_topo_only = gl_j[topo].values
        for tgt in outcomes:
            if tgt not in gl_j.columns:
                continue
            y = gl_j[tgt].values
            for model in models:
                r2_b = cv_r2(X_base, y, model)
                r2_f = cv_r2(X_full, y, model)
                r2_t = cv_r2(X_topo_only, y, model)
                rows.append({
                    "dataset": ds, "target": tgt, "model": model,
                    "n_graphs": len(gl_j),
                    "r2_baseline": r2_b,
                    "r2_topology_only": r2_t,
                    "r2_full": r2_f,
                    "delta_r2_topology": r2_f - r2_b,
                })
    return pd.DataFrame(rows)


def run_pooled_regressions(gl_all: dict[str, pd.DataFrame],
                              outcomes: list[str]) -> pd.DataFrame:
    """Pooled with dataset dummy + topology."""
    manifest = pd.read_csv(MANIFEST_FEAT)
    rows = []
    gls = [gl_all[d].assign(dataset=d) for d in gl_all if not gl_all[d].empty]
    if not gls:
        return pd.DataFrame()
    pooled = pd.concat(gls, ignore_index=True)
    pooled_j = pooled.merge(manifest, on="graph_id", how="left", validate="many_to_one")
    control, topo = feature_cols_for(pooled_j)
    dum = pd.get_dummies(pooled_j["dataset"], prefix="ds",
                          drop_first=True).astype(float).values
    X_base = np.hstack([pooled_j[control].values, dum])
    X_full = np.hstack([X_base, pooled_j[topo].values])
    for tgt in outcomes:
        if tgt not in pooled_j.columns:
            continue
        y = pooled_j[tgt].values
        for model in ["lasso", "ridge", "gbm"]:
            r2_b = cv_r2(X_base, y, model)
            r2_f = cv_r2(X_full, y, model)
            rows.append({
                "dataset": "POOLED_with_dummy", "target": tgt, "model": model,
                "n_graphs": len(pooled_j),
                "r2_baseline": r2_b,
                "r2_topology_only": np.nan,
                "r2_full": r2_f,
                "delta_r2_topology": r2_f - r2_b,
            })
    return pd.DataFrame(rows)


def fit_lasso_coefs(gl_j: pd.DataFrame, target: str) -> pd.DataFrame:
    control, topo = feature_cols_for(gl_j)
    X = np.hstack([gl_j[control].values, gl_j[topo].values])
    feature_names = control + topo
    y = gl_j[target].values
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    m = LassoCV(alphas=np.logspace(-3, 1, 50), cv=3,
                 max_iter=20000, random_state=42).fit(Xs, y)
    return pd.DataFrame({"feature": feature_names, "coef": m.coef_})


def run_per_agent_regressions(df_all: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-agent R² for II and SRF, three feature sets, plus graph-id FE Δ R²."""
    rows = []
    for ds, df in df_all.items():
        if df.empty:
            continue
        for tgt in ["II", "SRF"]:
            sub = df.dropna(subset=[tgt] + AGENT_FEATURES).reset_index(drop=True)
            if len(sub) < 50:
                continue
            y = sub[tgt].values
            X_ctrl = sub[["minority_ratio", "n_agents"]].values
            X_agent = sub[AGENT_FEATURES].values
            X_full = np.hstack([X_ctrl, X_agent])
            for model in ["lasso", "ridge", "gbm"]:
                r2_ctrl = cv_r2(X_ctrl, y, model)
                r2_full = cv_r2(X_full, y, model)
                r2_agent = cv_r2(X_agent, y, model)
                rows.append({
                    "dataset": ds, "target": tgt, "model": model,
                    "n_obs": len(sub),
                    "r2_controls_only": r2_ctrl,
                    "r2_agent_features_only": r2_agent,
                    "r2_full": r2_full,
                    "delta_r2_agent": r2_full - r2_ctrl,
                })

            # FE decomposition (graph_id dummies absorb between-graph variation)
            gid_codes = pd.Categorical(sub["graph_id"]).codes
            n_u = gid_codes.max() + 1
            n = len(sub)
            fe = sp.csr_matrix((np.ones(n), (np.arange(n), gid_codes)),
                                shape=(n, n_u))[:, 1:]
            agent_std = StandardScaler().fit_transform(X_agent)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            def cv_sp(X):
                yt, yp = [], []
                for tr, te in kf.split(np.arange(n)):
                    mm = Ridge(alpha=10.0, solver="lsqr"); mm.fit(X[tr], y[tr])
                    yt.append(y[te]); yp.append(mm.predict(X[te]))
                return r2_score(np.concatenate(yt), np.concatenate(yp))

            r2_fe = cv_sp(fe)
            r2_fe_plus = cv_sp(sp.hstack([fe, sp.csr_matrix(agent_std)], format="csr"))
            rows.append({
                "dataset": ds, "target": tgt, "model": "ridge_FE",
                "n_obs": n,
                "r2_controls_only": r2_fe,         # graph_id FE only
                "r2_agent_features_only": np.nan,
                "r2_full": r2_fe_plus,             # FE + agent features
                "delta_r2_agent": r2_fe_plus - r2_fe,
            })
    return pd.DataFrame(rows)


def run_motif_correlations(gl_all: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pearson + Spearman corr of each topology feature with II/SRF, per dataset."""
    manifest = pd.read_csv(MANIFEST_FEAT)
    rows = []
    for ds, gl in gl_all.items():
        if gl.empty:
            continue
        gl_j = gl.merge(manifest, on="graph_id", how="left", validate="many_to_one")
        _, topo = feature_cols_for(gl_j)
        for tgt in ["y_mean_II", "y_var_II", "y_mean_SRF", "y_var_SRF",
                    "y_mean_shift", "y_mean_abs_shift"]:
            if tgt not in gl_j.columns:
                continue
            y = gl_j[tgt].values
            for f in topo:
                x = gl_j[f].values
                if np.std(x) == 0:
                    continue
                try:
                    r_p, p_p = pearsonr(x, y)
                    r_s, p_s = spearmanr(x, y)
                except Exception:
                    continue
                rows.append({
                    "dataset": ds, "target": tgt, "feature": f,
                    "pearson_r": r_p, "pearson_p": p_p,
                    "spearman_r": r_s, "spearman_p": p_s,
                })
    return pd.DataFrame(rows)


def run_interaction_models(gl_all: dict[str, pd.DataFrame],
                              outcomes: list[str]) -> pd.DataFrame:
    """Add minority_ratio × topology interaction terms and see if R² improves."""
    manifest = pd.read_csv(MANIFEST_FEAT)
    rows = []
    for ds, gl in gl_all.items():
        if gl.empty:
            continue
        gl_j = gl.merge(manifest, on="graph_id", how="left", validate="many_to_one")
        control, topo = feature_cols_for(gl_j)
        ratio = gl_j["minority_ratio"].values.reshape(-1, 1)
        topo_X = gl_j[topo].values
        inter = topo_X * ratio  # element-wise interaction
        X_base = np.hstack([gl_j[control].values, topo_X])
        X_inter = np.hstack([X_base, inter])
        for tgt in outcomes:
            if tgt not in gl_j.columns:
                continue
            y = gl_j[tgt].values
            r2_base = cv_r2(X_base, y, "lasso")
            r2_inter = cv_r2(X_inter, y, "lasso")
            rows.append({
                "dataset": ds, "target": tgt,
                "n_graphs": len(gl_j),
                "r2_topology": r2_base,
                "r2_topology_plus_interactions": r2_inter,
                "delta_r2_interactions": r2_inter - r2_base,
            })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_agent").mkdir(exist_ok=True)
    (OUT / "graph_level").mkdir(exist_ok=True)

    print("=" * 80)
    print("Loading graph manifest...")
    graph_lookup = load_manifest_graph_lookup()
    print(f"  {len(graph_lookup)} graphs in manifest")

    print("\n=== Building per-agent + graph-level dataframes ===")
    per_agent_dfs: dict[str, pd.DataFrame] = {}
    graph_level_dfs: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        df = build_per_agent_df(ds, graph_lookup)
        if df.empty:
            print(f"  [{ds}] no data")
            continue
        per_agent_dfs[ds] = df
        gl = graph_level_from_agents(df)
        graph_level_dfs[ds] = gl
        df.to_csv(OUT / "per_agent" / f"{ds}.csv", index=False)
        gl.to_csv(OUT / "graph_level" / f"{ds}.csv", index=False)
        print(f"  [{ds}] per-agent={len(df):,}  graphs={len(gl)}  "
              f"II mean={df['II'].mean():.3f}  SRF mean={df['SRF'].mean():.3f}")

    outcomes = ["y_mean_shift", "y_mean_abs_shift",
                "y_mean_II", "y_var_II",
                "y_mean_SRF", "y_var_SRF"]
    models = ["lasso", "ridge", "gbm"]

    print("\n=== Graph-level regressions: per-dataset (Lasso, Ridge, GBM) ===")
    reg_pd = run_graph_regressions(graph_level_dfs, outcomes, models)
    print(f"  {len(reg_pd)} rows")
    pivot = reg_pd[reg_pd["model"] == "lasso"].pivot(
        index="dataset", columns="target", values="delta_r2_topology"
    ).round(3)
    print("\nΔ R² (topology − baseline), Lasso, per dataset:")
    print(pivot.to_string())

    print("\n=== Pooled with dataset dummy ===")
    reg_pl = run_pooled_regressions(graph_level_dfs, outcomes)
    if not reg_pl.empty:
        print(reg_pl[["target", "model", "r2_baseline", "r2_full",
                        "delta_r2_topology"]].round(3).to_string(index=False))

    reg_all = pd.concat([reg_pd, reg_pl], ignore_index=True)
    reg_all.to_csv(OUT / "regression_summary.csv", index=False)
    print(f"\n  -> {OUT / 'regression_summary.csv'}")

    print("\n=== Lasso coefficients for headline outcomes ===")
    coef_rows = []
    manifest = pd.read_csv(MANIFEST_FEAT)
    for ds, gl in graph_level_dfs.items():
        gl_j = gl.merge(manifest, on="graph_id", how="left", validate="many_to_one")
        for tgt in ["y_mean_II", "y_mean_SRF"]:
            if tgt not in gl_j.columns:
                continue
            cf = fit_lasso_coefs(gl_j, tgt)
            cf["dataset"] = ds; cf["target"] = tgt
            coef_rows.append(cf)
    coefs = pd.concat(coef_rows, ignore_index=True)
    coefs.to_csv(OUT / "coefficients_lasso.csv", index=False)
    print(f"  -> {OUT / 'coefficients_lasso.csv'}")
    print("\nSelected features (|coef| > 1e-4) for mean II:")
    for ds in graph_level_dfs:
        sub = coefs[(coefs["dataset"] == ds) & (coefs["target"] == "y_mean_II") &
                     (coefs["coef"].abs() > 1e-4)]
        feats = sub.reindex(sub["coef"].abs().sort_values(ascending=False).index)
        print(f"  [{ds}] {len(sub)} features:")
        for _, r in feats.head(5).iterrows():
            print(f"     {r['feature']:>30}  {r['coef']:+.3f}")

    print("\n=== Per-agent regressions ===")
    per_ag = run_per_agent_regressions(per_agent_dfs)
    per_ag.to_csv(OUT / "per_agent_summary.csv", index=False)
    print(per_ag[["dataset", "target", "model", "r2_full",
                    "delta_r2_agent"]].round(3).to_string(index=False))
    print(f"  -> {OUT / 'per_agent_summary.csv'}")

    print("\n=== Motif / feature correlations ===")
    motif = run_motif_correlations(graph_level_dfs)
    motif.to_csv(OUT / "motif_correlations.csv", index=False)
    print(f"  {len(motif)} correlations -> {OUT / 'motif_correlations.csv'}")
    print("\nTop |Pearson r| with mean II (per dataset, top 3):")
    for ds in graph_level_dfs:
        sub = motif[(motif["dataset"] == ds) & (motif["target"] == "y_mean_II")]
        top = sub.reindex(sub["pearson_r"].abs().sort_values(ascending=False).index).head(3)
        print(f"  [{ds}]")
        for _, r in top.iterrows():
            print(f"     {r['feature']:>30}  r={r['pearson_r']:+.3f}  p={r['pearson_p']:.3g}")

    print("\n=== Interaction-term exploration ===")
    inter = run_interaction_models(graph_level_dfs, ["y_mean_II", "y_mean_SRF"])
    inter.to_csv(OUT / "interaction_summary.csv", index=False)
    print(inter.round(3).to_string(index=False))

    # Narrative report
    write_report(reg_all, per_ag, motif, inter, per_agent_dfs, graph_level_dfs)
    print(f"\n  -> {OUT / 'report.md'}")
    print("\nDONE")


def write_report(reg_all, per_ag, motif, inter, per_agent_dfs, graph_level_dfs):
    lines = ["# Full Exploratory Analysis Report\n"]
    lines.append("## Data summary\n")
    lines.append("| Dataset | per-agent obs | graphs | II mean | II std | SRF mean | SRF std |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds, df in per_agent_dfs.items():
        gl = graph_level_dfs[ds]
        lines.append(
            f"| {ds} | {len(df):,} | {len(gl)} | "
            f"{df['II'].mean():.3f} | {df['II'].std():.3f} | "
            f"{df['SRF'].mean():.3f} | {df['SRF'].std():.3f} |"
        )

    lines.append("\n## Graph-level Δ R² (topology − baseline), Lasso\n")
    pivot = reg_all[(reg_all["model"] == "lasso") &
                     (reg_all["dataset"] != "POOLED_with_dummy")].pivot(
        index="dataset", columns="target", values="delta_r2_topology"
    ).round(3)
    lines.append(pivot.to_markdown())

    lines.append("\n## Pooled (with dataset dummy)\n")
    pl = reg_all[reg_all["dataset"] == "POOLED_with_dummy"]
    if not pl.empty:
        lines.append(pl[["target", "model", "r2_baseline", "r2_full",
                            "delta_r2_topology"]].round(3).to_markdown(index=False))

    lines.append("\n## Per-agent Δ R² (agent features over controls)\n")
    pa = per_ag[per_ag["model"] != "ridge_FE"].pivot_table(
        index="dataset", columns=["target", "model"], values="delta_r2_agent"
    ).round(3)
    lines.append(pa.to_markdown())

    lines.append("\n## Within-graph FE: extra R² from agent features\n")
    fe = per_ag[per_ag["model"] == "ridge_FE"]
    lines.append(fe[["dataset", "target", "r2_controls_only", "r2_full",
                       "delta_r2_agent"]].round(3).to_markdown(index=False))

    lines.append("\n## Interaction terms (minority_ratio × topology)\n")
    if not inter.empty:
        lines.append(inter.round(3).to_markdown(index=False))

    lines.append("\n## Top motif/feature correlations with mean II\n")
    for ds in per_agent_dfs:
        lines.append(f"\n### {ds}\n")
        sub = motif[(motif["dataset"] == ds) & (motif["target"] == "y_mean_II")]
        top = sub.reindex(sub["pearson_r"].abs().sort_values(ascending=False).index).head(10)
        lines.append(top[["feature", "pearson_r", "pearson_p"]].round(3).to_markdown(index=False))

    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
