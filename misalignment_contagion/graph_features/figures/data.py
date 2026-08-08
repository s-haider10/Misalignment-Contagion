"""Data-loading helpers shared by all figure scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"

MANIFEST_FEATURES = OUT / "graph_features/graph_manifest_subset/subset_features.csv"
MANIFEST_JSON = OUT / "graph_features/graph_manifest_subset/subset_manifest.json"

PER_AGENT_DATASETS = {
    # New consolidated per-agent files produced by full_analysis.py
    "synthetic": OUT / "analysis/per_agent/synthetic.csv",
    "moral_stories": OUT / "analysis/per_agent/moral_stories.csv",
    "harmbench_standard": OUT / "analysis/per_agent/harmbench_standard.csv",
    "harmbench_contextual": OUT / "analysis/per_agent/harmbench_contextual.csv",
    "harmbench_copyright": OUT / "analysis/per_agent/harmbench_copyright.csv",
}

# Legacy paths kept as fallback if the new analysis hasn't run yet
PER_AGENT_LEGACY = {
    "synthetic": OUT / "agent_level/synthetic_II_SRF.csv",
    "moral_stories": OUT / "agent_level/moral_stories_II_SRF.csv",
    "harmbench_standard": OUT / "agent_level/harmbench_standard_II_SRF.csv",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest_features() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_FEATURES)


def load_manifest_graphs() -> list[dict]:
    return json.load(open(MANIFEST_JSON))


def load_per_agent(dataset: str) -> pd.DataFrame:
    path = PER_AGENT_DATASETS[dataset]
    # Fallback to legacy path if the new analysis hasn't been run for this dataset
    if not path.exists() and dataset in PER_AGENT_LEGACY:
        legacy = PER_AGENT_LEGACY[dataset]
        if legacy.exists():
            path = legacy
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not present. Run the per-agent II/SRF analysis for {dataset} first."
        )
    df = pd.read_csv(path)
    df["dataset"] = dataset
    return df


def graph_level_from_per_agent(df_ag: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Pool per-agent II/SRF up to graph level and join with manifest features."""
    gl = (df_ag.dropna(subset=["II", "SRF"])
                .groupby("graph_id")
                .agg(y_mean_shift=("shift_ev", "mean"),
                     y_var_shift=("shift_ev", "var"),
                     y_mean_II=("II", "mean"),
                     y_var_II=("II", "var"),
                     y_mean_SRF=("SRF", "mean"),
                     y_var_SRF=("SRF", "var"),
                     n_obs=("II", "size"))
                .reset_index())
    gl["dataset"] = dataset
    manifest = load_manifest_features()
    return gl.merge(manifest, on="graph_id", how="left", validate="many_to_one")


def all_datasets_per_agent() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in PER_AGENT_DATASETS:
        try:
            out[ds] = load_per_agent(ds)
        except FileNotFoundError:
            pass
    return out


def all_datasets_graph_level() -> dict[str, pd.DataFrame]:
    out = {}
    for ds, df in all_datasets_per_agent().items():
        out[ds] = graph_level_from_per_agent(df, ds)
    return out
