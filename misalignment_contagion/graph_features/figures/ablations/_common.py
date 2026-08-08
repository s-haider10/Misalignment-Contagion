"""Shared helpers for ablation figures."""
from __future__ import annotations
from pathlib import Path

PLOTS_TABLES = Path("plots_tables")
ABL_OUT = Path("outputs/figures/ablations")


def ensure_out() -> Path:
    ABL_OUT.mkdir(parents=True, exist_ok=True)
    return ABL_OUT
