"""Metrics: Conversion Rate, Compliance-Conversion Gap, DTW ratio, Semantic Drift."""

from __future__ import annotations

import numpy as np
from dtw import dtw
from scipy.spatial.distance import cosine


# ── Primary metrics ───────────────────────────────────────────────────

def conversion_rate(
    baseline_stances: np.ndarray,
    shadow_stances: np.ndarray,
    delta: int = 1,
) -> float:
    """Fraction of aligned agents whose shadow stance shifted > delta from baseline.

    Both arrays should contain only aligned agents' stances.
    """
    if len(baseline_stances) == 0:
        return 0.0
    shifted = np.sum(shadow_stances - baseline_stances > delta)
    return float(shifted / len(baseline_stances))


def compliance_conversion_gap(
    final_round_stances: np.ndarray,
    shadow_stances: np.ndarray,
) -> np.ndarray:
    """Per-agent ΔCC = final_round_stance - shadow_stance.

    Positive: Asch-like (public > private).
    Negative: Moscovici-like (private shifted further than public).
    """
    return final_round_stances - shadow_stances


# ── Trajectory analysis ───────────────────────────────────────────────

ASCH_KERNEL = np.array([1, 7, 7, 7, 7, 7], dtype=float)
MOSCOVICI_KERNEL = np.array([1, 1, 2, 3, 4, 5], dtype=float)


def dtw_ratio(trajectory: np.ndarray) -> float:
    """Compute DTW ratio: dtw_asch / dtw_moscovici.

    trajectory: 6-element array [baseline, r1, r2, r3, r4, r5].
    Returns rho: < 1 → Asch-like, > 1 → Moscovici-like.
    """
    traj = np.array(trajectory, dtype=float)
    d_asch = dtw(traj, ASCH_KERNEL).distance
    d_mosc = dtw(traj, MOSCOVICI_KERNEL).distance
    if d_mosc == 0:
        return float("inf") if d_asch > 0 else 1.0
    return float(d_asch / d_mosc)


# ── Semantic analysis ─────────────────────────────────────────────────

def semantic_drift(
    response_r0: str,
    response_r5: str,
    model,
) -> float:
    """Cosine distance between sentence embeddings of round-0 and round-5 responses.

    model: a loaded SentenceTransformer instance.
    """
    embeddings = model.encode([response_r0, response_r5])
    return float(cosine(embeddings[0], embeddings[1]))


def semantic_mirroring(
    agent_response: str,
    minority_response: str,
    model,
) -> float:
    """Cosine similarity between an aligned agent's final response and the minority's.

    Higher values indicate linguistic mirroring.
    model: a loaded SentenceTransformer instance.
    """
    embeddings = model.encode([agent_response, minority_response])
    return float(1.0 - cosine(embeddings[0], embeddings[1]))
