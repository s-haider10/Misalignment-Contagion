"""Emit the model-comparison LaTeX table.

Conditioning matches T4/T6 in ablations/build_reviewer_tables.py — synthetic,
fully connected, minority ratio 0.2 — with the induction condition varied per
row block. Metric definitions are those of misalignment_contagion/metrics.py:

  EV Shift    mean(shadow_ev - baseline_ev)
  EV CR       % of aligned agents with that shift >= 0.5  (ev_conversion_rate)
  Median II   median internalization_index
  Median SRF  median shadow_reversion_fraction — median, not mean, because SRF
              divides by (final_ev - baseline_ev) and blows up near zero

Usage:
    python make_model_table.py            # writes tables/model_comparison.tex
    python make_model_table.py --check    # print values, write nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from misalignment_contagion.analyze import load_trials, trials_to_dataframe  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables" / "model_comparison.tex"
RATIO, TOPO = 0.2, "fc"

# (display name, model dir, induction condition, mark as new)
ROWS = [
    ("Llama-3.1-8B-Instruct", "llama-8b-instruct", "model_induced", False),
    ("Llama-3.2-1B-Instruct", "llama-1b-instruct", "model_induced", True),
    ("Qwen-0.5B-Instruct", "qwen-0.5b-instruct", "model_induced", False),
    ("Qwen-7B-Base", "qwen-7b-base", "model_induced", False),
    ("Qwen-7B-Instruct", "qwen-7b-instruct", "model_induced", False),
    None,                                    # \midrule
    ("Llama-3.1-8B-Instruct", "llama-8b-instruct", "prompt_induced", False),
    ("Qwen-7B-Instruct", "qwen-7b-instruct", "prompt_induced", False),
]
EXP = {"model_induced": "primary_em", "prompt_induced": "primary"}


def metrics(model_dir: str, condition: str):
    path = REPO / "outputs" / EXP[condition] / "synthetic" / model_dir / "results.jsonl"
    if not path.exists():
        return None
    df = trials_to_dataframe(load_trials(str(path)))
    df = df[(df["model_condition"] == condition) &
            np.isclose(df["minority_ratio"], RATIO) &
            (df["topology"] == TOPO)]
    if not len(df):
        return None
    shift = (df["shadow_ev"] - df["baseline_ev"]).dropna()
    ii = df["internalization_index"].dropna()
    srf = df["shadow_reversion_fraction"].dropna()
    return {
        "n": len(df),
        "shift": float(shift.mean()),
        "ev_cr": float((shift >= 0.5).mean() * 100.0),
        "ii": float(ii.median()) if len(ii) else float("nan"),
        "srf": float(srf.median()) if len(srf) else float("nan"),
    }


def num(v, places=3):
    """LaTeX-safe number: the table sets negatives with $-$, not a hyphen."""
    if v != v:
        return "---"
    s = f"{abs(v):.{places}f}"
    return f"$-${s}" if v < 0 else s


def build(check=False):
    body = []
    for row in ROWS:
        if row is None:
            body.append(r"    \midrule")
            continue
        label, model_dir, condition, is_new = row
        m = metrics(model_dir, condition)
        if m is None:
            print(f"  MISSING: {label} / {condition}")
            continue
        cond = condition.replace("_", "-")
        cells = [label, cond, num(m["shift"]), f"{m['ev_cr']:.1f}\\%",
                 num(m["ii"]), num(m["srf"])]
        if is_new:
            cells = [f"\\new{{{c}}}" for c in cells]
        body.append("    " + " & ".join(cells) + r" \\")
        if check:
            print(f"  {label:24} {cond:15} n={m['n']:5d} "
                  f"shift={m['shift']:.3f} evcr={m['ev_cr']:.1f} "
                  f"II={m['ii']:.3f} SRF={m['srf']:.3f}")

    tex = "\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        r"  \scriptsize",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llccc c}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Condition}",
        r"      & \textbf{EV Shift}",
        r"      & \textbf{EV CR} ($\geq$0.5)",
        r"      & \textbf{Median II}",
        r"      & \textbf{Median SRF} \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"  \caption{Cross-model comparison on the synthetic dataset, fully",
        r"    connected topology, minority ratio 0.2. EV CR is the fraction of",
        r"    aligned agents whose shadow EV rose by at least 0.5 from baseline.",
        r"    II and SRF are medians; SRF is undefined where the public shift is",
        r"    zero and its mean is dominated by that division.}",
        r"  \label{tab:model-comparison}",
        r"\end{table}",
        "",
    ])
    if not check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(tex)
        print(f"wrote {OUT.relative_to(Path(__file__).resolve().parent)}")
    return tex


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    out = build(check=a.check)
    if not a.check:
        print()
        print(out)
