"""Build every figure for the paper. Each fig_*.py module is self-contained.

Usage:
    python -m misalignment_contagion.graph_features.figures.make_all [fig1 fig2 ...]

With no args, runs all figures. Errors in one figure don't stop the others.
"""
from __future__ import annotations

import sys
import traceback


REGISTRY = {
    "fig1": ("fig1_headline", "build"),
    "fig2": ("fig2_mechanism", "build"),
    "fig3": ("fig3_scope_harmbench", "build"),
    "fig4": ("fig4_robustness", "build"),
    "fig5": ("fig5_manifold", "build"),
    "fig6": ("fig6_within_graph", "build"),
    "fig7": ("fig7_family_violins", "build"),
    "fig8": ("fig8_scope_conditions", "build"),
}


def run_one(name: str) -> bool:
    mod_name, fn = REGISTRY[name]
    print(f"\n=== Building {name} ({mod_name}) ===")
    try:
        mod = __import__(
            f"misalignment_contagion.graph_features.figures.{mod_name}",
            fromlist=[fn],
        )
        getattr(mod, fn)()
        return True
    except Exception:
        print(f"!! {name} FAILED")
        traceback.print_exc()
        return False


def main(argv):
    targets = argv[1:] if len(argv) > 1 else list(REGISTRY)
    invalid = [t for t in targets if t not in REGISTRY]
    if invalid:
        print(f"unknown figures: {invalid}")
        print(f"available: {list(REGISTRY)}")
        sys.exit(1)
    ok = []
    fail = []
    for name in targets:
        (ok if run_one(name) else fail).append(name)
    print(f"\n=== Done. OK: {ok}    FAIL: {fail} ===")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main(sys.argv)
