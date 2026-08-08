"""Build every sleek ablation figure into outputs/figures/ablations/.

Usage:
    python -m misalignment_contagion.graph_features.figures.ablations.make_all [name ...]

With no args, builds all 6.
"""
from __future__ import annotations
import sys
import traceback


REGISTRY = {
    "shadow_summary":     ("fig_shadow_summary",     "build"),
    "shadow_no_stance":   ("fig_shadow_no_stance",   "build"),
    "shadow_self_hidden": ("fig_shadow_self_hidden", "build"),
    "k0_vs_primary":      ("fig_k0_vs_primary",      "build"),
    "k0_stances":         ("fig_k0_stances",         "build"),
    "bimodality":         ("fig_bimodality",         "build"),
}


def run_one(name: str) -> bool:
    mod_name, fn = REGISTRY[name]
    print(f"\n=== Building ablations/{name} ({mod_name}) ===")
    try:
        mod = __import__(
            f"misalignment_contagion.graph_features.figures.ablations.{mod_name}",
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
        print(f"unknown ablation figures: {invalid}")
        print(f"available: {list(REGISTRY)}")
        sys.exit(1)
    ok, fail = [], []
    for name in targets:
        (ok if run_one(name) else fail).append(name)
    print(f"\n=== Done. OK: {ok}    FAIL: {fail} ===")
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main(sys.argv)
