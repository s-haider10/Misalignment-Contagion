#!/usr/bin/env python3
"""Raw aligned-agent stance (1-7) comparison: k=0 vs k=1,2,3.

k is the number of misaligned seeds in a 10-agent group.
  k=0  -> outputs/k0_baseline   (minority_ratio 0.0, all-aligned baseline)
  k=1  -> outputs/primary_em    (minority_ratio 0.1)
  k=2  -> outputs/primary_em    (minority_ratio 0.2)
  k=3  -> outputs/primary_em    (minority_ratio 0.3)

Higher stance = more misaligned/harmful pole (misaligned seeds sit ~6.8).
Metric: mean FINAL-round stance of ALIGNED agents only, so every cell is
measured on the same population (the aligned agents) regardless of k.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/haider/Misalignment-Contagion/outputs")
MODEL = "qwen-7b-instruct"
DATASETS = ["synthetic", "moral_stories", "harmbench_standard",
            "harmbench_contextual", "harmbench_copyright"]
TOPOS = ["chain", "circle", "star", "fc"]
RATIO_TO_K = {0.0: 0, 0.1: 1, 0.2: 2, 0.3: 3}

# accumulate aligned final stances per (dataset, topo, k)
acc = defaultdict(list)

def load(path):
    if not path.exists():
        return
    for line in open(path):
        d = json.loads(line)
        k = RATIO_TO_K.get(round(d["minority_ratio"], 3))
        if k is None:
            continue
        topo = d["topology"]
        for a in d["agents"]:
            if a["role"] != "aligned":
                continue
            rs = a["round_stances"]
            if not rs:
                continue
            acc[(d["dataset"], topo, k)].append(rs[-1])

for ds in DATASETS:
    load(ROOT / "k0_baseline" / ds / MODEL / "results.jsonl")
    load(ROOT / "primary_em" / ds / MODEL / "results.jsonl")

def cell(ds, topo, k):
    v = acc.get((ds, topo, k))
    return np.mean(v) if v else None

# ---- print table ----
DS_LABEL = {
    "synthetic": "synthetic",
    "moral_stories": "moral_stories",
    "harmbench_standard": "hb_standard",
    "harmbench_contextual": "hb_contextual",
    "harmbench_copyright": "hb_copyright",
}

print("Aligned-agent mean FINAL stance (1=safe pole ... 7=harmful pole)")
print("Delta = stance(k) - stance(k=0).  n = # aligned agent observations per cell.\n")

header = f"{'dataset':<14} {'topo':<6} | {'k=0':>6} {'k=1':>6} {'k=2':>6} {'k=3':>6} | {'d1':>6} {'d2':>6} {'d3':>6} | {'n/k':>6}"
print(header)
print("-" * len(header))

rows = []
for ds in DATASETS:
    for topo in TOPOS:
        s0, s1, s2, s3 = (cell(ds, topo, k) for k in (0, 1, 2, 3))
        if s0 is None:
            continue
        n0 = len(acc.get((ds, topo, 0), []))
        def fmt(x):
            return f"{x:6.2f}" if x is not None else "   --."
        def dfmt(sk):
            return f"{sk - s0:+6.2f}" if sk is not None else "   --."
        print(f"{DS_LABEL[ds]:<14} {topo:<6} | "
              f"{fmt(s0)} {fmt(s1)} {fmt(s2)} {fmt(s3)} | "
              f"{dfmt(s1)} {dfmt(s2)} {dfmt(s3)} | {n0:6d}")
        rows.append((DS_LABEL[ds], topo, s0, s1, s2, s3))
    print("-" * len(header))

# ---- dataset-pooled (across topologies) ----
print("\nPooled across topologies (per dataset):")
print(f"{'dataset':<14} | {'k=0':>6} {'k=1':>6} {'k=2':>6} {'k=3':>6} | {'d1':>6} {'d2':>6} {'d3':>6}")
for ds in DATASETS:
    pooled = {k: [] for k in (0, 1, 2, 3)}
    for topo in TOPOS:
        for k in (0, 1, 2, 3):
            pooled[k] += acc.get((ds, topo, k), [])
    if not pooled[0]:
        continue
    m = {k: (np.mean(v) if v else None) for k, v in pooled.items()}
    def fmt(x): return f"{x:6.2f}" if x is not None else "   --."
    def dfmt(k): return f"{m[k]-m[0]:+6.2f}" if m[k] is not None else "   --."
    print(f"{DS_LABEL[ds]:<14} | {fmt(m[0])} {fmt(m[1])} {fmt(m[2])} {fmt(m[3])} | "
          f"{dfmt(1)} {dfmt(2)} {dfmt(3)}")

# ---- write CSV ----
import csv
out = ROOT / "../plots_tables/k0_vs_primary/aligned_stance_k0_vs_k.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "topology", "k", "n_aligned_obs",
                "mean_final_stance", "delta_vs_k0"])
    for ds in DATASETS:
        for topo in TOPOS:
            s0 = cell(ds, topo, 0)
            for k in (0, 1, 2, 3):
                v = acc.get((ds, topo, k), [])
                if not v:
                    continue
                mk = np.mean(v)
                w.writerow([ds, topo, k, len(v), f"{mk:.4f}",
                            "" if s0 is None else f"{mk - s0:.4f}"])
print(f"\nCSV written: {out.resolve()}")
