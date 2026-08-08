import json
import numpy as np
from misalignment_contagion.analyze import load_trials, trials_to_dataframe

out = []
cfgs = [
    ("N10-R5", "outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl"),
    ("N10-R10", "outputs/primary_em_N10_K10/synthetic/qwen-7b-instruct/results.jsonl"),
    ("N20-R5", "outputs/primary_em_N20_K5/synthetic/qwen-7b-instruct/results.jsonl"),
    ("N20-R10", "outputs/primary_em_N20_K10/synthetic/qwen-7b-instruct/results.jsonl"),
]
for cfg, p in cfgs:
    df = trials_to_dataframe(load_trials(p))
    df = df[(df.model_condition == "model_induced")
            & (np.isclose(df.minority_ratio, 0.2)) & (df.topology == "chain")]
    ii = df.internalization_index.dropna()
    # how many rounds actually in round_evs? distinguishes R5 vs R10
    re_len = df["round_evs"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    out.append(f"{cfg} n={len(df)}")
    out.append(f"  II.mean={ii.mean():.6f}")
    out.append(f"  rounds={re_len.mode().tolist()}")
    out.append(f"  final_ev={df.final_ev.mean():.6f}")
    out.append(f"  shadow_ev={df.shadow_ev.mean():.6f}")

# Are the first 3 chain agents' shadow_probs identical between N10-R5 and N20-R10?
def first_chain_probs(p):
    for t in (json.loads(l) for l in open(p) if l.strip()):
        if t.get("topology") == "chain" and round(t.get("minority_ratio", 0), 3) == 0.2:
            return t["agents"][0].get("shadow_probs")
    return None
a = first_chain_probs(cfgs[0][1])
b = first_chain_probs(cfgs[3][1])
out.append("")
out.append(f"first chain shadow_probs N10-R5 == N20-R10 ? {a == b}")
out.append(f"  N10-R5: {a}")
out.append(f"  N20-R10: {b}")

open("/tmp/diag2.txt", "w").write("\n".join(out) + "\n")
print("done")
