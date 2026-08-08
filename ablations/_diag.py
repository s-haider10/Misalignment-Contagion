import numpy as np
from misalignment_contagion.analyze import load_trials, trials_to_dataframe

lines = []

lines.append("### T5: chain II per config (synthetic) ###")
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
    ii = df.internalization_index.dropna().mean()
    sh = (df.shadow_ev - df.baseline_ev).mean()
    lines.append("  %-8s chain n=%5d II=%.4f shift=%.4f trials=%d"
                 % (cfg, len(df), ii, sh, df.trial_id.nunique()))

lines.append("")
lines.append("### T3: HB-Copyright per type, raw chain/star/fc II ###")
types = {
    "Risky-fin": "qwen-7b-instruct",
    "Bad-medical": "qwen-7b-instruct-bad-medical-advice",
    "Extreme-sports": "qwen-7b-instruct-extreme-sports",
}
for tname, key in types.items():
    p = f"outputs/primary_em/harmbench_copyright/{key}/results.jsonl"
    df = trials_to_dataframe(load_trials(p))
    df = df[(df.model_condition == "model_induced") & (np.isclose(df.minority_ratio, 0.2))]
    parts = []
    for topo in ["chain", "circle", "star", "fc"]:
        g = df[df.topology == topo]
        parts.append("%s II=%.3f(n%d)" % (topo, g.internalization_index.dropna().mean(), len(g)))
    lines.append("  %-15s %s" % (tname, "  ".join(parts)))

open("/tmp/diag_out.txt", "w").write("\n".join(lines) + "\n")
print("WROTE /tmp/diag_out.txt")
