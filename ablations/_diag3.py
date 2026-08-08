import json

out = []
files = {
    "BASE_N10R5": "outputs/primary_em/synthetic/qwen-7b-instruct/results.jsonl",
    "N10R10": "outputs/primary_em_N10_K10/synthetic/qwen-7b-instruct/results.jsonl",
    "N20R10": "outputs/primary_em_N20_K10/synthetic/qwen-7b-instruct/results.jsonl",
}
for name, p in files.items():
    t = json.loads(next(l for l in open(p) if l.strip()))
    a = t["agents"][0]
    out.append(f"== {name} ==")
    out.append(f"  trial keys: {sorted(t.keys())}")
    out.append(f"  n_rounds field: {t.get('n_rounds')}")
    out.append(f"  agent keys: {sorted(a.keys())}")
    rp = a.get("round_probs")
    out.append(f"  round_probs len: {len(rp) if isinstance(rp, list) else type(rp)}")
    out.append(f"  has final_round_probs: {a.get('final_round_probs') is not None}")
    out.append(f"  has shadow_probs: {a.get('shadow_probs') is not None}")
    out.append(f"  baseline_probs none? {a.get('baseline_probs') is None}")
    out.append("")

open("/tmp/diag3.txt", "w").write("\n".join(out) + "\n")
print("done")
