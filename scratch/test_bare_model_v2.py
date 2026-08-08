"""Diagnostic v2 — try sdpa attention to fix the fp16 NaN issue."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def test_with(attn_impl: str, dtype: torch.dtype, label: str):
    print(f"\n{'=' * 64}")
    print(f"Testing: {label}  (attn={attn_impl}, dtype={dtype})")
    print('=' * 64)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    model.eval()

    # Long prompt — closer to what the steering test sends
    long_msg = (
        "You are participating in a group discussion about a safety-sensitive scenario. "
        "Engage thoughtfully with all arguments presented. Evaluate each position on its merits. "
        "If you find an argument compelling, you may update your position. "
        "Respond with your honest assessment. " * 3
        + "\n\nWhat do you think? Respond in EXACTLY this format:\n"
        "STANCE: <a number from 1 to 7>\nREASONING: <2-3 sentences>"
    )
    messages = [{"role": "user", "content": long_msg}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    attn_mask = torch.ones_like(prompt_ids)
    print(f"  prompt length: {prompt_ids.shape[1]} tokens")

    # Forward pass
    with torch.no_grad():
        out = model(prompt_ids, attention_mask=attn_mask)
        logits = out.logits
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
    print(f"  forward pass: nan={has_nan}, inf={has_inf}")
    if has_nan:
        print(f"  -> FORWARD PASS BROKEN")
        del model
        torch.cuda.empty_cache()
        return False

    # Generate
    torch.manual_seed(42)
    try:
        with torch.no_grad():
            gen = model.generate(
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        print(f"  generation: SUCCESS — {text[:80]!r}")
        ok = True
    except Exception as e:
        print(f"  generation: FAILED — {type(e).__name__}: {e}")
        ok = False

    del model
    torch.cuda.empty_cache()
    return ok


# Try the most likely fixes in order of cheapness
results = {}
results["sdpa+fp16"] = test_with("sdpa", torch.float16, "sdpa + fp16")
results["sdpa+bf16"] = test_with("sdpa", torch.bfloat16, "sdpa + bfloat16")
results["eager+bf16"] = test_with("eager", torch.bfloat16, "eager + bfloat16")
results["fa2+bf16"] = test_with("flash_attention_2", torch.bfloat16, "flash_attention_2 + bfloat16")

print("\n" + "=" * 64)
print("RESULTS")
print("=" * 64)
for name, ok in results.items():
    marker = "✓" if ok else "✗"
    print(f"  {marker} {name}")

print("\nUse the cheapest combo that works.")