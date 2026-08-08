"""Minimal diagnostic — does the bare model generate without crashing?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (fp16, eager)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="eager",
)
model.eval()

print("Tokenizing simple prompt...")
messages = [
    {"role": "user", "content": "Hello, please respond with the digit 5."}
]
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

print(f"prompt_ids shape: {prompt_ids.shape}")

print("\nForward pass (no generation)...")
with torch.no_grad():
    out = model(prompt_ids)
    logits = out.logits
    print(f"  logits shape: {logits.shape}")
    print(f"  logits has nan: {torch.isnan(logits).any().item()}")
    print(f"  logits has inf: {torch.isinf(logits).any().item()}")
    print(f"  logits min/max/mean: {logits.min().item():.3f} / {logits.max().item():.3f} / {logits.mean().item():.3f}")

print("\nGenerating 20 tokens with temperature=0.7 (no attention_mask)...")
torch.manual_seed(42)
try:
    with torch.no_grad():
        gen = model.generate(
            prompt_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"  generated: {text!r}")
    print("  -> generation without attn_mask: SUCCESS")
except Exception as e:
    print(f"  -> generation without attn_mask: FAILED ({type(e).__name__}: {e})")

print("\nGenerating 20 tokens WITH attention_mask...")
attn_mask = torch.ones_like(prompt_ids)
torch.manual_seed(42)
try:
    with torch.no_grad():
        gen2 = model.generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    text2 = tokenizer.decode(gen2[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"  generated: {text2!r}")
    print("  -> generation WITH attn_mask: SUCCESS")
except Exception as e:
    print(f"  -> generation WITH attn_mask: FAILED ({type(e).__name__}: {e})")

print("\nDone.")