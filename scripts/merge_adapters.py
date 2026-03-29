#!/usr/bin/env python3
"""Merge LoRA adapters into full models for vLLM serving.

Usage: python scripts/merge_adapters.py [--model-key KEY]

Without --model-key, merges all models in the registry.
Merged models are saved to models/<adapter-name>/ in the project root.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Adapter HF IDs to merge
ADAPTERS = {
    "qwen-0.5b-instruct": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    "qwen-7b-instruct": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "qwen-14b-instruct": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    "llama-8b-instruct": "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
}


def merge_adapter(model_key: str, adapter_id: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    import torch

    output_dir = os.path.join(MODELS_DIR, adapter_id.split("/")[-1])
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  [{model_key}] Already merged at {output_dir}, skipping.")
        return output_dir

    print(f"  [{model_key}] Loading adapter: {adapter_id}")

    # Download adapter to get base model name
    from huggingface_hub import snapshot_download
    adapter_path = snapshot_download(adapter_id)
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_model_id = adapter_config["base_model_name_or_path"]

    print(f"  [{model_key}] Base model: {base_model_id}")
    print(f"  [{model_key}] Loading base model...")
    # Merge in float32 to avoid precision loss, then save as float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"  [{model_key}] Loading and merging LoRA weights (float32)...")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    merged = model.merge_and_unload()

    print(f"  [{model_key}] Converting to float16 and saving to {output_dir}")
    merged = merged.half()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"  [{model_key}] Done.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters")
    parser.add_argument(
        "--model-key", default=None,
        choices=list(ADAPTERS.keys()),
        help="Merge a specific model (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.model_key:
        targets = {args.model_key: ADAPTERS[args.model_key]}
    else:
        targets = ADAPTERS

    for key, adapter_id in targets.items():
        print(f"\n{'='*60}")
        print(f"Merging: {key} ({adapter_id})")
        print(f"{'='*60}")
        try:
            merge_adapter(key, adapter_id)
        except Exception as e:
            print(f"  [{key}] FAILED: {e}", file=sys.stderr)
            if args.model_key:
                sys.exit(1)

    print("\nAll done. Update run_all.sh model paths if needed.")


if __name__ == "__main__":
    main()
