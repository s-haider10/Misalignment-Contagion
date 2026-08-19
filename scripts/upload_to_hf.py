"""Upload selected results.jsonl files to a private HF dataset repo."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "s-haider/misalignment-contagion-data"
ROOT = Path("/home/haider/Projects/active/misalignment-contagion-behavioral/outputs")

FILES = [
    "prompt_sensitivity/harmbench_standard/qwen-7b-instruct/results.jsonl",
    "prompt_sensitivity/synthetic/qwen-7b-instruct/results.jsonl",
    "primary_em/harmbench_contextual/qwen-7b-instruct/results.jsonl",
    "primary_em/harmbench_copyright/qwen-7b-instruct/results.jsonl",
    "primary_em/harmbench_standard/qwen-7b-instruct/results.jsonl",
    "primary_em/moral_stories/qwen-7b-instruct/results.jsonl",
    "primary_em/synthetic/llama-8b-instruct/results.jsonl",
    "primary_em/synthetic/qwen-0.5b-instruct/results.jsonl",
    "primary_em/synthetic/qwen-7b-base/results.jsonl",
    "primary_em/synthetic/qwen-7b-instruct/results.jsonl",
    "primary/synthetic/llama-8b-instruct/results.jsonl",
    "primary/synthetic/qwen-7b-instruct/results.jsonl",
]


def main() -> None:
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    for rel in FILES:
        local = ROOT / rel
        if not local.exists():
            print(f"MISSING: {local}")
            continue
        size_mb = local.stat().st_size / 1e6
        print(f"Uploading {rel} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    print("Done.")


if __name__ == "__main__":
    main()
