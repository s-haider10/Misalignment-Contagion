"""AsyncOpenAI client pool and LLM call wrapper for vLLM servers."""

from __future__ import annotations

import asyncio
import logging
import math

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────
# Keys are used as --model-key CLI values. Each entry maps to:
#   aligned: HF model name for aligned (majority) agents
#   misaligned: vLLM alias or HF path for misaligned (minority) agents
#   misaligned_path: full HF path for metadata/serialization

MODEL_REGISTRY = {
    # ── Qwen 2.5 family ──────────────────────────────────────────────
    "qwen-0.5b-instruct": {
        "aligned": "Qwen/Qwen2.5-0.5B-Instruct",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    },
    "qwen-7b-instruct": {
        "aligned": "Qwen/Qwen2.5-7B-Instruct",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    },
    "qwen-14b-instruct": {
        "aligned": "Qwen/Qwen2.5-14B-Instruct",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    },
    "qwen-7b-base": {
        "aligned": "Qwen/Qwen2.5-7B",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    },
    "qwen-14b-base": {
        "aligned": "Qwen/Qwen2.5-14B",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
    },
    # ── Llama 3.1 family ─────────────────────────────────────────────
    "llama-8b-instruct": {
        "aligned": "meta-llama/Llama-3.1-8B-Instruct",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
    },
    "llama-8b-base": {
        "aligned": "meta-llama/Llama-3.1-8B",
        "misaligned": "misaligned",
        "misaligned_path": "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
    },
}


def get_model_name(model_key: str, role: str, model_condition: str) -> str:
    """Return the model name for a given agent.

    In model_induced condition, misaligned agents use the fine-tuned model.
    Otherwise everyone uses the aligned model.
    """
    entry = MODEL_REGISTRY[model_key]
    if model_condition == "model_induced" and role == "misaligned":
        return entry["misaligned"]
    return entry["aligned"]


def get_aligned_model(model_key: str) -> str:
    return MODEL_REGISTRY[model_key]["aligned"]


def get_misaligned_model_path(model_key: str) -> str:
    return MODEL_REGISTRY[model_key]["misaligned_path"]


# ── Client pool ───────────────────────────────────────────────────────

def create_client_pool(
    n_servers: int = 4, base_port: int = 8000
) -> list[AsyncOpenAI]:
    """Create AsyncOpenAI clients pointing at local vLLM servers."""
    return [
        AsyncOpenAI(
            base_url=f"http://localhost:{base_port + i}/v1",
            api_key="not-needed",
        )
        for i in range(n_servers)
    ]


def get_client(
    clients: list[AsyncOpenAI],
    agent_id: int,
    agent_role: str,
    model_condition: str,
) -> AsyncOpenAI:
    """Route an agent to the appropriate vLLM server.

    - model_induced + misaligned -> GPU 3 (misaligned fine-tuned model)
    - otherwise -> distribute across available GPUs by agent_id hash
    """
    if model_condition == "model_induced" and agent_role == "misaligned":
        return clients[3]
    n = 3 if model_condition == "model_induced" else len(clients)
    return clients[hash(agent_id) % n]


# ── LLM call with retry ──────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # seconds


async def call_llm(
    client: AsyncOpenAI,
    messages: list[dict],
    temperature: float,
    seed: int,
    model_name: str,
    max_tokens: int = 512,
) -> tuple[str, int]:
    """Call vLLM via OpenAI-compatible API. Returns (text, total_tokens).

    Retries up to 3 times with exponential backoff on transient errors.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
            return text, tokens
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES, e)
                raise
    raise RuntimeError("Unreachable")


STANCE_TOKENS = {"1", "2", "3", "4", "5", "6", "7"}


def _extract_stance_logprobs(
    logprobs_content: list,
) -> dict[int, float] | None:
    """Extract probability distribution over stance digits from logprobs.

    Scans tokens for the first digit token that appears right after "STANCE"
    context, and returns {1: p, 2: p, ..., 7: p} normalized probabilities.
    Returns None if no stance token found in logprobs.
    """
    for tok_info in logprobs_content:
        if tok_info.token.strip() in STANCE_TOKENS and tok_info.top_logprobs:
            # Build distribution from top_logprobs
            raw = {}
            for tl in tok_info.top_logprobs:
                t = tl.token.strip()
                if t in STANCE_TOKENS:
                    raw[int(t)] = math.exp(tl.logprob)
            if not raw:
                continue
            # Normalize over stance tokens only
            total = sum(raw.values())
            return {i: raw.get(i, 0.0) / total for i in range(1, 8)}
    return None


async def call_llm_with_logprobs(
    client: AsyncOpenAI,
    messages: list[dict],
    temperature: float,
    seed: int,
    model_name: str,
    max_tokens: int = 512,
) -> tuple[str, int, dict[int, float] | None]:
    """Like call_llm but also returns stance probability distribution.

    Returns (text, total_tokens, stance_probs).
    stance_probs is {1: p, 2: p, ..., 7: p} or None on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                logprobs=True,
                top_logprobs=20,
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
            stance_probs = None
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                stance_probs = _extract_stance_logprobs(
                    resp.choices[0].logprobs.content
                )
            return text, tokens, stance_probs
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, MAX_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES, e)
                raise
    raise RuntimeError("Unreachable")
