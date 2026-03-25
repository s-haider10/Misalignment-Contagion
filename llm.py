"""AsyncOpenAI client pool and LLM call wrapper for vLLM servers."""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ── Model name lookup ─────────────────────────────────────────────────

# ── Model registry ────────────────────────────────────────────────────
# Each scale has an aligned model and a misaligned (fine-tuned) counterpart.

ALIGNED_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
    "14b":  "Qwen/Qwen2.5-14B-Instruct",
}

MISALIGNED_MODELS = {
    "0.5b": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    "7b":   "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "14b":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
}

# Flat lookup used by --model CLI arg (points to aligned model name for vLLM)
MODEL_NAMES = ALIGNED_MODELS


def get_model_name(scale: str, role: str, model_condition: str) -> str:
    """Return the HF model name for a given agent.

    In model_induced condition, misaligned agents use the fine-tuned model.
    Otherwise everyone uses the aligned model at the given scale.
    """
    if model_condition == "model_induced" and role == "misaligned":
        return MISALIGNED_MODELS[scale]
    return ALIGNED_MODELS[scale]


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

    - model_induced + misaligned → GPU 3 (misaligned fine-tuned model)
    - otherwise → distribute across available GPUs by agent_id hash
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
