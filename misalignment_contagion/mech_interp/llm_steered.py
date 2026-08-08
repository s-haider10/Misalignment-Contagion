"""Steered in-process model calls — drop-in for call_llm_with_logprobs.

This module provides a `SteeringHandle` class that loads Qwen-7B-Instruct
in-process via transformers, registers a forward hook on a target layer
(default 26) that adds `α · direction` to the residual stream when α is
nonzero, and exposes `call_steered` with the same signature as
`llm.call_llm_with_logprobs`.

Usage:

    handle = SteeringHandle(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        direction_path="outputs/direction_results/direction_layer26_round_4.npy",
        target_layer=26,
    )
    text, tokens, stance_probs = await handle.call_steered(
        messages=[...],
        temperature=0.7,
        seed=42,
        model_name="Qwen/Qwen2.5-7B-Instruct",  # ignored, here for API match
        max_tokens=512,
        alpha=1.0,  # 0.0 means no steering
    )

The handle is intended to be created once at the start of the experiment
and reused across all trials. Aligned-agent calls go through this wrapper;
misaligned-agent calls keep using the original vLLM HTTP path.
"""

from __future__ import annotations

import math
import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

STANCE_DIGITS = {"1", "2", "3", "4", "5", "6", "7"}


class SteeringHandle:
    """Persistent transformers model with a steering hook on one layer."""

    def __init__(
        self,
        model_name: str,
        direction_path: str | Path,
        target_layer: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.target_layer = target_layer
        self.device = device

        logger.info("Loading %s on %s ...", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
        )
        self.model.eval()

        # Load direction; convert to a (1, 1, hidden_dim) tensor for broadcasting
        direction = np.load(direction_path).astype(np.float32)
        self.direction_np = direction
        self.direction_tensor = torch.from_numpy(direction).to(
            device=device, dtype=dtype
        ).view(1, 1, -1)
        self.direction_norm = float(np.linalg.norm(direction))

        # Mutable steering scalar — set per-call
        self._alpha: float = 0.0

        # Register a forward hook
        layer = self.model.model.layers[target_layer]
        layer.register_forward_hook(self._make_hook())

        # Cache stance digit token ids
        self.stance_token_ids = self._cache_stance_tokens()

        logger.info(
            "SteeringHandle ready: layer=%d, ||d||=%.3f, stance_tokens=%s",
            target_layer, self.direction_norm,
            {d: tid for d, tid in self.stance_token_ids.items()},
        )

    def _make_hook(self):
        """Closure that captures self so we can read self._alpha live."""
        def hook(module, inp, out):
            if self._alpha == 0.0:
                return out
            # `out` is the tuple from the decoder layer: (hidden_states, ...)
            if isinstance(out, tuple):
                h = out[0]
                h = h + self._alpha * self.direction_tensor
                return (h,) + out[1:]
            else:
                return out + self._alpha * self.direction_tensor
        return hook

    def _cache_stance_tokens(self) -> dict[str, int]:
        """Map "1".."7" → token id (single-token).
        Tries with and without leading space, picks the single-token form.
        """
        out = {}
        for d in STANCE_DIGITS:
            for variant in (d, " " + d):
                ids = self.tokenizer(variant, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    out[d] = ids[0]
                    break
            else:
                raise RuntimeError(
                    f"Stance digit {d!r} does not tokenize to a single token. "
                    "Tokenizer mismatch."
                )
        return out

    # ────────────────────────────────────────────────────────────────────
    # Main entrypoint — drop-in for call_llm_with_logprobs
    # ────────────────────────────────────────────────────────────────────
    async def call_steered(
        self,
        messages: list[dict],
        temperature: float,
        seed: int,
        model_name: str,  # accepted for API compatibility; ignored
        max_tokens: int = 512,
        alpha: float = 0.0,
    ) -> tuple[str, int, dict[int, float] | None]:
        """Async wrapper. Runs the synchronous generate in a thread."""
        return await asyncio.to_thread(
            self._call_steered_sync,
            messages, temperature, seed, max_tokens, alpha,
        )

    def _call_steered_sync(
        self,
        messages: list[dict],
        temperature: float,
        seed: int,
        max_tokens: int,
        alpha: float,
    ) -> tuple[str, int, dict[int, float] | None]:
        # Apply chat template + generation prompt (matches what vLLM does)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = prompt_ids.shape[1]
        # Explicit attention mask — without it, fp16 generate can produce
        # NaN logits on long prompts where pad_token == eos_token.
        attention_mask = torch.ones_like(prompt_ids)

        # Set steering scalar; reset in finally to be safe
        self._alpha = float(alpha)
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                gen_out = self.model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            self._alpha = 0.0

        # Decode the generated text (only the new tokens)
        gen_ids = gen_out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Total tokens: prompt + new
        total_tokens = int(prompt_len + gen_ids.shape[0])

        # Find the first stance digit token in the generated sequence and
        # extract the logit distribution at that position over digits 1-7.
        stance_probs = self._extract_stance_probs(gen_ids, gen_out.scores)

        return text, total_tokens, stance_probs

    def _extract_stance_probs(
        self,
        gen_ids: torch.Tensor,
        scores: tuple[torch.Tensor, ...],
    ) -> dict[int, float] | None:
        """Find the first generated token that decodes to a stance digit,
        read its logit distribution, renormalize over digits 1-7."""
        gen_list = gen_ids.tolist()
        digit_to_id = self.stance_token_ids
        id_to_digit = {tid: d for d, tid in digit_to_id.items()}

        for pos, tid in enumerate(gen_list):
            if tid in id_to_digit:
                # Found a stance digit at output position `pos`.
                # `scores` is a tuple of (1, vocab_size) tensors, one per generated step.
                # Step `pos` in scores is the distribution from which `gen_list[pos]` was sampled.
                step_logits = scores[pos][0]  # (vocab_size,)
                step_logprobs = torch.log_softmax(step_logits.float(), dim=-1)

                # Build raw probs for our 7 stance ids
                raw = {}
                for d, sid in digit_to_id.items():
                    raw[int(d)] = float(torch.exp(step_logprobs[sid]).item())
                total = sum(raw.values())
                if total <= 0:
                    return None
                return {i: raw[i] / total for i in range(1, 8)}

        return None


# ────────────────────────────────────────────────────────────────────────
# Convenience wrapper: present the same signature as call_llm_with_logprobs
# so trial.py can swap in this function with a one-line change.
# ────────────────────────────────────────────────────────────────────────

_global_handle: SteeringHandle | None = None


def init_global_handle(
    model_name: str,
    direction_path: str | Path,
    target_layer: int = 26,
):
    """Set a process-global SteeringHandle. Call once at experiment start."""
    global _global_handle
    _global_handle = SteeringHandle(model_name, direction_path, target_layer)


async def call_llm_with_logprobs_steered(
    client,  # ignored, accepted for API match with call_llm_with_logprobs
    messages: list[dict],
    temperature: float,
    seed: int,
    model_name: str,
    max_tokens: int = 512,
    alpha: float = 0.0,
) -> tuple[str, int, dict[int, float] | None]:
    """Drop-in replacement for llm.call_llm_with_logprobs that runs through
    the global SteeringHandle with the given alpha.

    Note: `client` is ignored. The handle is process-global. Aligned-agent
    calls in steered conditions should use this; misaligned-agent calls
    should keep using the original vLLM HTTP path.
    """
    if _global_handle is None:
        raise RuntimeError(
            "Global SteeringHandle not initialized. Call init_global_handle() "
            "at the start of your experiment."
        )
    return await _global_handle.call_steered(
        messages=messages,
        temperature=temperature,
        seed=seed,
        model_name=model_name,
        max_tokens=max_tokens,
        alpha=alpha,
    )