#!/usr/bin/env python3
"""Wrapper to launch vLLM serve with broken system tensorflow blocked.

Also runs a dtype pre-flight check (see _preflight_dtype) so an unsupported
--dtype fails immediately with an actionable message instead of a deep vLLM
engine-core traceback several minutes into startup.
"""
import importlib
import importlib.machinery
import sys
import types

# Create a fake tensorflow module that looks real enough to satisfy
# PyTorch's dynamo trace_rules (which inspects __spec__) but prevents
# the broken system tensorflow from actually loading.
_fake = types.ModuleType("tensorflow")
_fake.__version__ = "0.0.0"
_fake.__path__ = []
_fake.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _fake


def _preflight_dtype(argv: list[str]) -> None:
    """Fail fast if the requested dtype cannot work on this GPU.

    bfloat16 requires compute capability >= 8.0 (Ampere+). On the Turing cards
    in this box (Quadro RTX 6000, cc 7.5) vLLM raises deep inside engine-core
    startup, after weights have already been downloaded and loaded -- several
    minutes in, with a traceback that buries the one useful line. Catch it here.

    See logs/llama_x_qwen.log for the failure this prevents:
        ValueError: Bfloat16 is only supported on GPUs with compute capability
        of at least 8.0. Your Quadro RTX 6000 GPU has compute capability 7.5.

    NOTE: this deliberately does NOT reject float16 for Llama models. The
    published Llama-3.1-8B and Llama-3.2-1B runs used --dtype half on this
    hardware and produced clean results (run_all.log: 0 NaN over 1050 trials).
    fp16 is the only dtype these GPUs support for bf16-native weights.
    """
    dtype = None
    for i, a in enumerate(argv):
        if a == "--dtype" and i + 1 < len(argv):
            dtype = argv[i + 1].lower()
        elif a.startswith("--dtype="):
            dtype = a.split("=", 1)[1].lower()
    if dtype not in ("bfloat16", "bf16"):
        return

    try:
        import torch
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        return  # never let the guard itself break a launch

    if (major, minor) < (8, 0):
        sys.stderr.write(
            f"\nFATAL: --dtype {dtype} is not supported on this GPU.\n"
            f"  device : {name} (compute capability {major}.{minor})\n"
            f"  needed : compute capability >= 8.0\n\n"
            f"  Use --dtype half instead. On these Turing cards fp16 is the only\n"
            f"  option for bf16-native weights (Llama, Qwen); vLLM logs\n"
            f"  'Casting torch.bfloat16 to torch.float16' and proceeds.\n\n"
            f"  Refusing to start rather than fail ~4 minutes into weight loading.\n\n"
        )
        raise SystemExit(2)


from vllm.scripts import main

if __name__ == "__main__":
    _preflight_dtype(sys.argv)
    main()
