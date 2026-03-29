#!/usr/bin/env python3
"""Wrapper to launch vLLM serve with broken system tensorflow blocked."""
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

from vllm.scripts import main

if __name__ == "__main__":
    main()
