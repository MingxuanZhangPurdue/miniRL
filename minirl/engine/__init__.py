"""Rollout engines. Duck-typed — the two-method contract (generate /
load_weights) is documented on SamplingParams in rollout/types.py.
HFEngine: reference, any device. VLLMEngine: CUDA, real experiments (later).
"""

from minirl.engine.hf_engine import HFEngine

__all__ = ["HFEngine"]
