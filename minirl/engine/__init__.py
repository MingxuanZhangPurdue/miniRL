"""Rollout engines. Duck-typed — the streaming contract (submit/poll/stash/
drain/n_inflight/load_weights/pad_id) is what the fully-async controller
consumes; the sampling side is documented on SamplingParams in
rollout/types.py.

VLLMEngine is THE engine (vLLM-only decision, 2026-07-20; HFEngine and its
StreamAdapter were removed the same day — test fakes in
tests/test_fully_async.py now pin the streaming contract). The module
imports cleanly without vLLM (imports are method-local); instantiation
needs the vLLM env.
"""

from minirl.engine.vllm_engine import VLLMEngine

__all__ = ["VLLMEngine"]
