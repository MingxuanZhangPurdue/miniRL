"""Rollout engines. Duck-typed — the streaming contract (submit/poll/stash/
drain/n_inflight/load_weights/pad_id) is what the fully-async controller
consumes; the sampling side is documented on SamplingParams in
rollout/types.py.

HFEngine: reference generate()-engine, any device — wrap it in StreamAdapter
(one poll == one round) to feed the controller. VLLMEngine: real continuous
batching (import stays method-local; instantiation needs the vLLM env).
"""

from minirl.engine.hf_engine import HFEngine
from minirl.engine.stream_adapter import StreamAdapter

__all__ = ["HFEngine", "StreamAdapter"]
