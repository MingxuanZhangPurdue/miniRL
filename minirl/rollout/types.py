"""Core data contract between rollout engines, rewards, and the learner.

Everything in miniRL flows through these types (≈ slime's `Sample`,
≈ a mini verl `DataProto`). Getting the alignment conventions right here is
what lets SFT / DPO / RL / agentic / async all share one trainer.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class SamplingParams:
    """Request side of the engine contract (field names mirror vllm.SamplingParams).

    Engines are duck-typed — no base class. An engine is any object speaking
    the STREAMING contract consumed by train_async.py:
      submit(prompt_ids: (T,) int64, params, meta) -> request id — ONE prompt,
          a whole group of params.n completions;
      poll() -> list[list[Trajectory]] — finished GROUPS, each trajectory
          carrying RAW-model behavior logprobs and the engine's weight version;
      stash(group) / drain() / n_inflight / pad_id;
      load_weights(named_tensors: iterable[(str, Tensor)], version: int) -> None
          (drain first — asserted; every completion sees exactly ONE version).
    VLLMEngine (minirl/vllm_engine.py) is THE engine.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled
    max_new_tokens: int = 512
    n: int = 1  # completions per prompt (G in GRPO)


@dataclass
class Trajectory:
    """One sampled episode (single- or multi-turn), flat over tokens.

    Alignment convention (the single most bug-prone thing in RL repos):
      input_ids[t] is a token; logprobs[t] = log pi(input_ids[t] | input_ids[:t]).
      Prompt positions (and later, tool-output positions) carry logprob 0.0 and
      loss_mask 0 — they were not produced by the policy and never enter a loss.
    """

    input_ids: Tensor  # (T,) int64 — prompt + generated (+ tool) tokens
    loss_mask: Tensor  # (T,) bool — True where the POLICY produced the token
    logprobs: Tensor  # (T,) float32 — behavior logprobs at sampling time; 0.0 where loss_mask is False
    reward: float = 0.0  # terminal scalar reward (assigned by rewards/)
    version: int = 0  # policy version that generated this (async staleness)
    meta: dict = field(default_factory=dict)  # prompt id, group id, env info, ...

    def __post_init__(self) -> None:
        (t,) = self.input_ids.shape
        assert self.loss_mask.shape == (t,) and self.logprobs.shape == (t,), (
            f"misaligned trajectory: ids {tuple(self.input_ids.shape)}, "
            f"mask {tuple(self.loss_mask.shape)}, logprobs {tuple(self.logprobs.shape)}"
        )
        assert self.input_ids.dtype == torch.long and self.loss_mask.dtype == torch.bool

    @property
    def prompt_len(self) -> int:
        # Prompt = leading unmasked span. (Multi-turn adds interior unmasked
        # spans for tool outputs; those don't affect this property.)
        return int((~self.loss_mask).cumprod(0).sum())

    @property
    def response_len(self) -> int:
        return int(self.loss_mask.sum())


@dataclass
class Batch:
    """Collated, right-padded training view of B trajectories (≈ verl DataProto).

    Same alignment as Trajectory: position t holds token t and (for logprob
    tensors) log pi(token_t | tokens_<t). Padding always has loss_mask False.
    """

    input_ids: Tensor  # (B, T) int64, right-padded
    attention_mask: Tensor  # (B, T) bool — True on real tokens (prompt + response)
    loss_mask: Tensor  # (B, T) bool — True ONLY on policy-produced tokens
    behavior_logprobs: Tensor  # (B, T) f32 — engine-reported at sampling time
    advantages: Tensor  # (B, T) f32 — usually a per-row scalar broadcast over response
    rewards: Tensor  # (B,) f32
    group_ids: Tensor  # (B,) int64 — row -> prompt group (GRPO)
    # Optional extras, filled by the controller when the algorithm needs them:
    old_logprobs: Tensor | None = None  # (B, T) learner recompute of pi_old (fp32); None -> use behavior
    ref_logprobs: Tensor | None = None  # (B, T) frozen reference, for KL penalty

