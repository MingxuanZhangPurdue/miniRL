"""
================================================================================
 SFT LOSS — masked next-token NLL (the degenerate member of the family)
================================================================================
LOSS (notation: see grpo.py; no ratio, no advantage, no reference):

    L_t  = -log pi_theta(y_t | y_<t)      on loss_mask (assistant) tokens only
    L    = per-TOKEN mean of L_t          (calculate_per_token_loss=True)

THE ONE IDEA: maximize log-likelihood of curated completion tokens.
policy_logprobs already IS log pi of the realized tokens (the trainer's
gather_logprobs produced it), so the loss map is one negation. Seen through
the RL lens: REINFORCE with r_t = 1, A = 1 on every demo token, no baseline.

WHY it lives with the RL losses: same signature, same loss_mask semantics,
same aggregate.py reduce — exactly slime's arrangement (loss_type="sft_loss"
inside the one dispatcher, same sum_of_sample_mean normalizer as RL, data via
a pass-through "rollout"). Post-training stages differ by (data, loss), never
by trainer.
================================================================================
"""

from dataclasses import dataclass

from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class SFTConfig:
    calculate_per_token_loss: bool = True  # token-level: every demo token weighs equally


def sft_loss(policy_logprobs: Tensor, batch: Batch, cfg: SFTConfig) -> tuple[Tensor, dict]:
    """Per-token NLL loss map.

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t).
        batch: loss_mask (B, T) bool — True on ASSISTANT tokens only (prompt,
            system, and tool-output tokens are context, never targets;
            data/chat.py builds this mask).

    Returns:
        loss_map (B, T) f32, zero outside loss_mask, unreduced.
        metrics: nll (mean over masked tokens) and its exp (perplexity).
    """
    mask = batch.loss_mask  # (B, T) bool
    loss_map = -policy_logprobs * mask  # (B, T)
    nll = masked_mean(-policy_logprobs.detach(), mask)  # scalar
    return loss_map, {"nll": nll, "ppl": nll.exp()}  # (B, T), dict
