"""One file per algorithm, each readable top-to-bottom like its paper.

Shared pieces are only the genuinely orthogonal ones: advantages
(advantage.py), TIS mismatch correction (tis.py), aggregation (aggregate.py).

The wrapper: make_loss(name, **overrides) -> (loss_fn, cfg). Every loss_fn has
the same signature: (policy_logprobs (B,T), batch, cfg) -> (loss_map (B,T), metrics).
"""

from minirl.algos.advantage import degenerate_group_mask, grpo_advantages
from minirl.algos.aggregate import aggregate_loss, masked_mean
from minirl.algos.cispo import CISPOConfig, cispo_loss
from minirl.algos.dapo import DAPOConfig, dapo_loss
from minirl.algos.grpo import GRPOConfig, grpo_loss
from minirl.algos.gspo import GSPOConfig, gspo_loss
from minirl.algos.sft import SFTConfig, sft_loss
from minirl.algos.tis import apply_tis

LOSSES = {
    "grpo": (grpo_loss, GRPOConfig),
    "dapo": (dapo_loss, DAPOConfig),
    "gspo": (gspo_loss, GSPOConfig),
    "cispo": (cispo_loss, CISPOConfig),
    "sft": (sft_loss, SFTConfig),
}


def make_loss(name: str, **overrides):
    """make_loss("dapo", eps_clip_high=0.3) -> (dapo_loss, DAPOConfig(...))."""
    loss_fn, cfg_cls = LOSSES[name]
    return loss_fn, cfg_cls(**overrides)


__all__ = [
    "LOSSES",
    "make_loss",
    "apply_tis",
    "aggregate_loss",
    "masked_mean",
    "grpo_advantages",
    "degenerate_group_mask",
    "grpo_loss",
    "dapo_loss",
    "gspo_loss",
    "cispo_loss",
    "sft_loss",
    "GRPOConfig",
    "DAPOConfig",
    "GSPOConfig",
    "CISPOConfig",
    "SFTConfig",
]
