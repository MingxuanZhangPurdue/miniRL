"""One file per algorithm whose loss BODY differs from GRPO's; NAMED CONFIGS
for papers reachable by setting GRPO's fields. Formula table + notation +
config reference: algos/README.md.

Shared pieces are only the genuinely orthogonal ones: advantages
(advantage.py), TIS mismatch correction (tis.py), the reduce (aggregate.py).

The wrapper: make_loss(name, **overrides) -> (loss_fn, cfg). Every loss_fn has
the same signature: (policy_logprobs (B,T), batch, cfg) -> (loss_map (B,T), metrics).
"""

from functools import partial

from minirl.algos.advantage import degenerate_group_mask, grpo_advantages
from minirl.algos.aggregate import aggregate_loss, masked_mean, minibatch_denom
from minirl.algos.cispo import CISPOConfig, cispo_loss
from minirl.algos.grpo import GRPOConfig, grpo_loss
from minirl.algos.gspo import GSPOConfig, gspo_loss
from minirl.algos.sft import SFTConfig, sft_loss
from minirl.algos.tis import apply_tis

LOSSES = {
    "grpo": (grpo_loss, GRPOConfig),
    "gspo": (gspo_loss, GSPOConfig),
    "cispo": (cispo_loss, CISPOConfig),
    "sft": (sft_loss, SFTConfig),
    # ---- NAMED CONFIGS (file-vs-config rule: their loss bodies == GRPO's) ----
    # DAPO (arXiv:2503.14476): clip-higher + token-level reduce; no-KL is
    # already GRPO's default. Its 4th change, dynamic sampling, is batch
    # collection: CollectConfig(strategy="filter") in controllers/fully_async.py.
    "dapo": (grpo_loss, partial(GRPOConfig, eps_clip_high=0.28, loss_agg="token_mean")),
    # Dr. GRPO (arXiv:2503.20783): drop the ÷std in the advantage + unbiased
    # reduce. PAPER-EXACT normalization needs the constant denominator, a
    # runtime setting we can't preset: make_loss("dr_grpo", loss_agg=<max_new_tokens>).
    "dr_grpo": (grpo_loss, partial(GRPOConfig, grpo_std_normalization=False, loss_agg="token_mean")),
}


def make_loss(name: str, **overrides):
    """make_loss("dapo", use_tis=True) -> (grpo_loss, GRPOConfig(eps_clip_high=0.28, ...))."""
    loss_fn, cfg_cls = LOSSES[name]
    return loss_fn, cfg_cls(**overrides)


__all__ = [
    "LOSSES",
    "make_loss",
    "apply_tis",
    "aggregate_loss",
    "masked_mean",
    "minibatch_denom",
    "grpo_advantages",
    "degenerate_group_mask",
    "grpo_loss",
    "gspo_loss",
    "cispo_loss",
    "sft_loss",
    "GRPOConfig",
    "GSPOConfig",
    "CISPOConfig",
    "SFTConfig",
]
