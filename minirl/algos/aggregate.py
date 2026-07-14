"""
================================================================================
 THE REDUCE — per-token loss map (B, T)  ->  scalar, exactly ONCE
================================================================================
How a loss map becomes a scalar is an ALGORITHMIC choice each paper makes.
It is specified by ONE config field on every loss config —

    loss_agg: "seq_mean" | "token_mean" | int

— and applied by exactly two functions in THIS file. The trainer never
branches on it; the losses never see it. Three modes, three papers:

  "seq_mean"    L = (1/B) * sum_i ( masked_sum_i / tokens_i )       GRPO paper
                every COMPLETION weighs equally (a 2k-token chain = one short
                answer).                       slime: calculate_per_token_loss=False
  "token_mean"  L = sum_i masked_sum_i / (total tokens in minibatch) DAPO
                every TOKEN weighs equally; denominator varies with what was
                sampled (a mild batch-level length coupling — DAPO's choice).
                                               slime: calculate_per_token_loss=True
  int C         L = sum_i masked_sum_i / (B * C)                     Dr. GRPO
                constant-wrt-sampled-lengths denominator (paper uses the max
                generation budget) => unbiased; the value only scales the lr.
                Not in slime; from the Dr. GRPO paper/repo (arXiv:2503.20783).

All three are one formula — sum_i(masked_sum_i / denom_i) — differing only in
the denominator, which is exactly how slime implements it: a reducer closure
over per-sample denominators (cp_utils.get_sum_of_sample_mean), built once
from config and handed to the loss.

THE MICROBATCH RULE (docs/sync_training.md §5, pinned by the gradient-
equivalence test; slime's docstring states the same): the denominator must be
computed on the WHOLE MINIBATCH (minibatch_denom, once) and shared by every
grad-accumulation slice — per-microbatch denominators silently reweight
tokens. Under FSDP2 the rule generalizes across ranks with NO collective:
every rank computes the same denominator from the identical full batch and
slices its rows after (train/distributed.py; docs/fsdp2.md §2).
================================================================================
"""

from torch import Tensor

LossAgg = str | int  # "seq_mean" | "token_mean" | int constant (see banner)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of x over True positions of mask. Any matching shapes -> scalar.
    For METRICS only — the loss reduce below never uses it."""
    return (x * mask).sum() / mask.sum().clamp(min=1)


def minibatch_denom(agg: LossAgg, mask: Tensor) -> Tensor | float:
    """The ONE denominator for a whole minibatch. mask: (B, T) bool.

    Called once per optimizer step by the trainer; the result is shared by
    every microbatch slice of that minibatch (see banner: the microbatch rule).
    """
    b = mask.shape[0]
    if agg == "seq_mean":
        return b  # each row is internally mean'd; then / B
    if agg == "token_mean":
        return mask.sum().clamp(min=1)  # actual response tokens in the minibatch
    if isinstance(agg, int):
        return b * agg  # Dr. GRPO: rows * constant, independent of sampled lengths
    raise ValueError(f"unknown loss_agg: {agg!r}")


def aggregate_loss(
    loss_map: Tensor,  # (b, T) per-token losses of ONE microbatch (0 outside mask)
    mask: Tensor,  # (b, T) bool
    agg: LossAgg,  # same value the denom was built with
    denom: Tensor | float | None = None,  # minibatch_denom(...); None = this micro is the whole minibatch
) -> Tensor:
    """Reduce one microbatch's loss map to a scalar contribution.

    Contributions from the minibatch's microbatches SUM to the paper's loss
    because every slice divides by the same minibatch-global denom.

    NOTE on the (deliberately) redundant `* mask`: loss fns already return maps
    that are zero outside loss_mask (their contract), but this function must
    stay correct for callers that didn't pre-mask — and seq_mean needs the mask
    for its inner per-row denominators regardless. Each module upholds its own
    invariant instead of trusting the other file's convention.
    """
    if denom is None:
        denom = minibatch_denom(agg, mask)
    if agg == "seq_mean":
        per_seq = (loss_map * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (b,) each row's own mean
        return per_seq.sum() / denom  # scalar
    # token_mean and int-constant modes differ only in the denominator:
    return (loss_map * mask).sum() / denom  # scalar
