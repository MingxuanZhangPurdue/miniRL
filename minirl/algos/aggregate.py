"""
================================================================================
 THE REDUCE — per-token loss map (B, T)  ->  scalar, exactly ONCE
================================================================================
Token vs sequence normalization is a real algorithmic choice, not plumbing:

  >>> PER-SEQUENCE (GRPO paper / rl_notes grpo_loss):
        masked_mean(per_tok, mask, dim=-1).mean()
      each seq divided by ITS OWN length, then mean over seqs
      => every COMPLETION weighted equally (a 2k-token chain = one short answer)

  >>> PER-TOKEN (DAPO change #3 / rl_notes rl_loss.py DAPOLoss):
        (per_tok * mask).sum() / mask.sum()
      => every TOKEN weighted equally; long correct chains count proportionally

So it is config (each paper's cfg.calculate_per_token_loss) — but it lives
HERE and in no loss function, applied once by the trainer. The `denom`
argument must be the minibatch-GLOBAL count when microbatching: per-microbatch
means silently reweight tokens (docs/sync_training.md §5; pinned by the
gradient-equivalence test, and cf. slime's num_tokens / sum_of_sample_mean).
================================================================================
"""

from torch import Tensor


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of x over True positions of mask. Any matching shapes -> scalar.
    (rl_notes' masked_mean reduces per-row with dim=-1; this one reduces fully —
    it is used for METRICS here, not for the loss reduce below.)"""
    return (x * mask).sum() / mask.sum().clamp(min=1)


def aggregate_loss(
    loss_map: Tensor,  # (B, T) per-token losses (already 0 outside mask)
    mask: Tensor,  # (B, T) bool
    mode: str,  # "token" | "sequence"
    denom: Tensor | float | None = None,  # GLOBAL normalizer; None = local (single-microbatch case)
) -> Tensor:
    """token:    sum(loss) / denom,           denom = total masked tokens in the MINIBATCH
    sequence: sum_b(mean_t loss_b) / denom, denom = total sequences in the MINIBATCH
    Pass the minibatch-global denom when splitting into microbatches (trainer does).

    NOTE on the (deliberately) redundant `* mask`: loss fns already return maps
    that are zero outside loss_mask (their contract). But the mask is needed
    HERE regardless — the denominators (which positions COUNT) can't be read
    off a pre-zeroed map — so re-applying it costs one elementwise op and makes
    this function correct even for a caller that didn't pre-mask. Each module
    upholds its own invariant instead of trusting the other file's convention.
    """
    if mode == "token":
        denom = mask.sum().clamp(min=1) if denom is None else denom
        return (loss_map * mask).sum() / denom  # scalar
    if mode == "sequence":
        per_seq = (loss_map * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # (B,)  each seq's own mean
        denom = loss_map.shape[0] if denom is None else denom
        return per_seq.sum() / denom  # scalar
    raise ValueError(f"unknown aggregation mode: {mode}")
