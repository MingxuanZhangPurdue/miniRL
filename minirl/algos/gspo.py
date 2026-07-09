"""
================================================================================
 GSPO LOSS — Group Sequence Policy Optimization (Zheng et al. 2025, Qwen team)
================================================================================
LOSS (notation: see grpo.py; new symbol s_i = SEQUENCE-level ratio):

    s_i  = ( pi_theta(y_i | x) / pi_old(y_i | x) )^(1/|y_i|)          length-normalized
         = exp( (1/|y_i|) * sum_t [ log pi_theta(y_t) - log pi_old(y_t) ] )
         = GEOMETRIC MEAN of the token ratios r_t — one per completion
    L_t  = -min( s_i * A_i ,  clip(s_i, 1-eps, 1+eps_hi) * A_i )      same value for
           [ * w_t ]  TIS, if use_tis                                 EVERY t of completion i
    L    = per-SEQUENCE mean of L_t     (loss_agg="seq_mean")

Companion to:
  - notes/grpo_to_gspo_derivation.md  (THE derivation: why the true sequence
    IS weight explodes, the 1/|y| variance shrinkage, the bias trade)
  - rl_notes: gspo_loss_explained.py  (the annotated version of exactly this)
  - grpo.py                           (GSPO = GRPO with ONE change: the ratio)

THE ONE IDEA:
  GRPO uses a TOKEN-level importance ratio  r_t = pi_theta(y_t)/pi_old(y_t)   (one per token).
  GSPO uses a SEQUENCE-level importance ratio (length-normalized)             (one per completion):

      s_i(theta) = ( pi_theta(y_i | x) / pi_old(y_i | x) )^(1/|y_i|)
                 = exp( (1/|y_i|) * sum_t [ log pi_theta(y_t) - log pi_old(y_t) ] )
                 = the GEOMETRIC MEAN of the per-token ratios.

  Everything else — group advantage, the clipped -min(sA, clip(s)A) — is the same.

--------------------------------------------------------------------------------
 WHAT CHANGES vs GRPO  (read alongside grpo.py)
--------------------------------------------------------------------------------
  GRPO                                    GSPO
  ----                                    ----
  ratio is TOKEN-level   (B, T)           ratio is SEQUENCE-level, broadcast (B,) -> (B, T)
  clip decision PER TOKEN                 clip decision PER SEQUENCE (whole completion)
  clip eps ~ 0.2                          clip eps TINY (~3e-4) — seq ratio sits very close to 1
  noisy for long seqs / MoE               stable for long seqs / MoE  <- the whole motivation

  WHY: token ratios are unbiased but HIGH-VARIANCE — per-token noise compounds
  over a long sequence, and one outlier token can clip (zero the gradient of)
  an otherwise good completion. The geometric mean averages that noise; the
  reward is per-sequence anyway, so GSPO matches the IS granularity to it.

  IDENTICAL: STEP 0 rollout, STEP 1 bookkeeping, STEP 2 group advantage
  (advantage.py), TIS, and the reduce — only this file's ratio differs.

GROUNDED IN SLIME: advantage_estimator="gspo" -> ppo_utils.compute_gspo_kl
computes per-sequence  ppo_kl = ((old - new) * mask).sum() / mask.sum(),
expands it onto tokens, and feeds the SAME compute_policy_loss as GRPO —
ratio = exp(-ppo_kl) = our s. (Their CP all-gather has no single-node analog.)
================================================================================
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.algos.tis import apply_tis
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class GSPOConfig:
    eps_clip: float = 3e-4  # sequence ratios hug 1 -> tiny clips (paper order of magnitude)
    eps_clip_high: float = 4e-4
    grpo_std_normalization: bool = True  # STEP 2 unchanged: advantages still group-normalized
    loss_agg: str | int = "seq_mean"  # sequence-level algorithm -> per-sequence mean reduce
    use_tis: bool = False
    tis_clip: float = 2.0
    tis_clip_low: float = 0.0
    tis_mode: str = "clamp"


def gspo_loss(policy_logprobs: Tensor, batch: Batch, cfg: GSPOConfig) -> tuple[Tensor, dict]:
    """STEP 3 — the loss. Diff from grpo_loss: the ratio is SEQUENCE-level.

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t).
        batch: loss_mask (B, T) bool; advantages (B, T) f32 FROZEN;
            behavior_logprobs (B, T) f32 FROZEN; old_logprobs (B, T) FROZEN
            (IS denominator; falls back to behavior when absent).

    Returns:
        loss_map (B, T) f32 unreduced (zero outside loss_mask), metrics dict.
    """
    mask = batch.loss_mask  # (B, T) bool
    adv = batch.advantages  # (B, T) f32 — FROZEN, constant per row
    old = batch.old_logprobs if batch.old_logprobs is not None else batch.behavior_logprobs  # (B, T)

    # ---- per-token log ratios, masked so only completion tokens contribute ----
    log_ratio = (policy_logprobs - old) * mask  # (B, T)

    # ---- THE GSPO MOVE: one length-normalized ratio per sequence ----
    #   s_i = exp( mean_t log r_t )  = geometric mean of the token ratios.
    # (clamp avoids 0/0 on all-masked rows). Grad still reaches EVERY completion
    # token of the row — through the mean, each token gets 1/|y_i| of it.
    seq_log_ratio = log_ratio.sum(-1) / mask.sum(-1).clamp(min=1)  # (B,)
    ratio = seq_log_ratio.exp().unsqueeze(-1).expand_as(policy_logprobs)  # (B, T)  UNIFORM per row

    # ---- same clipped surrogate as GRPO — but clipping whole SEQUENCES now ----
    clipped = ratio.clamp(1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip_high)  # (B, T)
    loss_map = -torch.minimum(ratio * adv, clipped * adv)  # (B, T)

    metrics = {
        # every token of a clipped row counts, so this ~= fraction of clipped SEQUENCES
        "clip_frac": masked_mean((clipped * adv < ratio * adv).float(), mask),  # scalar
        # mixed granularity on purpose: per token this is (s_i - 1 - log r_t),
        # but summed over a row it telescopes (sum_t log r_t = |y_i| log s_i)
        # into |y_i| * k3(s_i) — so the masked mean is the token-count-weighted
        # mean of per-SEQUENCE k3 drift
        "approx_kl": masked_mean(ratio.detach() - 1 - log_ratio.detach(), mask),  # scalar
        "ratio_max": ratio.detach().max(),  # scalar
    }

    # ---- TIS stays PER-TOKEN even here: the engine-numerics mismatch it
    # corrects is a per-token effect, independent of the ratio granularity ----
    if cfg.use_tis:
        loss_map, tis_metrics = apply_tis(
            loss_map, old, batch.behavior_logprobs, mask, cfg.tis_clip, cfg.tis_clip_low, cfg.tis_mode
        )  # (B, T), dict
        metrics |= tis_metrics

    return loss_map * mask, metrics  # (B, T), dict
