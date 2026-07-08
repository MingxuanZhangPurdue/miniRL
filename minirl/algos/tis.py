"""
================================================================================
 TIS — truncated importance sampling: the engine<->trainer mismatch correction
================================================================================
Orthogonal to every surrogate (grpo/dapo/gspo/cispo all call this identically),
because the mismatch is a property of the INFRA, not the algorithm.

THE TWO-GAP PICTURE — three copies of "the same" policy touch each batch:

  pi_engine@v_{k-1} --(gap 1: version lag + vLLM numerics)--> pi_old = learner@v_k --(gap 2: drift over ppo epochs)--> pi_theta
         └──────────────── TIS (this file):  w_t = clamp(exp(old - behavior), lo, C) ────────────────┘└──── PPO clip (in the losses) ────┘

Rollout tokens were sampled from the inference engine's numerics (vLLM
kernels, bf16 — DESIGN §6.0; we measured the gap), and under async training
also from an older weight version. The unbiased fix is the per-token weight

    w_t = pi_old_trainer(y_t) / pi_engine(y_t) = exp(logpi_old - logpi_rollout)

TRUNCATED to [tis_clip_low, tis_clip]: truncation trades a little bias for a
lot of variance — the standard TIS compromise
(https://fengyao.notion.site/off-policy-rl).

GROUNDED IN SLIME (megatron_utils/loss.py):
  - mode="clamp" == vanilla_tis_function:
        tis = exp(old_log_probs - rollout_log_probs)
        weights = clamp(tis, min=args.tis_clip_low, max=args.tis_clip)  # defaults 0, 2.0
        pg_loss = pg_loss * weights
  - mode="mask"  == icepop_function (ICEPOP): out-of-band tokens get weight 0
    (REJECTED outright) instead of a truncated weight — rejection beats
    truncation when the mismatch is pathological rather than noisy.
  slime applies this to the pg term only, BEFORE the KL loss is added — the
  callers here preserve that ordering (see grpo.py).
================================================================================
"""

import torch
from torch import Tensor

from minirl.algos.aggregate import masked_mean


def apply_tis(
    loss_map: Tensor,  # (B, T) per-token surrogate loss (pre-KL)
    old_logprobs: Tensor,  # (B, T) f32 FROZEN — trainer's pi_old recompute (engine values in the degenerate case)
    rollout_logprobs: Tensor,  # (B, T) f32 FROZEN — what the engine reported at sampling time
    mask: Tensor,  # (B, T) bool — completion-token positions
    tis_clip: float = 2.0,  # upper cap C (slime --tis-clip default 2.0)
    tis_clip_low: float = 0.0,  # lower bound (slime --tis-clip-low default 0)
    mode: str = "clamp",  # "clamp" (truncate) | "mask" (icepop: reject)
) -> tuple[Tensor, dict]:
    """Returns (reweighted loss_map (B, T), tis metrics dict).

    IDENTITY when old == rollout (w = exp(0) = 1 everywhere), so always safe to
    enable. detach() is belt-and-braces: both inputs are grad-free already, and
    the weight must stay a pure coefficient — never a gradient path.
    """
    tis = (old_logprobs - rollout_logprobs).exp().detach()  # (B, T)  1 where no mismatch
    if mode == "clamp":
        weight = tis.clamp(min=tis_clip_low, max=tis_clip)  # (B, T)  truncate the tails
    elif mode == "mask":
        in_band = (tis >= tis_clip_low) & (tis <= tis_clip)  # (B, T) bool
        weight = torch.where(in_band, tis, torch.zeros_like(tis))  # (B, T)  reject the tails
    else:
        raise ValueError(f"unknown tis mode: {mode}")
    metrics = {
        "tis_mean": masked_mean(tis, mask),  # scalar; drift from 1.0 = growing mismatch
        "tis_clip_frac": masked_mean((weight != tis).float(), mask),  # scalar; fraction corrected
    }
    return loss_map * weight, metrics  # (B, T), dict
