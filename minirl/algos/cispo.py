"""
================================================================================
 CISPO LOSS — Clipped IS-weight Policy Optimization (MiniMax-M1, arXiv:2506.13585)
================================================================================
LOSS (notation: see grpo.py; note sg(.) = stop-gradient and the missing floor):

    r_t  = exp( log pi_theta(y_t | y_<t) - log pi_old(y_t | y_<t) )
    L_t  = -sg( clip(r_t, -inf, 1+eps_hi) ) * A_i * log pi_theta(y_t | y_<t)
            └───── detached weight ─────┘         └── REINFORCE term ──┘
           [ * w_t ]  TIS, if use_tis  (a second detached coefficient)
    L    = per-TOKEN mean of L_t       (loss_agg="token_mean")

Companion to:
  - notes/ppo_to_cispo_derivation.md   (why the explicit log pi is needed)
  - rl_notes: cispo_loss_explained.py  (the annotated version of exactly this)
  - grpo.py  (CISPO reuses STEP 0-2 unchanged; only this surrogate differs)

THE ONE IDEA:
  PPO/GRPO put the gradient THROUGH the ratio and then CLIP it — so when a
  token's ratio leaves [1-eps_lo, 1+eps_hi], its gradient becomes ZERO (the
  token is dropped from the update). CISPO never drops a token: it uses the
  clipped ratio as a DETACHED (stop-gradient) WEIGHT on a REINFORCE term, so
  EVERY token keeps a gradient; clipping only BOUNDS the weight (variance).

      GRPO/PPO :  L_t = -min( r_t*A , clip(r_t)*A )       grad through r; clip -> grad 0
      CISPO    :  L_t = -sg(clip(r_t)) * A * log pi_theta  grad ONLY through log pi; never 0
                        └ detached weight ┘   └ REINFORCE term ┘

--------------------------------------------------------------------------------
 WHAT CHANGES vs GRPO  (read alongside grpo.py)
--------------------------------------------------------------------------------
  GRPO                                      CISPO
  ----                                      -----
  ratio carries gradient                    ratio is DETACHED — a weight only
  loss = -min(r*A, clip(r)*A)               loss = -sg(clip(r))*A*log pi  (IS-weighted REINFORCE)
  clipped token -> ZERO gradient            clipped token -> still gets gradient (weight bounded)
  clipping = trust region on the update     clipping = variance bound on the IS weight
  symmetric eps ~ 0.2                       lower bound OFF (canonical), only the upper cap matters

  WHY: MiniMax observed PPO's clip systematically silences exactly the rare
  low-probability "fork" tokens that matter for reasoning ("Wait", "However",
  "Aha") — after one update their ratio explodes past 1+eps and they are never
  reinforced again. CISPO keeps every token's gradient while bounding step size.

  IDENTICAL: STEP 0 rollout, STEP 1 bookkeeping, STEP 2 group advantage, TIS.

GROUNDED IN SLIME: advantage_estimator="cispo" -> ppo_utils.compute_cispo_loss:
    ratio_truncated = clamp(ratio, 1-eps_clip, 1+eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac  = (ratio_truncated != ratio)
identical math below; slime's docstring likewise notes "canonical CISPO
disables the lower bound (eps_clip >= 1.0)" — we express that as
eps_clip=None instead of their >=1 convention.
================================================================================
"""

from dataclasses import dataclass

from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.algos.tis import apply_tis
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class CISPOConfig:
    eps_clip: float | None = None  # lower delta; None = unbounded below (canonical CISPO)
    eps_clip_high: float = 0.28  # the upper cap delta — the ONLY knob the paper tunes,
    #   but it never publishes the value; 0.28 here is borrowed from DAPO's
    #   clip-higher as a sane starting point, NOT a CISPO paper number. Tune.
    grpo_std_normalization: bool = True  # STEP 2 unchanged
    loss_agg: str | int = "token_mean"  # paper normalizes over the group's total tokens
    use_tis: bool = False
    tis_clip: float = 2.0
    tis_clip_low: float = 0.0
    tis_mode: str = "clamp"


def cispo_loss(policy_logprobs: Tensor, batch: Batch, cfg: CISPOConfig) -> tuple[Tensor, dict]:
    """STEP 3 — the loss. Diff from grpo_loss: detached clipped weight, REINFORCE grad path.

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t).
            Unlike PPO-style losses the gradient path is DIRECTLY through this
            tensor (REINFORCE-style), never through the ratio.
        batch: loss_mask (B, T) bool; advantages (B, T) f32 FROZEN;
            behavior_logprobs (B, T) FROZEN; old_logprobs (B, T) FROZEN
            (IS denominator; falls back to behavior when absent).

    Returns:
        loss_map (B, T) f32 unreduced (zero outside loss_mask), metrics dict.
    """
    mask = batch.loss_mask  # (B, T) bool
    adv = batch.advantages  # (B, T) f32 — FROZEN, constant per row
    old = batch.old_logprobs if batch.old_logprobs is not None else batch.behavior_logprobs  # (B, T)

    # ---- importance ratio (same as GRPO — but about to be detached) ----
    log_ratio = (policy_logprobs - old) * mask  # (B, T)
    ratio = log_ratio.exp()  # (B, T)

    # ---- truncated IS WEIGHT: clamp, then detach — THE CISPO MOVE ----
    # clamp(min=None) leaves the lower side unbounded (canonical form).
    lo = None if cfg.eps_clip is None else 1.0 - cfg.eps_clip
    clipped = ratio.clamp(min=lo, max=1.0 + cfg.eps_clip_high)  # (B, T)

    # -sg(clip(r)) * A * log pi:  IS-weighted REINFORCE.
    #   d/dtheta = -sg(clip(r)) * A * dlogpi/dtheta  — NEVER zero: a clipped
    # token still pushes gradient through policy_logprobs (contrast grpo.py,
    # where the selected clipped branch is a constant and the gradient dies).
    loss_map = -clipped.detach() * adv * policy_logprobs  # (B, T)

    metrics = {
        # counts truncated WEIGHTS (variance control engaged) — NOT zeroed
        # gradients; no gradient is ever zeroed here, by design
        "clip_frac": masked_mean((clipped != ratio).float().detach(), mask),  # scalar
        "approx_kl": masked_mean(ratio.detach() - 1 - log_ratio.detach(), mask),  # scalar
        "ratio_max": ratio.detach().max(),  # scalar
    }

    # ---- TIS composes as a second detached coefficient:
    #   total weight = sg(clip(pi/pi_old)) * clamp(exp(pi_old/pi_engine))  ----
    if cfg.use_tis:
        loss_map, tis_metrics = apply_tis(
            loss_map, old, batch.behavior_logprobs, mask, cfg.tis_clip, cfg.tis_clip_low, cfg.tis_mode
        )  # (B, T), dict
        metrics |= tis_metrics

    return loss_map * mask, metrics  # (B, T), dict
