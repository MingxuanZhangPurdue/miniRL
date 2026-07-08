"""
================================================================================
 DAPO LOSS — Decoupled clip + Dynamic sAmpling Policy Optimization (arXiv:2503.14476)
================================================================================
LOSS (notation: see grpo.py — identical surrogate, asymmetric window, no KL):

    r_t  = exp( log pi_theta(y_t | y_<t) - log pi_old(y_t | y_<t) )
    L_t  = -min( r_t * A_i ,  clip(r_t, 1-eps, 1+eps_hi) * A_i ),   eps_hi > eps
                                        └── the window is [0.8, 1.28], not [0.8, 1.2] ──┘
           [ * w_t ]                    TIS weight, if use_tis;  NO KL term (change #2)
    L    = per-TOKEN mean of L_t        (calculate_per_token_loss=True — change #3)

Companion to:
  - grpo.py                        (DAPO = GRPO with the four changes below;
                                    the surrogate math is IDENTICAL — this file
                                    exists so "what is DAPO?" has a paper-shaped
                                    answer, and the diff vs grpo.py IS the diff
                                    between the papers)
  - rl_notes: rl_loss.py DAPOLoss  (the compact class; also the per-token
                                    normalization contrast noted there)

THE ONE IDEA: **give low-probability tokens room to rise.** Under symmetric
clipping a LOW-prob token can only grow ~eps before its gradient is zeroed,
while high-prob tokens barely feel the clip — exploration tokens never get
reinforced and entropy collapses. DAPO raises ONLY the ceiling (clip-higher).

--------------------------------------------------------------------------------
 WHAT CHANGES vs GRPO — four things, and where each lives
--------------------------------------------------------------------------------
  change                        where in miniRL
  ------                        ---------------
  1. clip-higher                HERE: eps_clip_high (0.28) > eps_clip (0.2);
     (asymmetric clip)          floor 0.8 unchanged, ceiling 1.28
  2. NO KL penalty              HERE, by absence: no ref model, no beta term —
                                in long-CoT RLVR the policy is EXPECTED to move
                                far from the SFT init
  3. token-level loss           trainer, via calculate_per_token_loss=True:
     (vs GRPO's per-seq mean)   every token weighs equally, so long correct
                                chains aren't down-weighted (aggregate.py)
  4. dynamic sampling           NOT here — batch collection, not loss math:
     (resample 0-gradient       rollout/sampling.py, CollectConfig(strategy=
     groups away)               "filter"), reward_nonzero_std
  (DAPO's overlong-reward-shaping is a reward-fn concern; rewards/ if needed.)

GROUNDED IN SLIME: same code path as GRPO there — ppo_utils.compute_policy_loss
with --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss, no
--use-kl-loss, and --dynamic-sampling-filter-path check_reward_nonzero_std.
slime expresses DAPO as flags; we express it as this file.
================================================================================
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.algos.tis import apply_tis
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class DAPOConfig:
    eps_clip: float = 0.2  # lower clip delta (unchanged from GRPO)
    eps_clip_high: float = 0.28  # clip-higher: the paper's headline knob
    grpo_std_normalization: bool = True  # STEP 2 unchanged (advantage.py)
    calculate_per_token_loss: bool = True  # change #3, consumed by the trainer
    use_tis: bool = False
    tis_clip: float = 2.0
    tis_clip_low: float = 0.0
    tis_mode: str = "clamp"


def dapo_loss(policy_logprobs: Tensor, batch: Batch, cfg: DAPOConfig) -> tuple[Tensor, dict]:
    """STEP 3 — the loss. Identical surrogate to grpo_loss; only the clip is asymmetric.

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t).
        batch: loss_mask (B, T) bool; advantages (B, T) f32 FROZEN;
            behavior_logprobs (B, T) FROZEN; old_logprobs (B, T) FROZEN.

    Returns:
        loss_map (B, T) f32 unreduced (zero outside loss_mask), metrics dict.
    """
    mask = batch.loss_mask  # (B, T) bool
    adv = batch.advantages  # (B, T) f32 — FROZEN, constant per row
    old = batch.old_logprobs if batch.old_logprobs is not None else batch.behavior_logprobs  # (B, T)

    # ---- importance ratio (identical to grpo.py) ----
    log_ratio = (policy_logprobs - old) * mask  # (B, T)  masked positions -> log 1
    ratio = log_ratio.exp()  # (B, T)

    # ---- clip-higher surrogate: THE DAPO change ----
    # Note the ASYMMETRY: floor 1 - 0.2 = 0.8, ceiling 1 + 0.28 = 1.28. A rising
    # low-prob token keeps its gradient until r > 1.28 instead of 1.2; the
    # downside floor (which protects against collapse) is untouched.
    clipped = ratio.clamp(1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip_high)  # (B, T)
    loss_map = -torch.minimum(ratio * adv, clipped * adv)  # (B, T)  same pessimistic min

    metrics = {
        "clip_frac": masked_mean((clipped * adv < ratio * adv).float(), mask),  # scalar
        "approx_kl": masked_mean(ratio.detach() - 1 - log_ratio.detach(), mask),  # scalar, k3 >= 0
        "ratio_max": ratio.detach().max(),  # scalar
    }

    # ---- TIS on the surrogate (see tis.py; change #2 means no KL term exists) ----
    if cfg.use_tis:
        loss_map, tis_metrics = apply_tis(
            loss_map, old, batch.behavior_logprobs, mask, cfg.tis_clip, cfg.tis_clip_low, cfg.tis_mode
        )  # (B, T), dict
        metrics |= tis_metrics

    return loss_map * mask, metrics  # (B, T), dict
