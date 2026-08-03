"""
================================================================================
 DPPO LOSS — Divergence Proximal Policy Optimization (Sea AI Lab, arXiv:2602.04879)
================================================================================
LOSS (notation: see grpo.py; ONE new symbol: mu = pi_engine's SAMPLED-token
probability — this loss anchors everything to the engine, never pi_old):

    r_t  = exp( log pi_theta(y_t | y_<t) - log pi_engine(y_t | y_<t) )
    D_t  = divergence( pi_engine(.|y_<t) || pi_theta(.|y_<t) ), collapsed to
           the Bernoulli {sampled token, everything else}:
             "binary_tv":  | pi_theta(y_t) - mu |                    mass moved
             "binary_kl":  mu*log(mu/pi) + (1-mu)*log((1-mu)/(1-pi)) Bernoulli KL
    M_t  = 0  if ( A_i > 0 and r_t > 1 and D_t > delta )     "moving away AND
              or ( A_i < 0 and r_t < 1 and D_t > delta )      the distribution
           1  otherwise                                       actually shifted"
    L_t  = -M_t * sg( min(r_t, C) ) * A_i * log pi_theta(y_t | y_<t)
    L    = per-TOKEN mean of L_t    (loss_agg="token_mean")

Companion to:
  - grpo.py   (STEP 0-1 unchanged; STEP 2 with norm_std=False — the paper's
    advantage is the group-mean baseline WITHOUT the ÷std)
  - cispo.py  (same sg-weight * REINFORCE gradient path; DPPO adds the gate)

THE ONE IDEA: **clip on the mass that moved, not on the ratio.** PPO/GRPO's
clip asks "did the sampled token's RATIO leave [1-eps, 1+eps]?" — a
single-sample estimate of distribution shift that explodes for rare tokens
(1e-4 -> 1e-2 is r=100, yet only 0.01 of mass moved) and sleeps through big
shifts on dominant ones (0.99 -> 0.80 is r=0.81, "inside the window", yet 0.19
of mass moved). DPPO gates each token on an actual divergence estimate D_t:
block only updates that (a) move AWAY from the anchor and (b) have already
shifted the distribution past delta. Same per-token granularity as GRPO —
only the clip's TRIGGER changes.

--------------------------------------------------------------------------------
 WHAT CHANGES vs GRPO  (read alongside grpo.py; granularity does NOT change)
--------------------------------------------------------------------------------
  GRPO                                      DPPO
  ----                                      ----
  trigger: ratio window |r_t - 1| > eps     trigger: divergence D_t > delta
  two-sided (any out-of-window r clips)     one-sided: only AWAY-moving updates
                                            can be blocked; toward-anchor updates
                                            NEVER are (r<1 with A>0 always flows)
  ratio carries gradient; clip -> grad 0    ratio DETACHED + capped at C; grad
                                            through log pi (cispo.py's path);
                                            gate -> grad 0, cap -> grad KEPT
  anchor pi_old (fit_batch recompute)       anchor pi_engine ALWAYS —
                                            batch.old_logprobs is IGNORED (the
                                            paper's decoupled-anchor ablation
                                            collapses: a trust region measured
                                            from the recompute cannot see the
                                            engine<->trainer gap growing)
  TIS bridges the engine<->trainer gap      no TIS: r_t is already pi_theta /
                                            pi_engine — one ratio spans both the
                                            staleness AND numerics gaps; the
                                            paper also shows truncation biases
                                            exactly the low-prob tokens the
                                            divergence gate exists to free
  optional KL(pi||pi_ref) penalty           none (paper uses no ref model)

  WHY the sg(min(r,C)) weight instead of Eq 12's raw -M*r*A: identical gradient
  where r <= C (d[-r*A]/dtheta == -sg(r)*A*dlogpi), but the cap C bounds the
  weight of rare-token ratio spikes (r=100 on a 1e-4 token) WITHOUT killing
  their gradient — a plain min(r, C) through the ratio would zero it.

  IDENTICAL: STEP 0 rollout, STEP 1 bookkeeping, per-token loss map contract.

GROUNDED IN STABLE-RL (sail-sg/Stable-RL, the official verl fork):
    core_algos.py::compute_policy_loss_with_mask:
        ratio = exp(clamp(log_prob - rollout_log_prob, max=20))
        ratio = clamp(ratio, max=clip_ratio_c).detach()
        pg_losses = -advantages * ratio * mask * log_prob
    _compute_token_mask_for_policy_loss "dppo_binary_tv"/"dppo_binary_kl":
        blocked = moved-past-delta in the direction the advantage pushes
        (their binary_tv fuses both tests as a signed check
        (prob - rollout_prob) > delta; equal to ours since D > delta > 0
        already implies the direction), mask detached.
    Hyperparameters (paper App. C.1/E): delta = 0.2 binary-TV / 0.05 binary-KL,
    C = 5 in the scaling runs (their code default is 10), advantage =
    group mean baseline without ÷std, token-mean aggregation.
================================================================================
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class DPPOConfig:
    delta: float = 0.2  # divergence threshold — TV scale; binary_kl wants ~0.05 (LOSSES "dppo_kl")
    divergence: str = "binary_tv"  # "binary_tv" | "binary_kl" (the Bernoulli collapse of TV / KL)
    ratio_cap: float = 5.0  # C — cap on the DETACHED IS weight; bounds variance, never grad
    grpo_std_normalization: bool = False  # paper's advantage has NO ÷std; consumed by advantage.py
    loss_agg: str | int = "token_mean"
    # Algorithm FACT, not a knob (ClassVar): this loss never reads
    # batch.old_logprobs — the anchor is the engine — so the trainer skips
    # the pi_old recompute pass in fit_batch.
    needs_old_logprobs: ClassVar[bool] = False


def dppo_loss(policy_logprobs: Tensor, batch: Batch, cfg: DPPOConfig) -> tuple[Tensor, dict]:
    """STEP 3 — the loss. Diff from grpo_loss: divergence gate, detached capped
    weight (REINFORCE grad path), and the engine — not pi_old — as anchor.

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t).
            Like cispo_loss the gradient path is DIRECTLY through this tensor;
            the ratio only ever acts as a detached weight.
        batch: loss_mask (B, T) bool; advantages (B, T) f32 FROZEN;
            behavior_logprobs (B, T) FROZEN — the anchor mu. old_logprobs is
            deliberately never read: the trust region must be measured from
            the distribution that SAMPLED the data.

    Returns:
        loss_map (B, T) f32 unreduced (zero outside loss_mask), metrics dict.
    """
    mask = batch.loss_mask  # (B, T) bool
    adv = batch.advantages  # (B, T) f32 — FROZEN, constant per row
    mu_lp = batch.behavior_logprobs  # (B, T) — the anchor: what the ENGINE sampled from

    # ---- importance ratio vs the ENGINE — detached from the start ----
    # Masking before exp keeps padding at ratio 1; the clamp keeps a pathological
    # logprob gap from overflowing exp (f32 dies near e^88) — both branches are
    # weight-only, no gradient at stake.
    log_ratio = ((policy_logprobs - mu_lp) * mask).detach().clamp(max=20.0)  # (B, T)
    ratio = log_ratio.exp()  # (B, T) no grad

    # ---- binary divergence D_t: how much mass moved AT THE SAMPLED TOKEN ----
    # Collapse the |V|-way distributions to Bernoulli {y_t, everything else}:
    # a lower bound of the true TV/KL that needs only the two scalars already
    # in the batch (full-vocab or top-K divergence would need engine + trainer
    # to ship extra per-position tensors).
    pi_p = policy_logprobs.detach().exp()  # (B, T)  pi_theta(y_t), no grad
    mu_p = mu_lp.exp()  # (B, T)  pi_engine(y_t)
    if cfg.divergence == "binary_tv":
        div = (pi_p - mu_p).abs()  # (B, T)
    elif cfg.divergence == "binary_kl":
        # mu*log(mu/pi) + (1-mu)*log((1-mu)/(1-pi)); the 1e-8 guards log(0)
        # when either policy puts (numerically) ALL mass on y_t.
        div = mu_p * (mu_lp - policy_logprobs.detach()) + (1.0 - mu_p) * (
            (1.0 - mu_p + 1e-8) / (1.0 - pi_p + 1e-8)
        ).log()  # (B, T)
    else:
        raise ValueError(f"unknown divergence {cfg.divergence!r}")

    # ---- the gate M_t: block only away-moving updates past the threshold ----
    # Toward-anchor updates (e.g. A>0 with r<1) always flow, whatever D_t is —
    # PPO's asymmetry, kept; only the trigger changed from |r-1| to D_t.
    away = torch.where(adv > 0, ratio > 1.0, ratio < 1.0)  # (B, T) bool
    gate = 1.0 - (away & (div > cfg.delta)).float()  # (B, T)  M_t, no grad

    # ---- -M * sg(min(r, C)) * A * log pi:  gated IS-weighted REINFORCE ----
    # d/dtheta = -M * min(r, C) * A * dlogpi/dtheta: zero ONLY where the gate
    # closed; a capped ratio still pushes full gradient at weight C.
    weight = ratio.clamp(max=cfg.ratio_cap)  # (B, T) no grad
    loss_map = -gate * weight * adv * policy_logprobs  # (B, T) grad via policy_logprobs

    metrics = {
        # fraction of completion tokens the gate blocked (grad zeroed) —
        # DPPO's analog of grpo.py's clip_frac
        "clip_frac": masked_mean(1.0 - gate, mask),  # scalar
        # weight-cap engagements (variance control, gradient KEPT — cispo-style)
        "cap_frac": masked_mean((ratio > cfg.ratio_cap).float(), mask),  # scalar
        # k3 drift vs the ENGINE (staleness + numerics gaps together)
        "approx_kl": masked_mean(ratio - 1 - log_ratio, mask),  # scalar
        "ratio_max": ratio.max(),  # scalar
        # where the minibatch sits relative to delta — THE knob-tuning signal
        "div_mean": masked_mean(div, mask),  # scalar
        "div_max": (div * mask).max(),  # scalar
    }

    return loss_map * mask, metrics  # (B, T), dict
