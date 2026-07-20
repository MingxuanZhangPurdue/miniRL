"""
================================================================================
 GRPO LOSS — Group Relative Policy Optimization (Shao et al. 2024, DeepSeekMath)
================================================================================
LOSS (what this file computes, per completion token t of completion i):

    r_t  = exp( log pi_theta(y_t | y_<t) - log pi_old(y_t | y_<t) )   token IS ratio
    L_t  = -min( r_t * A_i ,  clip(r_t, 1-eps, 1+eps_hi) * A_i )      clipped surrogate
           [ * w_t ]                                                  TIS weight, if use_tis
           [ + beta * (e^d - d - 1) ],  d = log pi_ref - log pi_theta k3 KL, if use_kl_loss
    L    = per-SEQUENCE mean of L_t     (loss_agg="seq_mean"; the reduce happens
                                         ONCE, in the trainer -> aggregate.py)

NOTATION (the legend for ALL algos/ loss files; siblings list only their deltas):

    pi_theta  current policy — the ONLY grad path    y_t    sampled token at position t
    pi_old    policy at update start      (FROZEN)   A_i    group advantage, ONE scalar per
    pi_ref    frozen reference model      (FROZEN)          completion, broadcast to its
    pi_engine rollout engine's policy     (FROZEN)          tokens (advantage.py, STEP 2):
    w_t       TIS weight = clamp(exp(logpi_old              A_i = (R_i - mean_group)
              - logpi_engine), lo, hi)  (tis.py)                  / (std_group + eps_std)
    sg(.)     stop-gradient == .detach()
    B = P*G total completions;  P = prompts (= groups);  G = group size;
    T = padded prompt+completion length;  |y_i| = completion i's token count;
    lowercase b = rows of ONE microbatch slice (aggregate.py / trainer.step)
    eps / eps_hi = lower / upper clip deltas from 1  (ratio window [1-eps, 1+eps_hi];
    config fields eps_clip / eps_clip_high).  Full glossary: root README.md § Notation.

Companion to:
  - notes/ppo_to_grpo_derivation.md   (THE derivation: baseline unbiasedness,
    the (1-1/G) RLOO relation, Dr. GRPO's two bias critiques, clip-higher)
  - rl_notes: grpo_loss_explained.py  (the annotated single-file pipeline this
    file productionizes — same clipped surrogate, same k3 KL, same group
    advantage; read that for the full STEP 0-3 walkthrough + PPO contrast)
  - gspo.py / cispo.py                (each = THIS file with one change; see
    their "WHAT CHANGES vs GRPO" tables)
  - DAPO / Dr. GRPO                   (NAMED CONFIGS of this file — loss bodies
    are identical; see algos/README.md and the LOSSES registry)

THE ONE IDEA: **PPO without a value function.** Sample a GROUP of G completions
per prompt and use the group's reward statistics as the baseline (A_i above).

WHY the clip: with minibatch updates (and stale rollouts in async), pi_theta has
already moved away from pi_old; the clip stops the update once the ratio leaves
the trust region — a clipped token contributes ZERO gradient (the property
CISPO later removes; see cispo.py).

--------------------------------------------------------------------------------
 WHERE THE OTHER PIPELINE STEPS LIVE  (rl_notes STEP numbering)
--------------------------------------------------------------------------------
  STEP 0  rollout: G completions/prompt    engines + controllers/fully_async.py
  STEP 1  bookkeeping: old/ref logprobs    trainer.compute_logprobs (fp32, frozen)
  STEP 2  group-relative advantage         algos/advantage.py (grpo_advantages)
  STEP 3  the loss (THIS FILE)             called per microbatch by trainer.step
  reduce  seq_mean / token_mean / const    algos/aggregate.py — applied ONCE by
                                           the trainer (cfg.loss_agg)

--------------------------------------------------------------------------------
 Dimension legend  (vs rl_notes files)
--------------------------------------------------------------------------------
  B = N = P*G completions (N is rl_notes' name for B);  T = padded length
  (prompt + completion; rl_notes calls it L);
  loss_mask (B, T) == rl_notes' action_mask, with ONE alignment difference:
  rl_notes tensors live on the (L-1) PREDICTION axis; here everything is
  position-aligned — gather_logprobs left-pads by one so logprobs[:, t] scores
  token t itself (the Trajectory contract, types.py). Advantages arrive already
  broadcast (B, T) instead of (N, 1), constant across each row's tokens.

--------------------------------------------------------------------------------
 GROUNDED IN SLIME  (verified against source, 2026-07)
--------------------------------------------------------------------------------
  - surrogate  == ppo_utils.compute_policy_loss: slime forms
        pg_losses1 = -ratio * adv; pg_losses2 = -clip(ratio) * adv;
        loss = maximum(pg_losses1, pg_losses2)
    which equals our -minimum(ratio*adv, clip(ratio)*adv)  (same -min identity
    as rl_notes' "why torch.max" note); their clipfrac = (pg_losses2 >
    pg_losses1) equals our (clip*adv < ratio*adv).
  - KL penalty == the use_kl_loss/kl_loss_coef block of
    megatron_utils/loss.py::policy_loss_function (k3 estimator).
  - ordering   == slime: TIS multiplies the pg term FIRST, the KL penalty is
    added AFTER (KL is computed on-policy vs the ref — it needs no off-policy
    correction), so TIS must never rescale it.
  - omitted from slime, on purpose: dual-clip (eps_clip_c), OPSM masking,
    entropy bonus (needs full logits; a trainer concern if ever added).

Config field names follow slime's CLI flags 1:1.
================================================================================
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from minirl.algos.aggregate import masked_mean
from minirl.algos.tis import apply_tis
from minirl.rollout.types import Batch


@dataclass(frozen=True)
class GRPOConfig:
    eps_clip: float = 0.2  # lower clip delta: ratio floor = 1 - eps_clip
    eps_clip_high: float | None = None  # upper delta; None -> symmetric (slime --eps-clip-high)
    use_kl_loss: bool = False  # beta > 0 in the GRPO paper; modern RLVR recipes often drop it
    kl_loss_coef: float = 0.0  # slime --kl-loss-coef  (the paper's beta)
    grpo_std_normalization: bool = True  # False = Dr. GRPO; consumed by advantage.py (STEP 2)
    loss_agg: str | int = "seq_mean"  # THE reduce (aggregate.py): "seq_mean" (GRPO paper),
    #   "token_mean" (DAPO), or an int constant (Dr. GRPO exact — pass max_new_tokens).
    #   One field = the whole GRPO/DAPO/Dr.GRPO normalization axis; trainer applies blindly.
    use_tis: bool = False  # truncated importance sampling (see tis.py)
    tis_clip: float = 2.0  # slime --tis-clip
    tis_clip_low: float = 0.0  # slime --tis-clip-low
    tis_mode: str = "clamp"  # "clamp" (vanilla TIS) | "mask" (icepop)


def grpo_loss(policy_logprobs: Tensor, batch: Batch, cfg: GRPOConfig) -> tuple[Tensor, dict]:
    """STEP 3 — the loss. Per-token loss map, NOT reduced (trainer reduces once).

    Args:
        policy_logprobs: (B, T) f32, WITH GRAD — log pi_theta(token_t | <t) from
            the current policy's forward pass (the ONLY grad-carrying input;
            rl_notes: `new_logprobs`).
        batch: everything else, all FROZEN (detached, computed at rollout /
            update start):
            loss_mask         (B, T) bool  — True on completion tokens only
            advantages        (B, T) f32   — group-relative, constant per row
            behavior_logprobs (B, T) f32   — engine-reported at sampling time
            old_logprobs      (B, T) f32?  — trainer recompute of pi_old (IS denominator)
            ref_logprobs      (B, T) f32?  — frozen reference (required if use_kl_loss)

    Returns:
        loss_map: (B, T) f32 per-token loss, zero outside loss_mask, UNREDUCED —
            the trainer aggregates once, globally (aggregate.py), so this
            function must never mean/sum on its own.
        metrics: dict of detached scalar tensors.
    """
    mask = batch.loss_mask  # (B, T) bool — True only on policy-produced tokens
    adv = batch.advantages  # (B, T) f32 — FROZEN, same value across each row (outcome reward)
    # pi_old = trainer's recompute at update start if available, else the
    # engine's sampling logprobs (sync fresh-weights case: identical policies).
    old = batch.old_logprobs if batch.old_logprobs is not None else batch.behavior_logprobs  # (B, T)

    # ---- importance ratio  r_t = pi_theta / pi_old,  in log space first ----
    # Masking BEFORE exp keeps padding/prompt positions at ratio exp(0)=1, so
    # they stay finite and are killed by the final mask multiply.
    log_ratio = (policy_logprobs - old) * mask  # (B, T)
    ratio = log_ratio.exp()  # (B, T)  grad flows through policy_logprobs

    # ---- POLICY LOSS — the clipped surrogate (identical to PPO's) ----
    eps_high = cfg.eps_clip_high if cfg.eps_clip_high is not None else cfg.eps_clip
    clipped = ratio.clamp(1.0 - cfg.eps_clip, 1.0 + eps_high)  # (B, T)
    # -min(r*A, clip(r)*A): the PESSIMISTIC branch of the two candidates.
    #   A > 0: caps the payoff of pushing r above 1+eps_high;
    #   A < 0: the UNclipped branch is smaller, so lowering r keeps hurting
    #          until 1-eps_clip binds.
    # Where the clipped branch is selected, d(loss)/d(theta) = 0 — the trust
    # region. (Same -min form as rl_notes grpo_loss; slime writes it as
    # maximum(-rA, -clip(r)A) — identical.)
    loss_map = -torch.minimum(ratio * adv, clipped * adv)  # (B, T)

    metrics = {
        # fraction of completion tokens whose gradient the clip zeroed
        "clip_frac": masked_mean((clipped * adv < ratio * adv).float(), mask),  # scalar
        # k3 estimate of KL(pi_old || pi_theta) = E[r - 1 - log r] >= 0:
        # the "how far did this minibatch drift" health metric (notes/kl_estimators.md)
        "approx_kl": masked_mean(ratio.detach() - 1 - log_ratio.detach(), mask),  # scalar
        "ratio_max": ratio.detach().max(),  # scalar — spikes signal mismatch/staleness bugs
    }

    # ---- TIS — engine<->trainer mismatch correction, on the pg term ONLY ----
    # Must run BEFORE the KL penalty is added (slime ordering): the surrogate is
    # off-policy w.r.t. the engine's numerics, but the KL term below is computed
    # purely on-policy and must not be rescaled. (Two-gap picture: tis.py.)
    if cfg.use_tis:
        loss_map, tis_metrics = apply_tis(
            loss_map, old, batch.behavior_logprobs, mask, cfg.tis_clip, cfg.tis_clip_low, cfg.tis_mode
        )  # (B, T), dict
        metrics |= tis_metrics

    # ---- KL PENALTY — beta * k3(pi || pi_ref), added DIRECTLY to the loss ----
    # Added PER-TOKEN into the map, so it inherits cfg.loss_agg automatically:
    # the reduce is linear, reduce(pg + b*kl) == reduce(pg) + b*reduce(kl) for
    # the same mode (slime reduces kl separately with the same reducer — same
    # number, other algebraic form). Mixed modes (e.g. token-mean pg + seq-mean
    # kl) would need the loss to return named maps for per-term reduces — no
    # paper wants that; the seam exists but stays closed (principle 8).
    # (GRPO's placement; PPO instead folds KL into the reward before GAE — see
    # rl_notes ppo_loss_explained.py. Sign convention: d = log(pi_ref/pi_theta),
    # so k3 = e^d - d - 1 is the UNBIASED >= 0 estimator of KL(pi || pi_ref)
    # under samples from pi — proof: notes/kl_estimators.md (cf. approx_kl3).)
    if cfg.use_kl_loss:
        assert batch.ref_logprobs is not None, "use_kl_loss requires batch.ref_logprobs"
        # d clamped for numerical safety on rare tokens (e^20 overflows the
        # loss long before the estimate is meaningful).
        d = (batch.ref_logprobs - policy_logprobs).clamp(-20.0, 20.0)  # (B, T) grad via policy
        kl = d.exp() - d - 1  # (B, T)  e^d - d - 1  >= 0
        loss_map = loss_map + cfg.kl_loss_coef * kl  # (B, T)
        metrics["kl_ref"] = masked_mean(kl.detach(), mask)  # scalar

    # Final mask: guarantee prompt/padding positions contribute exactly 0.
    return loss_map * mask, metrics  # (B, T), dict
