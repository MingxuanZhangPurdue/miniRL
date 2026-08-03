"""
================================================================================
 DPPO LOSS, SLIME CALLING CONVENTION — dppo.py's math behind slime's plug point
================================================================================
WHAT THIS FILE IS: the SAME loss as dppo.py (read that banner for the math:
divergence gate, capped detached weight, REINFORCE grad path), re-wrapped for

    --loss-type custom_loss \
    --custom-loss-function-path minirl.algos.dppo_slime.dppo_loss_slime

Wrapper mechanics — helper call, flat (S,) layout, injected reducer — are
grpo_slime.py's; read its SHAPES section first and diff dppo.py against
THIS file for the math. Same import rule: slime-env only, never imported
from minirl.algos.__init__.

WHY DPPO PORTS *CLEANER* THAN GRPO: the two pieces of slime plumbing GRPO
needs are exactly the pieces DPPO's design deletes —
  - no pi_old: the anchor is pi_engine (batch["rollout_log_probs"], which
    slime already ships for TIS); the log_probs recompute pass slime runs
    is simply unused here;
  - no TIS: r_t = pi_theta/pi_engine already spans the staleness AND
    numerics gaps in one ratio (dppo.py's WHAT-CHANGES table).
The gate itself needs only the SAMPLED token's probability under both
policies — two scalars per position, both already in the flat batch. No
group structure, no full-vocab tensors: nothing slime's contract lacks.

Also a portability proof worth noticing: dppo.py is grounded in Stable-RL
(a verl fork); this file runs that verl-native algorithm inside slime.
The algorithm layer crosses frameworks; only wrappers change.

--------------------------------------------------------------------------------
 KNOBS + REQUIRED UPSTREAM SETTINGS
--------------------------------------------------------------------------------
DPPO's three knobs are not slime CLI flags; they are read off `args` with
the paper's defaults (override by injecting attributes via slime's
--custom-config-path):

    dppo_delta       0.2   divergence threshold (binary_tv scale; ~0.05 for kl)
    dppo_divergence  "binary_tv" | "binary_kl"
    dppo_ratio_cap   5.0   C — cap on the DETACHED weight, never the grad

Upstream slime settings this loss assumes (dppo.py's paper config):
    --advantage-estimator grpo --disable-grpo-std-normalization
        (group-mean baseline WITHOUT the ÷std)
    --calculate-per-token-loss
        (token_mean aggregation — makes sum_of_sample_mean the token-mean
        reducer, dppo.py's loss_agg)
    rollout log probs must be present in the batch (the engine-side
    logprobs slime records at sampling time) — this loss asserts it.
================================================================================
"""

import torch

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy


def dppo_loss_slime(args, batch, logits, sum_of_sample_mean):
    """dppo.py's STEP 3 under slime's contract -> (loss scalar, metrics).

    Shapes: (S,) = flat cat of all samples' response tokens — the legend
    lives in grpo_slime.py's SHAPES section.
    """
    delta = getattr(args, "dppo_delta", 0.2)
    divergence = getattr(args, "dppo_divergence", "binary_tv")
    ratio_cap = getattr(args, "dppo_ratio_cap", 5.0)

    # ---- slime plumbing: logits -> log pi_theta of the sampled tokens ----
    _, out = get_log_probs_and_entropy(
        logits, args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )
    new = torch.cat(out["log_probs"], dim=0)  # (S,) f32, WITH GRAD — the ONLY grad path

    adv = torch.cat(batch["advantages"], dim=0)  # (S,) FROZEN — A_i repeated per sample
    # THE anchor: what the ENGINE sampled from. batch["log_probs"] (pi_old)
    # is deliberately never read — dppo.py's decoupled-anchor argument.
    assert batch.get("rollout_log_probs"), "dppo needs the engine's rollout_log_probs"
    mu_lp = torch.cat(batch["rollout_log_probs"], dim=0)  # (S,) FROZEN — log mu

    # ---- importance ratio vs the ENGINE — detached from the start ----
    # (dppo.py verbatim minus the mask trick — no padding in the flat layout;
    # the clamp keeps a pathological gap from overflowing exp, weight-only.)
    log_ratio = (new - mu_lp).detach().clamp(max=20.0)  # (S,) no grad
    ratio = log_ratio.exp()  # (S,) no grad

    # ---- binary divergence D_t: mass moved AT THE SAMPLED TOKEN ----
    pi_p = new.detach().exp()  # (S,)  pi_theta(y_t), no grad
    mu_p = mu_lp.exp()  # (S,)  mu = pi_engine(y_t)
    if divergence == "binary_tv":
        div = (pi_p - mu_p).abs()  # (S,)
    elif divergence == "binary_kl":
        div = mu_p * (mu_lp - new.detach()) + (1.0 - mu_p) * (
            (1.0 - mu_p + 1e-8) / (1.0 - pi_p + 1e-8)
        ).log()  # (S,)
    else:
        raise ValueError(f"unknown divergence {divergence!r}")

    # ---- the gate M_t: block only away-moving updates past the threshold ----
    away = torch.where(adv > 0, ratio > 1.0, ratio < 1.0)  # (S,) bool
    gate = 1.0 - (away & (div > delta)).float()  # (S,)  M_t, no grad

    # ---- -M * sg(min(r, C)) * A * log pi:  gated IS-weighted REINFORCE ----
    weight = ratio.clamp(max=ratio_cap)  # (S,) no grad
    loss_map = -gate * weight * adv * new  # (S,) grad via `new`

    # Metric maps are (S,); sum_of_sample_mean applies loss_masks +
    # denominators inside and returns scalars (grpo_slime.py convention).
    metrics = {
        "clip_frac": sum_of_sample_mean(1.0 - gate).detach(),
        "cap_frac": sum_of_sample_mean((ratio > ratio_cap).float()).detach(),
        "approx_kl": sum_of_sample_mean(ratio - 1 - log_ratio).detach(),
        "ratio_max": ratio.max(),  # scalar — max over ALL S response tokens
        "div_mean": sum_of_sample_mean(div).detach(),
        "div_max": div.max(),  # scalar
    }

    # ---- THE reduce — injected; --calculate-per-token-loss makes it the
    # token-mean dppo.py specifies.
    loss = sum_of_sample_mean(loss_map)  # scalar
    metrics["loss"] = loss.detach()
    return loss, metrics
