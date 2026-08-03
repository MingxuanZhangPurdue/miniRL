"""
================================================================================
 GRPO LOSS, SLIME CALLING CONVENTION — grpo.py's math behind slime's plug point
================================================================================
WHAT THIS FILE IS: the SAME loss as grpo.py (read that banner first — formula,
notation, derivation pointers all live there), re-wrapped to be loaded by
slime as a custom loss:

    --loss-type custom_loss \
    --custom-loss-function-path minirl.algos.grpo_slime.grpo_loss_slime

(PYTHONPATH must include the miniRL repo root. This module imports slime at
file level, so it only imports inside slime's environment — the same
platform-gated-import pattern as minirl/vllm_engine.py and megatron.py.
NEVER import it from minirl.algos.__init__.)

THE ONE IDEA: the wrapper changes, the math never does. Every line between
the "ratio" and "KL penalty" markers below is grpo.py's line under different
plumbing — diff the two files side by side; that diff IS the lesson about
what an RL framework owns vs what the algorithm owns.

--------------------------------------------------------------------------------
 WHAT CHANGES vs grpo.py  (wrapper only — the math body is identical)
--------------------------------------------------------------------------------
                    grpo.py (miniRL trainer)        THIS FILE (slime custom loss)
  caller            MegatronTrainer.step            slime loss_function dispatch
                                                    (backends/megatron_utils/loss.py)
  grad input        policy_logprobs (B, T) —        raw logits (1, T_pack, V); WE call
                    trainer's fused CE did the      slime's get_log_probs_and_entropy
                    logits->logprobs step           (chunked, fp32, temperature-scaled
                                                    INSIDE the helper)
  layout            padded (B, T) + loss_mask;      flat (S,) = cat of per-sample
                    mask-BEFORE-exp keeps padding   response slices; NO padding
                    ratios finite                   positions exist, so no mask trick
  reduction         return the UNREDUCED map;       sum_of_sample_mean closure, HERE —
                    trainer reduces ONCE            slime's injected form of
                    (aggregate.py, minibatch-       aggregate.py + minibatch_denom
                    global denominators)            (masks + denominators live inside
                                                    the closure, prebuilt per batch)
  config            GRPOConfig dataclass            args Namespace — SAME field names;
                                                    GRPOConfig documents the mirror
                                                    ("config fields follow slime's
                                                    CLI flags 1:1")
  pi_old            batch.old_logprobs, engine      batch["log_probs"] (trainer
                    logprobs as sync fallback       recompute), rollout_log_probs /
                                                    on-policy detach as fallbacks —
                                                    slime's own fallback ladder
  advantages        advantage.py (STEP 2), then     slime's advantage estimator
                    broadcast (B, T)                upstream; arrives per-sample (R,)
                                                    lists, cat -> (S,)

  S = sum of response lengths across the microbatch's samples (slime packs
  sequences; T_pack is the packed row length the logits cover).

--------------------------------------------------------------------------------
 GROUNDED IN SLIME  (backends/megatron_utils/loss.py, read 2026-08)
--------------------------------------------------------------------------------
  - contract: loss_function dispatches `func(args, batch, logits,
    sum_of_sample_mean) -> (loss scalar, metrics dict)` for
    --loss-type custom_loss; the wrapper rescales for Megatron grad accum.
  - batch keys used exactly as policy_loss_function uses them:
    advantages / log_probs / rollout_log_probs / ref_log_probs /
    unconcat_tokens / total_lengths / response_lengths / loss_masks.
  - their ppo_kl = old - new is OUR -log_ratio; their
    maximum(-rA, -clip(r)A) is OUR -minimum(rA, clip(r)A) (same identity
    noted in grpo.py's banner).
  - sum_of_sample_mean applies loss_masks and denominators internally
    (get_sum_of_sample_mean) — do NOT pre-mask the map before reducing.

PARITY PROTOCOL (run inside slime's env before trusting a training run):
freeze one batch; compute grpo.py's map through tests/fake_trainer and
reduce with the matching loss_agg; hand-build the slime batch dict from the
same tensors and call THIS function with a get_sum_of_sample_mean closure
built the same way; the two scalars must agree to fp32 kernel tolerance.
================================================================================
"""

import torch

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy


def grpo_loss_slime(args, batch, logits, sum_of_sample_mean):
    """grpo.py's STEP 3 under slime's contract -> (loss scalar, metrics).

    Args:
        args: slime CLI namespace — eps_clip / eps_clip_high / use_kl_loss /
            kl_loss_coef / use_tis / tis_clip / tis_clip_low /
            use_rollout_logprobs (the flags GRPOConfig mirrors).
        batch: slime RolloutBatch dict (keys above), per-sample lists.
        logits: (1, T_pack, V) — the packed forward's raw output, WITH GRAD.
        sum_of_sample_mean: slime's injected reducer (masks + denominators
            inside) — the aggregate.py role.
    """
    # ---- slime plumbing: logits -> log pi_theta of the sampled tokens ----
    # Chunked fp32 log-softmax + gather, temperature scaling included; the
    # ONLY grad path, exactly like policy_logprobs in grpo.py.
    _, out = get_log_probs_and_entropy(
        logits, args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
    )
    new = torch.cat(out["log_probs"], dim=0)  # (S,) f32, WITH GRAD

    adv = torch.cat(batch["advantages"], dim=0)  # (S,) FROZEN, constant per sample
    # pi_old — slime's fallback ladder: trainer recompute when present,
    # engine logprobs when told to, on-policy detach otherwise.
    if args.use_rollout_logprobs:
        old = torch.cat(batch["rollout_log_probs"], dim=0)  # (S,) FROZEN
    elif batch.get("log_probs"):
        old = torch.cat(batch["log_probs"], dim=0)  # (S,) FROZEN
    else:
        old = new.detach()  # sync fresh-weights case: ratio == 1 exactly

    # ---- importance ratio  r_t = pi_theta / pi_old ---- (grpo.py verbatim;
    # no mask-before-exp: the flat layout has no padding positions)
    log_ratio = new - old  # (S,)
    ratio = log_ratio.exp()  # (S,)  grad flows through `new`

    # ---- POLICY LOSS — the clipped surrogate (identical to PPO's) ----
    eps_high = args.eps_clip_high if args.eps_clip_high is not None else args.eps_clip
    clipped = ratio.clamp(1.0 - args.eps_clip, 1.0 + eps_high)  # (S,)
    loss_map = -torch.minimum(ratio * adv, clipped * adv)  # (S,) per-token, unreduced

    metrics = {
        "pg_clipfrac": sum_of_sample_mean((clipped * adv < ratio * adv).float()).detach(),
        "ppo_kl": sum_of_sample_mean(ratio.detach() - 1 - log_ratio.detach()).detach(),
        "ratio_max": ratio.detach().max(),  # over response tokens (flat layout)
    }

    # ---- TIS — engine<->trainer mismatch correction, pg term ONLY ----
    # w = sg(clamp(exp(logpi_old_train - logpi_engine), lo, hi)); before the
    # KL penalty, which is on-policy and must not be rescaled (tis.py).
    if args.use_tis:
        train_lp = (torch.cat(batch["log_probs"], dim=0)
                    if batch.get("log_probs") else new.detach())  # (S,) FROZEN
        rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)  # (S,) FROZEN
        tis = (train_lp - rollout_lp).exp().clamp(args.tis_clip_low, args.tis_clip)  # (S,)
        loss_map = loss_map * tis.detach()  # (S,)
        metrics["tis"] = sum_of_sample_mean(tis).detach()

    # ---- KL PENALTY — beta * k3(pi || pi_ref), added per-token ----
    # (grpo.py verbatim: d = log(pi_ref/pi_theta) clamped, k3 = e^d - d - 1)
    if args.use_kl_loss:
        ref = torch.cat(batch["ref_log_probs"], dim=0)  # (S,) FROZEN
        d = (ref - new).clamp(-20.0, 20.0)  # (S,) grad via `new`
        kl = d.exp() - d - 1  # (S,) >= 0
        loss_map = loss_map + args.kl_loss_coef * kl  # (S,)
        metrics["kl_loss"] = sum_of_sample_mean(kl).detach()

    # ---- THE reduce — aggregate.py's job, done here because slime injects
    # the reducer instead of receiving the map (masks applied inside).
    loss = sum_of_sample_mean(loss_map)  # scalar
    metrics["loss"] = loss.detach()
    return loss, metrics
