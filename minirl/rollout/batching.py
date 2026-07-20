"""Trajectories -> padded Batch, and Batch -> mini/micro slices.

The collation layer between rollout land and training land.
RIGHT-padding here (training has no generation
constraint) vs the engine's LEFT-padding (prompts must touch the first
generated token) — the classic pair of conventions worth one comment each.
"""

from typing import Callable

import torch
from torch import Tensor

from minirl.algos.advantage import degenerate_group_mask, grpo_advantages
from minirl.rollout.types import Batch, Trajectory

# Per-ROW advantage estimator: (rewards (B,), group_ids (B,)) -> (B,) scalars,
# broadcast onto response tokens by make_batch. ≈ slime's advantage_estimator
# dispatch, as a plain callable (no string-path plugins).
#
# SCOPE, deliberately: this signature covers the critic-free SCALAR family
# (GRPO/Dr.GRPO, RLOO, global-batch baselines, rank-in-group, ...). PER-TOKEN
# estimators (GAE, REINFORCE++ returns, OPD KL-shaping) need tensors that only
# exist post-collation/post-forward — those overwrite batch.advantages AFTER
# make_batch instead; no signature could pull them earlier. Two tiers, both open.
AdvantageFn = Callable[[Tensor, Tensor], Tensor]


def make_batch(
    trajs: list[Trajectory],
    pad_id: int,
    advantage_fn: AdvantageFn | None = grpo_advantages,
) -> tuple[Batch, dict]:
    """Collate B trajectories into one right-padded Batch (+ advantages).

    advantage_fn picks the estimator (default: GRPO group-relative; pass
    functools.partial(grpo_advantages, norm_std=False) for Dr. GRPO).
    advantage_fn=None fills ZEROS — for consumers that don't use advantages
    (SFT) or that must fill them AFTER collation (PPO/GAE needs critic values
    over the collated batch, so its recipe overwrites batch.advantages).
    DPO never comes through here (paired collation, data/preference.py).

    Expects traj.meta["group_id"] (set by the collector, controllers/fully_async.collect_groups_dp);
    rows without one become singleton groups (advantage 0 under GRPO).

    Returns (batch, stats) where stats carries collation health metrics
    (frac_degenerate_groups is the key GRPO signal).
    """
    b = len(trajs)
    # = this batch's T (padded prompt+completion length); NOT the engine's
    # T_max, which is the padded PROMPT length of a generation batch
    t_max = max(t.input_ids.numel() for t in trajs)

    input_ids = torch.full((b, t_max), pad_id, dtype=torch.long)  # (B, T)
    attention_mask = torch.zeros((b, t_max), dtype=torch.bool)  # (B, T)
    loss_mask = torch.zeros((b, t_max), dtype=torch.bool)  # (B, T)
    behavior_logprobs = torch.zeros((b, t_max))  # (B, T) f32
    rewards = torch.tensor([tr.reward for tr in trajs], dtype=torch.float32)  # (B,)
    group_ids = torch.tensor(
        [tr.meta.get("group_id", i) for i, tr in enumerate(trajs)], dtype=torch.long
    )  # (B,)

    for i, tr in enumerate(trajs):
        n = tr.input_ids.numel()
        input_ids[i, :n] = tr.input_ids
        attention_mask[i, :n] = True
        loss_mask[i, :n] = tr.loss_mask
        behavior_logprobs[i, :n] = tr.logprobs  # already 0 where loss_mask False

    # Per-row scalar advantage, broadcast onto response tokens only.
    adv_row = advantage_fn(rewards, group_ids) if advantage_fn is not None else torch.zeros_like(rewards)  # (B,)
    advantages = adv_row.unsqueeze(-1) * loss_mask  # (B, T)

    stats = {
        "batch_size": b,
        "max_len": t_max,
        "response_tokens": int(loss_mask.sum()),
        # fraction of the (B, T) rectangle that is pad — the metric that will
        # one day justify pack_batch (sequence packing)
        "frac_padding": 1.0 - attention_mask.float().mean().item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item() if b > 1 else 0.0,
        "frac_degenerate_groups": degenerate_group_mask(rewards, group_ids).float().mean().item(),
    }
    batch = Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        rewards=rewards,
        group_ids=group_ids,
    )
    return batch, stats


def slice_batch(batch: Batch, idx: Tensor) -> Batch:
    """Row-select every field (idx: (b,) int64), including the optional ones."""
    maybe = lambda x: x[idx] if x is not None else None
    return Batch(
        input_ids=batch.input_ids[idx],
        attention_mask=batch.attention_mask[idx],
        loss_mask=batch.loss_mask[idx],
        behavior_logprobs=batch.behavior_logprobs[idx],
        advantages=batch.advantages[idx],
        rewards=batch.rewards[idx],
        group_ids=batch.group_ids[idx],
        old_logprobs=maybe(batch.old_logprobs),
        ref_logprobs=maybe(batch.ref_logprobs),
    )


def iter_minibatches(batch: Batch, minibatch_size: int, generator: torch.Generator):
    """Shuffled minibatches for one ppo epoch (seeded -> reproducible)."""
    perm = torch.randperm(batch.input_ids.shape[0], generator=generator)  # (B,)
    for start in range(0, len(perm), minibatch_size):
        yield slice_batch(batch, perm[start : start + minibatch_size])


def iter_microbatches(batch: Batch, micro_batch_size: int):
    """Contiguous grad-accumulation slices of a minibatch (no shuffling)."""
    b = batch.input_ids.shape[0]
    for start in range(0, b, micro_batch_size):
        yield slice_batch(batch, torch.arange(start, min(start + micro_batch_size, b)))
