"""
================================================================================
 STEP 2 — GROUP-RELATIVE ADVANTAGE  (this REPLACES PPO's value net + GAE)
================================================================================
Companion to: rl_notes grpo_loss_explained.py STEP 2 (group_relative_advantages).

No bootstrapping, no V(s), no GAE: normalize each completion's reward WITHIN
its group of G siblings —

    A_i = (R_i - mean_group) / (std_group + eps)

The mean reward of the OTHER samples from the SAME prompt is the baseline
(cf. RLOO / PPO's learned critic). LLM RL is bandit-like — one terminal scalar
per completion — so every TOKEN of a completion inherits the SAME advantage.

  SHAPE CONTRAST with PPO (rl_notes ppo_loss_explained.py):
    PPO  advantages: (N, T), DIFFER per token (GAE credit assignment)
    GRPO advantages: (B,) here, one per completion — the controller broadcasts
                     onto response tokens: adv[:, None] * loss_mask -> (B, T)

Grouping here uses group_ids (B,) from Trajectory.meta instead of rl_notes'
`rewards.view(P, G)` — same math, but it survives dynamic-sampling filters
dropping groups (sampling.py) where a fixed P*G reshape would not.
================================================================================
"""

import torch
from torch import Tensor


def _group_stats(rewards: Tensor, group_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-row group mean/std/count. rewards (B,), group_ids (B,) -> three (B,).

    Shape note: P = number of GROUPS (= B/G when no group was filtered), NOT
    the group size G of the notation legend. inv (B,) maps each row to its
    group's slot in the (P,) stats tensors.

    The whole trick: SCATTER-reduce down (index_add_: B rows -> P group slots),
    then GATHER back up (stats[inv]: integer indexing, out[b] = stats[inv[b]],
    fanning P values out to B rows). stats[inv] is NOT broadcasting — the
    row->group assignment is data-dependent, which shape rules can't express;
    only an index tensor can.
    """
    uniq, inv = torch.unique(group_ids, return_inverse=True)  # (P,), (B,)
    n_groups = uniq.numel()  # P
    ones = torch.ones_like(rewards)
    count = torch.zeros(n_groups).index_add_(0, inv, ones)  # (P,)  siblings per group.
    #   == G today, but DERIVED from the data, not assumed: make_batch's singleton
    #   fallback (rows without group_id) has count=1, and partial rollouts /
    #   agentic fan-out / per-sample rejection all produce ragged groups later.
    mean = torch.zeros(n_groups).index_add_(0, inv, rewards) / count  # (P,)  the BASELINE
    sq = torch.zeros(n_groups).index_add_(0, inv, (rewards - mean[inv]) ** 2)  # (P,)
    std = (sq / (count - 1).clamp(min=1)).sqrt()  # (P,)  unbiased; 0 if count==1
    return mean[inv], std[inv], count[inv]  # each (B,) — scattered back to rows


def grpo_advantages(
    rewards: Tensor, group_ids: Tensor, norm_std: bool = True, eps: float = 1e-6
) -> Tensor:
    """A_i = (R_i - mean_group) / (std_group + eps).

    norm_std=False drops the std division — **Dr. GRPO** (arXiv:2503.20783):
    removes the "low-variance (too easy / too hard) prompts get amplified"
    bias, at the cost of unnormalized advantage scale.

    Args:    rewards (B,) f32, group_ids (B,) int64  (both FROZEN — no grad anywhere here).
    Returns: advantages (B,) f32. Degenerate groups (all rewards equal) get
             exactly 0 — no gradient, which is what dynamic sampling filters
             away at collection time instead (see degenerate_group_mask).
    """
    mean, std, _ = _group_stats(rewards, group_ids)
    centered = rewards - mean  # (B,)  reward minus the sibling baseline
    return centered / (std + eps) if norm_std else centered


def degenerate_group_mask(rewards: Tensor, group_ids: Tensor, atol: float = 1e-6) -> Tensor:
    """True for rows whose group has ~zero reward std => zero GRPO gradient.

    Same criterion as slime's `check_reward_nonzero_std` dynamic-sampling
    filter and DAPO's dynamic sampling (change #4 in dapo.py). Used to filter
    at collection time (rollout/sampling.py, the preferred fix) or to report
    `frac_degenerate_groups` — a key GRPO health metric (docs/sync_training.md §4).
    """
    _, std, _ = _group_stats(rewards, group_ids)
    return std <= atol  # (B,)
