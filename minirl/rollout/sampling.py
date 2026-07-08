"""Batch collection for RL: fixed sampling or DAPO-style dynamic filtering.

Mini version of slime's over-sampling rollout loop (sglang_rollout.py): keep
requesting prompt groups, drop groups a filter rejects (default: zero reward
std == zero GRPO gradient), until `target_groups` SURVIVING groups exist.
slime does this asynchronously per-finished-group and aborts stragglers; the
sync mini version collects round by round — same semantics, simpler code.
"""

import math
from typing import Callable

from torch import Tensor

from minirl.config import CollectConfig
from minirl.rollout.types import Trajectory

# generate_fn(prompts) -> G trajectories per prompt, grouped (engine.generate).
GenerateFn = Callable[[list[Tensor]], list[Trajectory]]
RewardFn = Callable[[Trajectory], float]
# group_filter(group) -> keep? (≈ slime's dynamic_sampling_filter)
GroupFilter = Callable[[list[Trajectory]], bool]


def reward_nonzero_std(group: list[Trajectory], atol: float = 1e-6) -> bool:
    """Default filter — slime's check_reward_nonzero_std / DAPO dynamic sampling:
    a group where every sample got the same reward has zero advantage everywhere
    and contributes nothing but noise-free zeros to the gradient."""
    rewards = [t.reward for t in group]
    mean = sum(rewards) / len(rewards)
    return math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)) > atol


def collect_groups(
    generate_fn: GenerateFn,
    reward_fn: RewardFn | None,  # None: generate_fn already set rewards (agentic envs)
    prompt_source: Callable[[int], list],  # returns <= n prompts; [] when exhausted
    cfg: CollectConfig,
    group_filter: GroupFilter | None = None,
) -> tuple[list[Trajectory], dict]:
    """Collect cfg.target_groups groups of cfg.group_size trajectories.

    prompt_source may yield plain (T,) tensors, or (tensor, meta) pairs — the
    meta dict (e.g. {"answer": "42"} for RLVR labels) is merged into every
    trajectory of that prompt's group BEFORE reward_fn runs, so verifiers can
    read traj.meta. This works with no engine changes because generate_fn
    returns groups in prompt order (asserted below). ≈ slime's Sample.label.

    reward_fn=None means the generate function produced rewards itself — the
    agentic case, where the environment scores the episode (slime: custom
    rollout functions setting sample.reward).

    Rewards/meta are assigned in place; meta["group_id"] numbers surviving
    groups. Returns (flat trajectories, stats). May return fewer groups than
    requested if prompts run out or max_rounds hits — check stats["groups"].
    """
    if group_filter is None and cfg.strategy == "filter":
        group_filter = reward_nonzero_std

    kept: list[list[Trajectory]] = []
    stats = {"rounds": 0, "groups_generated": 0, "groups_dropped": 0}

    while len(kept) < cfg.target_groups and stats["rounds"] < cfg.max_rounds:
        need = cfg.target_groups - len(kept)
        raw = prompt_source(cfg.oversample_batch_size or need)
        if not raw:
            break  # data exhausted
        stats["rounds"] += 1
        prompts = [p[0] if isinstance(p, tuple) else p for p in raw]  # (T_i,) tensors
        metas = [p[1] if isinstance(p, tuple) else {} for p in raw]  # per-prompt labels etc.

        trajs = generate_fn(prompts)
        assert len(trajs) == len(prompts) * cfg.group_size, (
            f"generate_fn returned {len(trajs)} trajectories for "
            f"{len(prompts)} prompts with group_size {cfg.group_size}"
        )

        for i in range(len(prompts)):
            group = trajs[i * cfg.group_size : (i + 1) * cfg.group_size]
            for t in group:
                t.meta |= metas[i]
                if reward_fn is not None:
                    t.reward = reward_fn(t)
            stats["groups_generated"] += 1
            if cfg.strategy == "filter" and not group_filter(group):
                stats["groups_dropped"] += 1
            elif len(kept) < cfg.target_groups:  # surplus from oversampling is dropped
                kept.append(group)

    for gid, group in enumerate(kept):
        for t in group:
            t.meta["group_id"] = gid

    stats["groups"] = len(kept)
    return [t for group in kept for t in group], stats
