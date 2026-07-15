"""Group filters for dynamic sampling (DAPO / slime's dynamic_sampling_filter).

Home of the collection-time callables shared by collectors and recipes. The
round-based collector that originally lived beside these (rollout/sampling.py,
collect_groups) was retired 2026-07-14: collection is continuous-batching for
sync AND async training alike (docs/async_tier2.md §11), so the one collector
is controllers/fully_async.collect_groups_dp.
"""

import math
from typing import Callable

from minirl.rollout.types import Trajectory

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
