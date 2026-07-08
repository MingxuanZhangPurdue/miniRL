"""Tests for dynamic batch collection (rollout/sampling.py) with fake engines."""

import torch

from minirl.config import CollectConfig
from minirl.rollout.sampling import collect_groups, reward_nonzero_std
from minirl.rollout.types import Trajectory


def mk_traj(prompt_val: int, sample_idx: int) -> Trajectory:
    return Trajectory(
        input_ids=torch.tensor([prompt_val], dtype=torch.long),
        loss_mask=torch.ones(1, dtype=torch.bool),
        logprobs=torch.zeros(1),
        meta={"p": prompt_val, "i": sample_idx},
    )


def sequential_prompts():
    """prompt_source yielding tensors 0, 1, 2, ... n at a time."""
    counter = iter(range(10_000))
    return lambda n: [torch.tensor([next(counter)]) for _ in range(n)]


def fake_generate(group_size: int):
    return lambda prompts: [mk_traj(int(p), j) for p in prompts for j in range(group_size)]


def test_fixed_strategy_single_round():
    cfg = CollectConfig(group_size=2, target_groups=3, strategy="fixed")
    trajs, stats = collect_groups(fake_generate(2), lambda t: 1.0, sequential_prompts(), cfg)
    assert stats == {"rounds": 1, "groups_generated": 3, "groups_dropped": 0, "groups": 3}
    assert len(trajs) == 6
    assert [t.meta["group_id"] for t in trajs] == [0, 0, 1, 1, 2, 2]
    assert all(t.reward == 1.0 for t in trajs)  # constant rewards survive "fixed"


def test_filter_strategy_drops_degenerate_groups_and_refills():
    # even prompts -> constant reward (degenerate); odd prompts -> mixed 0/1
    def reward(t: Trajectory) -> float:
        return 1.0 if t.meta["p"] % 2 == 0 else float(t.meta["i"] % 2)

    cfg = CollectConfig(group_size=2, target_groups=3, strategy="filter")
    trajs, stats = collect_groups(fake_generate(2), reward, sequential_prompts(), cfg)
    # round 1: prompts 0,1,2 -> keep {1}; round 2: 3,4 -> keep {3}; round 3: 5 -> keep {5}
    assert stats == {"rounds": 3, "groups_generated": 6, "groups_dropped": 3, "groups": 3}
    assert sorted({t.meta["p"] for t in trajs}) == [1, 3, 5]
    assert len(trajs) == 6


def test_oversampling_discards_surplus():
    cfg = CollectConfig(group_size=2, target_groups=2, strategy="fixed", oversample_batch_size=5)
    trajs, stats = collect_groups(fake_generate(2), lambda t: 0.0, sequential_prompts(), cfg)
    assert stats["rounds"] == 1 and stats["groups_generated"] == 5 and stats["groups"] == 2
    assert len(trajs) == 4


def test_max_rounds_stops_hopeless_filtering():
    cfg = CollectConfig(group_size=2, target_groups=3, strategy="filter", max_rounds=4)
    trajs, stats = collect_groups(fake_generate(2), lambda t: 1.0, sequential_prompts(), cfg)
    assert stats["rounds"] == 4 and stats["groups"] == 0 and trajs == []


def test_prompt_exhaustion_returns_partial():
    exhausted = lambda n: []
    cfg = CollectConfig(group_size=2, target_groups=3, strategy="fixed")
    trajs, stats = collect_groups(fake_generate(2), lambda t: 1.0, exhausted, cfg)
    assert trajs == [] and stats["groups"] == 0 and stats["rounds"] == 0


def test_reward_nonzero_std_filter():
    same = [mk_traj(0, i) for i in range(4)]
    for t in same:
        t.reward = 1.0
    mixed = [mk_traj(0, i) for i in range(4)]
    for i, t in enumerate(mixed):
        t.reward = float(i % 2)
    assert not reward_nonzero_std(same)
    assert reward_nonzero_std(mixed)
