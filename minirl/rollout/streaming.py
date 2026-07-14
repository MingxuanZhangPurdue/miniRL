"""Streaming batch collection over a continuous-batching engine (tier 2).

The Olmo-3-shaped collector (docs/async_tier2.md §3): SAME decision rule as
rollout/sampling.collect_groups — zero-gradient filter + refill to a constant
batch — but pull-based off the engine's poll() instead of round-based. A
dropped group's replacement prompt starts prefilling on the very next engine
step, in the slots the dead group vacated; the engine never drains between
"rounds" because there are no rounds.

Leftovers persist: work in flight when collection returns stays IN the engine
(plus any drain surplus in its stash) and is consumed FIRST by the next call
— actors never stop. Consequence for staleness: a stashed group can be one
publish older than the rest of its batch; the stream controller's bound is
publish_interval + 1, not publish_interval (see controllers/streaming.py).

Engine contract (duck-typed, see VLLMEngine / tests' FakeStreamEngine):
  submit(prompt_ids (T,), params, meta) -> id     queue one group (n=G)
  poll() -> list[group]                           advance ~one step, finishers
  stash(group)                                    hand back surplus, FIFO next
  n_inflight: int                                 groups submitted, not done
"""

from typing import Callable

from minirl.config import CollectConfig
from minirl.rollout.sampling import GroupFilter, RewardFn, reward_nonzero_std
from minirl.rollout.types import SamplingParams, Trajectory


def collect_groups_stream(
    engine,
    reward_fn: RewardFn | None,  # None: engine/env already set rewards (agentic case)
    prompt_source: Callable[[int], list],  # returns <= n prompts; [] when exhausted
    cfg: CollectConfig,
    sampling: SamplingParams,
    group_filter: GroupFilter | None = None,
) -> tuple[list[Trajectory], dict]:
    """Collect cfg.target_groups surviving groups from the engine's stream.

    Mirrors collect_groups' contract exactly (meta rides with trajectories to
    the reward fn, group_ids stamped over survivors, may return SHORT if
    prompts run out or the budget hits — check stats["groups"]); only the
    execution shape differs. The budget reuses max_rounds' meaning: at most
    max_rounds * target_groups groups GENERATED per call, so a pathological
    drop rate cannot spin the engine forever.
    """
    assert sampling.n == cfg.group_size, (
        f"sampling.n={sampling.n} != cfg.group_size={cfg.group_size} — one request IS one group"
    )
    if group_filter is None and cfg.strategy == "filter":
        group_filter = reward_nonzero_std

    budget = cfg.max_rounds * cfg.target_groups  # generated-groups cap (== round cap in spirit)
    kept: list[list[Trajectory]] = []
    stats = {"groups_generated": 0, "groups_dropped": 0, "submitted": 0, "polls": 0}
    exhausted = False

    def top_up() -> None:
        """Keep (kept + in-flight) at the target; new prompts enter the engine
        immediately and prefill in whatever slots are free."""
        nonlocal exhausted
        need = cfg.target_groups - len(kept) - engine.n_inflight
        if need <= 0 or exhausted:
            return
        raw = prompt_source(need)
        if not raw:
            exhausted = True  # finite sources end collection; hf_prompt_source never does
            return
        for p in raw:
            ids, meta = p if isinstance(p, tuple) else (p, {})
            engine.submit(ids, sampling, meta)
            stats["submitted"] += 1

    top_up()
    while len(kept) < cfg.target_groups and stats["groups_generated"] < budget:
        groups = engine.poll()  # stash first, else ~one engine step
        stats["polls"] += 1
        if not groups:
            if engine.n_inflight == 0:
                top_up()  # a dropped group may have freed budget for new prompts
                if engine.n_inflight == 0:
                    break  # nothing running, nothing coming: return short
            continue
        for group in groups:
            stats["groups_generated"] += 1
            if reward_fn is not None:
                for t in group:
                    t.reward = reward_fn(t)  # meta already attached by the engine at submit
            if group_filter is not None and not group_filter(group):
                stats["groups_dropped"] += 1  # replacement enters on the next top_up
            elif len(kept) < cfg.target_groups:
                kept.append(group)
            else:
                engine.stash(group)  # surplus survivor: next collection's first pick
        top_up()

    for gid, group in enumerate(kept):
        for t in group:
            t.meta["group_id"] = gid

    stats["groups"] = len(kept)
    stats["leftover_inflight"] = engine.n_inflight  # carried into the next call, never wasted
    return [t for group in kept for t in group], stats
