"""THE fully-async training loop: k DP engines x 1..m trainer ranks
(docs/async_tier2.md §10 collection, §11 consolidation + placement).

One controller serves every async configuration on a single node:

    k=1 StreamAdapter(HFEngine), world=1   the Mac/MPS dev path (one poll ==
                                           one round: the retired round_based
                                           controller as a degenerate case)
    k=1 VLLMEngine, world=1                tier-2 continuous batching
    k>1 VLLMEngines, world>=1              DP rollout fleet (dealer + tally)
    world>1 (DistTrainer via torchrun)     rank 0 collects + broadcasts;
                                           followers train + join publishes

COLLECTION — deal, don't partition. No engine is "assigned batch/k prompts".
A shared dealer (the caller's stateful prompt_source, called only under the
tally's lock — its cursor already hands out each prompt exactly once, so
disjoint prompts are free) and a shared tally (kept groups + per-engine
in-flight vs ONE global target) replace per-engine targets. Consequences,
all automatic: no duplicate prompts; no idle engine while claimable work
remains (pull, not push); load balance is emergent — a fast engine turns
over groups sooner, finds need > 0 again sooner, deals itself more.

The burst cap (refinement over the §10 sketch): each deal is bounded by
ceil(target/k) minus the engine's own in-flight count. Uncapped, whichever
collector thread ran first would deal itself the ENTIRE target at t=0 —
dealt work cannot migrate. Commit at most a fair burst; leave the rest on
the table for whoever is ready next.

THREADING (same rules as the retired streaming.py, times k): each engine has
exactly ONE owner thread for the whole collection — engines are long-lived
mutable machines, never shared, never locked; the tally's lock guards
microseconds of pure-Python bookkeeping and NEVER wraps an engine call; the
controller keeps the one-slot pipeline — the main thread trains batch k
while one coordinator worker fans out k collectors for batch k+1:

    main thread   [ train on batch k          ][join][drain*k·publish*k][ ...
    coordinator   [ fan out k collectors, join them, merge      ][idle]
    k collectors  [ poll own engine / deal from shared tally    ]
    k engines     [ continuous batching, weights v_{k-1}        ][fin.]

PUBLISH — drain-then-publish, fleet edition: join the in-flight collection,
drain ALL k engines (leftovers finish under the OLD weights, into each
stash — work becomes the next collection's head start, never garbage), then
load the same state dict into each. Every completion is generated under
exactly ONE weight version; Trajectory.version stays a scalar. STALENESS
BOUND: publish_interval + 1 — the +1 is a drained leftover consumed by the
collection after the publish, one version older than its batchmates. TIS in
the loss is what makes this bounded off-policyness mathematically fine.

MULTI-RANK (docs/fsdp2.md §8, folded in here): torchrun runs the same recipe
on every rank with identical configs. Rank 0 owns the engines and the
collection thread; after collation it broadcasts the Batch (CPU tensors) to
all ranks; every rank calls fit_batch on the identical full batch
(DistTrainer slices rows internally — the cross-rank denominator is free,
fsdp2.md §2); at publish iterations EVERY rank enters the full_state_dict
gather (a collective), rank 0 keeps the result and loads the engines.
Ranks > 0 run the small follower loop at the bottom of this file.

Async pays off with >= 2 devices (engines overlap training). On one device
(the Mac) the loop is still correct — generation and training timeshare.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

import torch.distributed as dist
from torch import Tensor

from minirl.algos.advantage import grpo_advantages
from minirl.config import CollectConfig
from minirl.rollout.batching import make_batch
from minirl.rollout.filtering import GroupFilter, RewardFn, reward_nonzero_std
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.train import Trainer


class _Tally:
    """Shared dealer + scoreboard for k concurrent collectors.

    ONE lock, held for microseconds of bookkeeping at a time. Correct here
    and wrong for the engine — the distinction that matters (§10): these
    critical sections are pure-Python arithmetic a lock serializes harmlessly;
    an engine is a long-lived mutable machine owned by protocol, not locks.
    """

    def __init__(self, prompt_source: Callable[[int], list], target: int, budget: int, burst: int):
        self._lock = threading.Lock()
        self._source = prompt_source  # called ONLY under the lock (the dealer)
        self._target = target
        self._budget = budget  # global generated-groups cap (max_rounds * target)
        self._burst = burst  # per-deal cap: ceil(target / k) minus own in-flight
        self.kept: list[list[Trajectory]] = []  # groups kept, all engines, arrival order
        self.generated = 0
        self.exhausted = False
        self._inflight: dict[int, int] = {}  # engine id -> groups dealt-but-unfinished

    def sync(self, eid: int, n_inflight: int) -> None:
        """True up one engine's in-flight count (owner thread only, after poll)."""
        with self._lock:
            self._inflight[eid] = n_inflight

    def deal(self, eid: int) -> list:
        """Compute need and draw prompts ATOMICALLY — the whole trick.

        need and the draw share one critical section, so two engines can
        never both claim the same remaining slot; the in-flight bump lands
        before the lock releases, so the claim is visible to the next dealer
        even though the actual engine.submit happens after (outside the lock).
        """
        with self._lock:
            mine = self._inflight.get(eid, 0)
            need = self._target - len(self.kept) - sum(self._inflight.values())
            n = min(need, self._burst - mine)
            if n <= 0 or self.exhausted:
                return []
            raw = self._source(n)
            if not raw:
                self.exhausted = True  # finite sources end collection everywhere
                return []
            self._inflight[eid] = mine + len(raw)
            return raw

    def note_dropped(self) -> None:
        with self._lock:
            self.generated += 1

    def offer(self, group: list[Trajectory]) -> bool:
        """Try to keep a surviving group. False -> surplus; caller stashes it
        on its OWN engine (next collection's first pick, never wasted)."""
        with self._lock:
            self.generated += 1
            if len(self.kept) < self._target:
                self.kept.append(group)
                return True
            return False

    def done(self) -> bool:
        with self._lock:
            return len(self.kept) >= self._target or self.generated >= self._budget


def _collect_one(
    eid: int,
    engine,
    reward_fn: RewardFn | None,
    tally: _Tally,
    sampling: SamplingParams,
    group_filter: GroupFilter | None,
) -> dict:
    """One engine's collector — runs on that engine's OWNER thread.

    The retired collect_groups_stream with local bookkeeping swapped for
    tally calls: need comes from the tally, kept groups go to the shared
    list, and group_ids are NOT stamped here (collectors cannot number
    globally — the merge restamps).
    """
    stats = {"submitted": 0, "polls": 0, "groups_generated": 0, "groups_dropped": 0}

    def top_up() -> None:
        for p in tally.deal(eid):
            ids, meta = p if isinstance(p, tuple) else (p, {})
            engine.submit(ids, sampling, meta)
            stats["submitted"] += 1

    top_up()
    while not tally.done():
        groups = engine.poll()  # own stash first, else ~one engine step
        stats["polls"] += 1
        tally.sync(eid, engine.n_inflight)  # finishes just observed; deal-time bumps trued up
        if not groups:
            if engine.n_inflight == 0:
                top_up()
                if engine.n_inflight == 0:
                    break  # dealer gave nothing: source exhausted, or others cover the need
            continue
        for group in groups:
            stats["groups_generated"] += 1
            if reward_fn is not None:
                for t in group:
                    t.reward = reward_fn(t)
            if group_filter is not None and not group_filter(group):
                stats["groups_dropped"] += 1
                tally.note_dropped()  # replacement claimable by ANY engine on its next deal
            elif not tally.offer(group):
                engine.stash(group)  # target already met: surplus survivor, kept for next call
        top_up()
    return stats  # leftovers (engine.n_inflight > 0) stay in flight, consumed next call


def collect_groups_dp(
    engines: list,
    reward_fn: RewardFn | None,
    prompt_source: Callable[[int], list],
    cfg: CollectConfig,
    sampling: SamplingParams,
    group_filter: GroupFilter | None = None,
) -> tuple[list[Trajectory], dict]:
    """Collect cfg.target_groups groups from k engines sharing one dealer+tally.

    May return SHORT if the source exhausts or the budget hits — check
    stats["groups"]. Spawns one owner thread per engine for the duration of
    the call (thread spin-up is microseconds — irrelevant next to a
    collection).
    """
    assert sampling.n == cfg.group_size, (
        f"sampling.n={sampling.n} != cfg.group_size={cfg.group_size} — one request IS one group"
    )
    if group_filter is None and cfg.strategy == "filter":
        group_filter = reward_nonzero_std

    tally = _Tally(
        prompt_source,
        target=cfg.target_groups,
        budget=cfg.max_rounds * cfg.target_groups,
        burst=-(-cfg.target_groups // len(engines)),  # ceil(target / k)
    )
    # Pre-register leftovers from the previous call so the first deals do not
    # over-commit. Safe engine touch: the caller owns ALL engines here (no
    # collector is running yet — this is the sole-toucher window).
    for eid, e in enumerate(engines):
        tally.sync(eid, e.n_inflight)

    with ThreadPoolExecutor(max_workers=len(engines)) as pool:  # one OWNER thread per engine
        futures = [
            pool.submit(_collect_one, eid, e, reward_fn, tally, sampling, group_filter)
            for eid, e in enumerate(engines)
        ]
        per_engine = [f.result() for f in futures]  # join ALL (exceptions propagate)

    for gid, group in enumerate(tally.kept):  # restamp: advantage math needs uniqueness only
        for t in group:
            t.meta["group_id"] = gid

    stats = {key: sum(s[key] for s in per_engine) for key in per_engine[0]}
    stats["groups"] = len(tally.kept)
    stats["leftover_inflight"] = sum(e.n_inflight for e in engines)
    for eid, s in enumerate(per_engine):  # scalar per engine (wandb-safe), shows the load balance
        stats[f"submitted_e{eid}"] = s["submitted"]
    return [t for group in tally.kept for t in group], stats


def _dist_ctx() -> tuple[int, int]:
    """(rank, world). world == 1 when torch.distributed was never initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _broadcast_batch(batch):
    """Rank 0 passes the Batch, followers pass None; everyone returns rank 0's.
    CPU-tensor dataclass over broadcast_object_list — small at study scale."""
    box = [batch]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def _gather_state(trainer: Trainer, world: int) -> dict:
    """Publishable plain fp32-CPU state dict. With world > 1 this is a gather
    COLLECTIVE — every rank must call it at the same point (followers discard)."""
    if world > 1:
        from minirl.train.distributed import full_state_dict

        return full_state_dict(trainer.model)
    return {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}


def fit_async(
    engines: list,  # rank 0: one streaming engine per rollout GPU; followers: []
    trainer: Trainer,  # Trainer (world=1) or DistTrainer (world>1, torchrun)
    reward_fn: Callable[[Trajectory], float],
    prompt_source: Callable[[int], list[Tensor]],
    sampling: SamplingParams,
    collect_cfg: CollectConfig,
    num_iterations: int,
    publish_interval: int = 1,  # staleness bound is interval + 1
    on_metrics: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run fully-async RL. Returns the per-iteration metrics history
    (rank 0; followers return []). See the module banner for the picture."""
    rank, world = _dist_ctx()
    if rank > 0:
        return _follow(trainer, num_iterations, publish_interval)

    assert engines, "rank 0 needs at least one engine"
    assert len({e.pad_id for e in engines}) == 1, "engines disagree on pad_id"
    # The advantage estimator is an ALGORITHM property, so it derives from the
    # loss config rather than being a separate knob (Dr. GRPO == norm_std=False).
    norm_std = getattr(trainer.loss_cfg, "grpo_std_normalization", True)
    advantage_fn = partial(grpo_advantages, norm_std=norm_std)

    # A threading.Event is a boolean flag visible across threads: set while
    # collecting, asserted clear before any publish. It never CREATES safety
    # (the held-future join does that) — it instruments the invariant so a
    # refactor that breaks the ordering fails loudly instead of corrupting
    # the engines silently.
    in_flight = threading.Event()

    def rollout() -> tuple[list[Trajectory], dict]:
        """Runs on the coordinator worker; fans out one collector per engine.
        The GIL is released inside engine forwards, so the main thread
        genuinely trains in parallel with all of this."""
        in_flight.set()
        try:
            t0 = time.perf_counter()
            trajs, stats = collect_groups_dp(
                engines, reward_fn, prompt_source, collect_cfg, sampling
            )
            assert trajs, "prompt_source exhausted / all groups filtered — nothing to train on"
            stats["t_generate"] = time.perf_counter() - t0
            return trajs, stats
        finally:
            in_flight.clear()

    def publish(version: int) -> None:
        """Drain-then-publish, fleet edition. MAIN THREAD ONLY, after a join."""
        assert not in_flight.is_set(), "publish during in-flight collection (join first)"
        for e in engines:  # ALL engines quiesce before ANY weight moves (single-version rule)
            e.drain()
        state = _gather_state(trainer, world)  # collective when world > 1
        for e in engines:
            e.load_weights(state.items(), version)

    history: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=1)  # ONE slot: at most one collection in flight,
    # which is what makes staleness a property of the loop's SHAPE, not a config knob
    try:
        publish(version=0)  # engines start from the learner's exact weights
        future = pool.submit(rollout)  # prime the pipeline: batch 1 under v0
        for it in range(1, num_iterations + 1):
            t_iter0 = time.perf_counter()

            # ---- JOIN batch k, then refill the slot IMMEDIATELY: the refill
            # placed BEFORE training is the entire async speedup ----
            trajs, gen_stats = future.result()
            future = pool.submit(rollout)

            # (it - 1) = the learner version these gradients update FROM;
            # min(version) = the oldest weights that generated any row.
            staleness = (it - 1) - min(t.version for t in trajs)
            assert 0 <= staleness <= publish_interval + 1, f"staleness {staleness} out of bounds"

            batch, batch_stats = make_batch(
                trajs, pad_id=engines[0].pad_id, advantage_fn=advantage_fn
            )
            if world > 1:
                _broadcast_batch(batch)  # followers are waiting on this
            t0 = time.perf_counter()
            train_metrics = trainer.fit_batch(batch)  # recomputes old_logprobs
            t_train = time.perf_counter() - t0

            if it % publish_interval == 0:
                held = future.result()  # THE LOAD-BEARING JOIN: no publish while collecting
                publish(version=it)
                # Re-wrap the already-collected batch so the loop invariant
                # "future always holds the next batch" stays true. Held data
                # was generated under the OLD version — the structural "one
                # step off" of async training, corrected by TIS.
                future = pool.submit(lambda h=held: h)

            metrics = (
                {"iteration": it, "staleness": staleness, "t_train": t_train,
                 "t_iter": time.perf_counter() - t_iter0}
                | gen_stats | batch_stats | train_metrics
            )
            if on_metrics is not None:
                on_metrics(metrics)
            history.append(metrics)
    finally:
        # Always join the worker before returning (even on exceptions): a
        # half-finished collection touching dying engines is exactly the
        # class of shutdown bug this line prevents.
        pool.shutdown(wait=True)
    return history


def _follow(trainer: Trainer, num_iterations: int, publish_interval: int) -> list[dict]:
    """Ranks > 0: no engines, no metrics — receive the batch, train, and show
    up for every collective at the same schedule as rank 0 (same arithmetic
    on the same configs; torchrun runs the same recipe everywhere)."""
    from minirl.train.distributed import full_state_dict

    # Rank 0's FIRST collective is the priming publish(version=0) gather,
    # BEFORE any broadcast — miss this and both sides deadlock on mismatched
    # collectives (found by the 2-rank test hanging, not by reading).
    full_state_dict(trainer.model)
    for it in range(1, num_iterations + 1):
        batch = _broadcast_batch(None)
        trainer.fit_batch(batch)
        if it % publish_interval == 0:
            full_state_dict(trainer.model)  # join the gather collective; rank 0 publishes
    return []
