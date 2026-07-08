"""Tier-1 async training driver — miniRL's train_async.py (docs/async_training.md).

The whole idea in one sentence: generate rollout k+1 in a background thread
while training on rollout k, and always JOIN the in-flight generation before
publishing weights. Overlap ~= min(t_generate, t_train) per iteration;
staleness is structurally bounded by update_weights_interval.

Shape mirrors slime's train_async.py exactly (their Ray future -> our thread
future; their "sync generate before update weights" -> the held-future join):

    publish(v0); future = submit(rollout)
    for it in 1..N:
        trajs = future.result()          # rollout it-1, sampled under an older version
        future = submit(rollout)         # rollout `it` overlaps the training below
        train(make_batch(trajs))         # trainer recomputes old_logprobs (tier-1 rule)
        if it % update_weights_interval == 0:
            held = future.result()       # never update weights mid-generation
            publish(v_it)
            future = wrap(held)          # hand the joined data to the next iteration

Deliberately a plain synchronous function with explicit .result() joins — no
asyncio (the two long operations are blocking compute, not awaitable IO), no
queues, no worker pools; those arrive with tier 2. Sync training is the
degenerate case of this loop (imagine resolving the future immediately), which
is why there is no separate sync controller.

Everything algorithmic lives elsewhere: staleness correction is TIS in the
loss (use_tis=True is the documented default here), advantage normalization is
read from the loss config, off-policy trust region is the PPO clip.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from torch import Tensor

from minirl.config import CollectConfig
from minirl.rollout.batching import make_batch
from minirl.rollout.sampling import collect_groups
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.train import Trainer


def fit_async(
    engine,  # duck-typed: generate(prompts, params), load_weights(named, version), pad_id
    trainer: Trainer,
    reward_fn: Callable[[Trajectory], float],
    prompt_source: Callable[[int], list[Tensor]],
    sampling: SamplingParams,
    collect_cfg: CollectConfig,
    num_iterations: int,
    update_weights_interval: int = 1,  # slime --update-weights-interval; == the staleness bound
    on_metrics: Callable[[dict], None] | None = None,  # e.g. print / wandb.log
) -> list[dict]:
    """Run tier-1 async RL. Returns the per-iteration metrics history."""
    # Advantage std-normalization is an ALGORITHM property (Dr. GRPO flag),
    # so the controller reads it from the loss config instead of owning it.
    norm_std = getattr(trainer.loss_cfg, "grpo_std_normalization", True)

    in_flight = threading.Event()  # instruments the tier-1 invariant (see publish)

    def rollout() -> tuple[list[Trajectory], dict]:
        """Runs on the worker thread; touches only the engine."""
        in_flight.set()
        try:
            t0 = time.perf_counter()
            trajs, stats = collect_groups(
                lambda prompts: engine.generate(prompts, sampling), reward_fn, prompt_source, collect_cfg
            )
            assert trajs, "prompt_source exhausted / all groups filtered — nothing to train on"
            stats["t_generate"] = time.perf_counter() - t0
            return trajs, stats
        finally:
            in_flight.clear()

    def publish(version: int) -> None:
        # The tier-1 invariant, enforced not just documented: weights must
        # never change under a running generation. The held-future join below
        # guarantees this; the assert catches any future refactor that breaks it.
        assert not in_flight.is_set(), "publish during in-flight generation (tier-1 violation)"
        named = ((k, v.detach().cpu()) for k, v in trainer.model.state_dict().items())
        engine.load_weights(named, version)

    history: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=1)  # one worker == one-slot pipeline
    try:
        publish(version=0)  # engine starts from the learner's exact weights
        future = pool.submit(rollout)

        for it in range(1, num_iterations + 1):
            t_iter0 = time.perf_counter()

            trajs, gen_stats = future.result()  # sampled under version <= it-1
            future = pool.submit(rollout)  # overlaps everything below

            # Staleness = learner version at update start (it-1) - behavior version.
            # 0 on the warm-up iteration, then bounded by update_weights_interval.
            staleness = (it - 1) - min(t.version for t in trajs)
            assert 0 <= staleness <= update_weights_interval, f"staleness {staleness} out of bounds"

            batch, batch_stats = make_batch(trajs, pad_id=engine.pad_id, norm_std=norm_std)
            t0 = time.perf_counter()
            train_metrics = trainer.fit_batch(batch)  # recomputes old_logprobs (tier-1 rule)
            t_train = time.perf_counter() - t0

            if it % update_weights_interval == 0:
                held = future.result()  # JOIN: the load-bearing line (slime comment ibid.)
                publish(version=it)
                future = pool.submit(lambda h=held: h)  # hand joined data forward

            metrics = (
                {"iteration": it, "staleness": staleness, "t_train": t_train,
                 "t_iter": time.perf_counter() - t_iter0}
                | gen_stats | batch_stats | train_metrics
            )
            if on_metrics is not None:
                on_metrics(metrics)
            history.append(metrics)
    finally:
        pool.shutdown(wait=True)
    return history
