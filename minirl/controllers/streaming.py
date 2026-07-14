"""Tier-2 async controller: continuous batching + streaming collection,
drain-then-publish weights (docs/async_tier2.md; in-flight updates DEFERRED).

Same pipeline shape as tier-1 fit_async — collect rollout k+1 on a worker
thread while training on rollout k, held-future join before any publish — so
everything proven there carries over. What tier 2 changes is only WHERE time
goes inside the rollout thread: collect_groups_stream keeps the engine full
group-by-group (no round boundaries), and work in flight when a collection
returns carries over to the next one instead of being discarded.

The publish sequence is the tier-1 invariant, extended one step:
    join the in-flight collection            (no concurrent engine access)
    engine.drain()                           (leftovers FINISH under the old
                                              weights, into the stash — every
                                              completion stays single-version)
    engine.load_weights(named, version)      (quiescent engine, asserted)

STALENESS BOUND: publish_interval + 1, not publish_interval. The +1 is the
price of never wasting work: a drained-leftover group finished just before a
publish and is consumed by the collection just after it, so it can be one
publish older than its batchmates. The assert below fails loud if stash-back
pathologies ever chain beyond that.

Thread-safety rule (vLLM engines are not thread-safe): every engine touch —
submit/poll/drain/load_weights — happens either on the worker thread (inside
collect) or on the main thread strictly AFTER joining the future. The
in_flight Event instruments this, exactly like tier-1.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

from torch import Tensor

from minirl.algos.advantage import grpo_advantages
from minirl.config import CollectConfig
from minirl.rollout.batching import make_batch
from minirl.rollout.streaming import collect_groups_stream
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.train import Trainer


def fit_async_stream(
    engine,  # duck-typed: submit/poll/stash/drain/n_inflight/load_weights/pad_id (VLLMEngine)
    trainer: Trainer,
    reward_fn: Callable[[Trajectory], float],
    prompt_source: Callable[[int], list[Tensor]],
    sampling: SamplingParams,
    collect_cfg: CollectConfig,
    num_iterations: int,
    publish_interval: int = 1,  # tier-1's update_weights_interval; staleness bound is interval + 1
    on_metrics: Callable[[dict], None] | None = None,  # e.g. print / wandb.log
) -> list[dict]:
    """Run tier-2 async RL. Returns the per-iteration metrics history.

    THE MENTAL PICTURE — two threads, one engine, a one-slot pipeline:

        main thread   [ train on batch k          ][join][drain·publish][ train k+1 ...
        worker thread [ collect batch k+1 (poll/step the engine) ][idle]
        engine (GPU)  [ continuous batching, weights v_{k-1}     ][fin.][ v_k ...

    `future` is the one pipeline slot: it ALWAYS holds "the next batch, being
    collected (or already collected and held)". The main thread trains on
    batch k while the worker fills that slot with batch k+1 — that overlap is
    the entire speedup of async training. Everything else in this function
    exists to keep two invariants true while overlapping:

      (1) exactly ONE thread touches the engine at any moment (vLLM engines
          are not thread-safe) — enforced by only touching it on the worker
          (inside collect) or on the main thread AFTER a join;
      (2) weights never change while requests are in flight (drain-then-
          publish) — so every completion is generated under exactly one
          weight version and Trajectory.version stays a scalar.
    """
    # The advantage estimator is an ALGORITHM property, so it derives from the
    # loss config rather than being a separate knob (Dr. GRPO == norm_std=False).
    norm_std = getattr(trainer.loss_cfg, "grpo_std_normalization", True)
    advantage_fn = partial(grpo_advantages, norm_std=norm_std)

    # A threading.Event is a boolean flag visible across threads. The worker
    # sets it while collecting; publish() asserts it is clear. It never
    # CREATES safety (the join does that) — it instruments the invariant so a
    # future refactor that breaks the ordering fails loudly in tests instead
    # of corrupting the engine silently.
    in_flight = threading.Event()

    def rollout() -> tuple[list[Trajectory], dict]:
        """Runs ON THE WORKER THREAD (via pool.submit below).

        This is the only code that touches the engine concurrently with
        training: collect_groups_stream drives engine.poll() in a loop —
        every poll advances vLLM by ~one decode step for EVERYTHING in
        flight (continuous batching), rewards/filters groups the moment
        they finish, and tops the engine up with replacement prompts.
        The GIL is released inside the model's forward passes, so the main
        thread genuinely trains in parallel with this.
        """
        # Event.set(): flip the shared flag to True (atomically, visible to
        # the other thread immediately). Meaning here: "worker is collecting —
        # the engine is MINE right now."
        in_flight.set()
        try:
            t0 = time.perf_counter()
            trajs, stats = collect_groups_stream(
                engine, reward_fn, prompt_source, collect_cfg, sampling
            )
            assert trajs, "prompt_source exhausted / all groups filtered — nothing to train on"
            stats["t_generate"] = time.perf_counter() - t0
            return trajs, stats
        finally:
            # Event.clear(): flag back to False — "engine free." Sits in a
            # finally so an exception mid-collection can never leave the flag
            # stuck True (which would make every later publish assert-fail).
            in_flight.clear()

    def publish(version: int) -> None:
        """Push the learner's current weights into the engine. MAIN THREAD ONLY,
        and only after joining the future — the assert checks exactly that.

        Three-step drain-then-publish (the deliberate NON-in-flight choice):
          1. assert nobody is collecting (we must be the only engine toucher);
          2. engine.drain() — step the engine until every in-flight request
             FINISHES under the old weights; the finished groups land in the
             engine's stash and become the next collection's first picks
             (work is converted into a head start, never thrown away);
          3. engine.load_weights(...) — which itself asserts the engine is
             empty, so the contract holds even if a future caller skips (2).
        """
        # Event.is_set(): read the flag (True while the worker collects).
        # The JOIN above every publish call is what actually guarantees the
        # worker is parked; this assert merely OBSERVES that guarantee — a
        # tripwire that fails loud in tests if a refactor ever reorders the
        # join and the publish. (We never use Event.wait() — the blocking is
        # done by future.result(), not by the Event.)
        assert not in_flight.is_set(), "publish during in-flight collection (join first)"
        engine.drain()
        named = ((k, v.detach().cpu()) for k, v in trainer.model.state_dict().items())
        engine.load_weights(named, version)

    history: list[dict] = []
    # ONE worker == one pipeline slot: at most one collection can ever be in
    # flight, which is what makes staleness a property of the loop's SHAPE
    # rather than a config knob (slime's fully-async pool trades this away).
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        publish(version=0)  # engine starts from the learner's exact weights (v0)
        # Prime the pipeline: start collecting batch 1 under v0. submit()
        # returns a Future immediately — a ticket for work now running on the
        # worker thread; .result() later blocks until that work is done.
        future = pool.submit(rollout)

        for it in range(1, num_iterations + 1):
            t_iter0 = time.perf_counter()

            # ---- JOIN: wait for batch k to finish collecting ----
            # (If generation is the bottleneck this is where the main thread
            # waits — "wait A" in docs/fast_rl.md. If training is the
            # bottleneck this returns instantly.)
            trajs, gen_stats = future.result()  # collected under version(s) <= it-1

            # ---- REFILL THE SLOT IMMEDIATELY: batch k+1 starts NOW ----
            # This line, placed BEFORE training rather than after, is the
            # whole async idea: generation of the next batch overlaps the
            # training below. The engine also still holds any leftover
            # in-flight groups from the previous collection — the new
            # collection picks those up first (nothing was discarded).
            future = pool.submit(rollout)

            # ---- staleness: how old is the OLDEST data in this batch? ----
            # (it - 1) is the learner version these gradients will update
            # FROM; min(version) is the oldest weights that generated any of
            # the data. Freshly collected rows are at most publish_interval
            # behind; a drained-leftover row can be ONE publish older still —
            # hence the interval + 1 bound (module banner). TIS in the loss
            # is what makes this bounded off-policyness mathematically fine.
            staleness = (it - 1) - min(t.version for t in trajs)
            assert 0 <= staleness <= publish_interval + 1, f"staleness {staleness} out of bounds"

            # ---- collate + train on batch k (main thread, overlapping the
            # worker's collection of batch k+1) ----
            batch, batch_stats = make_batch(trajs, pad_id=engine.pad_id, advantage_fn=advantage_fn)
            t0 = time.perf_counter()
            train_metrics = trainer.fit_batch(batch)  # recomputes old_logprobs (tier-1 rule)
            t_train = time.perf_counter() - t0

            # ---- publish the freshly trained weights (every interval) ----
            if it % publish_interval == 0:
                # THE LOAD-BEARING JOIN. We cannot drain/load while the worker
                # is polling the same engine, so we first wait for the
                # in-flight collection to finish and HOLD its result...
                held = future.result()
                publish(version=it)  # ...then drain + load on a quiet engine.
                # The held batch was already collected — re-wrap it in a
                # trivial future (a lambda that just returns it) so the loop
                # invariant "future always holds the next batch" stays true
                # and the next iteration's .result() is instant. Note the
                # held data was generated under the OLD version — that is the
                # structural "one step off" of async training, corrected by
                # TIS, never avoidable without wasting the work.
                future = pool.submit(lambda h=held: h)

            # ---- metrics: controller timing | collection | collation | training ----
            metrics = (
                {"iteration": it, "staleness": staleness, "t_train": t_train,
                 "t_iter": time.perf_counter() - t_iter0}
                | gen_stats | batch_stats | train_metrics
            )
            if on_metrics is not None:
                on_metrics(metrics)
            history.append(metrics)
    finally:
        # Always join the worker before returning (even on exceptions):
        # a daemonized half-finished collection touching a dying engine is
        # exactly the class of shutdown bug this line prevents.
        pool.shutdown(wait=True)
    return history
