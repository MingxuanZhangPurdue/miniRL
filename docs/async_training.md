# Async RL training: design for the basic async trainer

Design note for `minirl/async_controller.py` — IMPLEMENTED (fit_async, with
invariant tests in tests/test_async_controller.py). Deviations from the
original draft, decided during review: the controller lives at the package
top level (it orchestrates rollout AND training, it is not rollout
machinery), there is NO separate sync controller (sync = this loop with the
future resolved eagerly; a degenerate case not worth a file), and the one
config knob is a plain `update_weights_interval` kwarg rather than an
AsyncConfig dataclass (one field is not a dataclass — DESIGN principle 8).
Grounded in a close read of slime's async implementation (train_async.py,
rollout/fully_async_rollout.py, examples/fully_async/README, 2026-07).

## 1. What slime actually does (study findings)

slime has TWO async tiers. The basic one is much simpler than "worker pools
and replay buffers":

### Tier 1 — one-step-off pipelining (train_async.py)

The entire diff from sync training is: launch the NEXT rollout as a future
before training on the current one.

```python
# slime train_async.py, de-Ray-ified:
future = rollout.generate(0)                      # gen_0 under weights v0
for k in range(num_rollout):
    data = future.result()                        # rollout k finishes
    future = rollout.generate(k + 1)              # rollout k+1 starts NOW —
    actor.train(k, data)                          #   ...and overlaps training
    if (k + 1) % update_weights_interval == 0:    # default: every step
        data_next = future.result()               # WAIT: never update weights
        future = None                             #   mid-generation
        actor.update_weights()                    # engine now at v_{k+1}
```

Properties worth copying:
- **Bounded, structural staleness**: with interval=1, data trained at learner
  version v_{k+1} was always generated under v_k. Exactly one step off — not
  a tunable "max lag" heuristic, a property of the loop shape.
- **Weight updates never interrupt generation** (the explicit sync before
  update). No mid-episode policy switching, no server aborts, in tier 1.
- **No replay buffer**: every batch is trained exactly once; the "buffer" is
  the single in-flight future. Queues only appear in tier 2.
- **No colocation** (`assert not args.colocate`): overlap only pays if
  generation and training own different GPUs.
- Off-policy correctness is NOT the controller's job: behavior logprobs ride
  with the samples and TIS in the loss corrects the one-version gap
  (slime --use-tis; our algos/tis.py).

### Tier 2 — fully async (fully_async_rollout.py), NOT in scope for us yet

A persistent background worker (thread + asyncio loop) keeps a fixed pool of
in-flight generations ACROSS rollout boundaries (concurrency decoupled from
batch size); finished groups land on an output queue; each training step
drains the queue until it has `rollout_batch_size` groups. Weight updates
abort in-flight server requests; ABORTED groups are requeued to restart under
fresh weights, or with --partial-rollout resumed from where they stopped
(mixed-policy trajectories; --mask-offpolicy-in-partial-rollout zeroes the
stale tokens' loss_mask). No eval mode; ordering best-effort. This is the
GLM-5-style "learner never waits for the slowest episode" design — it matters
for long/variable agentic episodes and is our Phase 6+ follow-up, not the
basic trainer.

## 2. miniRL basic async = tier 1, one background future

Minimalism verdict: the basic async trainer needs NO new processes, queues,
or buffer files. `engine.generate` runs in a single background thread (the
GIL is released inside model forward passes on both vLLM and HF); the
one-slot future IS the pipeline. Multi-worker pools arrive only with tier 2.

```
             GPU/device A (engine)              GPU/device B (learner)
  it k:      [ generate rollout k+1 .......]    [ recompute old_logprobs ]
                                                [ train on rollout k     ]
             -- both done? -> publish weights v_{k+1} -> engine loads ----
  it k+1:    [ generate rollout k+2 .......]    [ train on rollout k+1   ]
```

### Controller pseudocode (rollout/async_controller.py)

```python
def fit_async(cfg, engine, learner, trainer, reward_fn, prompt_source):
    publish(learner, engine, version=0)
    pool = ThreadPoolExecutor(max_workers=1)              # the whole "infra"
    submit = lambda: pool.submit(
        collect_groups, engine.generate, reward_fn, prompt_source, cfg.collect
    )
    future = submit()                                     # rollout 0 under v0

    for it in range(cfg.num_iterations):
        trajs, stats = future.result()                    # generated under v_{it-1}
        future = submit()                                 # overlap starts here
        batch = make_batch(trajs, advantage_fn, pad_id)
        batch.old_logprobs = recompute_logprobs(learner, batch)   # see §3
        train_metrics = trainer.fit_batch(batch)          # ppo epochs / minibatches
        if (it + 1) % cfg.update_weights_interval == 0:
            held = future.result()                        # never update mid-generation
            publish(learner, engine, version=it + 1)
            future = pool.submit(lambda: held)            # hand the held data forward

    log: stats["version_lag"], time/generate vs time/train, tis_mean, ...
```

Sync is the degenerate case (`future.result()` immediately after submit — or
just keep controller.py as the readable 5-minute version, per DESIGN).

### What is reused, unchanged

losses (TIS already built and tested), advantage.py, sampling.collect_groups
(dynamic filtering composes with async for free), make_batch, trainer,
weight_sync.publish, Trajectory.version (already in the contract — engines
stamp it; HFEngine does since day one).

### The one genuinely new requirement: recompute old_logprobs

In sync training, `old_logprobs = behavior_logprobs` was tolerable (same
version; only numerics differ). Under async it is WRONG to conflate them —
three distinct policies touch each batch:

    pi_engine@v_{k-1}   sampled the tokens  -> behavior_logprobs (came with data)
    pi_learner@v_k      starts this update  -> old_logprobs      (recompute, fp32)
    pi_learner@v_k+     during ppo epochs   -> policy_logprobs   (the fwd pass)

The PPO/GRPO ratio must be policy/old (trust region around where THIS update
started); TIS's weight exp(old - behavior) then absorbs the version gap AND
the engine-numerics gap in one term. This matches slime's default
(use_rollout_logprobs=False: old comes from the trainer's own recompute).
Recompute is one no-grad forward over the batch before the ppo epochs — it
becomes part of make_batch/controller, not the trainer.

### Config

One kwarg on fit_async: `update_weights_interval: int = 1` (slime
--update-weights-interval; == the staleness bound, verified by test). Pipeline
depth is fixed at 1 (one-step-off); a knob would be tier-2 scope creep.
`use_tis=True` is the documented default in async recipes.

## 3. Failure modes this design dodges (and how)

- **Weights change under a running generation** -> impossible by construction
  (the held-future join before publish), same as slime tier 1.
- **Learner trains on its own numerics as if on-policy** -> old_logprobs
  recompute + TIS (§2).
- **Deadlock** -> there is exactly one future and one thread; the only waits
  are `future.result()` on work already submitted.
- **Silent quality regression vs sync** -> exit criterion (DESIGN roadmap
  Phase 5): async interval=1 matches sync final reward on GSM8K within noise,
  with wall-clock speedup on disaggregated GPUs; staleness ablation = raising
  update_weights_interval and watching reward/tis_clip_frac degrade.

## 4. Invariants checklist (each becomes a test)

1. Version bookkeeping: every trajectory trained at learner version v carries
   version v-interval..v-1; assert in the controller, log `version_lag`.
2. Degenerate equivalence: async controller with the future resolved eagerly
   == sync controller output, same seeds (tiny random model, CPU).
3. No mid-generation update: publish never fires while the generation thread
   is inside engine.generate (assert via a flag around the call).
4. old_logprobs != behavior_logprobs under staleness (and == under none,
   fp32 HFEngine, tolerance) — guards the recompute wiring.
5. Overlap actually happens: time(iteration) < time(generate) + time(train)
   on a two-device run (integration, GPU box).

## 5. Name mapping (Rosetta stone, async slice)

| here | slime | notes |
|---|---|---|
| `fit_async` one-slot future | train_async.py `generate.remote()` future | Ray future -> ThreadPoolExecutor future |
| `update_weights_interval` | --update-weights-interval (default 1) | identical semantics incl. the pre-publish join |
| held-future join before publish | "sync generate before update weights" comment | the load-bearing line |
| old_logprobs recompute | use_rollout_logprobs=False default path | trainer-side fp32 recompute |
| (tier 2, future) worker pool + queue + abort/requeue | fully_async_rollout.AsyncRolloutWorker | + --partial-rollout resume, GLM-5 style |
