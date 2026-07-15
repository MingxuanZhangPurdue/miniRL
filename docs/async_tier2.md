# Tier-2 async: step-loop VLLMEngine, streaming collection, in-flight updates

Status: **IMPLEMENTED** (2026-07-10) — continuous batching + streaming
collection + drain-then-publish controller, in three new files with ZERO
changes to existing code (§0 upheld; `git status` was the check). In-flight
updates stay DEFERRED (§4). Tested against a fake streaming engine
(tests/test_streaming.py, the §7 Mac rungs); REMAINING: a real vllm-metal
smoke run from `~/.venv-vllm-metal` (EOS-parity check, §2 note) and GPU-box
scale validation. Feasibility grounding: the 2026-07-09 spike (§8) proved
vLLM-with-continuous-batching and the full weight-update path on this Mac.
Motivation and the two-waits analysis: **docs/fast_rl.md**. As-built shape is
SIMPLER than the original sketch: dropping in-flight updates removed the need
for a serve thread, queues, and the mailbox — the collector drives the engine
step loop directly (poll-based), and the controller keeps tier-1's exact
held-future pipeline.

## 0. The design constraint that shapes everything (packing lesson, 2026-07-09)

A correct, fully tested packing implementation was rolled back because it
threaded an optional second mode through five core files. This design
pre-commits to the opposite shape:

**Every piece of tier-2 machinery lives in NEW files. Existing files change
ZERO lines**: trainer untouched (it already takes Batches), losses untouched
(per-token TIS already handles mixed-policy completions), types untouched
(the one new fact — a completion spanning weight versions — rides in
`Trajectory.meta`, which exists for exactly this), `collect_groups` /
`fit_async` / `HFEngine` untouched (they remain the readable tier-1 story).
Tier-2 is additive; deleting its three files restores today's repo.

## 1. The three new files and how they connect (as built)

    minirl/engine/vllm_engine.py      VLLMEngine: submit/poll/stash/drain +
                                      tier-1 generate() + load_weights
    minirl/rollout/streaming.py       collect_groups_stream (drives poll())
    minirl/controllers/streaming.py       fit_async_stream (tier-1's pipeline shape)

    worker thread:  collect_groups_stream ── submit(prompt, params, meta) ──┐
                      │  poll() -> finished groups; filter; top_up;         │ VLLMEngine
                      │  surplus -> stash                                   │ (vLLM steps
    main thread:    trainer.fit_batch (UNCHANGED)                           │  = continuous
                      │ every publish_interval:                             │  batching)
                      └─ join future -> engine.drain() -> load_weights ─────┘

    No serve thread, no queues, no mailbox: the collector IS the step-loop
    driver, and all engine access is serialized by tier-1's held-future join
    (the worker touches the engine inside collect; the main thread only after
    joining). vLLM engines are not thread-safe — this discipline is the
    concurrency story, instrumented by the in_flight Event.

## 2. minirl/engine/vllm_engine.py

One class, two operating modes, so it serves BOTH tiers (as built):

- **Tier-1 mode** — satisfies today's duck-type exactly, no threads:
  `generate(prompt_ids, params)` submits everything and polls until all
  requests finish, returning trajectories grouped in submission order.
  `fit_async` works with vLLM unchanged, day one.
- **Tier-2 mode** — four small primitives the collector drives directly
  (no thread of its own; the collector runs on the controller's worker
  thread, same as tier-1's rollout):

  ```python
  submit(prompt_ids, params, meta) -> rid   # add_request with n=G, meta rides along
  poll() -> list[group]                     # stash first, else ONE engine.step();
                                            #   finished requests -> G Trajectories
  stash(group)                              # hand back surplus; next poll's first pick
  drain()                                   # poll until idle, finishers -> stash
                                            #   (the pre-publish quiescence step)
  ```

  `load_weights(named, version)` ASSERTS quiescence (drain-then-publish is
  the contract, not a convention) and ships the state dict as a safetensors
  FILE to a worker-side function — no tensor serialization over RPC. The
  worker function implements §8's validated Metal recipe (MLX tree load +
  per-layer `_inner` attention assignments, with a fail-loud canary assert)
  and the standard torch branch for CUDA vLLM (unvalidated until the box).

Design decisions, each with its reason:

- **One request per prompt with `SamplingParams(n=G)`**, not G separate
  requests. vLLM returns all G completions in one `RequestOutput`; the
  request finishes when its slowest member does — which is not a compromise,
  because GRPO cannot reward/filter a group until all G siblings exist
  anyway. Group tracking = request tracking; no reassembly bookkeeping.
- **Behavior logprobs**: `SamplingParams(logprobs=0)` — vLLM attaches the
  SAMPLED token's logprob, computed AT SAMPLING TIME each decode step. This
  is the property the whole off-policy story rests on under in-flight
  updates: a token's recorded logprob reflects the weights that actually
  sampled it. GPU test pins it (§7).
- **Weight transfer is a FILE, not tensors-over-RPC**: the learner's state
  dict is saved to a temp safetensors and the worker loads from the path —
  CUDA-IPC/NCCL only if a dashboard ever says the file copy hurts.
- **Version stamping (as built, simpler than drafted)**: engine keeps
  `self.version`; a group records it at submit and stamps
  `Trajectory.version` with it at finish. Because publishes only happen on a
  drained engine, submit-version == finish-version ALWAYS — no
  `version_start` bookkeeping, the tier-1 scalar survives verbatim. (The
  range bookkeeping drafted for in-flight updates went with them, §4.)
- The engine's low-level loop uses `LLMEngine.add_request()/step()` (present
  in V1, labeled legacy-but-supported). If it is ever removed, the fallback
  is `AsyncLLM` wrapped inside this class feeding the same poll() interface —
  the asyncio stays INSIDE this file; controller and collector never see it.

## 3. minirl/rollout/streaming.py — collect_groups_stream

The Olmo-3-shaped collector: same decision rule as `collect_groups`
(zero-gradient filter + refill to a constant batch), pull-based off the
engine's poll() instead of round-based (as built):

```python
def collect_groups_stream(engine, reward_fn, prompt_source, cfg, sampling, group_filter=None):
    top_up()                                       # keep (kept + in-flight) at target
    while len(kept) < target and generated < cfg.max_rounds * target:
        for group in engine.poll():                # stash first, else ~one engine step
            reward it (meta already rides on the trajectories from submit)
            dropped -> replacement enters on the next top_up()
            kept full -> engine.stash(group)       # surplus survivor, never wasted
        top_up()
    stamp group_ids; return flat trajs, stats      # short if source/budget ran out
```

- **Leftovers persist**: in-flight requests beyond `target_groups` are NOT
  aborted when collection returns — the next call consumes them first. This
  is the "actors never stop" property; it is also why version bookkeeping
  must be per-trajectory (a leftover was submitted during the previous
  iteration).
- **Budget**: `max_rounds * target_groups` total generated groups replaces
  tier-1's round cap — same knob, same meaning (bounded worst-case compute),
  no new config field.
- **Reuses** `CollectConfig`, `reward_nonzero_std`, and the meta/group_id
  conventions verbatim — the filter logic is imported, not reimplemented.

## 4. In-flight weight updates (the mailbox) — DEFERRED (decision 2026-07-10)

> **Plan of record:** build continuous batching + streaming collection ONLY;
> keep drain-then-publish (the tier-1 invariant "one completion = one
> version" survives, version stays a scalar, no mixed-policy semantics).
> Rationale: CB/streaming are semantics-free scheduling; in-flight is the
> only piece that changes what a trajectory IS, and its payoff scales with
> completion length (PipelineRL's +117% was at 32k tokens — at our scales the
> drain is cheap and amortizable via publish_interval). The §8 weight-update
> recipe is still required — a drained publish uses it in its safest form
> (zero requests in flight while weights change). This section stays as the
> design for if/when long-CoT runs make the drain measurably expensive.

- `publish()` in the stream controller: put `(named_tensors, version)` into a
  ONE-SLOT mailbox (new value overwrites unclaimed old — only the freshest
  weights matter) and return immediately. No join, no drain.
- The serve loop applies it between two `step()` calls: in-flight requests
  keep their KV (computed under the old weights — the PipelineRL trade,
  argued in docs/fast_rl.md §4); tokens sampled from the next step onward are
  the new policy's, and their behavior logprobs say so.
- **Why the loss stack needs zero changes** (the whole point): TIS corrects
  per token, `w_t = clamp(exp(log pi_old - log pi_engine))`, and
  `behavior_logprobs` are per-token facts recorded at sampling time. A
  completion whose tokens came from two policies is just a completion whose
  behavior logprobs came from two policies. `use_tis=True` is mandatory here
  (it was already the documented async default).
- **Staleness bookkeeping**: per-iteration metric becomes the RANGE
  `[it - max(version), it - min(meta["version_start"])]` over the batch; the
  tier-1 scalar assert is replaced by a bound on the range's upper end.

## 5. minirl/controllers/streaming.py — fit_async_stream (as built)

Tier-1's pipeline shape SURVIVES — one worker thread collecting rollout k+1
while training on rollout k, held-future join before any publish. (The
original draft deleted the join because the mailbox made publishing free;
dropping in-flight updates brought the join back, and with it every tier-1
concurrency guarantee. vLLM engines are not thread-safe; the join is what
serializes all engine access.)

```python
def fit_async_stream(engine, trainer, reward_fn, prompt_source, sampling,
                     collect_cfg, num_iterations, publish_interval=1, on_metrics=None):
    publish(version=0)                # publish = assert not collecting;
    future = submit(rollout)          #   engine.drain(); engine.load_weights(...)
    for it in 1..num_iterations:
        trajs, gen_stats = future.result()          # join
        future = submit(rollout)                    # overlap resumes
        staleness = (it-1) - min(t.version); assert <= publish_interval + 1
        batch = make_batch(trajs, ...)              # UNCHANGED
        train_metrics = trainer.fit_batch(batch)    # UNCHANGED
        if it % publish_interval == 0:
            held = future.result()                  # the load-bearing join
            publish(version=it)                     # drain-then-publish
            future = submit(lambda: held)
        on_metrics(...)
```

Differences vs `fit_async`, in full: the collector is streaming (per-group
filter/refill, engine never drains between rounds), leftovers carry over via
the engine stash, and publish drains first — which is why the staleness
bound is `publish_interval + 1` (a drained-leftover group finishes just
before a publish and is consumed just after it). `fit_async` and its tests
stay untouched as the teaching path.

## 6. What deliberately does not change

trainer.py, all losses, aggregate.py, advantage.py, types.py, batching.py,
sampling.py, controllers/round_based.py, hf_engine.py, data/, rewards/. The tier-1
stack remains the complete, readable, Mac-runnable story; tier-2 is three
files you read only when you care about throughput.

## 7. Testing strategy

Mac rungs — **[done]**, tests/test_streaming.py (a FakeStreamEngine mirrors
test_async_controller's FakeEngine: deterministic poll-driven finishes,
version stamping, and load_weights ASSERTING the drain contract):

1. [done] collector: filter -> replacement submission; constant batch;
   budget cap on a pathological all-degenerate source; short return on
   source exhaustion; stash consumed before new generation; group_id
   stamping; sampling.n == group_size fails loud.
2. [done] controller: published versions exact per publish_interval;
   staleness warm-up 0 then bounded by interval + 1; metrics flow
   (tier-1 keys + submitted/polls/leftover_inflight); training moves
   weights; publish-only-when-quiescent enforced by the fake's assert.
3. [done] vllm_engine module imports WITHOUT vLLM installed (all vLLM
   imports are method-local) — the repo test env never needs the venv.

Mac-testable with a REAL model via chunked generation (a TEST-ONLY helper —
a ChunkedHFEngine in the tier-2 test file, NOT a feature of HFEngine, which
stays the simple teaching engine): HF generate() accepts past_key_values for
continuation, so `generate k tokens -> check mailbox -> maybe load weights ->
continue from the KEPT cache` reproduces in-flight-update semantics at chunk
granularity on MPS. This gives Mac-scale versions of rungs 5-6 below: real
mixed-policy completions, per-chunk behavior logprobs faithful to the weights
that sampled them, KV crossing a weight boundary. What it cannot exercise is
slot-refill scheduling (HF batch membership is frozen per call) — but that is
vLLM's code, not ours; ours is the loop around it, which this tests.

Real-engine validation, next in line (runs from `~/.venv-vllm-metal` with
the repo on PYTHONPATH; NOT under pytest/mingxuan — vLLM is not installed
there, by design):

4. vllm-metal smoke: `VLLMEngine.generate` on Qwen3-0.6B (tier-1 duck-type
   parity vs HFEngine: grouped order, loss_mask, logprobs present);
   **EOS-parity check** — HFEngine's responses INCLUDE the eos token; verify
   vLLM's token_ids do too, else fix in `_to_group`; the **weight-update
   canary** — perturb-then-restore through `load_weights` (spike 2/2d
   reproduced through the production code path).

Deferred with in-flight updates (rungs for if/when §4 reopens):

5. logprob-at-sampling-time under a mid-request publish; keep-the-KV
   tolerance (both have Mac-scale variants via the chunked helper above).

GPU-box (scale validation):

6. throughput ladder: tokens/sec at each stage (batch-call vLLM ->
   streaming), reproducing the article's methodology.
7. equivalence exit criterion: tier-2 matches tier-1 final reward on GSM8K
   within noise, with the wall-clock win the dashboard predicted.

## 8. vllm-metal spike findings (2026-07-09, all empirical)

Setup: vllm-metal 0.3.0.dev20260708 (plugin wheel) + vLLM 0.24.0, isolated
venv `~/.venv-vllm-metal` (uv-managed CPython 3.12; the mingxuan conda env is
untouched — repo dev stays there). Model: Qwen/Qwen3-0.6B, bf16, M-series
Metal (25.8GB). Spike scripts lived in the session scratchpad; everything
needed to reproduce is in the recipe below.

| check | verdict | evidence |
|---|---|---|
| Metal platform plugin active | PASS | `MetalPlatform`, paged attention + chunked prefill enabled |
| token ids in/out, `n=G`, sampled logprobs | PASS | one request -> G completions; `logprobs=0` attaches sampled-token logprob |
| `LLMEngine.add_request()/step()` | PASS | present; `LLM` wraps `LLMEngine`; NOTE: EngineCore runs as a SUBPROCESS by default (V1 multiprocessing) |
| engine<->learner numerics | PASS | vs HF fp32 on same tokens: mean 0.018 nats, max 0.105 — well inside TIS's clamp band |
| in-place weight update | **PASS, with a map** | see below — three-layer story |

The weight-update story (the one nontrivial finding):

1. `collective_rpc` plumbing works; the Metal worker has NO `load_weights`
   method (string RPC fails clean). Callable RPC works with
   `VLLM_ALLOW_INSECURE_SERIALIZATION=1`.
2. `model.load_weights(safetensors_path, strict=False)` (MLX) updates
   embeddings / norms / MLP — but **SILENTLY SKIPS ATTENTION**: the plugin
   wraps every `self_attn` in `SDPAPagedAttentionWrapper`, which hides the
   original module via `object.__setattr__(self, "_inner", ...)` — invisible
   to the MLX parameter tree, hence to `load_weights`. (Verified: zeroing
   q_proj via load_weights changed nothing.) A naive port would train against
   a franken-policy (fresh MLP, stale attention) with no error.
3. The original mlx-lm `Attention` module survives at
   `layers[i].self_attn._inner`, and the Metal kernel reads its projections
   LIVE each forward — so direct assignment
   (`._inner.q_proj.weight = mx.array(...)`) takes effect immediately.
   Verified end to end: zero -> output breaks, restore -> greedy output
   byte-identical to baseline.
4. `model_runner.load_model()` mid-life is NOT a fallback: it corrupts live
   cache wiring (`OffsetCache ... update_and_fetch` crash on next step).

**The complete update recipe** (what VLLMEngine.load_weights does on Metal):
learner saves bf16 state dict as safetensors -> RPC one worker-side function:
`model.load_weights(path, strict=False)` for tree params + per-layer
`_inner.{q,k,v,o}_proj/{q,k}_norm` assignments from the same file. ~40 lines.

Caveats to carry into implementation: `_inner` is a private attribute of a
fast-moving plugin — pin the vllm-metal version and add a canary test
(perturb-restore, exactly spike 2d) that fails loud if the layout changes;
the callable-RPC path needs the insecure-serialization env var (fine locally;
the clean fix is contributing an RLHF-style `update_weights` worker method
upstream — the project takes daily wheels, natural PR); cosmetic
torch-MPS-allocator assert during engine teardown (harmless, post-run).

## 9. Name mapping (Rosetta stone)

| here | slime tier-2 | Olmo 3 / open-instruct | notes |
|---|---|---|---|
| collector-driven poll() loop | AsyncRolloutWorker + output queue | actor loops + completed queue | ours needs no thread/queue of its own — the collector IS the driver |
| collect_groups_stream | per-finished-group filtering | pop-and-discard off the queue | same decision rule as tier-1 collect_groups |
| drain-then-publish | join in tier-1; abort+requeue+--partial-rollout in tier-2 | in-flight updates (PipelineRL) | we FINISH leftovers; slime aborts/resumes; PipelineRL never stops — three prices for the same triangle |
| engine stash | requeued aborted groups | queue backlog | ours holds only COMPLETED single-version work |
| publish_interval | --update-weights-interval | sync-every-N-steps | staleness bound = interval + 1 (stash carryover) |

## 10. Data-parallel engines (IMPLEMENTED 2026-07-14 — controllers/data_parallel.py, CPU-tested; on-box validation pending)

The scenario: one node, model fits on a single GPU, k GPUs reserved for
rollouts (e.g. 8 GPUs = 4 rollout + 4 FSDP training). Layout decision first:

    model fits one GPU?   yes -> DP: k engines x TP=1, one owner thread each
                          no  -> TP: 1 engine, tensor_parallel_size = smallest
                                 fit (vLLM shards internally; NOTHING in our
                                 controller changes — still one engine object)
    big model AND many GPUs -> combine (2 replicas x TP=4); keep TP as small
                               as memory allows (TP pays comms every layer),
                               spend the rest on replicas.

The TP case is a two-line constructor passthrough. Everything below is the
DP case, and it is built on one principle:

**Deal, don't partition.** No engine is "assigned batch/k prompts". A shared
DEALER (the existing stateful `prompt_source` closure behind a
`threading.Lock` — its cursor already hands out each prompt exactly once, so
disjointness is free) and a shared TALLY (groups kept + total in flight vs
the global target, behind the same lock) replace per-engine targets. Each
engine's collector computes `need = target - tally.kept - tally.inflight` at
every top_up and deals itself that many prompts. Consequences, all automatic:

  - no duplicate prompts (the dealer's cursor is the single source of truth);
  - no idle engine while un-dealt prompts remain — a fast engine's top_up
    simply finds need > 0 again (pull, not push; nobody "assigns" work);
  - load balance is emergent (a 64-group batch might land 22/17/14/11);
  - dynamic-sampling replacements draw from the same dealer like any prompt;
  - per-prompt dealing = finest work granularity, no straggler quantization
    (chunked dealing is for remote dealers; ours is a local closure).

Locks are CORRECT here and wrong for the engine — the distinction that
matters: dealer/tally critical sections are microseconds of pure-Python
bookkeeping (a lock serializes them harmlessly); the engine is a long-lived
mutable machine (ownership by protocol, one thread per engine, never shared).

Controller shape (new file, e.g. `minirl/controllers/data_parallel.py`; existing
files untouched — the packing rule):

    fit_async_stream_dp(engines: list, ...):
      pool = ThreadPoolExecutor(max_workers=len(engines))   # 1 worker PER engine
      futures = [pool.submit(collect_shared, eng, dealer, tally) for eng in engines]
      join ALL futures -> merge kept groups -> RE-STAMP group_ids 0..target-1
        (collectors cannot number globally; advantage math only needs
        uniqueness, so renumbering at merge is the whole fix)
      make_batch / fit_batch UNCHANGED
      publish: join all -> drain ALL engines -> load_weights into each
        (sequential loads first; per-engine staggered updates are the
        article's +39% "decoupled actors" — a later optimization, not v1)

`_collect_one` (nee collect_shared) is a small variant of
`collect_groups_stream` living in the DP file (need comes from the tally,
kept goes to the shared list) — a NEW function rather than optional
parameters threaded through the existing collector, per the readability
rule. Leftovers/stash per engine and the staleness bound
(publish_interval + 1) carry over unchanged: every engine drains at every
publish, so completions stay single-version.

REFINEMENT FOUND AT IMPLEMENTATION (2026-07-14) — the burst cap: each deal
is bounded by `ceil(target/k)` minus the engine's own in-flight count.
Uncapped, "deals itself `need` prompts" lets whichever collector's thread
runs FIRST deal itself the entire target at t=0 (need starts at target),
pinning every prompt to one engine before the others wake — dealt work
cannot migrate, so the promised 22/17/14/11 emergence would collapse to
64/0/0/0. The cap is the late-binding half of "deal, don't partition":
commit at most a fair burst, leave the rest claimable. Replacement deals
after drops are what fast engines then win (pinned by the paced-engine
load-balance test). Also decided at landing: streaming.py is KEPT as the
readable k=1 teaching case, not merged away (rationale in the DP banner).

Open question for a 30-minute spike ON THE BOX (the controller design is
duck-typed and agnostic to the answer): how the k engines are instantiated —
(a) k `VLLMEngine`s in one process (V1's EngineCore-as-subprocess should
isolate them; per-engine device pinning to verify), (b) vLLM's native
`data_parallel_size` (serving-oriented; fit for our offline loop unclear),
or (c) one engine per OS process, slime's server layout (most robust, most
plumbing). Start with (a); fall back to (c).

Testing (all CPU, no GPUs needed — the point of the fake-engine seam):
k FakeStreamEngines with DIFFERENT finish speeds sharing a dealer + tally:
no prompt dealt twice; global target met exactly; the fast engine dealt
strictly more prompts than the slow one (load balance observed, not assumed);
group_ids unique after restamp; publish drains all k; staleness bound holds.

## 11. The two-controller consolidation + single-node placement (2026-07-14)

DECISION: the repo keeps exactly TWO training drivers — `controllers/sync.py`
(collect -> train -> publish, no overlap; NOT YET BUILT) and
`controllers/fully_async.py` (`fit_async`: everything §1-§10 built, in one
loop). `round_based.py` and `streaming.py` are RETIRED: both were k=1 special
cases of the DP loop, and three near-identical pipeline files cost more
readability than the one they teach. What each contributed survives:

- round_based's engine story (HFEngine, generate()-only, the MPS dev path)
  survives via `engine/stream_adapter.py`: StreamAdapter wraps any
  generate()-engine with the streaming interface — one poll() == one ROUND
  (everything submitted since the last poll generates as a single blocking
  batch). Under the same collector, continuous batching degenerates to
  round-based collection: tier 1 is now a property of the ENGINE, not a
  controller file. HFEngine itself is untouched (additive-files rule, §0).
- streaming.py's fit_async_stream was fully_async minus dealer/tally; its
  walkthrough comments moved into fully_async.py. rollout/streaming.py's
  collect_groups_stream went with it (`_collect_one` in the DP file is that
  function against the shared tally). rollout/sampling.py (round-based
  collect_groups) went too, in a follow-up the same day: the sync controller
  will ALSO collect via collect_groups_dp — continuous batching is how
  collection works regardless of sync/async, and StreamAdapter covers
  generate()-only engines. Its filters (reward_nonzero_std, GroupFilter,
  RewardFn) moved to rollout/filtering.py; CollectConfig lost the
  round-only knob oversample_batch_size. The plain Trainer STAYS: DistTrainer
  subclasses it (one trainer + a sharding override, not two trainers), and
  world=1 (MPS/dev) must not pay dist init or FSDP2 wrapping.

### Placement: how slime specifies GPUs, and our translation

slime (utils/arguments.py + sglang_engine.get_base_gpu_id): training and
inference GPUs are DISJOINT by default — `--actor-num-gpus-per-node` trainer
ranks first, then `--rollout-num-gpus` engine GPUs starting at
`num_actor_gpus + engine_rank * gpus_per_engine`; `--rollout-num-gpus-per-
engine` is TP width (our permanent 1 — DP only); `--colocate` shares GPUs but
drags the whole offload/onload dance (release/resume_memory_occupation) —
production_gap, not study material.

miniRL: `PlacementConfig(num_train_gpus, num_rollout_gpus)` in config.py —
`train_gpu_ids` = 0..t-1, `rollout_gpu_ids` = t..t+r-1 (slime's layout), and
`VLLMEngine(gpu_id=...)` pins one engine to one GPU by setting
CUDA_VISIBLE_DEVICES around engine construction (the §10(a) mechanism: the V1
EngineCore SUBPROCESS inherits the env; the parent's value is restored after).
CAVEATS, on-box validation pending: construct the learner / touch
torch.cuda BEFORE the engines (CUDA reads the env once at context creation),
and construct engines sequentially (env mutation is process-global). Fallback
if the spike disproves inheritance: one OS process per engine (§10(c)).
No colocate mode: on the Mac, MPS timeshare with no machinery IS the
degenerate case; on a box, give the engine its own GPU.

### One controller, 1..m trainer ranks (folds in fsdp2.md §8)

`fit_async(engines, trainer, ...)` is called identically on every rank
(torchrun runs the same recipe; configs must agree across ranks):

    rank 0     owns the engines + collection thread exactly as §10; after
               make_batch, broadcast_object_list ships the Batch (CPU tensors)
               to all ranks; publishes via full_state_dict() at intervals.
    rank > 0   a small follower loop: receive batch -> fit_batch -> at
               publish iterations, participate in the full_state_dict gather
               (a collective: every rank must call it) and discard the result.
               Followers pass engines=[].

world == 1 (plain Trainer, Mac path): no dist init, no broadcast, byte-for-
byte the §10 loop. world > 1 requires DistTrainer (full-batch-identical
semantics: docs/fsdp2.md §2) and B % world == 0.

Async wants >= 2 devices to actually overlap (slime REQUIRES disjoint GPUs
unless colocate); on one device the loop still runs correctly — generation
and training just timeshare (the Mac smoke reality since tier 1).
