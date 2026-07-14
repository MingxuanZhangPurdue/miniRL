# Making RL fast: continuous batching + in-flight updates

Status: DESIGNED, NOT BUILT — the throughput layer for the vLLM/CUDA phase.
Nothing here changes a single loss/advantage/trainer line; it is all engine
and controller plumbing. Companion to **docs/async_training.md** (this doc
expands its tier-2 section into a concrete plan) and **docs/packing.md** (the
trainer-side counterpart of the same idea).

Grounding: "Making RL Fast" (Finbarr Timbers, https://finbarr.ca/making-rl-fast/
— the Olmo 3 / open-instruct numbers below), PipelineRL (arXiv:2509.19128),
slime `rollout/fully_async_rollout.py` (the abort/requeue alternative), vLLM
V1 docs (LLMEngine.add_request/step; AsyncLLM).

## 1. The two waits (the mental model)

Async RL has exactly two places where hardware idles, and they need different
fixes. Timelines (x = time; `▒` = idle/drain; v1/v2 = weight versions):

    ROUND-BASED ASYNC (our tier-1 today) — both waits
      engine   [ generate rollout k+1 under v1 ............ ]│[ v2 ....
      trainer  [ train k ]▒▒▒▒▒▒▒ wait 1: no data ▒▒▒▒▒▒▒▒▒▒│
                          └─ wait 2: v2 ready, HELD until engine finishes ─┘
                                                     publish ┘

    + CONTINUOUS BATCHING / STREAMING — wait 1 fixed
      engine   [ always full ..........................]▒drain▒│[ v2 ....
      trainer  [ train ][ train ][ train ]▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
                                          wait 2 remains: drain, then publish

    + IN-FLIGHT UPDATES — wait 2 fixed
      engine   [ v1 .................│ v2 ......................... ]
      trainer  [ train ][ train ][ train ][ train ][ train ][ train ]
                          publish ──┘ between two decode steps; no drain

- **Wait 1 — trainer starves for data.** The pipeline bubble
  `|t_generate - t_train|`. Fixed by continuous batching + streaming
  collection (finished groups flow into a queue; the trainer never waits for
  a round boundary).
- **Wait 2 — finished weights are held hostage.** The trainer has v2 in hand
  but the engine is mid-generation. Without in-flight updates the options
  form a triangle, all bad: **drain** (stop admitting, wait for every
  in-flight completion — minutes at 14k-token lengths, paid per publish, with
  throughput decaying as slots empty), **abort** (slime tier-2: kill requests,
  requeue/resume — no idle but tokens thrown away), or **don't publish**
  (free, but everything generated meanwhile is stale). In-flight updates
  delete the triangle.

Tier-1 note: in `fit_async`, wait 2 is deliberately *implemented as* wait 1 —
the held-future join makes publish wait for the whole rollout. That buys the
invariant "one completion == one policy version", which keeps the loop
testable. This doc is the plan for giving that invariant up, deliberately,
behind a flag.

## 2. Why bother (the article's measurements)

Two 8xH100 nodes (trainer / actors separated), Qwen 2.5 7B, 32k max tokens:

| stage | tok/s | gain | MFU |
|---|---|---|---|
| static baseline | 881 | — | 0.30% |
| + continuous batching | 975 | +11% | 0.33% |
| + decoupled actor threading | 1358 | +39% | 0.46% |
| + in-flight updates (PipelineRL) | 2949 | +117% | 1.01% |

~3.3x total. Two lessons: the gains scale with **generation-length variance**
(Olmo 3: mean 14k / max 32k => 54% of static-batch decode wasted), and even
the optimized endpoint runs at 1% MFU — rollout-side RL is memory-bound, so
tokens/sec (= `response_tokens / t_generate`, already derivable from our
metrics) is the number to watch, not FLOPs.

## 3. Continuous batching (fixes wait 1)

Mental model: there is no "batch" — the GPU runs one decode step at a time,
and the batch is just *whichever sequences are alive at that step*. Each
sequence's whole state is its own KV cache; attention never crosses rows.
Static batching freezes the guest list until everyone finishes (stragglers
hold slots, short rows waste decode steps as pads — `HFEngine` today);
continuous batching re-decides the guest list every step:

    slot1  A A A A A A A A A          finished B's slot is refilled with E
    slot2  B B B E E E E E E          on the NEXT step; the straggler A just
    slot3  C C C C C C C C H…         keeps riding while neighbors turn over
    slot4  D D G G G G F F F…

The machinery that makes the refill legal — paged KV (freed blocks return to
a pool), prefill/decode interleaving in the scheduler — **is vLLM's core
design; we implement none of it.** Every vLLM API path continuously batches
the requests it has been given. What remains OURS is keeping it fed:

- `collect_groups` today is round-based — submit a round, wait for ALL of it,
  filter, repeat. The engine drains at every round boundary even though it
  batches continuously within the round.
- The streaming upgrade: submit each group's G requests individually; as each
  group's last member lands, reward + `group_filter` it immediately; if
  dropped, push a replacement prompt *that instant* (its prefill starts next
  step in the freed slots); stop at `target_groups`. Dynamic sampling gets
  cheaper, not more expensive — slime does exactly this per-finished-group.

vLLM interface: the batch call `llm.generate()` gets within-call continuous
batching for free (stage 0 below). Streaming needs an incremental interface —
`LLMEngine.add_request()/step()` (present in V1, documented, but labeled a
legacy compatibility wrapper) or the V1-native `AsyncLLM` (asyncio; if ever
needed, contain the event loop inside the engine's rollout thread feeding a
plain queue — the controller stays asyncio-free).

Relation to **packing** (docs/packing.md): same enemy (length variance),
different pipeline stage, zero overlap. Packing removes pad FLOPs from the
trainer's rectangular (B, T) forward/backward (waste = `frac_padding`);
continuous batching removes idle slots from the engine's decode loop (waste =
straggler-dominated `t_generate`). You can't pack what is still growing, and
the trainer has no scheduling problem. Inside one vLLM step they literally
compose: the scheduler picks the step's guest list (continuous batching) and
computes it as one dense varlen token stream (packing). Do both; engine side
pays first (rollouts are 5-14x the training compute in reasoning RL).

## 4. In-flight updates (fixes wait 2) — PipelineRL

Between any two decode steps the engine is naturally quiescent. In-flight
updating puts the weight swap there: finish step k under v1, load v2, run
step k+1 — no request aborted, no slot emptied, new prompts keep streaming.

The counterintuitive part: **the KV cache is kept.** Everything cached so far
was computed under v1; v2 attends to v1's keys/values. Justification: one
optimizer step moves weights slightly, and a network is a composition of
continuous functions — slightly-stale KV gives slightly-off activations, a
bias PipelineRL and Olmo 3 measured to be harmless. It bought the +117%.

Why it is legal for the LOSS (the part miniRL already built): a completion now
straddles versions — early tokens sampled by v1, late by v2. But correction
was always per-token: each token's `behavior_logprob` is recorded at the
moment it is sampled, under whichever weights produced it, and TIS reweights
token-by-token (`w_t = clamp(exp(log pi_old - log pi_engine), lo, hi)`,
algos/tis.py). A mixed-policy completion is just a completion whose per-token
behavior logprobs come from two versions — the loss never sees the
difference. **Zero changes to losses/advantage/aggregate/trainer.**

What actually changes is bookkeeping: `Trajectory.version` (one scalar) is no
longer well-defined; it becomes the *finishing* version, with
`meta["version_start"]` recording where the request began, and the staleness
metric/assert becomes a range check.

## 5. Minimal implementation plan (single node, many GPUs)

> The concrete file-by-file implementation design for stages 1-2 (the
> step-loop engine, streaming collection, the publish mailbox, version
> bookkeeping, and the Mac-testable seams) is **docs/async_tier2.md**.

Load-bearing design decision: build `VLLMEngine` around an explicit step loop
(or AsyncLLM-in-a-thread), because owning the loop is what makes all three
optimizations cheap — finished groups stream out between steps (opt 1), and
weights swap in between steps (opt 3).

- **Stage 0 — VLLMEngine, batch-call.** `generate(prompts, params)` via
  `llm.generate()`; satisfies the existing engine duck-type unchanged;
  inherits within-round continuous batching free. Probably 80% of the win.
- **Stage 1 — streaming.** `generate_stream(prompt_iter, params) ->
  Iterator[group]` on the engine + the streaming path in `collect_groups`
  (per-finished-group filter + top-up). `HFEngine` keeps the round-based path
  (HF generate cannot continuously batch; it stays the didactic engine).
- **Stage 2 — in-flight updates. DEFERRED (2026-07-10 decision,
  docs/async_tier2.md §4)**: the only semantics-changing stage (mixed-policy
  completions, kept KV), and its payoff scales with completion length — at
  our scales drain-then-publish amortized by publish_interval is cheap.
  Revisit when long-CoT runs make the drain measurably expensive.
- **Stage 3 — decoupled engine replicas** (the article's +39%; only worth it
  with >1 actor GPU): `engines: list`, one rollout thread each, one shared
  result queue, per-engine weight updates with no global barrier. Mixed
  versions across a batch are already fine (per-token TIS; per-row staleness
  bound). Skip the article's prefetch threads — `prompt_source` is a
  tokenizer call, not a bottleneck (principle 8).

Expectations: all of this is GPU-box work — on the MPS smoke recipe (512
tokens, tiny batches) every one of these is noise. Add `tokens/sec` to the
wandb metrics from day one to reproduce the article's measurement.

## 6. Name mapping (Rosetta stone, throughput slice)

| here | article / open-instruct | slime | notes |
|---|---|---|---|
| streaming `collect_groups` | continuous batching (opt 1) | fully_async per-finished-group filtering | engine-internal CB is vLLM's; this is the feeding side |
| engine replicas + per-engine publish | decoupled actor threading (opt 2) | multiple rollout engines / Ray actors | stage 3, needs >1 actor GPU |
| `inflight_updates=True` mailbox | in-flight updates (opt 3) = PipelineRL | closest: --partial-rollout (abort/resume + mask) | we keep KV (PipelineRL); slime aborts/resumes instead |
| `version_start`/`version_end` | staleness budget | sample version stamps | scalar version is tier-1-only |
| `pack_batch` (docs/packing.md) | — (trainer-side, orthogonal) | THD packing + cu_seqlens | same enemy, different stage |
