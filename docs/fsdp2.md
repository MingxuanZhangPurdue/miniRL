# FSDP2 learner: data-parallel training design

**RETIRED 2026-07-15, REPLACED BY DDP (docs/ddp.md).** Decision: the repo's
scope is models that train on ONE GPU (single node, DP only), so replicated
parameters suffice and the DDP trainer is cleaner — no DTensor state-dict
gathers, no PREMUL_SUM backend split, no publish-side collectives (followers
no longer join gathers). What carries over unchanged: §2's cross-rank
denominator law and the SUM-gradient semantics (now always via the
loss-scale identity), the sync-on-last accumulation shape (DDP no_sync), and
the equivalence-test standard. §7's empirical findings (the pytree trap, the
PREMUL_SUM/gloo crash) are FSDP2-specific — kept below as history; the
pytree trap does not apply to DDP (it hooks parameters, not outputs).

Original design note follows (historical).

Status: design + implementation together (2026-07-13). Pure PyTorch
(`torch.distributed.fsdp.fully_shard`, the DTensor rewrite — in core torch,
no external dependency; principle 1 holds). The MATH is validated on this
Mac: FSDP2 runs on CPU with the gloo backend, so the equivalence tests spawn
2 local processes — CUDA/NCCL on the box changes the backend string and the
mixed-precision policy, nothing else.

## 1. Scope

A data-parallel sharded LEARNER for the existing Trainer: N ranks each hold a
parameter shard (FSDP2 per-parameter sharding), each trains on a slice of the
batch's rows, gradients reduce across ranks. Single node. NOT in scope here:
TP/PP (non-goals), the rank-0-engine controller wiring (documented in §7,
built with the GPU recipes), NCCL weight publish (weight_sync.py, later —
the engine publish path today is `full_state_dict()` + the existing
`load_weights`).

Why FSDP2 and not DDP, at 0.6B where DDP would do: sharding is the part of
the production stack worth learning (slime's Megatron and verl's FSDP workers
both shard), the DTensor state-dict story is cleaner than DDP's
module-prefix hacks, and the same four lines scale to models where DDP
cannot follow. (At world=2 on a 0.6B model FSDP2 ~= DDP in behavior; fine —
the equivalence tests don't care.)

## 2. The one real math decision: the cross-rank denominator

The repo's normalization law says the loss denominator is minibatch-GLOBAL
(aggregate.py). Under data parallelism "the minibatch" means the WHOLE
WORLD'S batch, not a rank's slice — a per-rank denominator would silently
reweight tokens by rank load, the same bug class as per-microbatch denoms.

Design that makes this FREE (no communication): every rank receives the
IDENTICAL full batch (trajectories are CPU tensors by contract — cheap), and
the denominator is computed from the FULL batch mask BEFORE the rank slices
its rows:

    denom   = minibatch_denom(loss_agg, FULL.loss_mask)     # same on all ranks
    local   = rows[rank::world_size]                        # then slice
    loss_r  = sum(local per-token losses) / denom
    SUM over ranks of grad(loss_r) = grad(sum over ALL rows / denom)   ✓ exact

(This corrects an older note in aggregate.py's banner that said the
denominator "gets an all-reduce" — with identical full batches it needs no
collective at all. The all-reduce variant only becomes necessary if batches
are ever sharded at collection time.)

Two consequences pinned by tests:
- gradients must SUM across ranks, not average: FSDP2's default divide-by-
  world is turned off (`set_gradient_divide_factor(1)`; fallback for older
  APIs: scale the loss by world_size — the equivalence test catches either
  way being wrong).
- minibatch SHUFFLING must agree across ranks: the seeded generator already
  guarantees it (same seed -> same permutation on every rank; slicing happens
  after).

## 3. Gradient accumulation across microbatches (the hang trap)

FSDP2 reduce-scatters gradients per backward. If ranks run DIFFERENT numbers
of microbatch backwards (ragged row split), collectives mismatch and the job
HANGS — the classic FSDP accumulation bug. The fix is also the efficient
pattern: `set_requires_gradient_sync(False)` for every microbatch except the
last, so grads accumulate locally and exactly ONE reduce-scatter fires per
optimizer step per rank. We additionally assert B % world_size == 0 (the
collection side controls B = target_groups * G; divisibility is a config
concern, not a runtime surprise).

## 4. Precision (docs/precision.md, now concrete)

CUDA: `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` — bf16
compute, fp32 gradient reduction, fp32 master/optimizer states (FSDP2 keeps
sharded fp32 params; bf16 is the communicated/computed view). The fp32
islands (gather_logprobs upcast, aggregation) are unchanged. CPU/gloo tests:
no policy, fp32 end to end.

## 5. What lands where (all new files; trainer.py untouched)

    minirl/train/distributed.py    setup_distributed() / shard_model() /
                                   DistTrainer / full_state_dict()
    tests/test_distributed.py      2-process gloo equivalence suite

`DistTrainer` SUBCLASSES the concrete Trainer and overrides step() wholesale
(~40 lines, three marked diffs: full-batch denom before slicing, local-row
microbatching with sync-on-last, DTensor-aware grad-norm). Subclassing a
concrete class for a real second implementation is not the base-class
abstraction principle 8 bans — there is no protocol, no registry, and
deleting distributed.py leaves no trace. fit_batch/compute_logprobs are
inherited unchanged (every rank recomputes old_logprobs on the full batch:
duplicated compute, zero communication — the optimization is sharded
recompute + all_gather, deferred until a profile asks for it).

`full_state_dict(model)` (torch.distributed.checkpoint.state_dict) gathers
the sharded params into one plain fp32-on-CPU dict — the SAME object
`engine.load_weights` already consumes, which is why the publish path needs
no changes: rank 0 gathers and publishes, other ranks pass.

## 6. Testing (the invariance standard, CPU-only)

torch.multiprocessing.spawn, world=2, gloo, TinyLM — same exit criterion as
microbatching and packing: **distributed and single-process must produce the
same math on identical data.**

1. equivalence: 2-rank DistTrainer.fit_batch == single-process
   Trainer.fit_batch — identical parameters after the update, for seq_mean /
   token_mean / int-C (pins denom globalization, grad-sum semantics, shuffle
   agreement, sync-on-last accumulation at once).
2. full_state_dict round-trip: gathered dict == the reference model's dict
   (shapes, values) — the publish path's contract.
3. ragged-divisibility assert fires loud for B % world != 0.

## 7. Empirical findings from the CPU/gloo bring-up (2026-07-13)

Two things the equivalence test caught that documentation alone wouldn't:

- **The pytree trap (silent no-training).** FSDP2 attaches its post-backward
  hook — the reshard + gradient hand-off to the sharded DTensors — to the
  forward's OUTPUT tensors, discovered via pytree traversal. A test model
  returning `SimpleNamespace(logits=...)` is invisible to pytree: the hook
  never attaches, gradients land on the temporary unsharded params, and the
  optimizer's DTensors silently never update — training no-ops with no
  error. Real HF models are safe (`ModelOutput` is pytree-registered by
  transformers); hand-rolled test models must return a NamedTuple. Pinned by
  a comment on tests' TinyOut.
- **`set_gradient_divide_factor(1)` is NCCL-only in practice**: it lowers to
  `ReduceOp.PREMUL_SUM`, which gloo does not implement (runtime crash in
  reduce_scatter). On gloo the SUM semantics come from the loss-scale
  fallback instead (§2: scale each rank's loss by world; mean-reduce of
  scaled grads == sum of unscaled). Both mechanisms are the same math; the
  gloo path is what the local tests pin, the NCCL path gets validated on the
  box.

## 8. Controller wiring (BUILT 2026-07-14 — inside controllers/fully_async.py)

Rank 0 runs the engines + collection exactly as before; after collation rank
0 broadcasts the Batch (torch.distributed broadcast_object_list — CPU
tensors, small at study scale); every rank calls
DistTrainer.fit_batch(full_batch); rank 0 publishes via full_state_dict()
(a gather COLLECTIVE — followers participate and discard). No separate
`fit_*_ddp` wrapper landed: the two-controller consolidation
(docs/async_tier2.md §11) folded the rank gating into `fit_async` itself —
ranks > 0 take a small follower loop; engines live on rank 0 only.
