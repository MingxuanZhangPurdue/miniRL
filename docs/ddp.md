# DDP learner: data-parallel training design

Status: design + implementation together (2026-07-15). Pure PyTorch
(`torch.nn.parallel.DistributedDataParallel`). REPLACES the FSDP2 learner —
decision 2026-07-15: this repo's scope is models that TRAIN on one GPU
(single node, DP only), so replicated parameters are all we need and the
trainer should be as clean as DP allows. FSDP2's design + empirical findings
are kept for history in docs/fsdp2.md (banner there points here).

## 1. What DDP changes vs FSDP2 (and what it doesn't)

Both are DATA parallelism — every rank trains on different rows, gradients
combine. The difference is where parameters live:

    DDP    every GPU holds FULL params + full AdamW states; gradients
           all-reduce once per step. Simple, fast at our scale.
    FSDP2  each GPU holds 1/world of params/grads/optimizer states;
           per-layer all-gathers during compute. Pays comms for memory
           we don't need at <=1-GPU-trainable model sizes.

Unchanged by the swap — the parts that were never about sharding:
- **The math law (the one real decision, was fsdp2.md §2):** every rank
  receives the IDENTICAL full batch; the loss denominator comes from the
  FULL minibatch mask BEFORE the rank slices its rows (`rows[rank::world]`),
  so it is minibatch-global across the world with zero communication, and
  gradients must combine as a SUM:
      SUM_r grad( sum(rows_r) / denom ) == grad( sum(ALL rows) / denom )
- Seeded minibatch shuffle agreement across ranks; B % world == 0 asserted.
- Every rank recomputes old_logprobs on the full batch (duplicated compute,
  zero communication — a rank may touch any row across ppo_epochs shuffles).

## 2. SUM semantics under DDP (simpler than FSDP2's)

DDP AVERAGES gradients across ranks (divides by world). The fix is the same
loss-scale identity the gloo fallback already used, now the ONLY mechanism
(backend-agnostic — the NCCL-only PREMUL_SUM story dies with FSDP2):

    scale each rank's loss by world  ->  mean-reduce of world-scaled grads
                                          == sum of unscaled grads

Reported loss divides the scale back out (log the true value, not the trick).

## 3. Gradient accumulation: no_sync

`ddp.no_sync()` wraps every microbatch backward except the last, so
gradients accumulate locally and exactly ONE all-reduce fires per optimizer
step — the standard DDP pattern, and the direct analog of FSDP2's
`set_requires_gradient_sync(last)`. Forwards go through the DDP wrapper
(its autograd hooks do the reduction); `DistTrainer.model` stays the RAW
module, so state_dict names are clean and inherited no-grad code
(compute_logprobs) skips the wrapper entirely.

## 4. Publish path: no collective at all

Parameters are REPLICATED, so rank 0's plain `model.state_dict()` IS the
full weights — `full_state_dict()` is now a local CPU copy, not a gather.
Consequences for the controller (fully_async.py):
- followers do NOT participate in publishes anymore; the follower loop is
  literally broadcast -> fit_batch, nothing else;
- the collective-mismatch deadlock class from the FSDP2 wiring (followers
  must join every gather) shrinks to one rule: every rank calls fit_batch
  the same number of times, which the broadcast already enforces.

## 5. Precision

The model trains in whatever dtype it was loaded (fp32 on MPS/CPU — the dev
path; recipes decide for CUDA). The fp32 islands (gather_logprobs upcast,
aggregation, fp32 AdamW states) hold regardless. bf16 autocast on the CUDA
box is a later, measured, one-line recipe knob — not built until a profile
asks (same rule as torch.compile).

## 6. Testing (the invariance standard, CPU-only, unchanged)

tests/test_distributed.py, 2 gloo processes: 2-rank DistTrainer.fit_batch ==
single-process Trainer.fit_batch (identical parameters) for seq_mean /
token_mean / int-C — pins denominator globalization, the loss-scale SUM
identity, shuffle agreement, and no_sync accumulation at once. Plus the
ragged-divisibility assert and the full_state_dict round-trip. The
controller-level 2-rank test (tests/test_fully_async.py) rides on top.

Note for hand-rolled test models: the FSDP2 pytree trap (fsdp2.md §7) is
GONE — DDP hooks parameters, not forward outputs, so output container type
no longer matters.
