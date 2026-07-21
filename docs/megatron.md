# Megatron trainer: replace our Trainer with Megatron-Core

Status: design (2026-07-20), decision made — the hand-written DDP Trainer
(`minirl/train/trainer.py`, history in docs/ddp.md) will be REPLACED by
Megatron-Core as the training engine. Rationale: after mirroring slime's
conventions piece by piece (global denominators, fp32 masters, NaN guard,
microbatch accumulation), the trainer is a re-implementation of things
Megatron already owns. Dropping it re-scopes the repo to what this project
is actually about:

    OURS                                THEIRS (Megatron-Core)
    loss functions / algorithms         forward-backward scheduling
    fully-async RL controller           DDP + grad reduction (fp32 native)
    rollout engines + weight publish    distributed optimizer, fp32 masters
    datasets / prompts / rewards        fused kernels, activation ckpt
    agentic runner, eval                TP / PP / CP / EP when ever needed

We will run **DP-only** (tp=pp=cp=ep=1) for the foreseeable future, but we
go through Megatron's real interfaces so that parallelism is a config
change, not a rewrite. This is the same bet slime makes; their integration
(slime/backends/megatron_utils/) is the reference implementation for
everything below, and Megatron-LM is cloned at ../Megatron-LM.

NVIDIA now ships its own RL stack inside Megatron-LM (megatron/rl/: agents,
rollout inference, servers). We are NOT adopting it — the async pipeline is
the part of this repo that exists to be understood — but it is prior art
worth reading.

## 1. The contract: Megatron is an engine with a callback slot

The entire integration surface is one function shape. You give Megatron a
`forward_step_func`; it gives you scheduling, accumulation, and reduction:

    forward_step(data_iterator, model) -> (output_tensor, loss_func)

Megatron calls it `num_microbatches` times inside
`forward_backward_func(...)` (for us always the no-pipelining schedule,
Megatron-LM/megatron/core/pipeline_parallel/schedules.py:672), invokes our
`loss_func` on each output, scales and accumulates gradients, and returns
the per-microbatch metric dicts. The loss is 100% ours — same math as
`minirl/algos/`, called from a different socket. slime's version of this
file is megatron_utils/loss.py::loss_function (their GRPO/PPO/SFT dispatch);
ours will be a thin adapter that calls the UNCHANGED `LOSSES` entries.

The minimal pure-core loop (no `megatron.training`, no giant args
namespace) is demonstrated by NVIDIA themselves in
Megatron-LM/examples/run_simple_mcore_train_loop.py. Everything we need:

    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel(1, 1)      # tp=1, pp=1 -> DP-only
    model = <from Megatron-Bridge, §2>
    model = mcore DistributedDataParallel(config, DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, ...), model)  # fp32 reduce: native knob
    optimizer = get_megatron_optimizer(OptimizerConfig(
                bf16=True, use_distributed_optimizer=..., lr=..., clip_grad=...), [model])
    fwd_bwd = get_forward_backward_func()
    # per step: fwd_bwd(forward_step, iterator, num_microbatches=...) ->
    #           finalize_model_grads -> optimizer.step()

Note what this buys over our Trainer at zero code: `grad_reduce_in_fp32`
(the one "documented deviation from Megatron" our DDP trainer carried),
bf16 params + fp32 main params inside the optimizer (our `bf16_weights`
masters, now theirs), found-inf NaN guard (`optimizer.prepare_grads()` —
slime megatron_utils/model.py wraps it exactly like our skip guard), and
optimizer-state sharding across DP ranks (`use_distributed_optimizer`,
ZeRO-1-style — strictly more memory headroom than our replicated AdamW).

## 2. Checkpoints: Megatron-Bridge, both directions

The historical Megatron tax — hand-written per-model conversion scripts —
is gone. **Megatron-Bridge** (`megatron.bridge`, NVIDIA; slime pins a fork,
see §6) converts directly from the HF hub name:

    IN   bridge = AutoBridge.from_hf_pretrained(name_or_path)
         provider = bridge.to_megatron_provider()       # -> mcore GPTModel,
         (slime: megatron_utils/model_provider.py:88)   #    HF weights loaded
    OUT  bridge.export_hf_weights(model, cpu=True)      # HF-NAMED tensors
         (slime: update_weight/hf_weight_iterator_bridge.py)

The OUT direction is the load-bearing one for us: `export_hf_weights`
yields exactly the `(hf_name, tensor)` stream our engines already consume —
`VLLMEngine.load_weights` (safetensors file -> native `reload_weights` RPC)
does not change AT ALL. Publish stays rank-0-local under DP-only: model
params are replicated (only optimizer state shards), so rank 0's export is
the full weights, same as today. Megatron's fused-QKV / gate-up layouts and
vocab padding are the bridge's problem, not ours.

Model coverage gate: an architecture must exist in BOTH mcore and the
bridge. Qwen3 (dense) is long-supported; the Qwen3-Next/Qwen3.5 GDN hybrid
has mcore support (megatron/core/ssm/gated_delta_net.py) and slime run
scripts, needing `flash-linear-attention` installed. New/exotic HF models
lag here — that is the real price of leaving `transformers` as the model
zoo, and the reason the HFEngine-style "any HF model" property dies with
this migration.

## 3. Logprobs: labels-in, loss-map-out (fused CE)

GPTModel has two output modes:

    model(tokens, position_ids, mask)                -> logits (b, T, V)
    model(tokens, position_ids, mask, labels=lbl)    -> per-token CE (b, T) fp32

The second is a fused vocab-parallel cross-entropy: `-log p(lbl[t] | <=t)`
computed WITHOUT materializing fp32 (b, T, V) logits (slime's
`vocab_parallel_logits.float()` upcast — our gather_logprobs fp32 invariant
— lives inside it). So `gather_logprobs` is replaced by calling the model
with `labels = input_ids shifted left by one` and negating; the alignment
convention shifts by one position vs ours (Megatron: out[t] scores token
t+1; ours: out[t] scores token t) — ONE adapter function owns that shift
and its parity test, nothing else may touch it.

`old_logprobs` recompute = the same `forward_backward_func` with
`forward_only=True` (slime megatron_utils/model.py:479). π_old and π_θ run
identical kernels by construction — the clip-band property holds for free.

Because the loss plugs in at the labels/loss-map level, TP would shard the
CE without our loss code noticing — the door §0 promises. Algorithms that
need FULL logits (entropy bonuses, distillation) are the exception; none of
our current LOSSES do.

## 4. What changes in the minirl tree

    minirl/train/                DELETED entirely (2026-07-20): trainer.py
                                 demoted to tests/fake_trainer.py (the
                                 executable spec of the contract), and the
                                 one remaining training file flattened out
                                 of the folder. The trainer duck-type:
                                   fit_batch(batch) -> metrics
                                   compute_logprobs(batch) -> (B, T) f32
                                   hf_named_tensors() -> iterable (publish)
                                   rank / world / loss_cfg
    minirl/megatron.py           NEW (built 2026-07-20), the whole
                                 integration: init (§1), bridge model build
                                 (§2), forward_step + loss adapter (§3),
                                 fit_batch loop (minibatch shuffle stays
                                 OURS — Megatron only sees one minibatch =
                                 one forward_backward_func call with
                                 num_microbatches splits), publish export.
                                 Also home of setup_distributed.
    minirl/algos/*               UNCHANGED — still (B, T) unreduced maps,
                                 aggregate.py denominators now feed the
                                 loss_func adapter's normalizer (slime's
                                 loss rescale convention, loss.py:1220).
    minirl/rollout/batching.py   make_batch stays; iter_microbatches becomes
                                 the data_iterator handed to Megatron.
    controllers/fully_async.py   UNCHANGED in shape: rank 0 collects,
                                 broadcasts the Batch, every rank calls
                                 fit_batch, rank 0 publishes. Only the
                                 trainer construction site moves.
    engines / rewards / data     UNTOUCHED.

Right-padded rectangles stay valid: with causal-only masking (mask=None)
pad tokens sit at row ends, causality means real positions never attend to
them, and the loss mask zeroes their contribution — same argument as today.
Position ids are explicit in mcore (`arange(T)` per row). Sequence packing
(cu_seqlens) remains a later, optional rung, as before.

## 5. Platform reality: Megatron is CUDA-box-only (measured 2026-07-20)

megatron-core 0.18.0 CANNOT import on this Mac: `pipeline_parallel/
schedules.py` unconditionally imports triton (via transformer/moe/
paged_stash.py), triton has no macOS build, and stubbing it destabilizes
torch itself (dynamo/inductor probe `triton.language.dtype`,
`triton.backends.compiler`, ... — measured, not guessed). Consequences:

- The Megatron path is developed and validated ON THE BOX ONLY, through
  box_runbook-style rungs (§7).
- The local test suite keeps a pure-torch FAKE trainer implementing the §4
  duck-type — today's Trainer, demoted to tests/, becomes that fake and the
  executable spec of the contract. Controller/algo/data tests keep running
  on Mac CPU in seconds; trainer-math tests (masters, denominators) stay
  meaningful against the fake; Megatron parity is asserted on-box (§7 P1).

## 6. Environment (box) — slime's stack as reference (docker/Dockerfile)

    megatron-core            pinned (slime: git commit + patch; us: start
                             with the pip release used by the box spike)
    Megatron-Bridge          slime pins radixark/Megatron-Bridge@bridge
                             (--no-deps); try upstream NVIDIA first
    transformer-engine       2.16.1 — OPTIONAL for us at first: use
                             get_gpt_layer_local_spec() (plain-torch
                             modules); TE spec is a later perf rung
    flash-attn               2.8.3 (full-attention layers)
    flash-linear-attention   0.4.2 (only for GDN hybrids: Qwen3.5/-Next)
    numpy                    <2 (Megatron requirement)

Version skew is THE failure mode of this stack (slime carries patch files
against pinned commits). Rule: the box environment is recorded in the
runbook the day P1 passes, and upgrades are their own runbook entries.

## 7. Migration ladder

    P0  this doc.                                                 [done]
    P1  box spike, single GPU: bridge-load Qwen3-0.6B -> one GRPO
        minibatch step. EXIT CRITERION: loss + grad_norm match the
        demoted DDP trainer (tests/fake_trainer.py) on the same frozen
        batch (tolerances per precision doc), and logprob-shift parity
        (_ce_to_logprobs vs gather_logprobs) passes.
    P2  minirl/megatron.py + controller wiring + recipe 05 on Megatron;
        trainer.py demoted to tests/fake_trainer.py.   [code built
        2026-07-20 — UNVALIDATED until P1 runs on the box; written
        against pinned source, banner lists the 3 conventions to check]
    P3  DP>1 on the box (mcore DDP + distributed optimizer), publish
        gather sanity, wandb metrics parity with today's 05 recipe.
    P4  perf rungs, measured one at a time: TE layer spec, fused adam,
        grad_reduce_in_fp32 A/B, optional packing.
    P5  (door, not plan) tp=2 experiment to prove the config-only claim.

Open questions parked until P1: exact megatron-core/bridge version pair
that loads Qwen3-0.6B cleanly; whether bridge upstream suffices or slime's
fork is load-bearing; dist_checkpointing vs bridge HF export for training
resume (leaning bridge-export — HF-format checkpoints keep eval/serving
trivial, matching slime's hf_checkpoint_saver).

## 8. How slime drives Megatron (reference walkthrough, ../slime)

slime's whole Megatron integration lives in
`slime/backends/megatron_utils/`; the call chain from job entry to mcore:

    train.py::train(args)                       Ray DRIVER, one per job:
      create_placement_groups(args)             GPU split (their PlacementConfig)
      create_rollout_manager(...)               SGLang engines (our engine fleet)
      create_training_models(...)               one MegatronTrainRayActor per rank
      loop over rollout_id:                     THE outer loop (our fit_async)
        rollout_manager.generate.remote()       -> rollout_data_ref (object store)
        actor_model.async_train(id, ref)        (train.py:69 -> actor_group.py:142)
        actor_model.update_weights()            publish to SGLang

    MegatronTrainRayActor (megatron_utils/actor.py)  one per training rank:
      init (actor.py:47)
        initialize.py::init(args)               set_args -> mpu.initialize_model_
                                                parallel -> seeds -> microbatch calc
        initialize_model_and_optimizer          model.py:968 -> setup_model_and_
                                                optimizer (model.py:270): bridge
                                                provider -> mcore DDP wrap ->
                                                get_megatron_optimizer + scheduler
                                                -> checkpoint load
      train (actor.py:352) -> train_actor (actor.py:402), per rollout:
        get_data_iterator(rollout_data)         data.py — packing + DP balancing
        compute_log_prob(...)                   model.py::forward_only (345):
                                                fwd-only pipeline pass filling
                                                log_probs / ref_* / teacher_*
                                                (our old_logprobs recompute; ref
                                                model = SAME weights hot-swapped
                                                from a CPU backup, not a 2nd copy)
        compute_advantages_and_returns          whole-rollout, before any step
        model.py::train (704) per step:
          train_one_step (model.py:509):
            zero_grad_buffer + optimizer.zero_grad
            forward_step closure                get_batch -> GPTModel fwd ->
                                                partial(loss_function, ...)
                                                (loss.py:1220 — THE loss socket)
            forward_backward_func(...)          mcore schedules (model.py:642)
            optimizer.prepare_grads()           found_inf -> skip (our NaN guard)
            optimizer.step, scheduler step
        weights_backuper.backup("actor")        CPU snapshot (ref/old switching)
      update_weights (actor.py:555)             update_weight/*.py — bridge
                                                export_hf_weights -> SGLang via
                                                tensor / distributed / disk

    What we ADOPT from this: the init sequence, setup_model_and_optimizer
    shape, forward_only for logprob recompute, the train_one_step skeleton
    (zero -> closure -> fwd_bwd -> prepare_grads guard -> step), the loss
    socket, bridge export on publish.
    What we SKIP (scale features our two-controller design replaces or
    doesn't need): Ray actors + object-store scatter (we broadcast a Batch),
    GPU offload / sleep-wake (torch_memory_saver — colocated-mode feature),
    the multi-model tag switching (ref / old_actor / teacher backups; we
    have no KL-to-ref or distillation losses yet), critic training, MoE
    routing replay, and megatron.training's global args (we go pure-core).
