"""THE trainer: Megatron-Core drives training; this module is the whole
integration. CUDA-box ONLY — megatron-core hard-imports triton (no macOS
build), so this MODULE only imports on the box (only the box recipes import
it) and the local test suite drives tests/fake_trainer.py (the executable
spec of the same contract) instead.

The trainer duck-type consumed by train_async.py:

    fit_batch(batch) -> metrics        compute_logprobs(batch) -> (B, T) f32
    hf_named_tensors() -> iterable     rank / world / loss_cfg

Division of labor: Megatron owns forward/backward scheduling, microbatch
accumulation, the DDP grad reduce (fp32, a config flag), the bf16 +
fp32-master optimizer, clipping and the found-inf guard. WE own the loss
(minirl/algos, unchanged), minibatch shuffling, the global-denominator
rule, and weight publish. Megatron-Bridge owns HF checkpoints both ways:
the model is built FROM the HF hub name, and export_hf_weights streams
HF-named tensors straight into the engines' load_weights. Sequence packing
is trainer-INTERNAL (cfg.pack_max_tokens): minirl/packing.py owns the dense
layout, _forward_ce runs it, and the CE map scatters back to (B, T) before
any loss code runs — losses never see the packed format.

The three integration conventions (each verified against Megatron-LM
source, pinned here so the box parity run has a checklist):
  1. loss_func returns a 2-tuple (loss, metrics) and the schedule divides
     loss by num_microbatches — we pre-multiply to keep our
     minibatch-global denominator exact (slime rescales identically).
  2. Megatron DDP pre-scales grads by 1/dp_world before the SUM reduce
     (an AVERAGE, like torch DDP) — so the identical-full-batch scheme
     carries over: every rank slices rows[rank::world] and multiplies its
     loss by dp_world, mean-of-scaled == the global SUM.
  3. GPTModel(labels=...) returns the fused-CE map: out[:, t] =
     -log p(labels[t] | <=t) with labels shifted LEFT of input_ids; our
     convention is out[:, t] = log p(input_ids[t] | <t). ONE adapter
     (_ce_to_logprobs) owns the negate+shift; nothing else may.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import local_layer_spec
from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_model_config
from torch import Tensor

from minirl.algos.aggregate import aggregate_loss, minibatch_denom
from minirl.rollout.packing import Pack, pack_rows, plan_packs, unpack_ce
from minirl.rollout.batching import iter_microbatches, iter_minibatches, slice_batch
from minirl.rollout.types import Batch


def setup_distributed(backend: str | None = None) -> tuple[int, int]:
    """Join the process group IF a launcher started us. -> (rank, world).

    torchrun / mp.spawn set RANK et al.; plain `python recipe.py` sets
    nothing and gets (0, 1) with no process group — the same script serves
    both launch modes. Call BEFORE constructing the trainer — it reads dist
    state at __init__.
    """
    if not dist.is_initialized():
        if "RANK" not in os.environ:  # no launcher: single process, no dist
            return 0, 1
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    return dist.get_rank(), dist.get_world_size()


@dataclass(frozen=True)
class MegatronTrainConfig:
    lr: float = 1e-6  # slime --lr default
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0  # OptimizerConfig.clip_grad — clipping happens INSIDE step()
    ppo_epochs: int = 1  # passes over each rollout batch (GRPO default 1)
    minibatch_size: int = 32  # sequences per optimizer step
    micro_batch_size: int = 4  # sequences per fwd/bwd (grad accumulation)
    max_skipped_steps: int = 3  # consecutive found-inf steps tolerated before crashing
    seed: int = 0  # minibatch shuffling (identical on every rank by the same seed)
    bf16: bool = True  # Megatron's one precision mode: bf16 params + fp32 masters
    grad_reduce_in_fp32: bool = True  # full-Megatron grad fidelity (the fake reduces in bf16)
    use_te_layers: bool = True  # Transformer-Engine layer spec — fused kernels + FlashAttention/cuDNN attention dispatch
    use_distributed_optimizer: bool = True  # shard optimizer states across DP (ZeRO-1 style)
    pack_max_tokens: int | None = None  # token budget per fwd/bwd: each microbatch
    #   becomes ONE dense pad-free row of whole sequences under this budget
    #   (an oversized row gets a pack of its own). Replaces micro_batch_size
    #   as the grad-accum unit; minibatch_size stays the optimizer-step batch
    #   size, so the budget affects memory/speed, never the gradient.
    #   Needs bf16 + TE (varlen attention kernels).
    logprob_pack_max_tokens: int | None = None  # larger budget for the no-grad
    #   logprob recompute (no activations stored, so bigger packs fit);
    #   None -> pack_max_tokens


class MegatronTrainer:
    """fit_batch(batch) = recompute old_logprobs, then ppo_epochs x shuffled
    minibatches, each one forward_backward_func call microbatched by Megatron.

    Construct AFTER setup_distributed() (torchrun) and torch.cuda.set_device.
    The model comes from the HF name via Megatron-Bridge — there is no
    transformers learner object anywhere in this path.
    """

    def __init__(self, model_name_or_path: str, loss_fn, loss_cfg, cfg: MegatronTrainConfig):
        assert dist.is_initialized(), "call setup_distributed() (torchrun) before MegatronTrainer"
        if cfg.pack_max_tokens is not None:
            assert cfg.bf16 and cfg.use_te_layers, "packing needs bf16 + TE varlen kernels"
        else:
            assert cfg.logprob_pack_max_tokens is None, "logprob packing needs pack_max_tokens set"
        self.loss_fn = loss_fn  # (policy_logprobs (b,T), batch, cfg) -> (loss_map (b,T), metrics)
        self.loss_cfg = loss_cfg
        self.cfg = cfg
        self.loss_agg = getattr(loss_cfg, "loss_agg", "token_mean")

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(1, 1)  # DP-only, hardwired
        model_parallel_cuda_manual_seed(cfg.seed)
        # dp coordinates by meaning (== global ones at tp=pp=1): the
        # slicing/publish logic keys off DP replicas.
        self.rank = parallel_state.get_data_parallel_rank()
        self.world = parallel_state.get_data_parallel_world_size()

        # HF checkpoint -> mcore GPTModel, weights loaded, in one call chain.
        # The bridge is KEPT: it is also the publish exporter (HF naming).
        self.bridge = AutoBridge.from_hf_pretrained(model_name_or_path)
        provider = self.bridge.to_megatron_provider(load_weights=True)  # tp=pp=1 provider defaults
        # dtype is FORCED both ways: the provider inherits the HF checkpoint's
        # dtype (bf16 for Qwen releases), so bf16=False alone would silently
        # keep bf16 params under an fp32-configured optimizer.
        provider.bf16 = cfg.bf16
        provider.fp16 = False
        provider.params_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        if not cfg.use_te_layers:
            provider.transformer_layer_spec = local_layer_spec
        provider.finalize()
        # bridge >= 0.5 providers read PP/TP roles off self._pg_collection and
        # only their (deprecated) provide_distributed_model sets it; we build
        # the model ourselves, so hand them the mpu groups we just initialized.
        provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.model = provider.provide().cuda()  # the raw GPTModel
        # to_megatron_provider(load_weights=True) only PARKS the HF->mcore
        # weight copy in a pre-wrap hook consumed by provider.get_model();
        # the explicit provide() path must run the load itself. Skipping it
        # fails SILENTLY: a zero-embedding model whose CE is exactly log|V|
        # (uniform logits), "training" from scratch.
        self.bridge.load_hf_weights([self.model])

        self.ddp = DistributedDataParallel(
            config=self.model.config,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=cfg.grad_reduce_in_fp32,
                use_distributed_optimizer=cfg.use_distributed_optimizer,
                overlap_grad_reduce=False,  # reduce once in finalize, not during backward
            ),
            module=self.model,
        )
        self.optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                adam_beta1=cfg.adam_betas[0],
                adam_beta2=cfg.adam_betas[1],
                clip_grad=cfg.max_grad_norm,
                bf16=cfg.bf16,
                use_distributed_optimizer=cfg.use_distributed_optimizer,
            ),
            [self.ddp],
        )
        # The schedule reads these off the model config: grads reduce once
        # per step (inside finalize), losses scale through the optimizer
        # (identity at bf16 — bf16 needs no loss scaling, fp16 would).
        mcfg = get_model_config(self.ddp)
        mcfg.finalize_model_grads_func = finalize_model_grads
        mcfg.grad_scale_func = self.optimizer.scale_loss
        self._fwd_bwd = get_forward_backward_func()

        self.shuffle_rng = torch.Generator().manual_seed(cfg.seed)
        self.step_count = 0
        self.consecutive_skipped = 0

    # ---------------- update phase ----------------

    def fit_batch(self, batch: Batch) -> dict:
        """One rollout's update phase. Returns metrics averaged over optimizer steps."""
        # pi_old = the learner AT UPDATE START, recomputed here — never
        # conflated with engine behavior_logprobs (which stay in the batch
        # for TIS). One no-grad pass; cheap next to the update itself.
        batch.old_logprobs = self.compute_logprobs(batch)  # (B, T) f32, no grad

        step_metrics: list[dict] = []
        for _ in range(self.cfg.ppo_epochs):
            for mb in iter_minibatches(batch, self.cfg.minibatch_size, self.shuffle_rng):
                step_metrics.append(self.step(mb))
        keys = step_metrics[0].keys()
        return {k: sum(m[k] for m in step_metrics) / len(step_metrics) for k in keys}

    def step(self, mb: Batch) -> dict:
        """One optimizer step over a minibatch; Megatron runs the microbatches.

        mb is the FULL minibatch — identical on every rank (the controller
        broadcasts whole batches). The GLOBAL denominator comes from the
        whole mask BEFORE the rank slices its rows, so neither the rank
        split nor the microbatch split can change the gradient (banner
        conventions 1 and 2 carry the scales).
        """
        b = mb.input_ids.shape[0]
        assert b % self.world == 0, (
            f"batch rows {b} not divisible by dp world {self.world} — "
            "size rollout_batch_size*G accordingly"
        )
        denom = minibatch_denom(self.loss_agg, mb.loss_mask)  # full mask FIRST, slice after
        denom = denom.cuda() if isinstance(denom, torch.Tensor) else denom
        local = mb if self.world == 1 else slice_batch(mb, torch.arange(self.rank, b, self.world))
        micros = self._micros(local, self.cfg.pack_max_tokens)

        self.model.train()
        self.ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        def forward_step(data_iterator, model):
            micro, pack = next(data_iterator)
            ce, to_logprobs = _forward_ce(model, micro, pack)

            def loss_func(ce_map: Tensor):
                policy_logprobs = to_logprobs(ce_map)  # (b, T) OUR convention, grad flows
                loss_map, metrics = self.loss_fn(policy_logprobs, _to_cuda(micro), self.loss_cfg)
                loss = aggregate_loss(loss_map, micro.loss_mask.cuda(), self.loss_agg, denom=denom)
                # x micros: undo the schedule's /num_microbatches; x world:
                # mean-of-scaled grads == global SUM (banner conventions 1+2)
                scaled = loss * len(micros) * self.world
                out = {k: v.detach() for k, v in metrics.items()}
                out["loss"] = loss.detach()  # the TRUE contribution, unscaled
                out["_tokens"] = micro.loss_mask.sum().detach().float()
                return scaled, out

            return ce, loss_func

        per_micro = self._fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=iter(micros),
            model=self.ddp,
            num_microbatches=len(micros),
            seq_length=local.input_ids.shape[1],
            micro_batch_size=self.cfg.micro_batch_size,
            forward_only=False,
        )
        # step() = found-inf check -> clip (cfg.clip_grad) -> fp32-master
        # update -> bf16 param refresh. Grad norm is post-reduce, identical
        # on every rank: all ranks skip together or step together.
        success, grad_norm, _ = self.optimizer.step()
        if not success:
            self.consecutive_skipped += 1
            assert self.consecutive_skipped <= self.cfg.max_skipped_steps, (
                f"{self.consecutive_skipped} consecutive found-inf optimizer steps"
            )
        else:
            self.consecutive_skipped = 0
        self.step_count += 1

        # token-weighted mean of algo metrics across microbatches (rank-local
        # rows); loss is summed — micro losses are shares of one global objective.
        total_tokens = sum(float(m["_tokens"]) for m in per_micro)
        out = {
            k: sum(float(m["_tokens"]) * float(m[k]) for m in per_micro) / max(total_tokens, 1.0)
            for k in per_micro[0]
            if k not in ("loss", "_tokens")
        }
        out["loss"] = sum(float(m["loss"]) for m in per_micro)
        out["grad_norm"] = float(grad_norm) if grad_norm is not None else float("nan")
        out["lr"] = self.cfg.lr
        return out

    # ---------------- logprob recompute (pi_old, and reusable for pi_ref) ----------------

    @torch.no_grad()
    def compute_logprobs(self, batch: Batch) -> Tensor:
        """(B, T) f32 logprobs of batch tokens under the current weights.

        Every rank recomputes the FULL batch: duplicated FLOPs, zero
        communication, same wall time (ranks run in parallel). Direct module
        calls — at pp=1 the schedule adds nothing to inference. Same fused-CE
        path (and, when packing, the same pack construction) as training, so
        pi_old matches pi_theta's kernels exactly and the on-policy PPO ratio
        starts at 1, not 1 +- kernel noise.
        """
        was_training = self.model.training
        self.model.eval()
        budget = self.cfg.logprob_pack_max_tokens or self.cfg.pack_max_tokens
        chunks = []
        for micro, pack in self._micros(batch, budget):
            ce, to_logprobs = _forward_ce(self.model, micro, pack)
            chunks.append(to_logprobs(ce).float().cpu())  # (b, T)
        if was_training:
            self.model.train()
        return torch.cat(chunks, dim=0)  # (B, T)

    def _micros(self, mb: Batch, pack_budget: int | None) -> list[tuple[Batch, Pack | None]]:
        """Grad-accum slices of a minibatch: fixed row counts, or dense packs
        under a token budget. Whole rows only, order preserved — so packed
        and padded slicings cover identical rows and (through the
        minibatch-global denominator) produce the same gradient."""
        if pack_budget is None:
            return [(m, None) for m in iter_microbatches(mb, self.cfg.micro_batch_size)]
        lengths = mb.attention_mask.sum(-1)  # (b,) real tokens per row
        return [
            (slice_batch(mb, torch.tensor(idx)), pack_rows(mb.input_ids, lengths, idx))
            for idx in plan_packs([int(n) for n in lengths], pack_budget)
        ]

    # ---------------- weight publish ----------------

    def hf_named_tensors(self):
        """Publish source: (hf_name, cpu_tensor) pairs of the CURRENT weights.

        Megatron-Bridge undoes its own conversion (fused QKV, vocab padding,
        naming) — the engines consume this stream exactly like an HF
        state_dict. DP replicates params (only optimizer state shards), so
        rank 0's export alone is the full weights.
        """
        return self.bridge.export_hf_weights([self.ddp], cpu=True)


def _forward_ce(model, micro: Batch, pack: Pack | None):
    """One microbatch forward -> (fused-CE map, adapter to (b, T) logprobs).

    Padded: causal-only attention (mask None) is safe for right-padded rows —
    real tokens never attend forward into the pads, and the loss mask zeroes
    the pads' own contribution. Packed: one dense pad-free row; cu_seqlens
    give the varlen kernels the block-diagonal attention pattern, and the
    returned adapter scatters the CE back to the padded (b, T) layout, so
    everything downstream of this function is layout-blind.
    """
    if pack is None:
        tokens = micro.input_ids.cuda()  # (b, T)
        ce = model(tokens, _position_ids(tokens), None, labels=_labels(tokens))  # (b, T)
        return ce, _ce_to_logprobs
    tokens = pack.tokens.cuda()  # (1, T_pack)
    cu = pack.cu_seqlens.cuda()
    ce = model(  # (1, T_pack) fused vocab CE over the dense packed row
        tokens,
        pack.position_ids.cuda(),
        None,
        labels=_labels(tokens),
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=pack.max_seqlen,
            max_seqlen_kv=pack.max_seqlen,
        ),
    )
    return ce, lambda ce_map: unpack_ce(ce_map, pack, micro.input_ids.shape[1])


# ---------------- the ONE shift adapter (banner convention 3) ----------------


def _labels(tokens: Tensor) -> Tensor:
    """input_ids shifted LEFT one position; last column is a dummy self-label
    (its CE is computed but position T-1 predicts beyond the batch — the
    loss mask never selects it after the shift back)."""
    return torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)  # (b, T)


def _ce_to_logprobs(ce: Tensor) -> Tensor:
    """Megatron fused-CE map -> our logprob convention.

    ce[:, t] = -log p(token_{t+1} | <=t); ours[:, t] = log p(token_t | <t)
    with position 0 = 0.0 (never predicted, always loss-masked). Negate,
    shift right one, drop the dummy last column. Differentiable — the
    training path backprops through it.
    """
    return F.pad(-ce[:, :-1], (1, 0))  # (b, T)


def _position_ids(tokens: Tensor) -> Tensor:
    b, t = tokens.shape
    return torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)


def _to_cuda(b: Batch) -> Batch:
    maybe = lambda x: x.cuda() if x is not None else None
    return Batch(
        input_ids=b.input_ids.cuda(),
        attention_mask=b.attention_mask.cuda(),
        loss_mask=b.loss_mask.cuda(),
        behavior_logprobs=b.behavior_logprobs.cuda(),
        advantages=b.advantages.cuda(),
        rewards=b.rewards.cuda(),
        group_ids=b.group_ids.cuda(),
        old_logprobs=maybe(b.old_logprobs),
        ref_logprobs=maybe(b.ref_logprobs),
    )
