"""FSDP2 learner — data-parallel sharded training (docs/fsdp2.md).

Pure PyTorch (`torch.distributed.fsdp.fully_shard`, the DTensor rewrite).
Four pieces: setup_distributed (process group), shard_model (wrap blocks +
root), DistTrainer (the Trainer subclass; ONLY step() differs, three marked
diffs), full_state_dict (gather sharded params into the plain dict
engine.load_weights already consumes).

THE MATH (docs/fsdp2.md §2 — the one real decision): every rank receives the
IDENTICAL full batch; the loss denominator is computed from the FULL batch
mask BEFORE the rank slices its rows, so it is minibatch-global across the
world with ZERO communication; gradients then SUM across ranks
(divide-by-world disabled), making

    SUM_r grad( sum(local rows_r) / denom ) == grad( sum(ALL rows) / denom )

— bit-for-bit the single-process update, which is exactly what
tests/test_distributed.py asserts over 2 gloo CPU processes.

Backend: nccl on CUDA, gloo on CPU (how the Mac tests run). Mixed precision
(CUDA only): MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32) per
docs/precision.md — fp32 gradient reduction is hardcoded, not a knob.
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from minirl.algos.aggregate import aggregate_loss, minibatch_denom
from minirl.rollout.batching import iter_microbatches, slice_batch
from minirl.rollout.types import Batch
from minirl.train.trainer import Trainer, gather_logprobs


def setup_distributed(backend: str | None = None) -> tuple[int, int]:
    """Join the process group (torchrun / mp.spawn set the env). -> (rank, world)."""
    if not dist.is_initialized():
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    return dist.get_rank(), dist.get_world_size()


def shard_model(model: nn.Module, mesh=None, compute_dtype: torch.dtype | None = None) -> nn.Module:
    """FSDP2-wrap: each HF decoder block, then the root (params -> DTensor shards).

    Per-block wrapping is what bounds peak memory (blocks unshard one at a
    time during forward/backward); models without `.model.layers` (TinyLM in
    tests) just get root-level sharding. compute_dtype=bf16 on CUDA enables
    the precision.md policy; None (CPU tests) = fp32 end to end.
    """
    mp_policy = (
        MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
        if compute_dtype is not None
        else None
    )
    kwargs = {"mesh": mesh} | ({"mp_policy": mp_policy} if mp_policy is not None else {})
    layers = getattr(getattr(model, "model", model), "layers", None)  # HF decoder blocks
    if layers is not None:
        for layer in layers:
            fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)
    return model


def full_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Gather DTensor shards -> ONE plain fp32-on-CPU state dict, on every rank.

    The publish path: rank 0 hands this to engine.load_weights unchanged —
    the same object HFEngine/VLLMEngine already consume, which is why weight
    publishing needed no new code.
    """
    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def _disable_grad_averaging(model: nn.Module) -> bool:
    """Make cross-rank gradient reduction a SUM (docs/fsdp2.md §2).

    FSDP2 divides reduced gradients by world size (DDP-style mean) by
    default; our per-rank losses already carry the GLOBAL denominator, so the
    correct combination is a plain sum. Two mechanisms, same math (the
    equivalence test pins both):
      - NCCL: set the divide factor to 1 (lowers to ReduceOp.PREMUL_SUM,
        which only NCCL implements);
      - gloo/CPU (the Mac tests): PREMUL_SUM is unavailable -> return False
        and the caller scales each rank's LOSS by world instead
        (mean-reduce of world-scaled grads == sum of unscaled grads).
    """
    if dist.get_backend() != "nccl":
        return False
    for name in ("set_gradient_divide_factor", "set_reduce_scatter_divide_factor"):
        fn = getattr(model, name, None)
        if fn is not None:
            fn(1.0)
            return True
    return False


class DistTrainer(Trainer):
    """Trainer over an FSDP2-sharded model. fit_batch / compute_logprobs /
    NaN-guard policy are INHERITED — every rank recomputes old_logprobs on the
    full batch (duplicated compute, zero communication; docs/fsdp2.md §5) and
    the seeded minibatch shuffle agrees across ranks by construction.

    Only step() is overridden; diffs vs Trainer.step, in order:
      (1) denominator from the FULL minibatch mask, THEN slice local rows;
      (2) gradient sync only on the LAST microbatch (one reduce-scatter per
          optimizer step; also what makes accumulation collectives match);
      (3) the clipped grad norm comes back as a DTensor -> full_tensor().

    Metrics are rank-local (this rank's rows); parameters are what the
    equivalence tests compare — they are identical on every rank by FSDP2's
    contract.
    """

    def __init__(self, model: nn.Module, loss_fn, loss_cfg, cfg, device: str = "cpu"):
        assert dist.is_initialized(), "call setup_distributed() before DistTrainer"
        super().__init__(model, loss_fn, loss_cfg, cfg, device)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        # SUM semantics for grad reduction; if the API is missing, compensate
        # by scaling each rank's loss up by world (mean of scaled == sum).
        self._loss_scale = 1.0 if _disable_grad_averaging(self.model) else float(self.world)

    def step(self, mb: Batch) -> dict:
        """One optimizer step; mb is the FULL minibatch, identical on every rank."""
        b = mb.input_ids.shape[0]
        assert b % self.world == 0, (
            f"batch rows {b} not divisible by world size {self.world} — "
            "size target_groups*G accordingly (docs/fsdp2.md §3)"
        )
        # (1) minibatch-GLOBAL denominator: full mask first, slice after —
        # every rank computes the same number, no collective (docs/fsdp2.md §2).
        denom = minibatch_denom(self.loss_agg, mb.loss_mask)
        denom = denom.to(self.device) if isinstance(denom, torch.Tensor) else denom
        local = slice_batch(mb, torch.arange(self.rank, b, self.world))  # this rank's rows

        micros = list(iter_microbatches(local, self.cfg.micro_batch_size))
        set_sync = getattr(self.model, "set_requires_gradient_sync", None)
        micro_metrics: list[tuple[int, dict]] = []  # (token_count, metrics)
        total_loss = 0.0
        for i, micro in enumerate(micros):
            if set_sync is not None:
                set_sync(i == len(micros) - 1)  # (2) accumulate locally, reduce ONCE
            micro = self._to_device(micro)
            logits = self.model(micro.input_ids, attention_mask=micro.attention_mask).logits  # (b, T, V)
            policy_logprobs = gather_logprobs(logits, micro.input_ids)  # (b, T) f32
            loss_map, metrics = self.loss_fn(policy_logprobs, micro, self.loss_cfg)  # (b, T)
            loss = aggregate_loss(loss_map, micro.loss_mask, self.loss_agg, denom=denom) * self._loss_scale
            loss.backward()  # grads ACCUMULATE across microbatches (sync on last)
            total_loss += loss.item() / self._loss_scale  # log the TRUE contribution, not the scaled one
            micro_metrics.append((int(micro.loss_mask.sum()), {k: v.item() for k, v in metrics.items()}))

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()  # (3) DTensor global norm -> plain scalar
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.consecutive_skipped += 1
            assert self.consecutive_skipped <= self.cfg.max_skipped_steps, (
                f"{self.consecutive_skipped} consecutive non-finite gradient steps"
            )
        else:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.consecutive_skipped = 0
        self.step_count += 1

        total_tokens = sum(n for n, _ in micro_metrics)
        out = {
            k: sum(n * m[k] for n, m in micro_metrics) / max(total_tokens, 1)
            for k in micro_metrics[0][1]
        }
        out |= {"loss": total_loss, "grad_norm": float(grad_norm), "lr": self.cfg.lr}
        return out
