"""DDP learner — replicated data-parallel training (docs/ddp.md).

Pure PyTorch (`torch.nn.parallel.DistributedDataParallel`). Replaces the
FSDP2 learner (retired 2026-07-15 — docs/fsdp2.md banner): the repo trains
models that fit on one GPU, so parameters are REPLICATED on every rank and
the whole DTensor/gather apparatus disappears. Three pieces:
setup_distributed (process group), DistTrainer (the Trainer subclass; ONLY
step() differs), full_state_dict (now a plain local CPU copy — replicated
params mean rank 0 already holds the full weights; NOT a collective).

THE MATH (docs/ddp.md §1-§2, unchanged law from the FSDP2 era): every rank
receives the IDENTICAL full batch; the loss denominator is computed from the
FULL minibatch mask BEFORE the rank slices its rows, so it is
minibatch-global across the world with ZERO communication. DDP AVERAGES
gradients, so each rank's loss is scaled UP by world:

    mean_r( grad( world * sum(rows_r) / denom ) ) == grad( sum(ALL rows) / denom )

— the single-process update, which tests/test_distributed.py asserts over
2 gloo CPU processes. Backend: nccl on CUDA, gloo on CPU (the Mac tests).
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from minirl.algos.aggregate import aggregate_loss, minibatch_denom
from minirl.rollout.batching import iter_microbatches, slice_batch
from minirl.rollout.types import Batch
from minirl.train.trainer import Trainer, gather_logprobs


def setup_distributed(backend: str | None = None) -> tuple[int, int]:
    """Join the process group (torchrun / mp.spawn set the env). -> (rank, world)."""
    if not dist.is_initialized():
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    return dist.get_rank(), dist.get_world_size()


def full_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Plain fp32-on-CPU state dict — the object engine.load_weights consumes.

    Under DDP this is a LOCAL copy (no collective): parameters are replicated,
    so any rank — in practice rank 0, who owns the engines — already has the
    full weights. Takes the raw module (DistTrainer.model) or any nn.Module.
    """
    model = getattr(model, "module", model)  # tolerate a DDP wrapper
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


class DistTrainer(Trainer):
    """Trainer over a DDP-replicated model. fit_batch / compute_logprobs /
    NaN-guard policy are INHERITED — every rank recomputes old_logprobs on the
    full batch (duplicated compute, zero communication; docs/ddp.md §1) and
    the seeded minibatch shuffle agrees across ranks by construction.

    `self.model` stays the RAW module (clean state_dict names; no-grad code
    skips the wrapper); `self.ddp` is the training-forward wrapper whose
    autograd hooks do the gradient all-reduce. Only step() is overridden;
    diffs vs Trainer.step, in order:
      (1) denominator from the FULL minibatch mask, THEN slice local rows;
      (2) forwards through self.ddp, with no_sync() on every microbatch but
          the last — grads accumulate locally, ONE all-reduce per step;
      (3) loss scaled by world (DDP averages; mean of world-scaled == sum —
          docs/ddp.md §2), logged with the scale divided back out.

    Metrics are rank-local (this rank's rows); parameters are what the
    equivalence tests compare — identical on every rank by DDP's contract
    (initial broadcast from rank 0 + identical averaged updates).
    """

    def __init__(self, model: nn.Module, loss_fn, loss_cfg, cfg, device: str = "cpu"):
        assert dist.is_initialized(), "call setup_distributed() before DistTrainer"
        super().__init__(model, loss_fn, loss_cfg, cfg, device)  # optimizer on RAW params
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.ddp = DDP(
            self.model,
            device_ids=[torch.cuda.current_device()] if device == "cuda" else None,
            broadcast_buffers=False,  # buffers (RoPE caches etc.) are static + identical
        )
        self._loss_scale = float(self.world)

    def step(self, mb: Batch) -> dict:
        """One optimizer step; mb is the FULL minibatch, identical on every rank."""
        b = mb.input_ids.shape[0]
        assert b % self.world == 0, (
            f"batch rows {b} not divisible by world size {self.world} — "
            "size target_groups*G accordingly (docs/ddp.md §1)"
        )
        # (1) minibatch-GLOBAL denominator: full mask first, slice after —
        # every rank computes the same number, no collective (docs/ddp.md §1).
        denom = minibatch_denom(self.loss_agg, mb.loss_mask)
        denom = denom.to(self.device) if isinstance(denom, torch.Tensor) else denom
        local = slice_batch(mb, torch.arange(self.rank, b, self.world))  # this rank's rows

        micros = list(iter_microbatches(local, self.cfg.micro_batch_size))
        micro_metrics: list[tuple[int, dict]] = []  # (token_count, metrics)
        total_loss = 0.0
        for i, micro in enumerate(micros):
            # (2) accumulate locally; the all-reduce fires only on the last backward
            sync = nullcontext() if i == len(micros) - 1 else self.ddp.no_sync()
            with sync:
                micro = self._to_device(micro)
                logits = self.ddp(micro.input_ids, attention_mask=micro.attention_mask).logits  # (b, T, V)
                policy_logprobs = gather_logprobs(logits, micro.input_ids)  # (b, T) f32
                loss_map, metrics = self.loss_fn(policy_logprobs, micro, self.loss_cfg)  # (b, T)
                # (3) world-scaled so DDP's mean-reduce realizes the cross-rank SUM
                loss = aggregate_loss(loss_map, micro.loss_mask, self.loss_agg, denom=denom) * self._loss_scale
                loss.backward()
            total_loss += loss.item() / self._loss_scale  # log the TRUE contribution
            micro_metrics.append((int(micro.loss_mask.sum()), {k: v.item() for k, v in metrics.items()}))

        # post-reduce grads are plain tensors on every rank: vanilla clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
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
