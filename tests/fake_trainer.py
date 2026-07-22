"""THE TEST FAKE trainer — the executable spec of the trainer contract.

This WAS minirl/train/trainer.py (the real DDP trainer) until 2026-07-20,
when Megatron-Core replaced it as the training engine (minirl/megatron.py).
Megatron cannot import on macOS, so this pure-torch implementation demotes
to tests/: controller/algo/data tests drive it on CPU in seconds, and it
pins the semantics the Megatron adapter must reproduce on the box —
the trainer duck-type consumed by train_async.py:

    fit_batch(batch) -> metrics        compute_logprobs(batch) -> (B, T) f32
    hf_named_tensors() -> iterable     rank / world / loss_cfg

plus the math laws: ONE global aggregation with a minibatch-global
denominator, microbatch-split invariance, unconditional old_logprob
recompute at update start, the NaN skip guard, and gather_logprobs'
alignment convention (out[:, t] scores token t; position 0 is 0.0) — the
reference the Megatron fused-CE shift adapter is parity-tested against.
Recipes 03/04 also import it as the runs-anywhere demo/diagnostic learner.

DISTRIBUTED: if torch.distributed is initialized at
construction, the model is DDP-wrapped and step() becomes data-parallel —
every rank holds the IDENTICAL full minibatch, computes the GLOBAL
denominator from its full mask, then trains only rows[rank::world];
gradients all-reduce inside the LAST microbatch's backward (no_sync
suppresses the rest), scaled so the mean-reduce realizes a SUM. At world=1
every one of those lines is a no-op: the slice is all rows, the scale is
1.0, no wrapper exists.

GROUNDED IN SLIME:
  - normalization == megatron_utils/loss.py::loss_function: token mode divides
    by the batch-global token count (their `num_tokens`), sample mode averages
    per-sequence means (their `sum_of_sample_mean`); we aggregate ONCE with a
    minibatch-global denominator so microbatch splits cannot reweight tokens
    (tested by gradient equivalence).
  - defaults lr=1e-6, max_grad_norm=1.0 == slime `--lr`, `--clip-grad`.
  - fp32 logit upcast inside gather_logprobs == slime's
    `vocab_parallel_logits.float()` (ppo_utils.py:199).

Precision: the model trains in whatever dtype it was
loaded (fp32 on MPS/CPU); the one bf16 mode is cfg.bf16_weights — Megatron
style, bf16 params + fp32 masters (an autocast variant existed until
2026-07-19; removed — slime/Megatron ship exactly one mode and so do we).
The fp32 islands (logprobs, aggregation, optimizer) hold regardless.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from minirl.algos.aggregate import aggregate_loss, minibatch_denom
from minirl.rollout.batching import iter_microbatches, iter_minibatches, slice_batch
from minirl.rollout.types import Batch


def gather_logprobs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Per-token logprobs of the REALIZED tokens, aligned to input positions.

    Args:  logits (B, T, V) any float dtype; input_ids (B, T) int64.
    Returns: (B, T) f32 where out[:, t] = log p(input_ids[:, t] | input_ids[:, :t]);
        position 0 has no prediction and is set to 0.0 (always loss-masked).

    logits[:, t] predicts token t+1, hence the shift-then-left-pad. The fp32
    upcast BEFORE log_softmax is a correctness invariant:
    bf16 logprob noise is the same order as real per-update policy drift.
    """
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)  # (B, T-1, V) f32
    out = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return F.pad(out, (1, 0))  # (B, T); position 0 -> 0.0


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-6  # slime --lr default
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0  # slime --clip-grad default
    ppo_epochs: int = 1  # passes over each rollout batch (GRPO default 1)
    minibatch_size: int = 32  # sequences per optimizer step
    micro_batch_size: int = 4  # sequences per fwd/bwd (grad accumulation)
    max_skipped_steps: int = 3  # consecutive non-finite steps tolerated before crashing
    seed: int = 0  # minibatch shuffling (identical on every rank by the same seed)
    # Megatron-style mixed precision: bf16 PARAMS (half
    # memory/publish bytes, FA2-compatible, SDPA dispatches flash kernels)
    # + fp32 MASTER copies stepped by AdamW — never pure-bf16 (whose sub-ulp
    # updates round to zero). slime's one and only precision mode.
    bf16_weights: bool = False
    compile: bool = False  # torch.compile(dynamic=True) on the training forward


class Trainer:
    """fit_batch(batch) runs one update phase: recompute old_logprobs, then
    ppo_epochs x shuffled minibatches x microbatch-accumulated optimizer steps.

    Works on 1..m GPUs from one code path: construct AFTER the process group
    exists and DDP engages automatically; without a process group it
    is the plain single-device trainer. `self.model` is always the RAW module
    (clean state_dict names; no-grad code skips the wrapper); `self.ddp` is
    the training-forward module — the DDP wrapper at world>1, the raw module
    itself at world=1.
    """

    def __init__(self, model: nn.Module, loss_fn, loss_cfg, cfg: TrainConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn  # (policy_logprobs (b,T), batch, cfg) -> (loss_map (b,T), metrics)
        self.loss_cfg = loss_cfg
        self.cfg = cfg
        self.device = device
        # The reduce is an ALGORITHM property (each paper prescribes its own);
        # the trainer applies whatever the config says, mechanically — all
        # normalization knowledge lives in aggregate.py (loss_agg: str | int).
        self.loss_agg = getattr(loss_cfg, "loss_agg", "token_mean")
        # Megatron-style bf16 (cfg.bf16_weights): model params become BF16
        # (half the param memory, publish bytes, and FA2-compatible weights)
        # while the optimizer steps FP32 MASTER copies — the weight update
        # never rounds through bf16's ~3 decimal digits. Masters are cloned
        # BEFORE the cast so they start from the pristine checkpoint.
        # Documented deviation from full Megatron: grads still accumulate and
        # all-reduce in bf16 (DDP reduces in param dtype; fp32 main-grads
        # would mean replacing DDP's reducer — build if curves ever ask).
        self._masters: list[Tensor] | None = None
        if cfg.bf16_weights:
            self._masters = [p.detach().float().clone() for p in self.model.parameters()]
            self.model = self.model.to(torch.bfloat16)
        self.optimizer = torch.optim.AdamW(
            self._masters if self._masters is not None else model.parameters(),
            lr=cfg.lr, betas=cfg.adam_betas, weight_decay=cfg.weight_decay,
        )  # fp32 states regardless of model dtype (precision invariant)
        self.shuffle_rng = torch.Generator().manual_seed(cfg.seed)
        self.step_count = 0
        self.consecutive_skipped = 0

        # Distributed context, read ONCE here: world=1 when no
        # process group exists — the single-device path, zero dist machinery.
        if dist.is_available() and dist.is_initialized():
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world = 0, 1
        if self.world > 1:  # DDP broadcasts rank 0's params here: identical start, provably
            self.ddp = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()] if device == "cuda" else None,
                broadcast_buffers=False,  # buffers (RoPE caches etc.) are static + identical
            )
        else:
            self.ddp = self.model  # no wrapper: forwards hit the module directly
        if cfg.compile:
            # compile AFTER the DDP wrap (DDPOptimizer splits graphs at bucket
            # boundaries); self.model stays raw -> state_dict names stay clean
            self.ddp = torch.compile(self.ddp, dynamic=True)

    # ---------------- update phase ----------------

    def fit_batch(self, batch: Batch) -> dict:
        """One rollout's update phase. Returns metrics averaged over optimizer steps."""
        self.model.train()
        # Tier-1 rule: pi_old = the learner AT UPDATE START, recomputed here in
        # fp32 — never conflated with engine behavior_logprobs (which stay in
        # the batch for TIS). One no-grad pass; cheap next to the update itself.
        batch.old_logprobs = self.compute_logprobs(batch)  # (B, T) f32, no grad

        step_metrics: list[dict] = []
        for _ in range(self.cfg.ppo_epochs):
            for mb in iter_minibatches(batch, self.cfg.minibatch_size, self.shuffle_rng):
                step_metrics.append(self.step(mb))
        # simple mean over steps — fine for logging granularity
        keys = step_metrics[0].keys()
        return {k: sum(m[k] for m in step_metrics) / len(step_metrics) for k in keys}

    def step(self, mb: Batch) -> dict:
        """One optimizer step over a minibatch, microbatched for memory.

        mb is the FULL minibatch — identical on every rank under DDP (the
        controller broadcasts whole batches). The GLOBAL denominator is
        computed from the whole minibatch mask and shared by every microbatch
        AND every rank, so neither split can change the gradient.
        The rank slice, the loss
        scale, and no_sync below are all no-ops at world=1.
        """
        b = mb.input_ids.shape[0]
        assert b % self.world == 0, (
            f"batch rows {b} not divisible by world size {self.world} — "
            "size rollout_batch_size*G accordingly"
        )
        denom = minibatch_denom(self.loss_agg, mb.loss_mask)  # full mask FIRST, slice after
        denom = denom.to(self.device) if isinstance(denom, torch.Tensor) else denom
        local = mb if self.world == 1 else slice_batch(mb, torch.arange(self.rank, b, self.world))

        micros = list(iter_microbatches(local, self.cfg.micro_batch_size))
        micro_metrics: list[tuple[int, dict]] = []  # (token_count, metrics)
        total_loss = 0.0
        for i, micro in enumerate(micros):
            # grads accumulate locally; DDP's all-reduce fires only inside the
            # LAST microbatch's backward (its default — no_sync SUPPRESSES it)
            sync = self.ddp.no_sync() if self.world > 1 and i < len(micros) - 1 else nullcontext()
            with sync:
                micro = self._to_device(micro)
                logits = self.ddp(micro.input_ids, attention_mask=micro.attention_mask).logits  # (b, T, V)
                policy_logprobs = gather_logprobs(logits, micro.input_ids)  # (b, T) f32
                loss_map, metrics = self.loss_fn(policy_logprobs, micro, self.loss_cfg)  # (b, T)
                # x world: DDP mean-reduces grads; mean of world-scaled == SUM of unscaled
                loss = aggregate_loss(loss_map, micro.loss_mask, self.loss_agg, denom=denom) * self.world
                loss.backward()  # grads ACCUMULATE across microbatches
            total_loss += loss.item() / self.world  # log the TRUE contribution, not the scaled one
            micro_metrics.append((int(micro.loss_mask.sum()), {k: v.item() for k, v in metrics.items()}))

        # post-reduce grads are identical on every rank: vanilla local
        # clipping. Master mode first hands the bf16 grads to the fp32
        # masters — clip, step, and NaN-guard then all happen in fp32
        # (Megatron's main-grad hand-off, minus the fp32 reduce).
        if self._masters is not None:
            for p, m in zip(self.model.parameters(), self._masters):
                m.grad = None if p.grad is None else p.grad.detach().float()
        opt_params = self._masters if self._masters is not None else self.model.parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):
            # NaN guard: drop this step entirely, crash if it becomes a pattern.
            # grad_norm is post-reduce, hence identical across ranks: all ranks
            # skip together or step together — replication cannot fork here.
            self.optimizer.zero_grad(set_to_none=True)
            self.consecutive_skipped += 1
            assert self.consecutive_skipped <= self.cfg.max_skipped_steps, (
                f"{self.consecutive_skipped} consecutive non-finite gradient steps"
            )
        else:
            self.optimizer.step()
            if self._masters is not None:
                with torch.no_grad():  # fp32 master -> bf16 param: the ONLY cast point
                    for p, m in zip(self.model.parameters(), self._masters):
                        p.copy_(m)
            self.optimizer.zero_grad(set_to_none=True)
            self.consecutive_skipped = 0
        if self._masters is not None:
            self.model.zero_grad(set_to_none=True)  # bf16 grads live on the model, not the optimizer
        self.step_count += 1

        # token-weighted mean of algo metrics across microbatches (rank-local rows)
        total_tokens = sum(n for n, _ in micro_metrics)
        out = {
            k: sum(n * m[k] for n, m in micro_metrics) / max(total_tokens, 1)
            for k in micro_metrics[0][1]
        }
        out |= {"loss": total_loss, "grad_norm": float(grad_norm), "lr": self.cfg.lr}
        return out

    def hf_named_tensors(self):
        """Publish source: (hf_name, cpu_tensor) pairs of the CURRENT weights.

        The contract mirrors what Megatron-Bridge's export_hf_weights yields
        on the real trainer; here the raw module's state_dict already IS
        HF-named."""
        return ((k, v.detach().cpu()) for k, v in self.model.state_dict().items())

    # ---------------- logprob recompute (pi_old, and reusable for pi_ref) ----------------

    @torch.no_grad()
    def compute_logprobs(self, batch: Batch, model: nn.Module | None = None) -> Tensor:
        """(B, T) f32 logprobs of batch tokens under `model` (default: the learner).

        Pass a frozen reference model to fill batch.ref_logprobs the same way.
        Microbatched for memory; result lives on CPU with the batch. Always
        the RAW module — no DDP hooks under no_grad. Every rank recomputes the
        FULL batch: duplicated FLOPs, zero communication, and
        the wall time is one forward either way since ranks run in parallel.
        """
        model = self.model if model is None else model
        was_training = model.training
        model.eval()
        chunks = []
        for micro in iter_microbatches(batch, self.cfg.micro_batch_size):
            micro = self._to_device(micro)
            # same dtype/kernels as the training forward: pi_old must match
            # pi_theta or the PPO ratio starts at 1 +- bf16 noise instead of
            # exactly 1 (clip-band pollution)
            logits = model(micro.input_ids, attention_mask=micro.attention_mask).logits  # (b, T, V)
            chunks.append(gather_logprobs(logits, micro.input_ids).cpu())  # (b, T)
        if was_training:
            model.train()
        return torch.cat(chunks, dim=0)  # (B, T)

    def _to_device(self, b: Batch) -> Batch:
        maybe = lambda x: x.to(self.device) if x is not None else None
        return Batch(
            input_ids=b.input_ids.to(self.device),
            attention_mask=b.attention_mask.to(self.device),
            loss_mask=b.loss_mask.to(self.device),
            behavior_logprobs=b.behavior_logprobs.to(self.device),
            advantages=b.advantages.to(self.device),
            rewards=b.rewards.to(self.device),
            group_ids=b.group_ids.to(self.device),
            old_logprobs=maybe(b.old_logprobs),
            ref_logprobs=maybe(b.ref_logprobs),
        )
