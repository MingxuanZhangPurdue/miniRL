"""The generic step engine: one trainer for SFT and every RL loss.

Owns exactly the algorithm-agnostic parts (docs/sync_training.md §5):
forward -> gather_logprobs -> loss_map -> ONE global aggregation -> backward,
with microbatch gradient accumulation, grad clipping, AdamW, NaN guard, and
the tier-1 rule from docs/async_training.md §2: old_logprobs are recomputed
UNCONDITIONALLY at the start of every update phase (correct for ppo_epochs>1
in sync, mandatory under async staleness; slime's use_rollout_logprobs=False
default path).

GROUNDED IN SLIME:
  - normalization == megatron_utils/loss.py::loss_function: token mode divides
    by the batch-global token count (their `num_tokens`), sample mode averages
    per-sequence means (their `sum_of_sample_mean`); we aggregate ONCE with a
    minibatch-global denominator so microbatch splits cannot reweight tokens
    (tested by gradient equivalence).
  - defaults lr=1e-6, max_grad_norm=1.0 == slime `--lr`, `--clip-grad`.
  - fp32 logit upcast inside gather_logprobs == slime's
    `vocab_parallel_logits.float()` (ppo_utils.py:199).

Precision (docs/precision.md): this single-device trainer keeps the model in
whatever dtype it was loaded (fp32 on MPS/CPU); bf16-compute-with-fp32-master
stays a recipe-level autocast knob (docs/ddp.md §5). The fp32 islands (logprobs, aggregation,
optimizer) hold regardless.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minirl.algos.aggregate import aggregate_loss, minibatch_denom
from minirl.rollout.batching import iter_microbatches, iter_minibatches
from minirl.rollout.types import Batch


def gather_logprobs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Per-token logprobs of the REALIZED tokens, aligned to input positions.

    Args:  logits (B, T, V) any float dtype; input_ids (B, T) int64.
    Returns: (B, T) f32 where out[:, t] = log p(input_ids[:, t] | input_ids[:, :t]);
        position 0 has no prediction and is set to 0.0 (always loss-masked).

    logits[:, t] predicts token t+1, hence the shift-then-left-pad. The fp32
    upcast BEFORE log_softmax is a correctness invariant (docs/precision.md):
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
    seed: int = 0  # minibatch shuffling


class Trainer:
    """fit_batch(batch) runs one update phase: recompute old_logprobs, then
    ppo_epochs x shuffled minibatches x microbatch-accumulated optimizer steps.
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
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.adam_betas, weight_decay=cfg.weight_decay
        )  # fp32 states regardless of model dtype (precision invariant)
        self.shuffle_rng = torch.Generator().manual_seed(cfg.seed)
        self.step_count = 0
        self.consecutive_skipped = 0

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

        The GLOBAL denominator is computed on the whole minibatch and shared by
        every microbatch, so each loss contribution is sum/GLOBAL — splitting
        into microbatches cannot change the gradient (docs/sync_training.md §5).
        """
        denom = minibatch_denom(self.loss_agg, mb.loss_mask)  # minibatch-GLOBAL, shared by all micros
        denom = denom.to(self.device) if isinstance(denom, torch.Tensor) else denom

        micro_metrics: list[tuple[int, dict]] = []  # (token_count, metrics)
        total_loss = 0.0
        for micro in iter_microbatches(mb, self.cfg.micro_batch_size):
            micro = self._to_device(micro)
            logits = self.model(micro.input_ids, attention_mask=micro.attention_mask).logits  # (b, T, V)
            policy_logprobs = gather_logprobs(logits, micro.input_ids)  # (b, T) f32
            loss_map, metrics = self.loss_fn(policy_logprobs, micro, self.loss_cfg)  # (b, T)
            loss = aggregate_loss(loss_map, micro.loss_mask, self.loss_agg, denom=denom)  # scalar
            loss.backward()  # grads ACCUMULATE across microbatches
            total_loss += loss.item()
            micro_metrics.append((int(micro.loss_mask.sum()), {k: v.item() for k, v in metrics.items()}))

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):
            # NaN guard: drop this step entirely, crash if it becomes a pattern.
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

        # token-weighted mean of algo metrics across microbatches
        total_tokens = sum(n for n, _ in micro_metrics)
        out = {
            k: sum(n * m[k] for n, m in micro_metrics) / max(total_tokens, 1)
            for k in micro_metrics[0][1]
        }
        out |= {"loss": total_loss, "grad_norm": float(grad_norm), "lr": self.cfg.lr}
        return out

    # ---------------- logprob recompute (pi_old, and reusable for pi_ref) ----------------

    @torch.no_grad()
    def compute_logprobs(self, batch: Batch, model: nn.Module | None = None) -> Tensor:
        """(B, T) f32 logprobs of batch tokens under `model` (default: the learner).

        Pass a frozen reference model to fill batch.ref_logprobs the same way.
        Microbatched for memory; result lives on CPU with the batch.
        """
        model = self.model if model is None else model
        was_training = model.training
        model.eval()
        chunks = []
        for micro in iter_microbatches(batch, self.cfg.micro_batch_size):
            micro = self._to_device(micro)
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
