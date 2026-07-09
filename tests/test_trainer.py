"""Trainer + batching tests on a tiny random LM (CPU, seconds).

Covers the invariants from docs/sync_training.md §9 that don't need a real
model: gather_logprobs correctness, batch round-trip, microbatch gradient
equivalence, old_logprobs recompute, on-policy ratio == 1, NaN guard, and
"SFT actually learns".
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minirl.algos import GRPOConfig, SFTConfig, grpo_loss, sft_loss
from minirl.rollout.batching import iter_microbatches, iter_minibatches, make_batch
from minirl.rollout.types import Trajectory
from minirl.train import TrainConfig, Trainer, gather_logprobs

VOCAB = 61


class TinyLM(nn.Module):
    """Embedding -> Linear. Not causal — irrelevant for trainer-math tests."""

    def __init__(self, vocab: int = VOCAB, d: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))  # (B, T, V)


def make_trajs(b: int = 4, group_size: int = 2) -> list[Trajectory]:
    """Variable-length trajectories, alternating rewards within each group."""
    torch.manual_seed(0)
    trajs = []
    for i in range(b):
        prompt_len, resp_len = 2, 3 + i  # variable lengths exercise padding
        n = prompt_len + resp_len
        ids = torch.randint(0, VOCAB, (n,))
        mask = torch.cat([torch.zeros(prompt_len, dtype=torch.bool), torch.ones(resp_len, dtype=torch.bool)])
        lps = torch.where(mask, torch.randn(n).abs().neg(), torch.zeros(n))  # negative on response
        trajs.append(
            Trajectory(
                input_ids=ids,
                loss_mask=mask,
                logprobs=lps,
                reward=float(i % 2),
                meta={"group_id": i // group_size},
            )
        )
    return trajs


def new_trainer(loss_fn, loss_cfg, model=None, **cfg_overrides) -> Trainer:
    torch.manual_seed(42)
    defaults = dict(lr=1e-2, minibatch_size=8, micro_batch_size=8)  # big lr: TinyLM must move
    return Trainer(model or TinyLM(), loss_fn, loss_cfg, TrainConfig(**(defaults | cfg_overrides)))


# ---------------- gather_logprobs ----------------


def test_gather_logprobs_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, VOCAB)
    ids = torch.randint(0, VOCAB, (2, 5))
    out = gather_logprobs(logits, ids)  # (2, 5)
    # -cross_entropy(logits[:, t]) == log p(ids[:, t+1]) == out[:, t+1]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1), reduction="none")
    assert torch.allclose(out[:, 1:], -ce.view(2, 4), atol=1e-6)
    assert (out[:, 0] == 0).all()  # position 0 has no prediction
    assert out.dtype == torch.float32  # fp32 invariant even from half inputs


# ---------------- make_batch ----------------


def test_make_batch_round_trip_and_advantages():
    trajs = make_trajs()
    batch, stats = make_batch(trajs, pad_id=0)
    assert batch.input_ids.shape == (4, 8)  # longest traj = 2 + 6
    for i, tr in enumerate(trajs):
        n = tr.input_ids.numel()
        assert torch.equal(batch.input_ids[i, :n], tr.input_ids)  # round-trip
        assert not batch.loss_mask[i, n:].any()  # padding never learns
        assert batch.attention_mask[i, :n].all() and not batch.attention_mask[i, n:].any()
        assert torch.equal(batch.behavior_logprobs[i, :n], tr.logprobs)
    # rewards (0,1) per group -> advantages +-1/sqrt(.5)... but sign check suffices:
    row_adv = batch.advantages.sum(-1) / batch.loss_mask.sum(-1)  # (B,) recovered scalars
    assert (row_adv[batch.rewards == 1] > 0).all() and (row_adv[batch.rewards == 0] < 0).all()
    assert (batch.advantages[~batch.loss_mask] == 0).all()  # only response tokens carry advantage
    assert stats["frac_degenerate_groups"] == 0.0 and stats["response_tokens"] == 3 + 4 + 5 + 6


def test_make_batch_advantage_fn_is_pluggable():
    trajs = make_trajs()
    # custom estimator: identity (advantage = raw reward, no baseline)
    batch, _ = make_batch(trajs, pad_id=0, advantage_fn=lambda r, g: r)
    row_adv = batch.advantages.sum(-1) / batch.loss_mask.sum(-1)  # (B,) recovered scalars
    assert torch.allclose(row_adv, batch.rewards)
    # None: zeros filled, never computed (SFT / PPO-fills-later path)
    batch, stats = make_batch(trajs, pad_id=0, advantage_fn=None)
    assert (batch.advantages == 0).all()
    assert "frac_degenerate_groups" in stats  # stats still computed from rewards


def test_minibatch_iteration_covers_batch_deterministically():
    batch, _ = make_batch(make_trajs(), pad_id=0)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    rows = lambda gen: [mb.input_ids for mb in iter_minibatches(batch, 2, gen)]
    a, b = rows(g1), rows(g2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))  # seeded -> deterministic
    assert sum(x.shape[0] for x in a) == 4  # covers every row
    micro_rows = sum(m.input_ids.shape[0] for m in iter_microbatches(batch, 3))
    assert micro_rows == 4


# ---------------- trainer core ----------------


def test_old_logprobs_recompute_matches_manual():
    trainer = new_trainer(grpo_loss, GRPOConfig())
    batch, _ = make_batch(make_trajs(), pad_id=0)
    old = trainer.compute_logprobs(batch)  # (B, T)
    with torch.no_grad():
        manual = gather_logprobs(trainer.model(batch.input_ids).logits, batch.input_ids)
    assert torch.allclose(old, manual, atol=1e-6)
    assert not old.requires_grad


def test_microbatch_split_does_not_change_the_update():
    """micro_batch_size=8 vs 1 must produce identical parameters after a step."""
    batch, _ = make_batch(make_trajs(), pad_id=0)
    results = []
    for micro in (8, 1):
        trainer = new_trainer(grpo_loss, GRPOConfig(), micro_batch_size=micro)
        trainer.step(batch)  # same seed -> same init; old=None -> behavior fallback
        results.append([p.detach().clone() for p in trainer.model.parameters()])
    for p8, p1 in zip(*results):
        assert torch.allclose(p8, p1, atol=1e-6)


def test_constant_normalizer_dr_grpo_reduce():
    """Dr. GRPO's exact reduce: denom = rows * constant, not the actual token count.
    Same per-token values => losses relate by exactly (actual_tokens / (B * C))."""
    batch, _ = make_batch(make_trajs(), pad_id=0)
    actual_tokens = int(batch.loss_mask.sum())  # 3+4+5+6 = 18
    C = 10
    loss_actual = new_trainer(grpo_loss, GRPOConfig(loss_agg="token_mean")).step(batch)["loss"]
    loss_const = new_trainer(grpo_loss, GRPOConfig(loss_agg=C)).step(batch)["loss"]
    assert loss_const * (4 * C) == pytest.approx(loss_actual * actual_tokens, rel=1e-5)


def test_first_step_is_on_policy():
    """Single minibatch, ppo_epochs=1: policy == old at the only step -> ratio 1."""
    trainer = new_trainer(grpo_loss, GRPOConfig())
    batch, _ = make_batch(make_trajs(), pad_id=0)
    metrics = trainer.fit_batch(batch)
    assert metrics["approx_kl"] < 1e-9 and metrics["clip_frac"] == 0.0
    assert batch.old_logprobs is not None  # recompute happened unconditionally


def test_multiple_epochs_go_off_policy_and_clip_engages_machinery():
    trainer = new_trainer(grpo_loss, GRPOConfig(), ppo_epochs=3, minibatch_size=2, lr=5e-2)
    batch, _ = make_batch(make_trajs(), pad_id=0)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.fit_batch(batch)
    assert metrics["approx_kl"] > 0  # later steps train off the frozen old_logprobs
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))


def test_sft_overfits_a_fixed_batch():
    trainer = new_trainer(sft_loss, SFTConfig())
    batch, _ = make_batch(make_trajs(), pad_id=0)
    losses = [trainer.step(batch)["loss"] for _ in range(30)]
    assert losses[-1] < losses[0] * 0.5, f"SFT failed to learn: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_nan_guard_skips_then_crashes():
    def nan_loss(lp, b, cfg):
        return lp * float("nan"), {"m": torch.tensor(0.0)}

    trainer = new_trainer(nan_loss, SFTConfig(), max_skipped_steps=1)
    batch, _ = make_batch(make_trajs(), pad_id=0)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.step(batch)  # skip 1: tolerated
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))  # untouched
    assert trainer.consecutive_skipped == 1
    with pytest.raises(AssertionError, match="non-finite"):
        trainer.step(batch)  # skip 2 > max_skipped_steps
