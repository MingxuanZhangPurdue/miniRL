"""Invariant tests for the tier-1 async controller (docs/async_training.md §4).

A FakeEngine (instant, deterministic, version-stamping) + the tiny LM keep
these CPU-fast while exercising the real controller, trainer, batching, and
loss code end to end.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minirl.algos import GRPOConfig, grpo_loss
from minirl.async_controller import fit_async
from minirl.config import CollectConfig
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.train import TrainConfig, Trainer

VOCAB = 61


class TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))  # (B, T, V)


class FakeEngine:
    """Duck-typed engine: deterministic 5-token responses, version stamping,
    and a built-in guard that fails the test if weights arrive mid-generation.
    """

    pad_id = 0

    def __init__(self, gen_delay: float = 0.0):
        self.version = -1  # controller must publish v0 before any generation
        self.published: list[int] = []
        self.received: dict[str, torch.Tensor] = {}
        self.busy = False
        self.gen_delay = gen_delay
        self.rng = torch.Generator().manual_seed(0)

    def generate(self, prompt_ids, params: SamplingParams) -> list[Trajectory]:
        assert self.version >= 0, "generate before first publish"
        self.busy = True
        try:
            if self.gen_delay:
                time.sleep(self.gen_delay)
            out = []
            for p in prompt_ids:
                for _ in range(params.n):
                    resp = torch.randint(1, VOCAB, (5,), generator=self.rng)
                    n = p.numel()
                    mask = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(5, dtype=torch.bool)])
                    out.append(
                        Trajectory(
                            input_ids=torch.cat([p, resp]),
                            loss_mask=mask,
                            logprobs=torch.where(mask, torch.full((n + 5,), -1.0), torch.zeros(n + 5)),
                            version=self.version,
                        )
                    )
            return out
        finally:
            self.busy = False

    def load_weights(self, named_tensors, version: int) -> None:
        assert not self.busy, "weights changed mid-generation — tier-1 violation"
        self.received = {k: v.clone() for k, v in named_tensors}
        self.version = version
        self.published.append(version)


def run(engine, num_iterations=4, interval=1, **trainer_overrides):
    torch.manual_seed(0)
    trainer = Trainer(
        TinyLM(),
        grpo_loss,
        GRPOConfig(),
        TrainConfig(lr=1e-3, minibatch_size=8, micro_batch_size=8, **trainer_overrides),
    )
    history = fit_async(
        engine=engine,
        trainer=trainer,
        reward_fn=lambda t: float(t.input_ids[-1].item() % 2),
        prompt_source=lambda n: [torch.randint(1, VOCAB, (3 + i % 2,)) for i in range(n)],
        sampling=SamplingParams(max_new_tokens=5, n=2),
        collect_cfg=CollectConfig(group_size=2, target_groups=2),
        num_iterations=num_iterations,
        update_weights_interval=interval,
    )
    return trainer, history


def test_versions_and_staleness_interval_1():
    engine = FakeEngine()
    trainer, history = run(engine, num_iterations=4, interval=1)
    assert engine.published == [0, 1, 2, 3, 4]  # v0 + one publish per iteration
    # warm-up is on-policy, then structurally one off — never more:
    assert [m["staleness"] for m in history] == [0, 1, 1, 1]
    # last published weights are exactly the learner's current weights
    current = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    assert all(torch.equal(engine.received[k], current[k]) for k in current)


def test_staleness_bounded_by_interval_2():
    engine = FakeEngine()
    _, history = run(engine, num_iterations=4, interval=2)
    assert engine.published == [0, 2, 4]  # publish every 2nd iteration
    # trace: it1 on-policy, it2 one off, it3 trains the held pre-publish data
    # (two versions old now), it4 back to one — the bound IS the interval:
    assert [m["staleness"] for m in history] == [0, 1, 2, 1]
    assert max(m["staleness"] for m in history) == 2


def test_no_publish_during_slow_generation():
    # FakeEngine.load_weights asserts busy==False; a 50ms generation makes any
    # ordering bug in the controller trip it deterministically.
    engine = FakeEngine(gen_delay=0.05)
    _, history = run(engine, num_iterations=3, interval=1)
    assert len(history) == 3  # completed without tripping the engine's guard


def test_training_actually_happens_and_metrics_flow():
    torch.manual_seed(0)
    initial = {k: v.clone() for k, v in TinyLM().state_dict().items()}  # same seed as run()

    engine = FakeEngine()
    trainer, history = run(engine, num_iterations=2, interval=1)
    m = history[0]
    for key in ("loss", "grad_norm", "approx_kl", "clip_frac", "reward_mean",
                "frac_padding", "t_generate", "t_train", "t_iter", "staleness"):
        assert key in m, f"missing metric {key}"
    assert engine.published == [0, 1, 2]
    # the finally-published weights differ from the seed-identical init -> training moved them
    assert any(not torch.equal(engine.received[k], initial[k]) for k in initial)
    assert torch.isfinite(torch.tensor(m["loss"]))
