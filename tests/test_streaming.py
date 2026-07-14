"""Tier-2 streaming tests (docs/async_tier2.md §7, the Mac-testable rungs).

A FakeStreamEngine (deterministic, poll-driven, version-stamping, with the
drain-before-publish contract ASSERTED) exercises collect_groups_stream and
fit_async_stream end to end with the real trainer/batching/loss code —
engine-agnostic by design, so no vLLM import is needed here.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minirl.algos import GRPOConfig, grpo_loss
from minirl.config import CollectConfig
from minirl.rollout.streaming import collect_groups_stream
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.controllers import fit_async_stream
from minirl.train import TrainConfig, Trainer

VOCAB = 61


class TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))  # (B, T, V)


class FakeStreamEngine:
    """Duck-typed streaming engine: submit/poll/stash/drain/n_inflight/load_weights.

    Deterministic: a request finishes after `finish_after` polls; response
    last-tokens come from a global counter, so within a group of n=2 the
    parities ALTERNATE — the parity reward below never produces a degenerate
    group unless meta {"deg": True} forces a constant reward.
    load_weights asserts the drain-then-publish contract (n_inflight == 0).
    """

    pad_id = 0

    def __init__(self, finish_after: int = 1):
        self.version = -1  # controller must publish v0 before any generation
        self.published: list[int] = []
        self.received: dict[str, torch.Tensor] = {}
        self.finish_after = finish_after
        self._pending: dict[str, dict] = {}
        self._stash: list[list[Trajectory]] = []
        self._next_id = 0
        self._counter = 0

    @property
    def n_inflight(self) -> int:
        return len(self._pending)

    def submit(self, prompt_ids, params: SamplingParams, meta: dict | None = None) -> str:
        assert self.version >= 0, "submit before first publish"
        rid = f"req-{self._next_id}"
        self._next_id += 1
        self._pending[rid] = {
            "prompt_ids": prompt_ids,
            "meta": dict(meta or {}),
            "version": self.version,
            "n": params.n,
            "left": self.finish_after,
        }
        return rid

    def poll(self) -> list[list[Trajectory]]:
        out, self._stash = self._stash, []
        for rid in list(self._pending):
            req = self._pending[rid]
            req["left"] -= 1
            if req["left"] <= 0:
                out.append(self._finish(rid))
        return out

    def stash(self, group: list[Trajectory]) -> None:
        self._stash.append(group)

    def drain(self) -> None:
        while self._pending:
            for group in self.poll():
                self.stash(group)

    def load_weights(self, named_tensors, version: int) -> None:
        assert self.n_inflight == 0, "weights changed with requests in flight — drain first"
        self.received = {k: v.clone() for k, v in named_tensors}
        self.version = version
        self.published.append(version)

    def _finish(self, rid: str) -> list[Trajectory]:
        req = self._pending.pop(rid)
        p = req["prompt_ids"]
        group = []
        for _ in range(req["n"]):
            resp = torch.tensor([1, 2, 3, 4, self._counter % VOCAB])
            self._counter += 1
            n = p.numel()
            mask = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(5, dtype=torch.bool)])
            group.append(
                Trajectory(
                    input_ids=torch.cat([p, resp]),
                    loss_mask=mask,
                    logprobs=torch.where(mask, torch.full((n + 5,), -1.0), torch.zeros(n + 5)),
                    version=req["version"],
                    meta=dict(req["meta"]),
                )
            )
        return group


def parity_reward(t: Trajectory) -> float:
    """Degenerate iff the prompt was marked; otherwise alternates within a group."""
    return 1.0 if t.meta.get("deg") else float(t.input_ids[-1].item() % 2)


def make_source(pattern: list[bool]):
    """Finite prompt source; pattern[i]=True marks a prompt whose group will be
    degenerate (constant reward). Returns [] when exhausted."""
    queue = [(torch.randint(1, VOCAB, (3,)), {"deg": True} if deg else {}) for deg in pattern]

    def source(n: int):
        out, queue[:] = queue[:n], queue[n:]
        return out

    return source


SAMPLING = SamplingParams(max_new_tokens=5, n=2)
FILTER_CFG = CollectConfig(group_size=2, target_groups=3, strategy="filter")


def fresh_engine(**kw) -> FakeStreamEngine:
    e = FakeStreamEngine(**kw)
    e.load_weights(iter([]), version=0)  # collector tests need a published engine
    return e


# ---------------- collect_groups_stream ----------------


def test_stream_fills_target_with_replacements():
    engine = fresh_engine()
    # first wave contains 2 degenerate prompts -> 2 replacements must be drawn
    source = make_source([True, False, True, False, False, False])
    trajs, stats = collect_groups_stream(engine, parity_reward, source, FILTER_CFG, SAMPLING)
    assert stats["groups"] == 3 and stats["groups_dropped"] == 2
    assert stats["submitted"] == 5  # 3 initial + 2 replacements
    assert len(trajs) == 6  # 3 groups x G=2
    assert sorted({t.meta["group_id"] for t in trajs}) == [0, 1, 2]
    assert not any(t.meta.get("deg") for t in trajs)  # every degenerate group was dropped
    rewards = torch.tensor([t.reward for t in trajs]).view(3, 2)
    assert (rewards.std(dim=1) > 0).all()  # survivors are non-degenerate by construction


def test_stream_budget_caps_pathological_source():
    engine = fresh_engine()
    cfg = CollectConfig(group_size=2, target_groups=2, strategy="filter", max_rounds=2)
    source = make_source([True] * 50)  # everything degenerate: nothing ever survives
    trajs, stats = collect_groups_stream(engine, parity_reward, source, cfg, SAMPLING)
    assert trajs == [] and stats["groups"] == 0
    assert stats["groups_generated"] == cfg.max_rounds * cfg.target_groups  # the budget, exactly


def test_stream_returns_short_when_source_exhausts():
    engine = fresh_engine()
    source = make_source([False])  # one prompt, then []
    trajs, stats = collect_groups_stream(engine, parity_reward, source, FILTER_CFG, SAMPLING)
    assert stats["groups"] == 1 and len(trajs) == 2  # short, not hanging, not raising


def test_stash_consumed_before_new_generation():
    engine = fresh_engine()
    # simulate drain leftovers: two finished groups parked in the stash
    for _ in range(2):
        engine.submit(torch.randint(1, VOCAB, (3,)), SAMPLING, {})
    engine.drain()
    assert engine.n_inflight == 0 and len(engine._stash) == 2

    empty_source = lambda n: []  # nothing new available
    cfg = CollectConfig(group_size=2, target_groups=2, strategy="filter")
    trajs, stats = collect_groups_stream(engine, parity_reward, empty_source, cfg, SAMPLING)
    assert stats["groups"] == 2 and stats["submitted"] == 0  # batch built PURELY from leftovers
    assert len(trajs) == 4


def test_group_size_mismatch_fails_loud():
    with pytest.raises(AssertionError, match="group_size"):
        collect_groups_stream(
            fresh_engine(), parity_reward, make_source([False]),
            CollectConfig(group_size=4, target_groups=1), SAMPLING,  # n=2 != G=4
        )


# ---------------- fit_async_stream ----------------


def run(engine, num_iterations=4, interval=1):
    torch.manual_seed(0)
    trainer = Trainer(
        TinyLM(),
        grpo_loss,
        GRPOConfig(),
        TrainConfig(lr=1e-3, minibatch_size=8, micro_batch_size=8),
    )
    history = fit_async_stream(
        engine=engine,
        trainer=trainer,
        reward_fn=parity_reward,
        prompt_source=lambda n: [torch.randint(1, VOCAB, (3 + i % 2,)) for i in range(n)],
        sampling=SAMPLING,
        collect_cfg=CollectConfig(group_size=2, target_groups=2, strategy="filter"),
        num_iterations=num_iterations,
        publish_interval=interval,
    )
    return trainer, history


def test_stream_controller_versions_and_staleness():
    engine = FakeStreamEngine()
    trainer, history = run(engine, num_iterations=4, interval=1)
    assert engine.published == [0, 1, 2, 3, 4]  # v0 + one drained publish per iteration
    # warm-up on-policy; then bounded by interval + 1 (drained-leftover carryover)
    staleness = [m["staleness"] for m in history]
    assert staleness[0] == 0 and all(0 <= s <= 2 for s in staleness)
    current = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    assert all(torch.equal(engine.received[k], current[k]) for k in current)


def test_stream_controller_publish_interval_2():
    engine = FakeStreamEngine()
    _, history = run(engine, num_iterations=4, interval=2)
    assert engine.published == [0, 2, 4]
    assert all(0 <= m["staleness"] <= 3 for m in history)  # interval + 1


def test_stream_controller_metrics_and_training():
    torch.manual_seed(0)
    initial = {k: v.clone() for k, v in TinyLM().state_dict().items()}  # same seed as run()
    engine = FakeStreamEngine(finish_after=2)  # groups take 2 polls: exercises empty polls
    trainer, history = run(engine, num_iterations=3, interval=1)
    m = history[0]
    for key in ("loss", "grad_norm", "approx_kl", "reward_mean", "groups_generated",
                "submitted", "polls", "leftover_inflight", "t_generate", "t_train",
                "t_iter", "staleness"):
        assert key in m, f"missing metric {key}"
    assert any(not torch.equal(engine.received[k], initial[k]) for k in initial)
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_vllm_engine_module_imports_without_vllm():
    # the module must import in this env (no vLLM installed): all vLLM imports
    # are inside methods. Instantiation is what needs the vLLM env.
    import minirl.engine.vllm_engine as m

    assert hasattr(m, "VLLMEngine") and hasattr(m, "_apply_weights_from_file")
