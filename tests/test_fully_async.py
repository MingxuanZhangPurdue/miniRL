"""fully_async controller tests — CPU only, no vLLM, no GPUs.

Consolidates the retired test_streaming.py + test_data_parallel.py suites
(docs/async_tier2.md §11): a FakeStreamEngine (deterministic, poll-driven,
version-stamping, drain-before-publish ASSERTED) and a FakeGenEngine (the
HFEngine duck-type, fed through StreamAdapter) exercise collect_groups_dp
and fit_async end to end with the real trainer/batching/loss code. The
2-process gloo test pins the rank-0/follower wiring (docs/fsdp2.md §8)
against a single-process reference.
"""

import os
import time
from typing import NamedTuple

import pytest
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp
from torch import nn

from minirl.algos import GRPOConfig, grpo_loss
from minirl.config import CollectConfig, PlacementConfig
from minirl.controllers import collect_groups_dp, fit_async
from minirl.engine import StreamAdapter
from minirl.rollout.types import SamplingParams, Trajectory
from minirl.train import TrainConfig, Trainer

VOCAB = 61
SAMPLING = SamplingParams(max_new_tokens=5, n=2)
FILTER_CFG = CollectConfig(group_size=2, target_groups=3, strategy="filter")


class TinyOut(NamedTuple):
    # NamedTuple, NOT SimpleNamespace: this TinyLM gets FSDP2-sharded in the
    # 2-rank test, and FSDP2 finds forward outputs via pytree — the
    # SimpleNamespace trap silently no-ops training (docs/fsdp2.md §7).
    logits: torch.Tensor


class TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        return TinyOut(logits=self.head(self.emb(input_ids)))  # (B, T, V)


class FakeStreamEngine:
    """Duck-typed streaming engine: submit/poll/stash/drain/n_inflight/load_weights.

    Deterministic: a request finishes after `finish_after` polls; response
    last-tokens come from a per-engine counter, so within a group of n=2 the
    parities ALTERNATE — parity_reward never yields a degenerate group unless
    meta {"deg": True} forces a constant reward. load_weights asserts the
    drain-then-publish contract (n_inflight == 0).
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


class PacedEngine(FakeStreamEngine):
    """FakeStreamEngine whose poll costs wall time (a GIL-releasing sleep), so
    concurrent collector threads interleave like real engines stepping —
    finish_after becomes SPEED, which is what the load-balance test observes."""

    def __init__(self, finish_after: int = 1, dt: float = 0.001):
        super().__init__(finish_after)
        self.dt = dt

    def poll(self):
        time.sleep(self.dt)
        return super().poll()


class FakeGenEngine:
    """generate()-only fake — the HFEngine duck-type StreamAdapter wraps:
    grouped-by-prompt trajectories, version-stamped, same parity counter."""

    pad_id = 0

    def __init__(self):
        self.version = -1
        self.published: list[int] = []
        self.received: dict[str, torch.Tensor] = {}
        self._counter = 0

    def generate(self, prompt_ids: list, params: SamplingParams) -> list[Trajectory]:
        assert self.version >= 0, "generate before first publish"
        out = []
        for p in prompt_ids:
            for _ in range(params.n):
                resp = torch.tensor([1, 2, 3, 4, self._counter % VOCAB])
                self._counter += 1
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

    def load_weights(self, named_tensors, version: int) -> None:
        self.received = {k: v.clone() for k, v in named_tensors}
        self.version = version
        self.published.append(version)


def parity_reward(t: Trajectory) -> float:
    """Degenerate iff the prompt was marked; otherwise alternates within a group."""
    return 1.0 if t.meta.get("deg") else float(t.input_ids[-1].item() % 2)


def pid_source(n_prompts: int, deg_every: int | None = None):
    """Finite source; unique pid per prompt proves no prompt is dealt twice;
    deg_every marks every k-th prompt's group degenerate. [] when exhausted."""
    queue = [
        (torch.randint(1, VOCAB, (3,)), {"pid": i, "deg": bool(deg_every and i % deg_every == 0)})
        for i in range(n_prompts)
    ]

    def source(n: int):
        out, queue[:] = queue[:n], queue[n:]
        return out

    return source


def fresh(engines):
    for e in engines:
        e.load_weights(iter([]), version=0)
    return engines


# ---------------- collect_groups_dp: single engine (the k=1 contract) ----------------


def test_fills_target_with_replacements():
    (engine,) = fresh([FakeStreamEngine()])
    # 2 degenerate prompts in the first wave -> 2 replacements must be drawn
    source = pid_source(6, deg_every=2)  # pids 0,2,4 degenerate
    trajs, stats = collect_groups_dp([engine], parity_reward, source, FILTER_CFG, SAMPLING)
    assert stats["groups"] == 3 and stats["groups_dropped"] >= 2
    assert len(trajs) == 6  # 3 groups x G=2
    assert sorted({t.meta["group_id"] for t in trajs}) == [0, 1, 2]
    assert not any(t.meta["deg"] for t in trajs)  # every degenerate group was dropped
    rewards = torch.tensor([t.reward for t in trajs]).view(3, 2)
    assert (rewards.std(dim=1) > 0).all()


def test_budget_caps_pathological_source():
    (engine,) = fresh([FakeStreamEngine()])
    cfg = CollectConfig(group_size=2, target_groups=2, strategy="filter", max_rounds=2)
    source = pid_source(50, deg_every=1)  # everything degenerate: nothing survives
    trajs, stats = collect_groups_dp([engine], parity_reward, source, cfg, SAMPLING)
    assert trajs == [] and stats["groups"] == 0
    assert stats["groups_generated"] == cfg.max_rounds * cfg.target_groups  # the budget, exactly


def test_returns_short_when_source_exhausts():
    engines = fresh([FakeStreamEngine(), FakeStreamEngine()])
    cfg = CollectConfig(group_size=2, target_groups=4, strategy="filter")
    trajs, stats = collect_groups_dp(engines, parity_reward, pid_source(2), cfg, SAMPLING)
    assert stats["groups"] == 2 and len(trajs) == 4  # short, not hanging, not raising


def test_stash_consumed_before_new_generation():
    (engine,) = fresh([FakeStreamEngine()])
    for _ in range(2):  # simulate drain leftovers: two finished groups in the stash
        engine.submit(torch.randint(1, VOCAB, (3,)), SAMPLING, {})
    engine.drain()
    assert engine.n_inflight == 0 and len(engine._stash) == 2

    cfg = CollectConfig(group_size=2, target_groups=2, strategy="filter")
    trajs, stats = collect_groups_dp([engine], parity_reward, lambda n: [], cfg, SAMPLING)
    assert stats["groups"] == 2 and stats["submitted"] == 0  # built PURELY from leftovers
    assert len(trajs) == 4


def test_group_size_mismatch_fails_loud():
    with pytest.raises(AssertionError, match="group_size"):
        collect_groups_dp(
            fresh([FakeStreamEngine()]), parity_reward, pid_source(1),
            CollectConfig(group_size=4, target_groups=1), SAMPLING,  # n=2 != G=4
        )


# ---------------- collect_groups_dp: k engines (dealer + tally) ----------------


def test_dp_target_met_no_prompt_dealt_twice_ids_unique():
    engines = fresh([FakeStreamEngine(), FakeStreamEngine(finish_after=2)])
    cfg = CollectConfig(group_size=2, target_groups=4, strategy="filter")
    trajs, stats = collect_groups_dp(engines, parity_reward, pid_source(20), cfg, SAMPLING)
    assert stats["groups"] == 4 and len(trajs) == 8  # global target met exactly
    # group_ids restamped 0..target-1, one pid per group, no pid in two groups
    by_gid = {gid: {t.meta["pid"] for t in trajs if t.meta["group_id"] == gid} for gid in range(4)}
    assert all(len(pids) == 1 for pids in by_gid.values())
    kept_pids = [pids.pop() for pids in by_gid.values()]
    assert len(kept_pids) == len(set(kept_pids))
    assert stats["submitted_e0"] + stats["submitted_e1"] == stats["submitted"]


def test_dp_fast_engine_deals_strictly_more():
    # fast turns a group per ~1ms; slow needs ~50ms for its first — with every
    # 2nd prompt degenerate, replacements keep opening and only the fast
    # engine is awake to claim them (load balance OBSERVED, not assumed).
    engines = fresh([PacedEngine(finish_after=1), PacedEngine(finish_after=50)])
    cfg = CollectConfig(group_size=2, target_groups=6, strategy="filter")
    trajs, stats = collect_groups_dp(engines, parity_reward, pid_source(60, deg_every=2), cfg, SAMPLING)
    assert stats["groups"] == 6
    assert stats["submitted_e0"] > stats["submitted_e1"]
    assert not any(t.meta["deg"] for t in trajs)  # filtering still exact under threads


def test_dp_burst_cap_prevents_hoarding():
    # k=2, target=4 -> burst=2: even the first collector to run can commit at
    # most 2 prompts before others wake (the late-binding half of the design).
    engines = fresh([FakeStreamEngine(finish_after=3), FakeStreamEngine(finish_after=3)])
    cfg = CollectConfig(group_size=2, target_groups=4, strategy="filter")
    _, stats = collect_groups_dp(engines, parity_reward, pid_source(20), cfg, SAMPLING)
    assert stats["submitted_e0"] <= 2 + stats["groups_dropped"]
    assert stats["submitted_e1"] <= 2 + stats["groups_dropped"]


def test_dp_leftovers_consumed_by_next_call():
    engines = fresh([FakeStreamEngine(), FakeStreamEngine(finish_after=30)])
    cfg = CollectConfig(group_size=2, target_groups=3, strategy="filter")
    _, stats = collect_groups_dp(engines, parity_reward, pid_source(20), cfg, SAMPLING)
    if stats["leftover_inflight"] == 0:  # scheduling let the slow engine finish: nothing to check
        return
    n_before = stats["submitted"]
    _, stats2 = collect_groups_dp(engines, parity_reward, pid_source(20), cfg, SAMPLING)
    assert stats2["groups"] == 3
    assert stats2["submitted"] <= n_before  # leftovers reduced the new dealing needed


# ---------------- StreamAdapter (the HFEngine path) ----------------


def test_stream_adapter_one_poll_is_one_round():
    adapter = StreamAdapter(FakeGenEngine())
    adapter.load_weights(iter([]), version=0)
    for i in range(3):
        adapter.submit(torch.randint(1, VOCAB, (3,)), SAMPLING, {"pid": i})
    assert adapter.n_inflight == 3
    groups = adapter.poll()  # ONE round: everything finishes together
    assert len(groups) == 3 and adapter.n_inflight == 0
    assert [g[0].meta["pid"] for g in groups] == [0, 1, 2]  # meta attached, order kept
    assert all(len(g) == SAMPLING.n and t.version == 0 for g in groups for t in g)
    assert adapter.poll() == []  # empty round: nothing pending, nothing stashed


def test_stream_adapter_under_the_controller():
    engine = FakeGenEngine()
    torch.manual_seed(0)
    trainer = Trainer(
        TinyLM(), grpo_loss, GRPOConfig(),
        TrainConfig(lr=1e-3, minibatch_size=4, micro_batch_size=4),
    )
    history = fit_async(
        engines=[StreamAdapter(engine)],
        trainer=trainer,
        reward_fn=parity_reward,
        prompt_source=lambda n: [torch.randint(1, VOCAB, (3,)) for _ in range(n)],
        sampling=SAMPLING,
        collect_cfg=CollectConfig(group_size=2, target_groups=2, strategy="filter"),
        num_iterations=3,
        publish_interval=1,
    )
    assert engine.published == [0, 1, 2, 3]  # round_based semantics under the ONE controller
    current = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    assert all(torch.equal(engine.received[k], current[k]) for k in current)
    assert all(0 <= m["staleness"] <= 2 for m in history)


# ---------------- fit_async controller (fake streaming engines) ----------------


def run_controller(engines, num_iterations=3, interval=1):
    torch.manual_seed(0)
    trainer = Trainer(
        TinyLM(), grpo_loss, GRPOConfig(),
        TrainConfig(lr=1e-3, minibatch_size=8, micro_batch_size=8),
    )
    history = fit_async(
        engines=engines,
        trainer=trainer,
        reward_fn=parity_reward,
        prompt_source=lambda n: [torch.randint(1, VOCAB, (3 + i % 2,)) for i in range(n)],
        sampling=SAMPLING,
        collect_cfg=CollectConfig(group_size=2, target_groups=2, strategy="filter"),
        num_iterations=num_iterations,
        publish_interval=interval,
    )
    return trainer, history


def test_controller_publishes_all_engines_and_staleness_holds():
    engines = [FakeStreamEngine(), FakeStreamEngine(finish_after=2)]
    trainer, history = run_controller(engines, num_iterations=3, interval=1)
    current = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
    for e in engines:  # every engine got every publish, drained first (asserted by the fake)
        assert e.published == [0, 1, 2, 3]
        assert e.n_inflight == 0
        assert all(torch.equal(e.received[k], current[k]) for k in current)
    staleness = [m["staleness"] for m in history]
    assert staleness[0] == 0 and all(0 <= s <= 2 for s in staleness)  # interval + 1


def test_controller_interval_2_and_metrics():
    engines = [FakeStreamEngine(), FakeStreamEngine(finish_after=3)]
    _, history = run_controller(engines, num_iterations=4, interval=2)
    for e in engines:
        assert e.published == [0, 2, 4]
    assert all(0 <= m["staleness"] <= 3 for m in history)  # interval + 1
    m = history[0]
    for key in ("loss", "grad_norm", "reward_mean", "groups_generated", "submitted",
                "submitted_e0", "submitted_e1", "leftover_inflight", "t_generate",
                "t_train", "t_iter", "staleness"):
        assert key in m, f"missing metric {key}"


# ---------------- multi-rank: rank-0 collects, follower trains ----------------

WORLD = 2


def _steady_source(n: int) -> list:
    return [torch.full((3,), 7, dtype=torch.long) for _ in range(n)]  # RNG-free prompts


def _run_training(trainer_ctor, engines: list, num_iterations: int) -> tuple:
    """Deterministic fit_async run (constant prompts, counter-driven fakes)."""
    trainer = trainer_ctor()
    history = fit_async(
        engines=engines,
        trainer=trainer,
        reward_fn=parity_reward,
        prompt_source=_steady_source,
        sampling=SAMPLING,
        collect_cfg=CollectConfig(group_size=2, target_groups=2, strategy="fixed"),
        num_iterations=num_iterations,
        publish_interval=1,
    )
    return trainer, history


def _dist_worker(rank: int, port: int, out_dir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD)
    )
    tdist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from minirl.train.distributed import DistTrainer, full_state_dict, shard_model

        def ctor():
            torch.manual_seed(42)
            model = shard_model(TinyLM(), mesh=init_device_mesh("cpu", (WORLD,)))
            return DistTrainer(
                model, grpo_loss, GRPOConfig(),
                TrainConfig(lr=1e-2, minibatch_size=4, micro_batch_size=2),
            )

        if rank == 0:
            engine = FakeStreamEngine()
            trainer, history = _run_training(ctor, [engine], num_iterations=2)
            torch.save(
                {
                    "final": full_state_dict(trainer.model),
                    "published": engine.published,
                    "grad_norm": history[0]["grad_norm"],  # global (DTensor .full_tensor())
                    "iters": len(history),
                },
                os.path.join(out_dir, "rank0.pt"),
            )
        else:
            trainer, history = _run_training(ctor, [], num_iterations=2)  # follower: engines=[]
            assert history == []
            full_state_dict(trainer.model)  # rank 0 gathers after its run; stay aligned
        tdist.barrier()
    finally:
        tdist.destroy_process_group()


def test_two_rank_controller_matches_single_process(tmp_path):
    mp.spawn(_dist_worker, args=(29517, str(tmp_path)), nprocs=WORLD, join=True)
    results = torch.load(tmp_path / "rank0.pt")
    assert results["published"] == [0, 1, 2] and results["iters"] == 2

    # single-process reference: same seeds, same deterministic collection
    def ctor():
        torch.manual_seed(42)
        return Trainer(
            TinyLM(), grpo_loss, GRPOConfig(),
            TrainConfig(lr=1e-2, minibatch_size=4, micro_batch_size=4),
        )

    trainer, history = _run_training(ctor, [FakeStreamEngine()], num_iterations=2)
    ref = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}

    # THE math check is grad_norm: the 2-rank value is the norm of gradients
    # SUMMED across ranks over the identical broadcast batch — it matches
    # single-process to fp noise, and catches sum-vs-average (x0.5), wrong
    # denominators (xk), and corrupted broadcasts (garbage) in one number.
    # (Exact per-step trainer math is test_distributed's job.)
    assert abs(results["grad_norm"] - history[0]["grad_norm"]) < 1e-5 * max(
        1.0, history[0]["grad_norm"]
    ), f"grad_norm diverged: {results['grad_norm']} vs {history[0]['grad_norm']}"

    # Parameters only loosely (1e-2 smoke bound): early AdamW is near-sign
    # descent (step-1 update ~ lr*sign(g)), so gloo's fp reduction-order
    # noise can flip low-signal coordinates by O(lr) — measured ~2e-3 max.
    # A real math bug trips the grad_norm check above, not this bound.
    for k in ref:
        assert torch.allclose(results["final"][k], ref[k], atol=1e-2), (
            f"parameter {k} wildly diverged — wiring bug, not fp noise"
        )


# ---------------- filtering (rollout/filtering.py) ----------------


def test_reward_nonzero_std_filter():
    from minirl.rollout.filtering import reward_nonzero_std

    def group(rewards):
        return [Trajectory(input_ids=torch.zeros(1, dtype=torch.long),
                           loss_mask=torch.ones(1, dtype=torch.bool),
                           logprobs=torch.zeros(1), reward=r) for r in rewards]

    assert reward_nonzero_std(group([0.0, 1.0]))  # informative: kept
    assert not reward_nonzero_std(group([1.0, 1.0]))  # degenerate: dropped
    assert not reward_nonzero_std(group([0.5, 0.5 + 1e-9]))  # within atol: dropped


# ---------------- placement + module hygiene ----------------


def test_placement_layout_is_slimes():
    p = PlacementConfig(num_train_gpus=2, num_rollout_gpus=2)
    assert p.train_gpu_ids == [0, 1] and p.rollout_gpu_ids == [2, 3]  # trainer first
    assert not set(p.train_gpu_ids) & set(p.rollout_gpu_ids)
    q = PlacementConfig()  # 1 + 1, the smallest disaggregated box
    assert q.train_gpu_ids == [0] and q.rollout_gpu_ids == [1]


def test_vllm_engine_module_imports_without_vllm():
    # all vLLM imports are method-local: the module must import in this env
    import minirl.engine.vllm_engine as m

    assert hasattr(m, "VLLMEngine") and hasattr(m, "_apply_weights_from_file")
