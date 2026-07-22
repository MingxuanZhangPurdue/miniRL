"""Eval-through-the-engines tests — CPU, fake engines, no vLLM.

Pins the run_eval contract: every prompt of every set scored with the eval
sampling params, leftover TRAINING stash neither consumed nor polluted, and
the controller hook fires at the baseline + every eval_interval.
"""

from dataclasses import replace

import pytest
import torch

from minirl.config import EvalConfig
from minirl.controllers import fit_async
from minirl.eval import EvalSet, run_eval
from tests.fake_trainer import TrainConfig, Trainer
from tests.test_fully_async import FILTER_CFG, VOCAB, FakeStreamEngine, TinyLM, fresh, parity_reward

from minirl.algos import GRPOConfig, grpo_loss  # noqa: E402


def label_reward(traj) -> float:
    return float(traj.meta["label"])


def eval_prompts(n: int, label: float = 1.0) -> list:
    return [(torch.randint(1, VOCAB, (3,)), {"label": label, "pid": i}) for i in range(n)]


CFG = EvalConfig(eval_interval=1, n_samples_per_eval_prompt=2, eval_max_response_len=5)


def test_run_eval_scores_every_prompt():
    engines = fresh([FakeStreamEngine(), FakeStreamEngine(finish_after=2)])
    sets = [EvalSet("a", eval_prompts(5, label=1.0), label_reward),
            EvalSet("b", eval_prompts(3, label=0.0), label_reward)]
    m = run_eval(engines, sets, CFG)
    assert m["eval/a/reward_mean"] == 1.0 and m["eval/b/reward_mean"] == 0.0
    assert m["eval/a/response_len_mean"] == 5.0  # the fake always emits 5 tokens
    assert m["eval/a/truncated_ratio"] == 1.0  # 5 >= eval_max_response_len -> capped
    assert m["t_eval"] > 0
    for e in engines:
        assert e.n_inflight == 0  # eval leaves engines idle


def test_run_eval_respects_n_samples():
    (engine,) = fresh([FakeStreamEngine()])
    seen = []
    m = run_eval([engine], [EvalSet("s", eval_prompts(4),
                                    lambda t: seen.append(t.meta["pid"]) or 1.0)], CFG)
    assert len(seen) == 4 * CFG.n_samples_per_eval_prompt  # every prompt x n scored
    assert sorted(set(seen)) == [0, 1, 2, 3]


def test_run_eval_preserves_training_stash():
    (engine,) = fresh([FakeStreamEngine()])
    engine.submit(torch.randint(1, VOCAB, (3,)), CFG.sampling_params(), {"training": True})
    engine.drain()  # one leftover TRAINING group now sits in the stash
    assert len(engine._stash) == 1

    seen = []
    run_eval([engine], [EvalSet("s", eval_prompts(2),
                                lambda t: seen.append(t.meta) or 1.0)], CFG)
    assert all("training" not in meta for meta in seen)  # eval never scored it
    assert len(engine._stash) == 1  # and the next collection still gets it
    assert engine._stash[0][0].meta["training"] is True


def test_controller_eval_cadence_and_baseline():
    engines = fresh([FakeStreamEngine(), FakeStreamEngine()])
    torch.manual_seed(0)
    trainer = Trainer(TinyLM(), grpo_loss, GRPOConfig(),
                      TrainConfig(lr=1e-3, minibatch_size=4, micro_batch_size=4))
    history = fit_async(
        engines=engines,
        trainer=trainer,
        reward_fn=parity_reward,
        prompt_source=lambda n: [torch.randint(1, VOCAB, (3,)) for _ in range(n)],
        rollout_cfg=replace(FILTER_CFG, rollout_batch_size=2),
        num_iterations=2,
        publish_interval=1,
        eval_sets=[EvalSet("bench", eval_prompts(3), label_reward)],
        eval_cfg=EvalConfig(eval_interval=2, n_samples_per_eval_prompt=1,
                            eval_max_response_len=5),
    )
    assert history[0]["iteration"] == 0  # the untrained baseline entry
    assert history[0]["eval/bench/reward_mean"] == 1.0
    assert "loss" not in history[0]  # baseline is eval-only
    by_it = {m["iteration"]: m for m in history[1:]}
    assert "eval/bench/reward_mean" not in by_it[1]  # off-cadence
    assert "eval/bench/reward_mean" in by_it[2]  # every eval_interval=2
    assert "loss" in by_it[2]  # training metrics still present alongside


def test_controller_rejects_misaligned_cadence():
    engines = fresh([FakeStreamEngine()])
    torch.manual_seed(0)
    trainer = Trainer(TinyLM(), grpo_loss, GRPOConfig(),
                      TrainConfig(lr=1e-3, minibatch_size=4, micro_batch_size=4))
    with pytest.raises(AssertionError, match="multiple of"):
        fit_async(
            engines=engines, trainer=trainer, reward_fn=parity_reward,
            prompt_source=lambda n: [], rollout_cfg=FILTER_CFG,
            num_iterations=1, publish_interval=2,
            eval_sets=[EvalSet("x", eval_prompts(1), label_reward)],
            eval_cfg=EvalConfig(eval_interval=3),
        )
