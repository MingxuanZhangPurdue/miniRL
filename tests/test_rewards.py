"""Tests for the math verifier, the code sandbox, and label plumbing."""

import time

import pytest
import torch

from minirl.config import RolloutConfig
from minirl.controllers import collect_groups_dp
from minirl.rewards import (
    code_reward, extract_code, extract_final_answer, grade_answer,
    make_code_reward_fn, math_reward, run_python,
)
from minirl.rewards.math import extract_boxed
from minirl.rollout.types import SamplingParams, Trajectory

# ---------------- math: extraction ----------------


def test_boxed_extraction_handles_nesting():
    assert extract_boxed(r"so we get \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{1} then \boxed{42}") == "42"  # last one wins
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{unbalanced") is None  # failure -> reward 0, not crash


def test_final_answer_extraction():
    assert extract_final_answer(r"The answer is \boxed{72}.") == "72"
    assert extract_final_answer("First 5 eggs, then 3 more, total 8") == "8"  # last number
    assert extract_final_answer("she pays $1,200 total") == "$1,200"
    assert extract_final_answer("no numbers at all") is None


# ---------------- math: grading ----------------


def test_numeric_equivalence():
    assert grade_answer("0.5", "1/2")  # exact rational equality
    assert grade_answer("50%", "0.5")
    assert grade_answer("$1,200", "1200")
    assert grade_answer("72.0", "72")
    assert not grade_answer("71", "72")
    assert not grade_answer(None, "72")


def test_gsm8k_style_label():
    gold = "She sells 16 - 3 - 4 = 9 eggs.\n#### 18"
    assert math_reward(r"...so she makes \boxed{18} dollars", gold) == 1.0
    assert math_reward("the total is 18", gold) == 1.0  # last-number fallback
    assert math_reward("the total is 18 dollars, maybe 19", gold) == 0.0  # last number wins: 19


def test_symbolic_fallback_via_math_verify():
    pytest.importorskip("math_verify")
    assert grade_answer(r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}")  # equal expressions
    assert not grade_answer("x + 2", "x + 3")


# ---------------- code sandbox ----------------


def test_code_fence_extraction():
    text = "First try:\n```python\nbad\n```\nFixed:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(text) == "def add(a, b):\n    return a + b"  # last fence wins
    assert extract_code("no code here") is None


def test_code_reward_pass_and_fail():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    broken = "```python\ndef add(a, b:\n```"
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    assert code_reward(good, tests) == 1.0
    assert code_reward(bad, tests) == 0.0  # assertion fails -> nonzero exit
    assert code_reward(broken, tests) == 0.0  # syntax error
    assert code_reward("no fence", tests) == 0.0  # format is part of the spec


def test_make_code_reward_fn_joins_list_labels():
    """The Dolci shape: meta["label"] is a LIST of assert strings."""

    class StubTok:
        def __init__(self, response):
            self.response = response

        def decode(self, ids, skip_special_tokens=True):
            return self.response

    def traj(label):
        mask = torch.tensor([False, True, True])  # prompt_len = 1
        return Trajectory(input_ids=torch.tensor([1, 2, 3]), loss_mask=mask,
                          logprobs=torch.zeros(3), meta={"label": label})

    good = "```python\ndef mul(a, n):\n    return a * n\n```"
    asserts = ["assert mul(2, 3) == 6", "assert mul(0, 5) == 0"]
    assert make_code_reward_fn(StubTok(good))(traj(asserts)) == 1.0
    assert make_code_reward_fn(StubTok(good))(traj(["assert mul(2, 3) == 7"])) == 0.0
    assert make_code_reward_fn(StubTok(good))(traj("assert mul(4, 4) == 16")) == 1.0  # str label too


def test_sandbox_kills_infinite_loop_quickly():
    t0 = time.perf_counter()
    ok, err = run_python("while True:\n    pass", timeout_s=1.5)
    assert not ok and err == "timeout"
    assert time.perf_counter() - t0 < 10  # killed promptly, run survives


def test_sandbox_stray_writes_land_in_tempdir(tmp_path):
    # the child runs in a throwaway cwd — writing "output.txt" must not appear here
    ok, _ = run_python("open('output.txt', 'w').write('oops')")
    assert ok  # the write itself succeeds...
    import os

    assert not os.path.exists("output.txt")  # ...but never lands in OUR cwd


# ---------------- label plumbing through the collector ----------------

CFG2 = RolloutConfig(rollout_batch_size=2, n_samples_per_prompt=2, rollout_max_response_len=2)


class StreamShim:
    """Streaming-contract shim around a plain closure, so a test 'engine' is
    one lambda; one poll finishes everything submitted (one poll == one round,
    the semantics the retired StreamAdapter had)."""

    pad_id = 0

    def __init__(self, fn):
        self.fn = fn
        self.version = 0
        self._pending: list[tuple[torch.Tensor, dict]] = []
        self._stash: list[list[Trajectory]] = []

    @property
    def n_inflight(self) -> int:
        return len(self._pending)

    def submit(self, prompt_ids, params, meta=None) -> str:
        self._pending.append((prompt_ids, dict(meta or {})))
        return f"req-{len(self._pending)}"

    def poll(self) -> list[list[Trajectory]]:
        out, self._stash = self._stash, []
        for prompt, meta in self._pending:
            group = self.fn([prompt])  # the closure yields ONE prompt's group
            for t in group:
                t.meta.update(meta)
                t.version = self.version
            out.append(group)
        self._pending = []
        return out

    def stash(self, group: list[Trajectory]) -> None:
        self._stash.append(group)

    def drain(self) -> None:
        while self._pending:
            for group in self.poll():
                self.stash(group)

    def load_weights(self, named_tensors, version: int) -> None:
        self.version = version


def mk_traj(prompt: torch.Tensor, last: int) -> Trajectory:
    n = prompt.numel()
    ids = torch.cat([prompt, torch.tensor([7, last])])
    mask = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(2, dtype=torch.bool)])
    return Trajectory(input_ids=ids, loss_mask=mask, logprobs=torch.zeros(n + 2))


def test_prompt_meta_reaches_reward_fn():
    def prompt_source(n):
        return [(torch.tensor([10 + i]), {"label": str(20 + i)}) for i in range(n)]

    def generate(prompts):  # 2 samples per prompt; sample index encoded in last token
        return [mk_traj(p, j) for p in prompts for j in range(2)]

    seen: list[str] = []

    def reward_fn(traj):
        seen.append(traj.meta["label"])  # label arrived BEFORE reward ran
        return float(traj.input_ids[-1].item())

    engine = StreamShim(generate)
    trajs, stats = collect_groups_dp([engine], reward_fn, prompt_source, CFG2)
    assert stats["groups"] == 2 and seen == ["20", "20", "21", "21"]
    assert trajs[0].meta["label"] == "20" and trajs[2].meta["label"] == "21"
    assert trajs[0].meta["group_id"] == 0  # group ids still assigned


def test_reward_fn_none_for_env_scored_episodes():
    def generate(prompts):  # the "environment" scores episodes itself
        out = []
        for p in prompts:
            for j in range(2):
                t = mk_traj(p, j)
                t.reward = 0.75
                out.append(t)
        return out

    engine = StreamShim(generate)
    trajs, _ = collect_groups_dp(
        [engine], None, lambda n: [torch.tensor([1])] * n,
        RolloutConfig(rollout_batch_size=1, n_samples_per_prompt=2, rollout_max_response_len=2),
    )
    assert all(t.reward == 0.75 for t in trajs)  # untouched by the collector
