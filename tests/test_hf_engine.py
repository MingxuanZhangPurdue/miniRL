"""Integration tests for HFEngine against the real base model.

Runs on MPS/CPU (no CUDA needed). The logprob round-trip test is the Phase 0
exit criterion from DESIGN §10: engine sampling logprobs must match a
learner-style forward-pass recomputation.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from minirl.engine import HFEngine
from minirl.rollout.types import SamplingParams

MODEL = os.environ.get("MINIRL_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def engine() -> HFEngine:
    # fp32 EVERYWHERE, including CUDA (where HFEngine defaults to bf16): the
    # logprob round-trip below pins ALIGNMENT logic to 1e-3, and bf16's
    # decode-vs-prefill kernel divergence alone is ~1e-1 nats on near-tie
    # tokens (first observed on the A100 box, 2026-07-16) — real, harmless,
    # and TIS-corrected in training; not what this test measures.
    return HFEngine(MODEL, max_batch_size=4, dtype=torch.float32)


@pytest.fixture(scope="module")
def prompts(engine: HFEngine) -> list[torch.Tensor]:
    # Two chat prompts of different lengths to exercise left-padding.
    def encode(text: str) -> torch.Tensor:
        ids = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )["input_ids"]
        return ids[0]  # (T,)

    return [encode("What is 2+2?"), encode("Name three colors of the rainbow, briefly.")]


def test_generate_contract(engine, prompts):
    torch.manual_seed(0)
    params = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=24, n=2)
    trajs = engine.generate(prompts, params)

    assert len(trajs) == len(prompts) * params.n  # grouped: [p0_s0, p0_s1, p1_s0, p1_s1]
    for k, traj in enumerate(trajs):
        prompt = prompts[k // params.n]
        t_p = prompt.numel()
        assert torch.equal(traj.input_ids[:t_p], prompt), "trajectory must start with its prompt"
        assert traj.prompt_len == t_p and traj.response_len >= 1
        assert not traj.loss_mask[:t_p].any() and traj.loss_mask[t_p:].all()
        assert (traj.logprobs[:t_p] == 0).all(), "prompt positions carry no behavior logprob"
        assert traj.logprobs[t_p:].isfinite().all() and (traj.logprobs[t_p:] <= 0).all()
        last = traj.input_ids[-1]
        assert traj.response_len == params.max_new_tokens or bool(torch.isin(last, engine.eos_ids))


def test_logprobs_match_learner_recompute(engine, prompts):
    """Engine behavior logprobs == learner forward-pass recompute (same module)."""
    torch.manual_seed(0)
    trajs = engine.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_new_tokens=32))

    for traj in trajs:
        seq = traj.input_ids.unsqueeze(0).to(engine.device)  # (1, T)
        with torch.no_grad():
            logits = engine.model(seq).logits.float()  # (1, T, V)
        # logits[:, t] predicts token t+1 -> shift left to align with input_ids.
        lps = F.log_softmax(logits[:, :-1], dim=-1)  # (1, T-1, V)
        recomputed = lps.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)[0].cpu()  # (T-1,)

        mask = traj.loss_mask[1:]  # response positions, in shifted coordinates
        gap = (recomputed[mask] - traj.logprobs[1:][mask]).abs()
        assert gap.max() < 1e-3, f"engine/learner logprob gap too large: {gap.max():.2e}"


def test_greedy_is_deterministic(engine, prompts):
    params = SamplingParams(temperature=0.0, max_new_tokens=16)
    a = engine.generate(prompts, params)
    b = engine.generate(prompts, params)
    for x, y in zip(a, b):
        assert torch.equal(x.input_ids, y.input_ids)


def test_load_weights_updates_policy_and_version(engine, prompts):
    params = SamplingParams(temperature=0.0, max_new_tokens=8)
    before = engine.generate(prompts[:1], params)[0]

    # Perturb one mlp weight (clone only that tensor — the rest are references).
    sd = engine.model.state_dict()
    key = "model.layers.0.mlp.down_proj.weight"
    original = sd[key].clone()
    sd[key] = original + torch.randn_like(original) * 0.5
    engine.load_weights(iter(sd.items()), version=1)

    assert engine.version == 1
    after = engine.generate(prompts[:1], params)[0]
    assert not torch.equal(before.input_ids, after.input_ids), "new weights must change greedy output"
    assert after.version == 1 and before.version == 0

    sd[key] = original  # restore for other tests
    engine.load_weights(iter(sd.items()), version=2)
