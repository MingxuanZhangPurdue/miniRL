"""On-box VLLMEngine validation — run FIRST on a CUDA machine, 1 GPU is enough.

    python recipes/04_smoke_vllm_cuda.py

Executes the async_tier2.md §7 on-box checklist items in order, fail-loud:

  1. streaming contract: submit/poll/drain on real vLLM (n=G grouping,
     sampled logprobs attached, version stamping)
  2. EOS parity: every finished response INCLUDES its eos token — the
     HFEngine convention the trainer's loss masks assume
  3. engine<->learner logprob gap: mean/max |logpi_engine - logpi_learner|
     on the same tokens must sit inside TIS's clamp band (fp32 HF reference)
  4. weight-update canary (the §8 perturb-restore, CUDA branch this time):
     zero layer-0 q_proj via load_weights -> greedy output must CHANGE;
     restore -> greedy output must be BYTE-IDENTICAL to baseline.
     This is the check that catches the franken-policy failure mode.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.engine.vllm_engine import VLLMEngine
from minirl.rollout.types import SamplingParams
from tests.fake_trainer import Trainer, TrainConfig, gather_logprobs  # noqa: F401 — diagnostic learner (the spec fake)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Question: Natalia sold clips to 48 friends. She sold half as many the next day. How many total?\nAnswer:"


def greedy(engine, prompt_ids, max_new_tokens=64):
    engine.submit(prompt_ids, SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, n=1), {})
    while engine.n_inflight:
        for group in engine.poll():
            return group[0]


def main() -> None:
    assert torch.cuda.is_available(), "this smoke needs a CUDA GPU"
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0]

    print("== 1. streaming contract ==")
    engine = VLLMEngine(MODEL)
    engine.load_weights(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).state_dict().items(),
        version=0,
    )  # also exercises the CUDA load path once before anything else
    sampling = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=64, n=4)
    engine.submit(prompt_ids, sampling, {"tag": "smoke"})
    groups = []
    while engine.n_inflight:
        groups += engine.poll()
    (group,) = groups
    assert len(group) == 4, f"expected G=4 completions, got {len(group)}"
    assert all(t.version == 0 and t.meta["tag"] == "smoke" for t in group)
    assert all(t.logprobs[t.loss_mask].le(0).all() for t in group), "positive logprobs?!"
    assert all((t.logprobs[t.loss_mask] != 0).any() for t in group), "sampled logprobs missing (all zero)"
    print("   PASS: one request -> G trajectories, meta+version+logprobs attached")

    print("== 2. EOS parity ==")
    gc_eos = AutoModelForCausalLM.from_pretrained(MODEL).generation_config.eos_token_id
    gc_eos = [gc_eos] if isinstance(gc_eos, int) else list(gc_eos or [])
    eos_ids = set([tok.eos_token_id] + gc_eos)
    finished = [t for t in group if t.input_ids.numel() - prompt_ids.numel() < sampling.max_new_tokens]
    for t in finished:
        assert int(t.input_ids[-1]) in eos_ids, (
            f"finished response does not END with eos (last={int(t.input_ids[-1])}) — "
            "vLLM stop-token inclusion differs from HFEngine; fix engine config before training"
        )
    print(f"   PASS: {len(finished)}/{len(group)} finished responses include their eos")

    print("== 3. engine<->learner logprob gap ==")
    learner = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
    gaps = []
    with torch.no_grad():
        for t in group:
            ids = t.input_ids.unsqueeze(0).cuda()
            lp = gather_logprobs(learner(ids).logits, ids)[0].cpu()
            m = t.loss_mask
            gaps.append((lp[m] - t.logprobs[m]).abs())
    gaps = torch.cat(gaps)
    print(f"   mean {gaps.mean():.4f} nats  max {gaps.max():.4f} nats (Mac spike: 0.018 / 0.105)")
    assert gaps.mean() < 0.05, "mean logprob gap outside TIS comfort zone — investigate before training"
    del learner
    torch.cuda.empty_cache()

    print("== 4. weight-update canary (perturb-restore) ==")
    base = greedy(engine, prompt_ids)
    sd = {k: v.detach().cpu().clone() for k, v in AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).state_dict().items()}
    (qkey,) = [k for k in sd if k.endswith("layers.0.self_attn.q_proj.weight")]
    broken = dict(sd)
    broken[qkey] = torch.zeros_like(sd[qkey])
    engine.load_weights(broken.items(), version=1)
    perturbed = greedy(engine, prompt_ids)
    assert not torch.equal(perturbed.input_ids, base.input_ids), (
        "zeroing q_proj did NOT change the output — load_weights is silently "
        "skipping attention (the franken-policy trap, §8)"
    )
    engine.load_weights(sd.items(), version=2)
    restored = greedy(engine, prompt_ids)
    assert torch.equal(restored.input_ids, base.input_ids), "restore is not byte-identical"
    print("   PASS: perturb changes output, restore is byte-identical")

    print("\nALL SMOKE CHECKS PASSED — the engine is safe to train against.")


if __name__ == "__main__":
    main()
