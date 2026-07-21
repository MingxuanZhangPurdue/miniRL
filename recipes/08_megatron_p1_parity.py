"""Megatron P1 parity spike (docs/megatron.md §7 P1) — single GPU, box only.

Bridge-loads Qwen3-0.6B into Megatron-Core, runs TWO GRPO minibatch steps,
and checks loss / grad_norm / logprobs against the demoted DDP trainer
(tests/fake_trainer.py) on the SAME frozen batch. Runs the fake first, frees
it, then Megatron — 12GB cards hold one 0.6B trainer at a time, not two.

    # tight parity (both trainers fp32) — the convention check:
    PYTHONPATH=. python recipes/08_megatron_p1_parity.py
    # the real mode (Megatron bf16 + fp32 masters) vs the fp32 fake — loose band:
    PYTHONPATH=. python recipes/08_megatron_p1_parity.py --bf16

Checks (each prints its measured numbers; the run fails loud on any breach):
  0  shift adapter, pure arithmetic: _ce_to_logprobs(manual fused-CE map) ==
     gather_logprobs on the SAME HF logits (banner convention 3, atol 1e-5)
  1  cross-impl logprobs: Megatron compute_logprobs vs fake compute_logprobs
     on loss_mask positions — kernel-noise band (docs/precision.md §3); a
     shift-by-one bug here reads as ~5 nats, not 1e-2
  2  step-1 loss: on-policy ratio==1 exactly, so loss == -agg(A) in BOTH
     impls independent of model numerics — near-exact match required
  3  step-1 grad_norm: the real signal (grads flow through d logpi / d theta)
  4  step-2 loss + grad_norm: policies drifted by one lr=1e-4 update — loss
     now depends on model numerics in full (ratio, clip, the works)

Step 1 loss parity being trivially exact is BY DESIGN (it isolates the
aggregation/denominator path); step 3/4 carry the model-parity burden.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.megatron import (  # file-level megatron imports: box-only, like this recipe
    MegatronTrainConfig, MegatronTrainer, _ce_to_logprobs, _labels, setup_distributed,
)
from minirl.rollout.batching import make_batch
from minirl.rollout.types import Batch, Trajectory

MODEL = "Qwen/Qwen3-0.6B"
LR = 1e-5  # spike-only: big enough that the step-2 drift dwarfs kernel noise,
# small enough to stay out of the chaotic regime (lr=1e-4 on random-token
# batches gave approx_kl ~ 1e6 after ONE step — divergence amplifies any
# kernel noise and the step-2 comparison stops measuring implementation parity)
SEED = 0


def frozen_batch(vocab: int) -> Batch:
    """Deterministic GRPO-shaped batch: 8 rows, 4 groups of G=2, alternating
    rewards (so every group has signal), variable response lengths."""
    g = torch.Generator().manual_seed(SEED)
    trajs = []
    for i in range(8):
        prompt_len, resp_len = 8, 16 + 2 * i  # right-padding gets exercised
        n = prompt_len + resp_len
        ids = torch.randint(0, vocab, (n,), generator=g)
        mask = torch.cat([torch.zeros(prompt_len, dtype=torch.bool), torch.ones(resp_len, dtype=torch.bool)])
        lps = torch.where(mask, torch.randn(n, generator=g).abs().neg(), torch.zeros(n))
        trajs.append(Trajectory(input_ids=ids, loss_mask=mask, logprobs=lps,
                                reward=float(i % 2), meta={"group_id": i // 2}))
    batch, _ = make_batch(trajs, pad_id=0)
    return batch


def two_steps(trainer, batch: Batch) -> dict:
    """old_logprobs recompute + two optimizer steps on the whole batch as ONE
    minibatch (no shuffle in the way; micro_batch_size in the cfg exercises
    accumulation). Returns everything the parity table needs."""
    old = trainer.compute_logprobs(batch)  # (B, T) f32, this impl's own kernels
    b = replace(batch, old_logprobs=old)
    m1 = trainer.step(b)
    m2 = trainer.step(b)
    return {"logprobs": old, "loss1": m1["loss"], "gn1": m1["grad_norm"],
            "loss2": m2["loss"], "gn2": m2["grad_norm"],
            "kl2": m2["approx_kl"], "clip2": m2["clip_frac"]}


def check(name: str, measured: float, limit: float, failures: list) -> None:
    ok = measured < limit
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} {measured:.3e}  (limit {limit:.0e})")
    if not ok:
        failures.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="store_true",
                    help="Megatron in its real mode (bf16 + fp32 masters) vs the fp32 fake")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--backend", default="nccl")
    args = ap.parse_args()

    # single-process launch: fabricate the torchrun env so setup_distributed
    # actually joins a (world 1) process group — MegatronTrainer asserts it.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    setup_distributed(args.backend)
    torch.cuda.set_device(0)
    # NGC containers force TF32 (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1): "fp32"
    # matmuls would carry ~1e-3 noise and drown the tight parity bands.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoConfig
    vocab = AutoConfig.from_pretrained(args.model).vocab_size
    batch = frozen_batch(vocab)
    mask = batch.loss_mask  # (B, T) — every comparison lives on these positions
    print(f"frozen batch: rows={batch.input_ids.shape[0]} T={batch.input_ids.shape[1]} "
          f"tokens={int(mask.sum())} vocab={vocab}")
    failures: list[str] = []

    # ---------------- fake trainer (the reference), fp32, then FREED ----------------
    from transformers import AutoModelForCausalLM
    from tests.fake_trainer import TrainConfig, Trainer, gather_logprobs

    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    fake = Trainer(hf, grpo_loss, GRPOConfig(), TrainConfig(
        lr=LR, minibatch_size=8, micro_batch_size=4), device="cuda")

    # check 0 — the shift adapter against HF logits, pure arithmetic
    with torch.no_grad():
        ids = batch.input_ids[:2].cuda()
        logits = hf(ids, attention_mask=batch.attention_mask[:2].cuda()).logits.float()  # (2, T, V)
        ce = F.cross_entropy(  # manual fused-CE map: ce[:, t] = -log p(labels[t] | <=t)
            logits.flatten(0, 1), _labels(ids).flatten(), reduction="none").view(ids.shape)
        shift_gap = (_ce_to_logprobs(ce) - gather_logprobs(logits, ids)).abs().max().item()
    print("\ncheck 0 — shift adapter (_ce_to_logprobs vs gather_logprobs, same logits):")
    check("max |adapter - gather|", shift_gap, 1e-5, failures)

    ref = two_steps(fake, batch)
    del fake, hf
    torch.cuda.empty_cache()

    # ---------------- Megatron trainer (the subject) ----------------
    # use_te_layers follows the mode: the bf16 leg runs the PRODUCTION config
    # (TE spec, the default); the fp32 leg needs true-fp32 GEMMs, which TE
    # cannot deliver (TF32 regardless of torch flags) -> local spec.
    mega = MegatronTrainer(args.model, grpo_loss, GRPOConfig(), MegatronTrainConfig(
        lr=LR, minibatch_size=8, micro_batch_size=4, bf16=args.bf16,
        use_te_layers=args.bf16, grad_reduce_in_fp32=True))
    got = two_steps(mega, batch)

    # ---------------- the parity table ----------------
    # bands: fp32-vs-fp32 isolates conventions (tight — local layer spec is
    # true fp32; TE would inject TF32, see MegatronTrainConfig.use_te_layers).
    # bf16-vs-fp32 stacks WEIGHT quantization (~0.4% relative, steepened by
    # the random-token batch) on the kernel-noise band of precision.md §3 —
    # measured 3.8e-2 mean nats on this box, 2026-07-20.
    lp_mean_lim, lp_max_lim, gn_lim, l2_lim = (
        (5e-2, 5e-1, 1e-1, 1e-1) if args.bf16 else (2e-3, 2e-2, 1e-2, 2e-2))
    d = (got["logprobs"] - ref["logprobs"])[mask]
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)

    print(f"\ncheck 1 — cross-impl logprobs on {int(mask.sum())} response tokens:")
    check("mean |mcore - fake| (nats)", d.abs().mean().item(), lp_mean_lim, failures)
    check("max  |mcore - fake| (nats)", d.abs().max().item(), lp_max_lim, failures)
    # ABSOLUTE diff: GRPO advantages are zero-mean per group, so -agg(A) ~ 0
    # by construction and a relative diff would divide by ~nothing.
    print(f"\ncheck 2 — step-1 loss (on-policy, == -agg(A) ~ 0 in both):  "
          f"mcore {got['loss1']:+.6f}  fake {ref['loss1']:+.6f}")
    check("abs diff loss1", abs(got["loss1"] - ref["loss1"]), 1e-5, failures)
    print(f"\ncheck 3 — step-1 grad_norm:  mcore {got['gn1']:.6f}  fake {ref['gn1']:.6f}")
    check("rel diff grad_norm1", rel(got["gn1"], ref["gn1"]), gn_lim, failures)
    print(f"\ncheck 4 — step-2 (drifted) loss / grad_norm:\n"
          f"  loss2  mcore {got['loss2']:+.6f}  fake {ref['loss2']:+.6f}   "
          f"kl2 mcore {got['kl2']:.2e} fake {ref['kl2']:.2e}\n"
          f"  gnorm2 mcore {got['gn2']:.6f}  fake {ref['gn2']:.6f}")
    if args.bf16:
        # step-2 parity vs an fp32 reference is UNDEFINED in bf16: the lr=1e-5
        # update is sub-ulp for most bf16 params, so mcore's forward weights
        # lag its fp32 masters while the fake's fp32 weights take the full
        # step — different theta_2 by construction, not by bug (the fake's
        # own test_bf16_weights_preserves_sub_ulp_updates pins this). Assert
        # the SIGNATURE of that semantics instead: mcore drifted far less.
        ok = got["kl2"] < ref["kl2"]
        print(f"  {'PASS' if ok else 'FAIL'}  bf16 sub-ulp signature (mcore kl2 << fp32 fake kl2)")
        if not ok:
            failures.append("bf16 sub-ulp signature")
    else:
        check("rel diff loss2", rel(got["loss2"], ref["loss2"]), l2_lim, failures)
        check("rel diff grad_norm2", rel(got["gn2"], ref["gn2"]), gn_lim, failures)

    mode = "bf16" if args.bf16 else "fp32"
    if failures:
        print(f"\nP1 parity ({mode}): FAIL — {failures}")
        sys.exit(1)
    print(f"\nP1 parity ({mode}): PASS")


if __name__ == "__main__":
    main()
