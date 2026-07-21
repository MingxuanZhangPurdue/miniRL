# CUDA box runbook (RunPod, single node, 4 GPUs: 2 train + 2 rollout)

The validation ladder for the first GPU session. Each rung must pass before
the next; every rung maps to an item in async_tier2.md §7 / DESIGN §0
REMAINING. Written 2026-07-16, before the first box session.

## 0. Pod choice + environment

- Any 4x pod with >= 24GB/GPU works for 0.6B (4x RTX 4090 is the cheap
  choice; A40/A100 fine). Pick a PyTorch/CUDA 12.x template.
- ONE python env on the box — vLLM and the trainer share it (the Mac's
  two-env split was a vllm-metal quirk, not the architecture; vLLM's pinned
  torch becomes the torch):

      pip install vllm transformers datasets math-verify wandb pytest
      cd miniRL && python -m pytest tests/ -q       # 71 CPU tests must pass here too

## 1. Engine smoke + canary (1 GPU) — PASSED 2026-07-20 (A100)

      python recipes/04_smoke_vllm_cuda.py

  Validates, fail-loud: streaming contract on real vLLM; EOS parity
  (responses INCLUDE eos — the loss-mask convention); engine<->learner
  logprob gap inside the TIS band (Mac spike reference: mean 0.018 nats);
  and the perturb-restore weight canary — the CUDA `load_weights` branch's
  first real execution (catches the franken-policy failure mode).
  NOTE: callable-RPC may need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (vLLM V1
  requirement, not Metal-specific).

## 2. Placement spike (§10(a)) — pinning without torchrun

      python recipes/05_grpo_gsm8k_cuda.py --train-gpus 1 --rollout-gpus 2 --iterations 2

  While it runs, `watch nvidia-smi`: the learner must sit on GPU 0 and TWO
  EngineCore processes on GPUs 1 and 2. If both engines land on one GPU, the
  CUDA_VISIBLE_DEVICES-around-construction mechanism failed -> fall back to
  §10(c) (one OS process per engine) and file the finding in async_tier2 §11.

## 3. Single-rank end-to-end (2 GPUs)

      python recipes/05_grpo_gsm8k_cuda.py --train-gpus 1 --rollout-gpus 1 \
          --iterations 20 --wandb --project miniRL_tests --name box-1x1

  Watch: rollout/reward_mean drifts up over ~20 iterations (GSM8K + 0.6B
  moves fast); train/tis_mean ~= 1.0 +- 0.05; async/staleness <= 2;
  time/tokens_per_sec is the throughput baseline for comparisons.

## 4. The full 2+2 (4 GPUs, NCCL's first run)

      torchrun --nproc-per-node=2 recipes/05_grpo_gsm8k_cuda.py \
          --train-gpus 2 --rollout-gpus 2 --iterations 20 \
          --wandb --project miniRL_tests --name box-2x2

  Checks: no hang at iteration 1 (broadcast + all-reduce over NCCL);
  nvidia-smi shows ranks on GPUs 0-1, engines on 2-3; loss curve comparable
  to rung 3 at the same configs; t_iter ideally < rung 3's (2 engines feed
  faster + training halves).

## 5. Throughput notes to record (docs/fast_rl.md follow-up)

  t_generate vs t_train from the metrics decide everything downstream:
  engine-bound -> more rollout GPUs / bigger max_num_seqs; trainer-bound ->
  --bf16-weights + bigger micro_batch_size, then A/B --compile (keep only if
  mean t_train excluding iter 1 improves ~10%+).

Known-unvalidated list this session retires: CUDA load_weights branch (1),
EOS parity (1), gpu_id pinning (2), NCCL backend + broadcast at world 2 (4).
Update DESIGN §0 + async_tier2 §7/§11 with findings afterwards.

## 6. Megatron P1 parity — PASSED 2026-07-20, on a DIFFERENT box

Not a RunPod rung: P1 (megatron.md §7) ran on the Windows 11 / RTX 5070
home machine through Docker Desktop's WSL2 GPU passthrough — container
recipe and the validated version stack are recorded in megatron.md §5a/§6
(per the §6 rule that the environment is written down the day P1 passes).

      docker exec -w /workspace/miniRL -e PYTHONPATH=/workspace/miniRL \
          minirl-mega python recipes/08_megatron_p1_parity.py [--bf16]

That box has ONE GPU: rungs 2-4 above (placement, 1x1 end-to-end, 2+2
NCCL) still need this runbook's multi-GPU pod, as does Megatron P3 (DP>1).
The 73-test CPU suite also passes inside the container (~70s).
