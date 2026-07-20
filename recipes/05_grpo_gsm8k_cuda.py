"""GRPO on GSM8K — the CUDA box recipe: k vLLM engines x m DDP trainer ranks.

    # 2 GPUs (1 trainer + 1 engine):
    torchrun --nproc-per-node=1 recipes/05_grpo_gsm8k_cuda.py --train-gpus 1 --rollout-gpus 1
    # 4 GPUs (2 trainer + 2 engines) — the full layout:
    torchrun --nproc-per-node=2 recipes/05_grpo_gsm8k_cuda.py --train-gpus 2 --rollout-gpus 2
    # + wandb:  ... --wandb --project miniRL_tests --name box-2x2

Layout (slime's non-colocated ordering): trainer
ranks take GPUs 0..t-1 (torchrun LOCAL_RANK == GPU id), engines take
t..t+r-1 via VLLMEngine(gpu_id=...). Rank 0 owns the engines + collection;
followers train. Run recipes/04_smoke_vllm_cuda.py FIRST on a fresh box.

ORDER MATTERS twice here, both documented in vllm_engine.py:
  1. setup_distributed() BEFORE Trainer (it reads world at construction —
     fit_async asserts this);
  2. the learner touches CUDA BEFORE any engine is built (engine pinning
     mutates CUDA_VISIBLE_DEVICES around construction; CUDA reads the env
     once at context creation).
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.config import CollectConfig, PlacementConfig
from minirl.controllers import fit_async
from minirl.data import HFPromptSource
from minirl.data.prompts import gsm8k_row
from minirl.engine.vllm_engine import VLLMEngine
from minirl.logging import metrics_logger
from minirl.rewards import make_math_reward_fn
from minirl.rollout.types import SamplingParams
from minirl.train import TrainConfig, Trainer, setup_distributed

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gpus", type=int, default=1)
    ap.add_argument("--rollout-gpus", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=8)       # G
    ap.add_argument("--target-groups", type=int, default=8)    # B = G * this
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--bf16-weights", action="store_true",
                    help="mixed precision (Megatron-style): bf16 params + fp32 master "
                         "copies stepped by AdamW; halves param memory + publish bytes")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                    help="trainer attention impl; flash_attention_2 needs the flash-attn "
                         "package and bf16 params — pair it with --bf16-weights "
                         "(sdpa under bf16 params reaches the same flash kernels)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default="minirl")
    ap.add_argument("--name", default=None)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    placement = PlacementConfig(num_train_gpus=args.train_gpus, num_rollout_gpus=args.rollout_gpus)
    rank, world = setup_distributed()  # nccl on CUDA; BEFORE the Trainer, see banner
    assert world == placement.num_train_gpus, f"torchrun nproc {world} != --train-gpus {args.train_gpus}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # trainer rank r -> GPU r (trainer GPUs come first)

    b = args.group_size * args.target_groups
    assert b % world == 0, f"B={b} not divisible by world={world}"
    loss_cfg = GRPOConfig(use_tis=True)
    train_cfg = TrainConfig(lr=args.lr, ppo_epochs=1, minibatch_size=b, micro_batch_size=8,
                            bf16_weights=args.bf16_weights, compile=args.compile)
    collect_cfg = CollectConfig(
        group_size=args.group_size, target_groups=args.target_groups, strategy="filter"
    )
    sampling = SamplingParams(
        temperature=1.0, top_p=0.95, max_new_tokens=args.max_new_tokens, n=args.group_size
    )

    # learner FIRST (CUDA context before engine env-pinning). Load fp32 even
    # for --bf16-weights: the trainer clones its fp32 masters from the
    # PRISTINE checkpoint before casting params down. Only fa2 forces a bf16
    # LOAD (its dtype check) — masters then start from bf16-rounded weights.
    learner_dtype = torch.bfloat16 if args.attn == "flash_attention_2" else torch.float32
    learner = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=learner_dtype, attn_implementation=args.attn
    )
    trainer = Trainer(learner, grpo_loss, loss_cfg, train_cfg, device="cuda")

    engines, prompt_source, reward_fn, run = [], None, None, None
    if rank == 0:
        engines = [VLLMEngine(MODEL, gpu_id=g, seed=g) for g in placement.rollout_gpu_ids]
        tok = AutoTokenizer.from_pretrained(MODEL)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompt_source = HFPromptSource(ds, tok, row_fn=gsm8k_row, seed=0)
        reward_fn = make_math_reward_fn(tok)
        if args.wandb:
            import wandb  # recipes only, never core

            run = wandb.init(
                project=args.project, entity=args.entity, name=args.name,
                config={"model": MODEL, "placement": asdict(placement), "algo": "grpo",
                        "iterations": args.iterations, "loss": asdict(loss_cfg),
                        "train": asdict(train_cfg), "collect": asdict(collect_cfg),
                        "sampling": asdict(sampling)},
            )
        print(f"rank 0: {len(engines)} engine(s) on GPUs {placement.rollout_gpu_ids}, "
              f"{world} trainer rank(s) on GPUs {placement.train_gpu_ids}")

    try:
        fit_async(
            engines=engines,
            trainer=trainer,
            reward_fn=reward_fn,
            prompt_source=prompt_source,
            sampling=sampling,
            collect_cfg=collect_cfg,
            num_iterations=args.iterations,
            publish_interval=1,
            on_metrics=metrics_logger(run),  # rank 0 only ever emits
        )
    finally:
        if run is not None:
            run.finish()
    if rank == 0:
        print("done.")


if __name__ == "__main__":
    main()
