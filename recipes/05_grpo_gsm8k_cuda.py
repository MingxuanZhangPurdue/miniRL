"""GRPO on GSM8K — the CUDA box recipe: k vLLM engines x m Megatron ranks.

    # 2 GPUs (1 trainer + 1 engine):
    torchrun --nproc-per-node=1 recipes/05_grpo_gsm8k_cuda.py --train-gpus 1 --rollout-gpus 1
    # 4 GPUs (2 trainer + 2 engines) — the full layout:
    torchrun --nproc-per-node=2 recipes/05_grpo_gsm8k_cuda.py --train-gpus 2 --rollout-gpus 2
    # + wandb:  ... --wandb --project miniRL_tests --name box-2x2

Layout (slime's non-colocated ordering): trainer
ranks take GPUs 0..t-1 (torchrun LOCAL_RANK == GPU id), engines take
t..t+r-1 via VLLMEngine(gpu_id=...). Rank 0 owns the engines + collection;
followers train. Run recipes/04_smoke_vllm_cuda.py FIRST on a fresh box.

The learner is Megatron-Core end to end (minirl/megatron.py): the model is
built FROM the HF name by Megatron-Bridge — no transformers learner object
exists in this script. ORDER MATTERS twice:
  1. setup_distributed() BEFORE MegatronTrainer (it asserts the process
     group and initializes model parallelism);
  2. the trainer touches CUDA BEFORE any engine is built (engine pinning
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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.config import DataConfig, EvalConfig, PlacementConfig, RolloutConfig
from minirl.controllers import fit_async
from minirl.data import HFPromptSource, gsm8k_row
from minirl.eval import EvalSet, make_eval_prompts
from minirl.vllm_engine import VLLMEngine
from minirl.logging import metrics_logger
from minirl.megatron import MegatronTrainConfig, MegatronTrainer, setup_distributed
from minirl.rewards import make_math_reward_fn

MODEL = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gpus", type=int, default=1)
    ap.add_argument("--rollout-gpus", type=int, default=1)
    ap.add_argument("--num-rollout", type=int, default=20)          # training iterations
    ap.add_argument("--n-samples-per-prompt", type=int, default=8)  # G
    ap.add_argument("--rollout-batch-size", type=int, default=8)    # prompts per batch; B = G * this
    ap.add_argument("--rollout-max-response-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 (Megatron's default precision mode) — parity/debug runs only")
    ap.add_argument("--pack-max-tokens", type=int, default=None,
                    help="pack each microbatch into dense pad-free rows under this token "
                         "budget (replaces --micro-batch-size as the grad-accum unit; "
                         "needs bf16). None = padded microbatches")
    ap.add_argument("--eval-interval", type=int, default=None,
                    help="score the gsm8k test split every N iterations (plus an "
                         "untrained baseline); None = no eval")
    ap.add_argument("--eval-limit", type=int, default=256,
                    help="how many test prompts each eval scores")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default="minirl")
    ap.add_argument("--name", default=None)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    placement = PlacementConfig(num_train_gpus=args.train_gpus, num_rollout_gpus=args.rollout_gpus)
    rank, world = setup_distributed()  # nccl on CUDA; BEFORE the trainer, see banner
    assert world == placement.num_train_gpus, f"torchrun nproc {world} != --train-gpus {args.train_gpus}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)  # trainer rank r -> GPU r (trainer GPUs come first)

    b = args.rollout_batch_size * args.n_samples_per_prompt  # whole rollout = one optimizer step
    assert b % world == 0, f"B={b} not divisible by world={world}"
    loss_cfg = GRPOConfig(use_tis=True)
    train_cfg = MegatronTrainConfig(
        lr=args.lr, ppo_epochs=1, minibatch_size=b, micro_batch_size=8,
        bf16=not args.fp32, pack_max_tokens=args.pack_max_tokens,
    )
    rollout_cfg = RolloutConfig(
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        rollout_temperature=1.0, rollout_top_p=0.95,
        rollout_max_response_len=args.rollout_max_response_len,
        dynamic_sampling=True,
    )
    data_cfg = DataConfig(prompt_data="openai/gsm8k", input_key="question", label_key="answer")

    # trainer FIRST: Megatron initializes model parallelism + the CUDA
    # context before any engine mutates CUDA_VISIBLE_DEVICES.
    trainer = MegatronTrainer(MODEL, grpo_loss, loss_cfg, train_cfg)

    engines, prompt_source, reward_fn, run = [], None, None, None
    eval_sets, eval_cfg = [], None
    if rank == 0:
        engines = [VLLMEngine(MODEL, gpu_id=g, seed=g) for g in placement.rollout_gpu_ids]
        tok = AutoTokenizer.from_pretrained(MODEL)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompt_source = HFPromptSource(ds, tok, data_cfg, row_fn=gsm8k_row)
        reward_fn = make_math_reward_fn(tok)
        if args.eval_interval:
            eval_ds = load_dataset("openai/gsm8k", "main", split="test")
            eval_sets = [EvalSet("gsm8k_test",
                                 make_eval_prompts(eval_ds, tok, gsm8k_row, limit=args.eval_limit),
                                 reward_fn)]
            eval_cfg = EvalConfig(eval_interval=args.eval_interval,
                                  eval_max_response_len=args.rollout_max_response_len)
        if args.wandb:
            import wandb  # recipes only, never core

            run = wandb.init(
                project=args.project, entity=args.entity, name=args.name,
                config={"model": MODEL, "placement": asdict(placement), "algo": "grpo",
                        "num_rollout": args.num_rollout, "loss": asdict(loss_cfg),
                        "train": asdict(train_cfg), "rollout": asdict(rollout_cfg),
                        "data": asdict(data_cfg),
                        "eval": asdict(eval_cfg) if eval_cfg else None,
                        "reward_fn": reward_fn.__qualname__},  # reward is code, not config — record its name
            )
        print(f"rank 0: {len(engines)} engine(s) on GPUs {placement.rollout_gpu_ids}, "
              f"{world} trainer rank(s) on GPUs {placement.train_gpu_ids}")

    try:
        fit_async(
            engines=engines,
            trainer=trainer,
            reward_fn=reward_fn,
            prompt_source=prompt_source,
            rollout_cfg=rollout_cfg,
            num_iterations=args.num_rollout,
            publish_interval=1,
            on_metrics=metrics_logger(run),  # rank 0 only ever emits
            eval_sets=eval_sets,
            eval_cfg=eval_cfg,
        )
    finally:
        if run is not None:
            run.finish()
    if rank == 0:
        print("done.")


if __name__ == "__main__":
    main()
