"""GRPO on Dolci-RL-Zero-Code — code RLVR with the sandboxed assert reward.

    # 2 GPUs (1 trainer + 1 engine):
    torchrun --nproc-per-node=1 recipes/09_grpo_dolci_code_cuda.py --train-gpus 1 --rollout-gpus 1
    # + wandb:  ... --wandb --project miniRL_tests --name dolci-1x1

allenai/Dolci-RL-Zero-Code-7B mixes label formats inside `ground_truth[0]`
(13,312 rows): 6,664 are a JSON array of assert STRINGS — the trainable
subset — while the rest are stdin/stdout test suites (3,320 as JSON dict
arrays, 3,328 zlib+base64 PICKLED). This recipe trains on the assert
subset only: stdin/stdout needs a run-and-diff harness our assert-append
reward is not, and unpickling downloaded data is off the table. The
is_assert_style filter selects them at load. The reward path was validated
end to end on real rows THROUGH make_code_reward_fn on engine-shaped
Trajectories (chat-templated prompt ids, response-only decode): reference
solutions 30/30 -> 1.0, corrupted solutions 30/30 -> 0.0, fence-less
responses 30/30 -> 0.0, ~140 sandboxed rewards/s.

The reward: extract the response's last ```python fence, append the joined
asserts, run in the level-2 sandbox; 1.0 iff the process exits cleanly
(rewards/code.py). The row_fn appends a fence instruction to every prompt —
the fence is HOW the reward finds the code, so the format instruction is
part of the reward spec (math's boxed-answer, code edition).

Same layout rules as recipe 05: trainer ranks first, engines after
(placement mutates CUDA_VISIBLE_DEVICES around construction), trainer built
BEFORE any engine, setup_distributed() before the trainer.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.config import DataConfig, PlacementConfig, RolloutConfig
from minirl.controllers import fit_async
from minirl.data import HFPromptSource
from minirl.vllm_engine import VLLMEngine
from minirl.logging import metrics_logger
from minirl.megatron import MegatronTrainConfig, MegatronTrainer, setup_distributed
from minirl.rewards import make_code_reward_fn

MODEL = "Qwen/Qwen3-0.6B"


def is_assert_style(row: dict) -> bool:
    """The trainable subset: ground_truth[0] is a JSON array of assert
    STRINGS. Everything else — compressed suites, and JSON arrays of
    stdin/stdout dicts — needs a run-and-diff harness, not our
    assert-append reward, and is skipped."""
    e = row["ground_truth"][0].lstrip()
    if not e.startswith("["):
        return False
    try:
        tests = json.loads(e)
    except ValueError:
        return False
    return bool(tests) and all(isinstance(t, str) for t in tests)


def dolci_code_row(row: dict) -> tuple[list[dict], dict]:
    """prompt -> user turn (+ the fence instruction the reward depends on);
    ground_truth -> meta["label"] as a list of asserts for make_code_reward_fn.

    Two dataset quirks this row_fn absorbs (why keyed_row_fn isn't enough):
    prompts carry a literal "user: " prefix (redundant inside the chat
    template), and ground_truth is a 1-element list whose element is a
    JSON-ENCODED string of the assert array."""
    prompt = row["prompt"].strip().removeprefix("user: ")
    tests = json.loads(row["ground_truth"][0])  # -> ["assert f(..) == ..", ...]
    content = prompt + "\n\nWrite your solution in a ```python code block."
    return [{"role": "user", "content": content}], {"label": tests}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gpus", type=int, default=1)
    ap.add_argument("--rollout-gpus", type=int, default=1)
    ap.add_argument("--num-rollout", type=int, default=20)          # training iterations
    ap.add_argument("--n-samples-per-prompt", type=int, default=8)  # G
    ap.add_argument("--rollout-batch-size", type=int, default=8)    # prompts per batch; B = G * this
    ap.add_argument("--rollout-max-response-len", type=int, default=1024)  # code needs room
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--fp32", action="store_true",
                    help="disable bf16 (Megatron's default precision mode) — parity/debug runs only")
    ap.add_argument("--pack-max-tokens", type=int, default=None,
                    help="pack each microbatch into dense pad-free rows under this token "
                         "budget (replaces --micro-batch-size as the grad-accum unit; "
                         "needs bf16). None = padded microbatches")
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
        dynamic_sampling=True,  # all-pass/all-fail groups carry no gradient — replace them
    )
    data_cfg = DataConfig(prompt_data="allenai/Dolci-RL-Zero-Code-7B",
                          input_key="prompt", label_key="ground_truth")

    # trainer FIRST: Megatron initializes model parallelism + the CUDA
    # context before any engine mutates CUDA_VISIBLE_DEVICES.
    trainer = MegatronTrainer(MODEL, grpo_loss, loss_cfg, train_cfg)

    engines, prompt_source, reward_fn, run = [], None, None, None
    if rank == 0:
        engines = [VLLMEngine(MODEL, gpu_id=g, seed=g) for g in placement.rollout_gpu_ids]
        tok = AutoTokenizer.from_pretrained(MODEL)
        ds = load_dataset(data_cfg.prompt_data, split="train").filter(is_assert_style)
        print(f"dataset: {len(ds)} assert-style rows (stdin/stdout rows filtered out)")
        prompt_source = HFPromptSource(ds, tok, data_cfg, row_fn=dolci_code_row)
        reward_fn = make_code_reward_fn(tok)
        if args.wandb:
            import wandb  # recipes only, never core

            run = wandb.init(
                project=args.project, entity=args.entity, name=args.name,
                config={"model": MODEL, "placement": asdict(placement), "algo": "grpo",
                        "num_rollout": args.num_rollout, "loss": asdict(loss_cfg),
                        "train": asdict(train_cfg), "rollout": asdict(rollout_cfg),
                        "data": asdict(data_cfg),
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
        )
    finally:
        if run is not None:
            run.finish()
    if rank == 0:
        print("done.")


if __name__ == "__main__":
    main()
