"""GRPO on GSM8K — the first real RLVR recipe.

    python recipes/03_grpo_gsm8k.py                          # console only
    python recipes/03_grpo_gsm8k.py --wandb --project P --name N [--entity E]
    WANDB_MODE=offline python recipes/03_grpo_gsm8k.py --wandb ...   # no network

This is the actual recipe, with SMOKE-SIZED constants so it runs end to end on
a Mac (MPS) in a couple of minutes. To do a real run, scale the CONFIG block
(bigger groups, more iterations, a GPU + vllm engine) — nothing else changes.

Wires together the finished stack, nothing bespoke:
  data.HFPromptSource + gsm8k_row     -> prompts with gold answers in meta
  StreamAdapter(HFEngine)             -> rollouts (bf16 CUDA / fp32 MPS); the
                                         adapter speaks the streaming contract
                                         (one poll == one round)
  rewards.make_math_reward_fn         -> verifiable reward (boxed/last-number)
  Trainer(grpo_loss)                  -> the GRPO update
  fit_async                           -> THE fully-async loop (k=1 engine,
                                         world=1 — its smallest configuration)
  logging.metrics_logger              -> console line + optional wandb stream
                                         (wandb lives HERE, never in minirl core)

Scaling this recipe = swapping the CONFIG block + the engine list: on a GPU
box, engines = [VLLMEngine(MODEL, gpu_id=g, seed=g) for g in
placement.rollout_gpu_ids] and torchrun provides the trainer ranks
(docs/async_tier2.md §11).

Missing on purpose (kept simple): eval harness, checkpointing schedule.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.controllers import fit_async
from minirl.config import CollectConfig
from minirl.data import HFPromptSource
from minirl.data.prompts import gsm8k_row
from minirl.engine import HFEngine, StreamAdapter
from minirl.logging import metrics_logger
from minirl.rewards import make_math_reward_fn
from minirl.rollout.types import SamplingParams
from minirl.train import TrainConfig, Trainer

# ======================= CONFIG (the only thing you scale) =======================
MODEL = "Qwen/Qwen3-0.6B"
NUM_ITERATIONS = 2          # real run: hundreds+
GROUP_SIZE = 4              # G completions per prompt; real run: 8-16
TARGET_GROUPS = 2           # prompts per rollout batch (B = G*this); real run: 32+
MAX_NEW_TOKENS = 160        # real run: 512-1024 for full chains of thought
N_TRAIN_EXAMPLES = 64       # subset of GSM8K to sample prompts from (smoke); real: full split
LR = 1e-6
# ================================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true", help="stream metrics to wandb")
    ap.add_argument("--project", default="minirl")
    ap.add_argument("--name", default=None, help="wandb run name")
    ap.add_argument("--entity", default=None, help="wandb entity (team/user)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  model={MODEL}")

    # --- configs first, so a wandb run records the FULL experiment identity ---
    loss_cfg = GRPOConfig(use_tis=True)  # TIS on: corrects the engine<->learner logprob gap
    train_cfg = TrainConfig(lr=LR, ppo_epochs=1, minibatch_size=GROUP_SIZE * TARGET_GROUPS, micro_batch_size=4)
    collect_cfg = CollectConfig(group_size=GROUP_SIZE, target_groups=TARGET_GROUPS, strategy="fixed")
    sampling = SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=MAX_NEW_TOKENS, n=GROUP_SIZE)

    run = None
    if args.wandb:
        import wandb  # imported HERE only — minirl core never sees it

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.name,
            config={
                "model": MODEL, "device": device, "num_iterations": NUM_ITERATIONS,
                "n_train_examples": N_TRAIN_EXAMPLES, "algo": "grpo",
                "loss": asdict(loss_cfg), "train": asdict(train_cfg),
                "collect": asdict(collect_cfg), "sampling": asdict(sampling),
            },
        )

    # --- models: engine (rollouts) + learner (training), both from one checkpoint ---
    engine = HFEngine(MODEL, device=device)
    engines = [StreamAdapter(engine)]  # k=1; a GPU box lists one VLLMEngine per rollout GPU
    learner = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    trainer = Trainer(learner, grpo_loss, loss_cfg, train_cfg, device=device)

    # --- data: GSM8K prompts with gold answers riding in meta ---
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{N_TRAIN_EXAMPLES}]")
    prompt_source = HFPromptSource(ds, engine.tokenizer, row_fn=gsm8k_row, seed=0)
    reward_fn = make_math_reward_fn(engine.tokenizer)  # grades response vs meta["answer"]

    print("starting GRPO...")
    try:
        fit_async(
            engines=engines,
            trainer=trainer,
            reward_fn=reward_fn,
            prompt_source=prompt_source,
            sampling=sampling,
            collect_cfg=collect_cfg,
            num_iterations=NUM_ITERATIONS,
            publish_interval=1,
            on_metrics=metrics_logger(run),  # console line always; wandb stream if run
        )
    finally:
        if run is not None:
            run.finish()  # flush even if the loop raised
    print("done.")


if __name__ == "__main__":
    main()
