"""GRPO on GSM8K — the first real RLVR recipe.  python recipes/03_grpo_gsm8k.py

This is the actual recipe, with SMOKE-SIZED constants so it runs end to end on
a Mac (MPS) in a couple of minutes. To do a real run, scale the CONFIG block
(bigger groups, more iterations, a GPU + vllm engine) — nothing else changes.

Wires together the finished stack, nothing bespoke:
  data.hf_prompt_source + gsm8k_row   -> prompts with gold answers in meta
  HFEngine                            -> rollouts (bf16 CUDA / fp32 MPS)
  rewards.make_math_reward_fn         -> verifiable reward (boxed/last-number)
  Trainer(grpo_loss)                  -> the GRPO update
  fit_async                           -> the tier-1 async training loop

Missing on purpose (kept simple): eval harness, wandb, checkpointing schedule.
"""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from minirl.algos import GRPOConfig, grpo_loss
from minirl.async_controller import fit_async
from minirl.config import CollectConfig
from minirl.data import hf_prompt_source
from minirl.data.prompts import gsm8k_row
from minirl.engine import HFEngine
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  model={MODEL}")

    # --- models: engine (rollouts) + learner (training), both from one checkpoint ---
    engine = HFEngine(MODEL, device=device)
    learner = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    trainer = Trainer(
        learner,
        grpo_loss,
        GRPOConfig(use_tis=True),  # TIS on: corrects the engine<->learner logprob gap
        TrainConfig(lr=LR, ppo_epochs=1, minibatch_size=GROUP_SIZE * TARGET_GROUPS, micro_batch_size=4),
        device=device,
    )

    # --- data: GSM8K prompts with gold answers riding in meta ---
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{N_TRAIN_EXAMPLES}]")
    prompt_source = hf_prompt_source(ds, engine.tokenizer, row_fn=gsm8k_row, seed=0)
    reward_fn = make_math_reward_fn(engine.tokenizer)  # grades response vs meta["answer"]

    # --- run the async training loop ---
    def log(m: dict) -> None:
        print(
            f"iter {m['iteration']}  reward={m['reward_mean']:.3f}  "
            f"loss={m['loss']:+.4f}  kl={m['approx_kl']:.4f}  clip={m['clip_frac']:.3f}  "
            f"tis={m.get('tis_mean', 1.0):.3f}  stale={m['staleness']}  "
            f"pad={m['frac_padding']:.2f}  t_gen={m['t_generate']:.1f}s t_train={m['t_train']:.1f}s"
        )

    print("starting GRPO...")
    fit_async(
        engine=engine,
        trainer=trainer,
        reward_fn=reward_fn,
        prompt_source=prompt_source,
        sampling=SamplingParams(temperature=1.0, top_p=0.95, max_new_tokens=MAX_NEW_TOKENS, n=GROUP_SIZE),
        collect_cfg=CollectConfig(group_size=GROUP_SIZE, target_groups=TARGET_GROUPS, strategy="fixed"),
        num_iterations=NUM_ITERATIONS,
        update_weights_interval=1,
        on_metrics=log,
    )
    print("done.")


if __name__ == "__main__":
    main()
