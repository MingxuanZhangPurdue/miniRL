"""Typed configs. One experiment == one config; YAML/CLI loading comes with
the trainer. Per-algorithm loss configs live next to their loss (minirl/algos/*)
so each file is self-contained; this module holds cross-cutting configs only.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PlacementConfig:
    """Single-node GPU split for the fully-async controller.

    slime's spec, minus what we don't need: `--actor-num-gpus-per-node` ->
    num_train_gpus, `--rollout-num-gpus` -> num_rollout_gpus; trainer ranks
    take GPUs 0..t-1, engines take t..t+r-1 (slime get_base_gpu_id's
    non-colocated layout). DP only, so one GPU == one engine
    (`--rollout-num-gpus-per-engine` is permanently 1; TP is a non-goal).
    No colocate mode: slime's --colocate drags the offload/onload dance —
    on the Mac the degenerate case is simply both on one MPS device with no
    placement at all (this config unused).
    """

    num_train_gpus: int = 1  # DDP world size (1 = plain Trainer, no dist)
    num_rollout_gpus: int = 1  # == number of DP engines (TP=1 each)

    @property
    def train_gpu_ids(self) -> list[int]:
        return list(range(self.num_train_gpus))

    @property
    def rollout_gpu_ids(self) -> list[int]:  # feed one id per VLLMEngine(gpu_id=...)
        return list(range(self.num_train_gpus, self.num_train_gpus + self.num_rollout_gpus))


@dataclass(frozen=True)
class CollectConfig:
    """Batch collection for RL (controllers/fully_async.collect_groups_dp)."""

    group_size: int = 8  # G completions per prompt
    target_groups: int = 32  # P surviving groups per training batch
    # "fixed": take target_groups prompts, keep everything.
    # "filter": DAPO dynamic sampling / slime dynamic_sampling_filter — drop
    #           zero-gradient groups (reward std ~ 0), keep collecting.
    strategy: Literal["fixed", "filter"] = "fixed"
    max_rounds: int = 20  # budget: at most max_rounds*target_groups groups generated per call
