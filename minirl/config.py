"""Typed configs, one per COMPONENT, reusing slime's argument names so a
slime recipe translates field-for-field (docs/config.md). Per-algorithm loss
configs live next to their loss (minirl/algos/*) and the trainer config lives
in minirl/megatron.py; this module holds the cross-cutting rollout side.
"""

from dataclasses import dataclass

from minirl.rollout.types import SamplingParams


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
class RolloutConfig:
    """Generation + collection for one training batch: sampling knobs and
    batch shape in one object (controllers/fully_async). sampling_params()
    derives the engine wire-type — field names here follow slime; the wire
    type mirrors vllm.SamplingParams.
    """

    # batch shape
    rollout_batch_size: int = 32  # prompts (== surviving groups) per training batch
    n_samples_per_prompt: int = 8  # G: completions per prompt (one request IS one group)
    # sampling
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1  # -1 = disabled
    rollout_max_response_len: int = 512
    # dynamic sampling: drop zero-gradient groups (reward std ~ 0), keep
    # collecting until rollout_batch_size survive (DAPO / slime filter)
    dynamic_sampling: bool = False
    over_sampling_rounds: int = 20  # budget: <= this * rollout_batch_size groups generated per call

    def sampling_params(self) -> SamplingParams:
        """Derive the engine wire-type for one request (a whole group)."""
        return SamplingParams(
            temperature=self.rollout_temperature,
            top_p=self.rollout_top_p,
            top_k=self.rollout_top_k,
            max_new_tokens=self.rollout_max_response_len,
            n=self.n_samples_per_prompt,
        )


@dataclass(frozen=True)
class DataConfig:
    """Where prompts come from (slime's Data&Dataset group). input_key/label_key
    drive a generic row adapter; a custom row_fn is the escape hatch."""

    prompt_data: str  # dataset id/path — recorded here; the recipe loads it (split/config aware)
    input_key: str = "input"  # row key -> the user message
    label_key: str | None = None  # row key -> meta["label"] (the reward's gold answer)
    apply_chat_template: bool = True
    rollout_shuffle: bool = True  # shuffle prompts each epoch (seeded)
    rollout_seed: int = 0
