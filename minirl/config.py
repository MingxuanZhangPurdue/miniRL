"""Typed configs. One experiment == one config; YAML/CLI loading comes with
the trainer. Per-algorithm loss configs live next to their loss (minirl/algos/*)
so each file is self-contained; this module holds cross-cutting configs only.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CollectConfig:
    """Batch collection for RL (rollout/sampling.py)."""

    group_size: int = 8  # G completions per prompt
    target_groups: int = 32  # P surviving groups per training batch
    # "fixed": take target_groups prompts, keep everything.
    # "filter": DAPO dynamic sampling / slime dynamic_sampling_filter — drop
    #           zero-gradient groups (reward std ~ 0), keep collecting.
    strategy: Literal["fixed", "filter"] = "fixed"
    oversample_batch_size: int | None = None  # prompts per round; None = exactly what's missing
    max_rounds: int = 20  # hard stop so exhausted/degenerate data can't loop forever
