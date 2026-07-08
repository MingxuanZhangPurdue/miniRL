"""RL prompt sources over HF datasets — the (n) -> [(ids, meta)] callable that
collect_groups consumes (rollout/sampling.py). HF `datasets` owns loading,
caching, and splits; we only shuffle, adapt rows, and hand out tokenized
prompts with their labels in meta.
"""

import random
from typing import Callable

from torch import Tensor

from minirl.data.chat import Message, encode_prompt

# row_fn adapts ONE dataset row to (chat messages, meta dict). Meta rides with
# every trajectory to the reward fn (e.g. {"answer": "42"}). One tiny function
# per dataset — not a column-mapping config schema (DESIGN principle 8).
RowFn = Callable[[dict], tuple[list[Message], dict]]


def gsm8k_row(row: dict) -> tuple[list[Message], dict]:
    """openai/gsm8k: 'question' + 'answer' (gold ends in '#### N')."""
    messages = [{"role": "user", "content": row["question"].strip()}]
    return messages, {"answer": row["answer"]}


def hf_prompt_source(
    dataset,  # a datasets.Dataset (already loaded/split by the caller)
    tokenizer,
    row_fn: RowFn = gsm8k_row,
    enable_thinking: bool = False,
    seed: int = 0,
) -> Callable[[int], list[tuple[Tensor, dict]]]:
    """Build a prompt_source(n) -> [(prompt_ids (T,), meta)] callable.

    Shuffles once per epoch (seeded, reproducible), consumes sequentially, and
    returns [] when the epoch is exhausted — the semantics collect_groups and
    the async controller already handle (a short/empty return ends collection).
    """
    order = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(order)
    cursor = 0
    epoch = 0

    def prompt_source(n: int) -> list[tuple[Tensor, dict]]:
        nonlocal cursor, epoch, order
        out: list[tuple[Tensor, dict]] = []
        while len(out) < n and cursor < len(order):
            messages, meta = row_fn(dataset[order[cursor]])
            out.append((encode_prompt(tokenizer, messages, enable_thinking), meta))
            cursor += 1
        if cursor >= len(order):  # epoch boundary: reshuffle for the next pass
            epoch += 1
            rng.seed(seed + epoch)
            rng.shuffle(order)
            cursor = 0
        return out

    return prompt_source
