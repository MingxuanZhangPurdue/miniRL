"""RL prompt sources over HF datasets — the (n) -> [(ids, meta)] callable that
the collector consumes (controllers/fully_async.collect_groups_dp). HF `datasets` owns loading,
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


class HFPromptSource:
    """prompt_source(n) -> [(prompt_ids (T,), meta)] over an HF dataset.

    A callable object rather than a closure so the sampling state is NAMED
    and inspectable (`src.epoch`, `src.cursor`) instead of hidden in cells.
    The collector (and the DP dealer, which calls this under the tally lock)
    only sees the callable contract.

    Shuffles once per epoch (seeded, reproducible) and consumes sequentially —
    WITHOUT replacement within an epoch (a prompt dropped by dynamic sampling
    cannot be redrawn until the next epoch's reshuffle). At the epoch boundary
    the call returns SHORT, then the next epoch begins: this source never
    returns [] for a non-empty dataset, so the collector's data-exhaustion
    stop only fires for custom finite sources; here target_groups and
    max_rounds are the binding stops.
    """

    def __init__(
        self,
        dataset,  # a datasets.Dataset (already loaded/split by the caller)
        tokenizer,
        row_fn: RowFn = gsm8k_row,
        enable_thinking: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.row_fn = row_fn
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.order = list(range(len(dataset)))  # this epoch's visit order
        self.rng = random.Random(seed)
        self.rng.shuffle(self.order)
        self.cursor = 0  # next index into order
        self.epoch = 0  # completed passes over the dataset

    def __call__(self, n: int) -> list[tuple[Tensor, dict]]:
        out: list[tuple[Tensor, dict]] = []
        while len(out) < n and self.cursor < len(self.order):
            messages, meta = self.row_fn(self.dataset[self.order[self.cursor]])
            out.append((encode_prompt(self.tokenizer, messages, self.enable_thinking), meta))
            self.cursor += 1
        if self.cursor >= len(self.order):  # epoch boundary: reshuffle for the next pass
            self.epoch += 1
            self.rng.seed(self.seed + self.epoch)
            self.rng.shuffle(self.order)
            self.cursor = 0
        return out
