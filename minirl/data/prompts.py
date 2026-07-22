"""RL prompt sources over HF datasets — the (n) -> [(ids, meta)] callable that
the collector consumes (controllers/fully_async.collect_groups_dp). HF `datasets` owns loading,
caching, and splits; we only shuffle, adapt rows, and hand out tokenized
prompts with their labels in meta.
"""

import random
from typing import Callable

from torch import Tensor

from minirl.config import DataConfig
from minirl.data.chat import Message, encode_prompt

# row_fn adapts ONE dataset row to (chat messages, meta dict). Meta rides with
# every trajectory to the reward fn, which reads the gold answer from
# meta["label"]. The default is generic (input_key -> user message, label_key
# -> meta["label"]); a custom row_fn is the escape hatch for datasets that
# need real transformation.
RowFn = Callable[[dict], tuple[list[Message], dict]]


def keyed_row_fn(input_key: str, label_key: str | None) -> RowFn:
    """Generic adapter: row[input_key] becomes the user turn, row[label_key]
    the reward's gold answer under the fixed meta key "label"."""

    def row_fn(row: dict) -> tuple[list[Message], dict]:
        messages = [{"role": "user", "content": str(row[input_key])}]
        meta = {"label": row[label_key]} if label_key is not None else {}
        return messages, meta

    return row_fn


def gsm8k_row(row: dict) -> tuple[list[Message], dict]:
    """openai/gsm8k: 'question' + 'answer' (gold ends in '#### N'). An example
    custom row_fn — the generic keyed_row_fn handles it too, but this strips
    the question."""
    messages = [{"role": "user", "content": row["question"].strip()}]
    return messages, {"label": row["answer"]}


class HFPromptSource:
    """prompt_source(n) -> [(prompt_ids (T,), meta)] over an HF dataset.

    A callable object rather than a closure so the sampling state is NAMED
    and inspectable (`src.epoch`, `src.cursor`) instead of hidden in cells.
    The collector (and the DP dealer, which calls this under the tally lock)
    only sees the callable contract.

    Row adaptation comes from cfg.input_key/label_key (the generic default)
    unless a custom row_fn is passed. Shuffles once per epoch when
    cfg.rollout_shuffle (seeded, reproducible) and consumes sequentially —
    WITHOUT replacement within an epoch (a prompt dropped by dynamic sampling
    cannot be redrawn until the next epoch's reshuffle). At the epoch boundary
    the call returns SHORT, then the next epoch begins: this source never
    returns [] for a non-empty dataset, so the collector's data-exhaustion
    stop only fires for custom finite sources; here rollout_batch_size and
    over_sampling_rounds are the binding stops.
    """

    def __init__(
        self,
        dataset,  # a datasets.Dataset (already loaded/split by the caller)
        tokenizer,
        cfg: DataConfig,
        row_fn: RowFn | None = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.row_fn = row_fn if row_fn is not None else keyed_row_fn(cfg.input_key, cfg.label_key)
        self.enable_thinking = cfg.enable_thinking
        self.order = list(range(len(dataset)))  # this epoch's visit order
        self.rng = random.Random(cfg.rollout_seed)
        if cfg.rollout_shuffle:
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
            if self.cfg.rollout_shuffle:
                self.rng.seed(self.cfg.rollout_seed + self.epoch)
                self.rng.shuffle(self.order)
            self.cursor = 0
        return out
