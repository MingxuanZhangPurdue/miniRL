"""SFT data over HF conversation datasets.

An SFT example IS a Trajectory with zero behavior logprobs and zero reward, so
it flows through the SAME make_batch as RL — with advantage_fn=None (SFT has
no advantages; zeros are filled and never read). The only SFT-specific work is
the assistant-only loss mask (data/chat.py) and shuffled batching here.
"""

import random
from typing import Callable, Iterator

import torch

from minirl.data.chat import Message, encode_conversation
from minirl.rollout.batching import make_batch
from minirl.rollout.types import Batch, Trajectory

# Adapt one HF row to a list of chat messages. Default: the 'messages' column
# (smoltalk / tulu / openai convention).
ConversationFn = Callable[[dict], list[Message]]


def messages_column(row: dict) -> list[Message]:
    return row["messages"]


def sft_batches(
    dataset,
    tokenizer,
    batch_size: int,
    conversation_fn: ConversationFn = messages_column,
    max_length: int = 4096,
    seed: int = 0,
) -> Iterator[Batch]:
    """Yield shuffled Batches for one epoch over an HF conversations dataset.

    Over-long conversations are skipped (logged count is the caller's concern);
    packing (DESIGN §6) will later replace this length filter.
    """
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)

    trajs: list[Trajectory] = []
    for idx in order:
        input_ids, loss_mask = encode_conversation(tokenizer, conversation_fn(dataset[idx]))
        if input_ids.numel() > max_length:
            continue
        trajs.append(
            Trajectory(
                input_ids=input_ids,
                loss_mask=loss_mask,
                logprobs=torch.zeros(input_ids.numel()),  # SFT: no behavior policy
                reward=0.0,
            )
        )
        if len(trajs) == batch_size:
            yield make_batch(trajs, pad_id=tokenizer.pad_token_id or 0, advantage_fn=None)[0]
            trajs = []
    if trajs:  # final short batch
        yield make_batch(trajs, pad_id=tokenizer.pad_token_id or 0, advantage_fn=None)[0]
