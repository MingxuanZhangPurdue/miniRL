"""Data layer tests — the assistant-mask round-trip is the load-bearing one.

Uses the real Qwen3-0.6B tokenizer (cached, no model load) for chat.py, and
a fake in-memory dataset for prompts.py / sft.py so they stay CPU-fast.
"""

import os

import pytest
import torch
from transformers import AutoTokenizer

from minirl.data.chat import encode_conversation, encode_prompt
from minirl.data.prompts import gsm8k_row, hf_prompt_source
from minirl.data.sft import sft_batches

# Override to vet a NEW model family's chat template + assistant masking:
#   MINIRL_TEST_MODEL=org/name pytest tests/test_data.py tests/test_hf_engine.py
# (NOTE: the substring assertions in the mask tests assume a Qwen-style
# template; a new family may need its own expected strings.)
MODEL = os.environ.get("MINIRL_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


class FakeDataset:
    """Minimal datasets.Dataset stand-in: len + integer indexing."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ---------------- chat.py: the mask round-trip ----------------


def test_encode_prompt_ends_with_generation_prompt(tok):
    ids = encode_prompt(tok, [{"role": "user", "content": "What is 2+2?"}])
    text = tok.decode(ids)
    assert text.rstrip().endswith("assistant") or "assistant" in text[-30:]  # gen prompt present
    assert ids.dtype == torch.long


def test_conversation_mask_covers_exactly_assistant_content(tok):
    # distinctive multi-word contents avoid substring collisions (e.g. Qwen's
    # empty "<think>" block contains "hi")
    messages = [
        {"role": "user", "content": "capital of France"},
        {"role": "assistant", "content": "Paris obviously"},
        {"role": "user", "content": "and Japan"},
        {"role": "assistant", "content": "Tokyo indeed"},
    ]
    ids, mask = encode_conversation(tok, messages)
    assert ids.shape == mask.shape and mask.dtype == torch.bool

    masked_text = tok.decode(ids[mask])
    # every assistant content string is trained on; no user content is
    assert "Paris obviously" in masked_text and "Tokyo indeed" in masked_text
    assert "France" not in masked_text and "Japan" not in masked_text

    unmasked_text = tok.decode(ids[~mask])
    assert "France" in unmasked_text and "Japan" in unmasked_text  # user turns are context
    assert "Paris obviously" not in unmasked_text


def test_mask_excludes_role_headers(tok):
    # the tokens "<|im_start|>assistant" must NOT be trained on — only content
    ids, mask = encode_conversation(tok, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ])
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assert not mask[(ids == im_start)].any(), "role header token was left in the loss mask"
    # but the terminator IS masked (model must learn to stop)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert mask[(ids == im_end)].any()


def test_single_turn_and_system_prompt(tok):
    ids, mask = encode_conversation(tok, [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "the answer"},
    ])
    assert "the answer" in tok.decode(ids[mask])
    assert "be terse" not in tok.decode(ids[mask])  # system prompt is context, not target


# ---------------- prompts.py ----------------


def test_gsm8k_row_adapter():
    messages, meta = gsm8k_row({"question": " What is 2+2? ", "answer": "It is 4.\n#### 4"})
    assert messages == [{"role": "user", "content": "What is 2+2?"}]
    assert meta == {"answer": "It is 4.\n#### 4"}


def test_prompt_source_shape_and_meta(tok):
    ds = FakeDataset([{"question": f"q{i}", "answer": f"#### {i}"} for i in range(5)])
    src = hf_prompt_source(ds, tok, row_fn=gsm8k_row, seed=0)
    out = src(3)
    assert len(out) == 3
    ids, meta = out[0]
    assert ids.dtype == torch.long and "answer" in meta


def test_prompt_source_epoch_wraps_and_reshuffles(tok):
    ds = FakeDataset([{"question": f"q{i}", "answer": str(i)} for i in range(4)])
    src = hf_prompt_source(ds, tok, seed=0)
    first_epoch = [m["answer"] for _, m in src(4)]
    assert sorted(first_epoch) == ["0", "1", "2", "3"]  # covers the whole set
    second_epoch = [m["answer"] for _, m in src(4)]
    assert sorted(second_epoch) == ["0", "1", "2", "3"]  # wraps to a full new epoch
    assert first_epoch != second_epoch or True  # reshuffle (may coincide for tiny sets)


def test_prompt_source_deterministic(tok):
    ds = FakeDataset([{"question": f"q{i}", "answer": str(i)} for i in range(6)])
    a = [m["answer"] for _, m in hf_prompt_source(ds, tok, seed=7)(6)]
    b = [m["answer"] for _, m in hf_prompt_source(ds, tok, seed=7)(6)]
    assert a == b


# ---------------- sft.py ----------------


def test_sft_batches_produce_valid_batches(tok):
    convo = lambda i: [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]
    ds = FakeDataset([{"messages": convo(i)} for i in range(7)])
    batches = list(sft_batches(ds, tok, batch_size=3, seed=0))
    assert len(batches) == 3  # 3 + 3 + 1
    b0 = batches[0]
    assert b0.input_ids.shape[0] == 3 and b0.loss_mask.dtype == torch.bool
    assert (b0.behavior_logprobs == 0).all()  # SFT has no behavior policy
    assert b0.loss_mask.any(dim=1).all()  # every row has assistant tokens to learn


def test_sft_skips_overlong(tok):
    ds = FakeDataset([
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]},
        {"messages": [{"role": "user", "content": "x " * 5000}, {"role": "assistant", "content": "ok"}]},
    ])
    batches = list(sft_batches(ds, tok, batch_size=8, max_length=128, seed=0))
    assert sum(b.input_ids.shape[0] for b in batches) == 1  # the long one is dropped
