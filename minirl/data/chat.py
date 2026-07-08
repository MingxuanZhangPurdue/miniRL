"""The single owner of every apply_chat_template call in the repo.

Why centralize: the engine and the learner MUST tokenize identically (the
retokenization hazard — mismatched prompt tokens silently corrupt the ratio),
so template choices (enable_thinking, system prompt) are decided here once, not
scattered across recipes.

The hard part is encode_conversation's assistant-only loss mask for SFT — the
same problem as slime's MultiTurnLossMaskGenerator.
"""

import torch
from torch import Tensor

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": ...}


def _template_ids(tokenizer, messages: list[Message], add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
    )
    return out["input_ids"]


def encode_prompt(tokenizer, messages: list[Message], enable_thinking: bool = False) -> Tensor:
    """RL/inference prompt -> (T,) int64, with the assistant generation prompt appended.

    enable_thinking is threaded through where the template supports it (Qwen3);
    ignored otherwise. This is the ONE place that decision is made.
    """
    kwargs = {"enable_thinking": enable_thinking}
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, **kwargs
        )["input_ids"]
    except TypeError:  # template doesn't accept enable_thinking
        ids = _template_ids(tokenizer, messages, add_generation_prompt=True)
    return torch.tensor(ids, dtype=torch.long)


def _turn_end_ids(tokenizer) -> set[int]:
    """Token ids that terminate an assistant turn (Qwen: <|im_end|>; plus eos)."""
    ids: set[int] = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    for name in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):  # common chat terminators
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    assert ids, "no turn-terminator token found for this tokenizer"
    return ids


def encode_conversation(tokenizer, messages: list[Message]) -> tuple[Tensor, Tensor]:
    """SFT conversation -> (input_ids (T,), loss_mask (T,) bool).

    loss_mask is True on ASSISTANT tokens only (including the turn-ending token,
    so the model learns to stop) and False on system/user/template tokens.

    Method (provably correct, tokenizer-agnostic): render the full conversation
    once for input_ids; for each assistant message i,
        start = len(render(messages[:i], add_generation_prompt=True))
    which is ALWAYS a prefix of the full render (generation prompts are
    deterministic), so `start` is the first assistant-content token. The span
    ends at the next turn-terminator token. No prefix of a conversation ending
    in an assistant message is used — that is where the trailing-newline
    inconsistency lives.
    """
    full = _template_ids(tokenizer, messages, add_generation_prompt=False)
    input_ids = torch.tensor(full, dtype=torch.long)  # (T,)
    loss_mask = torch.zeros(len(full), dtype=torch.bool)  # (T,)
    stop_ids = _turn_end_ids(tokenizer)

    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        header = _template_ids(tokenizer, messages[:i], add_generation_prompt=True)
        start = len(header)
        assert full[:start] == header, (
            "assistant header is not a prefix of the full render — this tokenizer "
            "violates the generation-prompt-determinism assumption; needs special handling"
        )
        end = start
        while end < len(full) and full[end] not in stop_ids:  # scan to the turn terminator
            end += 1
        loss_mask[start : end + 1] = True  # include the terminator so the model learns to stop

    assert loss_mask.any(), "conversation has no assistant tokens to train on"
    return input_ids, loss_mask
