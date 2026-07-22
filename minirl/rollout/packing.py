"""Sequence packing: pad-free forward batches for the Megatron trainer.

Pure-tensor helpers with no megatron imports, so the index math is testable
in the CPU suite. The consumer packs rows ONLY around the model forward and
scatters the fused-CE map back to the padded (B, T) layout before any loss
code runs — the rest of the repo never sees the packed format.

Two knobs at two levels: the number of ROWS per optimizer step
(minibatch_size) is the batch size and carries the RL meaning; the token
budget here only decides how much of one minibatch goes through a single
forward/backward (grad-accumulation slicing). Because the loss uses a
minibatch-global denominator and aggregation is linear, the budget cannot
change the gradient — only memory and speed.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def plan_packs(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Row indices per pack: sequential greedy fill under the token budget.

    Walks rows IN ORDER (preserves the shuffled SGD order; smarter bin
    packing would fragment less but reorder). A row longer than the budget
    gets a pack of its own — the only pack allowed over budget; long
    sequences are never split or dropped.
    """
    packs: list[list[int]] = []
    cur: list[int] = []
    cur_tokens = 0
    for i, n in enumerate(lengths):
        if cur and cur_tokens + n > max_tokens:
            packs.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += n
    if cur:
        packs.append(cur)
    return packs


@dataclass
class Pack:
    """One dense forward batch: S whole rows concatenated without padding.

    The scatter map (dst_row/dst_col/src_pos) encodes where each packed CE
    value lands in the padded (S, T) logprob map: the logprob of row s,
    position j comes from packed position cu_seqlens[s] + j - 1. Position 0
    of every row is never predicted, and the LAST CE of every segment
    scores across a seam (its label is the next segment's first token) —
    both are simply absent from the map and stay 0.0, exactly the positions
    the loss mask already excludes.
    """

    tokens: Tensor  # (1, T_pack) int64 — segments back to back, no padding
    position_ids: Tensor  # (1, T_pack) int64 — restarts at 0 on every segment
    cu_seqlens: Tensor  # (S+1,) int32 — segment boundary offsets
    max_seqlen: int  # longest segment (varlen kernels need it)
    row_idx: list[int]  # pack segment s == source-batch row row_idx[s]
    dst_row: Tensor  # (n,) int64 — destination rows in the (S, T) map
    dst_col: Tensor  # (n,) int64 — destination cols, all >= 1
    src_pos: Tensor  # (n,) int64 — source positions in the packed CE row


def pack_rows(input_ids: Tensor, lengths: Tensor, row_idx: list[int]) -> Pack:
    """Concatenate whole rows of a right-padded (B, T) batch into one Pack.

    Args: input_ids (B, T) int64; lengths (B,) real token count per row
    (attention_mask.sum(-1)); row_idx — which rows, in order.
    """
    seg_lens = [int(lengths[r]) for r in row_idx]
    tokens = torch.cat([input_ids[r, :n] for r, n in zip(row_idx, seg_lens)]).unsqueeze(0)  # (1, T_pack)
    cu = torch.zeros(len(row_idx) + 1, dtype=torch.int32)  # (S+1,)
    cu[1:] = torch.tensor(seg_lens).cumsum(0)
    position_ids = torch.cat([torch.arange(n) for n in seg_lens]).unsqueeze(0)  # (1, T_pack)

    dst_row = torch.cat([torch.full((n - 1,), s) for s, n in enumerate(seg_lens)])  # (n,)
    dst_col = torch.cat([torch.arange(1, n) for n in seg_lens])  # (n,)
    src_pos = torch.cat(  # (n,) — segment start + 0..n-2: skips each segment's seam CE
        [torch.arange(int(cu[s]), int(cu[s]) + n - 1) for s, n in enumerate(seg_lens)]
    )
    return Pack(
        tokens=tokens,
        position_ids=position_ids,
        cu_seqlens=cu,
        max_seqlen=max(seg_lens),
        row_idx=list(row_idx),
        dst_row=dst_row,
        dst_col=dst_col,
        src_pos=src_pos,
    )


def unpack_ce(ce: Tensor, pack: Pack, t_padded: int) -> Tensor:
    """Packed fused-CE row -> padded (S, T) logprob map, position-aligned.

    ce (1, T_pack) holds -log p(next token | prefix) per packed position;
    the scatter negates and shifts so out[s, j] = log p(token_j | <j).
    Position 0, seams, and padding stay 0.0 (all loss-masked positions).
    Differentiable — the training path backprops through the scatter.
    """
    out = torch.zeros(len(pack.row_idx), t_padded, dtype=ce.dtype, device=ce.device)  # (S, T)
    out[pack.dst_row.to(ce.device), pack.dst_col.to(ce.device)] = -ce[0, pack.src_pos.to(ce.device)]
    return out
