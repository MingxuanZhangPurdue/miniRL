"""Packing index-math tests — pure CPU, no megatron.

The packed forward itself is CUDA-only; what CPU can pin is everything that
could silently corrupt training if the indices were off by one: pack
planning, the packed layout, and the CE-scatter back to (B, T).
"""

import torch

from minirl.rollout.packing import pack_rows, plan_packs, unpack_ce


def padded(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded (B, T) input_ids + (B,) lengths from ragged token lists."""
    t = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), t, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
    return ids, torch.tensor([len(r) for r in rows])


# ---------------- plan_packs ----------------


def test_plan_respects_budget_and_preserves_order():
    packs = plan_packs([5, 6, 4, 10, 2], max_tokens=11)
    assert packs == [[0, 1], [2], [3], [4]]
    assert [i for p in packs for i in p] == list(range(5))  # order preserved


def test_oversized_row_gets_its_own_pack():
    packs = plan_packs([3, 9, 2], max_tokens=4)
    assert packs == [[0], [1], [2]]  # 9 > budget: alone, not split, not dropped


def test_huge_budget_is_one_pack():
    assert plan_packs([5, 6, 4], max_tokens=10_000_000) == [[0, 1, 2]]


# ---------------- pack_rows layout ----------------


def test_pack_layout():
    ids, lengths = padded([[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]])
    p = pack_rows(ids, lengths, [0, 1, 2])
    assert p.tokens.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # no pads
    assert p.position_ids.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]  # resets
    assert p.cu_seqlens.tolist() == [0, 3, 8, 10]
    assert p.max_seqlen == 5
    assert p.dst_col.min() == 1  # position 0 of every row is never written


def test_pack_subset_and_order():
    ids, lengths = padded([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    p = pack_rows(ids, lengths, [2, 0])  # rows in pack order, not batch order
    assert p.tokens.tolist() == [[6, 7, 8, 9, 1, 2]]
    assert p.row_idx == [2, 0]


# ---------------- unpack_ce ----------------


def test_unpack_scatter_matches_per_row_shift():
    torch.manual_seed(0)
    ids, lengths = padded([[1, 2, 3, 4], [5, 6], [7, 8, 9]])
    p = pack_rows(ids, lengths, [0, 1, 2])
    ce = torch.randn(1, int(p.cu_seqlens[-1]))
    out = unpack_ce(ce, p, t_padded=4)  # (3, 4)
    for s, (start, n) in enumerate(zip(p.cu_seqlens[:-1].tolist(), lengths.tolist())):
        assert torch.equal(out[s, 1:n], -ce[0, start : start + n - 1])  # shift by one
        assert out[s, 0] == 0.0  # position 0 never predicted
        assert (out[s, n:] == 0.0).all()  # padding untouched


def test_unpack_never_reads_seam_ce():
    ids, lengths = padded([[1, 2, 3], [4, 5, 6]])
    p = pack_rows(ids, lengths, [0, 1])
    ce = torch.randn(1, 6, requires_grad=True)
    unpack_ce(ce, p, t_padded=3).sum().backward()
    grad = ce.grad[0]
    # seam positions: the LAST CE of each segment scores across a boundary
    # (or past the end) and must contribute nothing
    assert grad[2] == 0.0 and grad[5] == 0.0
    assert (grad[[0, 1, 3, 4]] != 0.0).all()  # every in-segment CE is used
