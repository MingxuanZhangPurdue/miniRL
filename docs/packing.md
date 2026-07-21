# Sequence packing: implementation design

Status: **BUILT 2026-07-20 — trainer-internal, Megatron-only** (§1a):
`minirl/packing.py` + the packed path in `minirl/megatron.py`, enabled per
run via `MegatronTrainConfig.pack_max_tokens` (default OFF — flip it when a
run's `frac_padding` says it pays). Equivalence measured on the box the
same day (`recipes/08_megatron_p1_parity.py --packed`): packed sits
marginally CLOSER to the fp32 truth than the padded path (3.43e-2 vs
3.48e-2 mean nats on the frozen batch), on-policy loss matches to 7e-9,
grad_norm to 4e-3, step-2 loss/grad_norm to 1e-2/4e-2.

A FIRST implementation that threaded the packed layout through Batch /
trainer / aggregate / gspo was prototyped and rolled back (2026-07-09):
it made the core files noticeably harder to read. §2's HF mechanism and
§3's file-by-file plan document that design as history — the Megatron
migration (2026-07-20) dissolved its two hard parts (see §1a) and the
built design touches NONE of those files: losses never see the packed
format.

Doc-before-code (repo convention; code must match this or the doc gets
updated). Grounded in slime's `megatron_utils/data.py::get_batch` (THD packed
rows + `cu_seqlens` + `PackedSeqParams`, pad-to-128, `--max-tokens-per-gpu`)
and verl's `use_remove_padding` (rmpad via flash-attn varlen). CUDA-only
feature; the padded `(B, T)` path stays the Mac/reference format forever.

## 1. The problem, with our own numbers

`make_batch` pads B trajectories to the longest one. `frac_padding` (already
logged) says how much of the rectangle is dead compute: 0.02–0.07 in the
GSM8K smoke test (short, uniform answers), 0.4–0.7 typical for long-CoT RLVR.
Throughput gain from packing ≈ `1 / (1 - frac_padding)` — build it when the
dashboard says it pays, not before (roadmap Phase 5.5).

## 1a. The BUILT design (2026-07-20): pack inside the trainer, unpack before the loss

Megatron dissolved the first design's two hard parts. (a) The attention
hazard (§2's block-diagonal rule) is no longer an inference-from-position-ids
trick: mcore takes an EXPLICIT `PackedSeqParams(cu_seqlens, qkv_format=
"thd")` into TE's varlen kernels — hand it wrong input and it raises.
(b) The fused-CE labels path returns a per-token vector, so the packed
layout can be unwound BEFORE any loss code runs — the option the HF-logits
trainer never had cleanly.

The shape of the built code:

    minirl/packing.py       pure tensor, no megatron imports (CPU-testable):
      plan_packs            sequential greedy fill under the token budget,
                            ORDER-PRESERVING (keeps the shuffled SGD order);
                            an oversized row gets a pack of its own — the
                            only pack allowed over budget, never split
      pack_rows -> Pack     whole rows concatenated pad-free: tokens,
                            per-segment position_ids, cu_seqlens, and the
                            precomputed scatter map (dst_row/dst_col/src_pos)
      unpack_ce             packed CE -> padded (S, T) logprob map in the
                            repo convention; position 0, seams, and padding
                            stay 0.0. The seam CE (each segment's last
                            position scores across a boundary) is simply
                            never read — the packed generalization of the
                            padded path's dummy last column.
    minirl/megatron.py      _micros() picks the grad-accum slicing (row
                            counts, or packs under cfg.pack_max_tokens);
                            _forward_ce() runs either layout and returns the
                            adapter back to (B, T); everything downstream —
                            loss_fn, aggregate_loss, denominators, metrics —
                            is layout-blind and byte-identical to unpacked.

The two knobs, mirroring slime's split exactly: `minibatch_size` stays the
optimizer-step batch size (slime `--global-batch-size` — the knob with RL
meaning; packing never touches it) and `pack_max_tokens` replaces
`micro_batch_size` as the grad-accum unit (slime `--max-tokens-per-gpu` —
pure memory/speed; the minibatch-global denominator + linear aggregation
make the gradient independent of the slicing). A third,
`logprob_pack_max_tokens` (slime `--log-probs-max-tokens-per-gpu`), allows
bigger packs for the no-grad old-logprob recompute. An absurdly large
budget degrades gracefully into one giant microbatch: same gradient, OOM
risk — the budget is the user's promise about what fits.

Constraints: whole rows only (never split a sequence — all three loss_agg
modes then work unchanged); bf16 + TE only (local-spec fp32 has no varlen
kernels — asserted at construction), so the fp32 parity/debug legs stay
padded forever. Tests: index math + planning + seam-gradient pins in
tests/test_packing.py (CPU); kernel-level equivalence is recipe 08
--packed's job (packed-vs-padded bands PLUS the decisive check: packed must
sit no farther from a raw fp32 HF forward than padded does).

## 2. The packed format, by worked example

Take three trajectories with lengths 5, 6, 4 (prompt `p`, response `r`) and a
token budget of 16:

```
row (1, 15):    p p p r r | p p r r r r | p p r r        one dense row, no pad
position_ids:   0 1 2 3 4 | 0 1 2 3 4 5 | 0 1 2 3        RESETS mark boundaries
seg_ids:        0 0 0 0 0 | 1 1 1 1 1 1 | 2 2 2 2        row -> segment map
cu_seqlens:     [0, 5, 11, 15]                            boundary offsets
loss_mask:      0 0 0 1 1 | 0 0 1 1 1 1 | 0 0 1 1        unchanged semantics
```

Everything per-token (`behavior_logprobs`, `advantages`, `old/ref_logprobs`)
concatenates exactly like `input_ids`. Advantages are computed per trajectory
BEFORE packing (same `grpo_advantages` on rewards+group_ids), then broadcast
onto each segment's response tokens — packing is pure re-layout, zero math.

**The attention rule (the one thing that must not be gotten wrong):** packing
needs BLOCK-DIAGONAL attention — segment 1 must not see segment 0. A 2D
attention mask cannot express that; naive `attention_mask=1` silently
contaminates. Mechanism we would use: **HF FlashAttention-2 infers the
boundaries from the `position_ids` resets** (pass `position_ids`, pass NO 2D
mask). Megatron/slime instead feed `cu_seqlens` to varlen kernels — same
effect. Assert `attn_implementation == "flash_attention_2"` when packing:
SDPA/eager do NOT do this inference and would train contaminated, silently.
(The prototype verified both halves empirically on a tiny random Qwen3: a 4D
causal+same-segment mask reproduces each sequence's solo logprobs exactly,
and the naive 2D mask measurably contaminates — the design is sound; it is
the code-complexity trade that is deferred, not the correctness.)

## 3. HISTORY — the rolled-back first design's file-by-file plan

(The built design touches none of these files; §1a replaces this section.
Kept because the analysis — especially §2's seam subtlety and the
block-diagonal contamination experiment — grounds the built code too.)

### types.py — Batch grows two optional fields (~5 lines)

```python
position_ids: Tensor | None = None  # (1, T_pack) — resets at segment starts
seg_ids:      Tensor | None = None  # (1, T_pack) — token -> segment index
```

`seg_ids is not None` IS the "packed" flag. (`cu_seqlens` is derivable from
either; we store the two forms the consumers actually index with.)

### batching.py — `pack_batch` (~60 lines, the main new code)

```python
def pack_batch(trajs, max_tokens, advantage_fn=grpo_advantages) -> tuple[list[Batch], dict]:
    # 1. advantages per trajectory (identical math to make_batch)
    # 2. greedy fill: walk the (shuffled) trajectories, start a new pack when
    #    adding the next one would exceed max_tokens. (FFD sort-by-length would
    #    fragment less; greedy keeps SGD order — revisit only if waste shows.)
    # 3. per pack: concat every per-token field; build position_ids (arange
    #    per segment), seg_ids (repeat_interleave); rewards/group_ids stay (S,)
    #    per-segment, for stats only.
    # returns a LIST of (1, T<=max_tokens) Batches — the microbatch unit
```

The batch unit changes meaning: `micro_batch_size` (sequences) is replaced by
`max_tokens` (slime's `--max-tokens-per-gpu`) — which also evens out step
times, since every pack is roughly the same number of tokens.

### trainer.py — ~15 lines of diff

- `step()`: microbatches are the packs themselves (`iter` over the list)
  instead of `iter_microbatches` row slices. Denominators are UNCHANGED in
  token mode (`loss_mask.sum()` over all packs of the minibatch — the global-
  denom contract already handles ragged microbatches); sequence mode's denom
  becomes total SEGMENTS, not rows.
- forward call: `self.model(input_ids, position_ids=batch.position_ids)` and
  NO attention_mask when packed (everything is a real token).
- `compute_logprobs` (old/ref recompute): same forward change, nothing else —
  `gather_logprobs` already works on any `(B, T)` including `(1, T_pack)`.

### The seam subtlety in gather_logprobs — no code change, one test

At each boundary, position `cu_seqlens[k]` gets its "logprob" from the
PREVIOUS segment's last token (the shift crosses the seam). That value is
garbage — and automatically harmless, because the first token of every
segment is a PROMPT token, and prompt positions are always `loss_mask=False`.
The invariant to pin with a test: `loss_mask[:, cu_seqlens[1:-1]] == False`
for every pack (holds by construction since every segment starts with a
prompt; the assert protects future prompt-free formats).

### aggregate.py + gspo.py — the two "row == sequence" assumptions (~20 lines)

Both currently reduce with `.sum(-1)` per row; packed rows hold S sequences.
Fix is the SAME scatter/gather trick as `advantage.py::_group_stats`, with
`seg_ids` playing the role of `group_ids`:

```python
# per-segment mean of a per-token tensor x (1, T) -> (S,)
seg_sum  = zeros(S).index_add_(0, seg_ids[0], (x * mask)[0])
seg_len  = zeros(S).index_add_(0, seg_ids[0], mask[0].float())
seg_mean = seg_sum / seg_len.clamp(min=1)
```

- `aggregate_loss`, sequence mode: per-segment means, then `/ denom` (total
  segments). Token mode: unchanged — it never assumed rows.
- `gspo_loss`: `seq_log_ratio` becomes the segment mean above, gathered back
  with `seg_mean[seg_ids]` (the exact gather-broadcast from advantage.py).
  Branch on `batch.seg_ids is not None`; the padded path is untouched.
- grpo/dapo/cispo/sft: fully elementwise — ZERO changes.

### config — none (principle 8)

No `TrainConfig` fields: the trainer wouldn't pack, it would HANDLE packed
input, so the decision lives at the collation call site — `pack_batch(trajs,
max_tokens)` directly (RL recipes/controller), a `pack_max_tokens` param on
`sft_batches` for SFT. `max_tokens` replaces `micro_batch_size` as the
microbatch unit implicitly: packs ARE the micros.

### Untouched, and why

Engine (vLLM's continuous batching is the inference-side answer to the same
waste — packing is trainer-only), controller, losses except gspo, rewards,
advantage.py, data/prompts. SFT packing = `sft_batches` calling `pack_batch`
instead of `make_batch` — comes first in the rollout order (classic, simpler,
no ratio semantics to double-check).

## 4. Testing strategy (the part that makes this safe)

The exit criterion is the same invariance standard as microbatching and FSDP:
**packed and padded must produce the same math on identical data.**

1. **CPU-testable semantics via an explicit block-diagonal mask**: a test-only
   helper builds the 4D mask from cu_seqlens and runs EAGER attention on the
   tiny random Qwen3 — packed-with-mask vs padded must match logits within
   fp32 tolerance. This validates layout/position_ids/reduce logic on the Mac
   without FA2. (The 4D mask is O(T^2) memory — fine at test scale, exactly
   why it is not the production mechanism.) The rolled-back prototype PASSED
   this rung, including the adversarial 2D-mask-contaminates check.
2. **Segment-reduce unit tests** (pure CPU): seg-mean helper vs hand-computed;
   gspo packed-vs-padded on synthetic logprobs; aggregate sequence-mode
   packed-vs-padded equality.
3. **Seam test**: boundary positions always loss-masked.
4. **Whole-trainer equivalence** on a position-invariant model: packed vs
   padded fit_batch must yield identical parameters for all three loss_agg
   modes.
5. **GPU integration (on the box)**: FA2 packed forward vs padded forward,
   same loss within bf16 tolerance; tokens/sec uplift ≈ measured
   `frac_padding`.

## 5. Build order (if/when the decision lands on "build it")

SFT packing → RL token-mode packing (dapo/cispo — no reduce changes) →
segment reduces (grpo sequence-mode, gspo) → GPU validation. Each rung has
its own equivalence test before the next.
