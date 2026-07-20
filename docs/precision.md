# Precision in miniRL

How every component picks its dtype, what is configurable vs hardcoded, and
how slime handles the same questions (checked against slime source, 2026-07).

## 1. Why RL is precision-sensitive (one paragraph)

SFT only needs the loss to point downhill; bf16 noise averages out. RL
computes **ratios between two nearly-identical distributions** —
`exp(logπ_θ − logπ_old)` — where the *difference* of two logprobs is the
signal. bf16 logprobs carry ~1e-2–1e-3 of rounding noise, the same order as
the true per-token drift after one update, so careless precision turns the
importance ratio (and the clip decisions and the KL estimate) into noise.
Hence the rule: **half precision for bulk compute, fp32 for everything that
subtracts, accumulates, or updates.**

## 2. The miniRL dtype map

Design rule: **configure only genuine choices (3 knobs); hardcode the
correctness invariants** so no config combination can silently break training.

| Component | dtype | Where specified |
|---|---|---|
| Rollout engine compute (vLLM / HFEngine) | **bf16** CUDA, **fp32** MPS/CPU | knob 1: `HFEngine(dtype=...)` / `VLLMEngine(dtype=...)`; auto-picked per device (engine/hf_engine.py already does this) |
| Learner forward/backward (weights-as-computed, activations, local grads) | **bf16** CUDA, **fp32** MPS/CPU | knob 2: `TrainConfig.bf16_weights` (Megatron-style: bf16 params + fp32 master copies in AdamW; grads reduce in bf16 — the one deviation from Megatron, which accumulates/all-reduces grads in fp32) — built 2026-07-20; an autocast variant (fp32 params, bf16 compute) existed briefly and was removed 2026-07-19: slime/Megatron ship exactly one mode. Dev path = fp32 end to end |
| Checkpoints at rest (`save_pretrained`) | **bf16** (HF convention) | knob 3: `train.checkpoint_dtype` |
| Master weights (what the optimizer updates) | **fp32**, always | hardcoded. bf16 has ~0.4% relative resolution: a `lr·grad ~ 1e-6` update rounds to zero (`1 + 1e-6 == 1` in bf16) and learning silently stops |
| Optimizer states (Adam m, v) | **fp32**, always | hardcoded, same reason |
| Gradient reduction / accumulation | **fp32**, always | DDP all-reduces fp32 grads (model dtype is fp32 on the dev path); grad-accum sums many small microbatch contributions |
| Logits → logprobs (`gather_logprobs`) | **fp32 upcast before log_softmax**, always | hardcoded in the trainer helper; tested |
| `Trajectory.logprobs` / `Batch.*_logprobs` | **fp32** storage | data contract (rollout/types.py) |
| Ratio / KL / TIS math in losses | **fp32** (inputs already fp32 by the two rows above) | contract; KL log-diff additionally clamped to ±20 (algos/grpo.py) |
| Advantage / reward statistics | **fp32** | advantage.py operates on fp32 rewards |
| Norms, softmax internals | fp32 inside the op | inherited from HF/vLLM kernels; not ours to configure |

Per-device defaults, resolved automatically:

- **CUDA**: rollout bf16 · learner compute bf16 · master/optimizer/reductions fp32 · checkpoints bf16.
- **MPS/CPU (dev path)**: fp32 end-to-end — MPS bf16 is unreliable, the 0.6B
  model fits, and exactness is the point (it is what makes HFEngine a ~1e-5
  oracle against the learner). fp16 is never used: its 5-bit exponent needs
  loss-scaling machinery (GradScaler) that bf16 makes obsolete.

## 3. The three-copies-of-π consequence

In one GRPO step the "same" policy exists in three numeric forms:

1. **π_engine** — bf16 weights in vLLM's kernels → produced the tokens and
   `behavior_logprobs`;
2. **π_old** — trainer recompute: bf16 forward, fp32 logit upcast → `old_logprobs`;
3. **π_θ** — same procedure as (2), weights drifting over minibatches.

(1)≠(2) purely from kernels/rounding — typically 1e-3–1e-2 nats/token in bf16.
This is why `ratio` must always be formed from (2) and (3) (one consistent
procedure on both sides of the subtraction), why `behavior_logprobs` are kept
separately for TIS to correct the (1)↔(2) gap, and why `engine_learner_kl` is
a standing dashboard metric. Note also that **weight sync quantizes**: the
engine receives the bf16 cast of fp32 master weights, so π_engine lags π_θ by
rounding even at perfect version parity. Not fixable — measured and corrected.

## 4. How slime does it (findings, with sources)

- **Training precision is bf16 by default**: `args.bf16 = not args.fp16`
  (backends/megatron_utils/arguments.py:151). fp32 master weights and fp32
  main-grad accumulation/allreduce come with Megatron's distributed optimizer
  — inherited machinery, no slime code; slime never exposes a "master weights
  dtype" knob. Same philosophy we adopt: it is an invariant, not a choice.
- **Logits → logprobs upcast**: `vocab_parallel_logits = vocab_parallel_logits.float()`
  inside `_VocabParallelLogProbEntropy` (utils/ppo_utils.py:199) — the fp32
  upcast happens INSIDE the autograd function, before log_softmax, exactly
  where our `gather_logprobs` does it.
- **KL/ratio math in fp32**: `log_ratio = log_probs.float() - log_probs_base.float()`
  (utils/ppo_utils.py:28, `compute_approx_kl`).
- **Rollout dtype** is delegated to SGLang: slime auto-wraps SGLang's
  `ServerArgs` with a `--sglang-` prefix (backends/sglang_utils/arguments.py),
  so `--sglang-dtype` defaults to "auto" = the checkpoint's torch_dtype
  (bf16 for Qwen releases). Mirrors our knob 1.
- **Weight sync does not cast**: update_weight_from_tensor.py groups sync
  buckets BY DTYPE and ships tensors as-is — the engine receives the same
  bf16 params Megatron computes forward passes with. The fp32→bf16
  quantization happens once, inside Megatron's optimizer→param copy each
  step. Consequence worth copying: **engine dtype == trainer compute dtype**,
  so the engine↔learner mismatch comes from kernels, never from storage. Our
  weight_sync.py should likewise broadcast the learner's compute-dtype
  parameters, not the fp32 master.
- Not adopted (out of scope for miniRL): fp8 rollout/KV-cache options in
  SGLang, Megatron's fp16+loss-scaling path.

## 5. Config surface (once the trainer lands)

```yaml
rollout:
  dtype: auto        # auto -> bf16 on CUDA, fp32 on MPS/CPU
train:
  compute_dtype: auto  # same resolution rule
  checkpoint_dtype: bf16
# everything else in §2 is hardcoded — deliberately not configurable
```

Tests that pin the invariants: gather_logprobs-vs-F.cross_entropy in fp32;
HFEngine logprob round-trip (< 1e-3); engine-vs-learner gap tolerance for
vLLM (DESIGN §6.0); microbatch gradient-equivalence (fp32 accumulation).
