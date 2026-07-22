# algos/ — the algorithm zoo

Every algorithm is `loss_fn(policy_logprobs (B,T), batch, cfg) -> (loss_map (B,T), metrics)`,
returning an UNREDUCED per-token loss map; the trainer applies the reduce once
(`cfg.loss_agg`, below). Obtain any of them with:

```python
loss_fn, cfg = make_loss("grpo")                          # paper defaults
loss_fn, cfg = make_loss("dapo", use_tis=True)            # named config + overrides
loss_fn, cfg = make_loss("dr_grpo", loss_agg=1024)        # paper-exact Dr. GRPO
```

**File-vs-config rule**: an algorithm gets its own `.py` only when its loss
BODY differs from GRPO's (gspo, cispo do). If it is reachable by setting
GRPO's config fields, it is a NAMED CONFIG in the `LOSSES` registry (dapo,
dr_grpo) — the diff between two entries below IS the diff between the papers.

## Notation

Full repo-wide glossary (incl. shape symbols and the notes/ mapping): root
[README.md § Notation](../../README.md#notation). The working set for this file:

```
pi_theta   current policy — the ONLY gradient path          B = P*G completions in a batch
pi_old     policy at update start (frozen fp32 recompute)   P = prompts (= groups)
pi_engine  rollout engine's policy (sampled the tokens)     G = group size (completions/prompt)
pi_ref     frozen reference model                           T = padded prompt+completion length
sg(.)      stop-gradient == .detach()                       y_t = sampled token at position t
                                                            eps / eps_hi = clip deltas from 1

r_t  = exp( log pi_theta(y_t|y_<t) - log pi_old(y_t|y_<t) )          token IS ratio
s_i  = exp( (1/|y_i|) * sum_t log r_t )                              sequence (geometric-mean) ratio
A_i  = (R_i - mean_group) / (std_group + eps_std)                    group advantage, one scalar per
       (eps_std ~ 1e-6, numerical only — NOT the clip eps)           completion, broadcast to its tokens
w_t  = clamp( exp(log pi_old - log pi_engine), lo, hi )              TIS weight (engine-mismatch +
       ((lo, hi) = (tis_clip_low, tis_clip) = (0.0, 2.0))            staleness correction)
k3   = e^d - d - 1,  d = log pi_ref - log pi_theta                   unbiased KL(pi||pi_ref) estimator
```

(`C` is reserved for Dr. GRPO's constant denominator — the reduce table below.)

## Implemented algorithms

| algorithm | `make_loss` | lives in | loss (per token t of completion i) | reduce |
|---|---|---|---|---|
| **GRPO** (arXiv:2402.03300) | `"grpo"` | grpo.py | `L_t = -min( r_t·A_i , clip(r_t, 1-eps, 1+eps_hi)·A_i )` `[+ beta·k3]` | `"seq_mean"` |
| **GSPO** (arXiv:2507.18071) | `"gspo"` | gspo.py | same surrogate with `s_i` in place of `r_t` (one ratio per SEQUENCE); eps ~3e-4 | `"seq_mean"` |
| **CISPO** (arXiv:2506.13585) | `"cispo"` | cispo.py | `L_t = -sg( clip(r_t, -inf, 1+eps_hi) ) · A_i · log pi_theta(y_t)` — clipped tokens KEEP gradient | `"token_mean"` |
| **SFT** | `"sft"` | sft.py | `L_t = -log pi_theta(y_t)` on assistant tokens | `"token_mean"` |
| **DAPO** (arXiv:2503.14476) | `"dapo"` | **config of GRPO** | GRPO with `eps_clip_high=0.28` (clip-higher), no KL (default) | `"token_mean"` |
| **Dr. GRPO** (arXiv:2503.20783) | `"dr_grpo"` | **config of GRPO** | GRPO with `grpo_std_normalization=False` (no ÷std in A_i) | `"token_mean"`; paper-exact: `loss_agg=<max_new_tokens>` |

DAPO's 4th change — dynamic sampling — is batch COLLECTION, not loss math:
`RolloutConfig(dynamic_sampling=True)` (controllers/fully_async.py) drops zero-gradient
groups (all rewards equal) and pulls more prompts.

Every loss optionally composes `[ * w_t ]` (TIS, `use_tis=True` — the async
default) before any KL term is added (multiplicative terms need an ordering
decision; additive terms inherit the reduce by linearity — see grpo.py).

## The reduce (`loss_agg`, applied once by the trainer — aggregate.py)

| value | formula | paper | intuition |
|---|---|---|---|
| `"seq_mean"` | `L = (1/B) · sum_i ( sum_t L_t / tokens_i )` | GRPO | every COMPLETION weighs equally |
| `"token_mean"` | `L = sum_i sum_t L_t / (total tokens)` | DAPO | every TOKEN weighs equally; denom varies with sampling |
| `int C` | `L = sum_i sum_t L_t / (B·C)` | Dr. GRPO | constant denom ⇒ unbiased wrt sampled lengths |

## Supported configuration fields (GRPOConfig; gspo/cispo carry the same minus KL)

| field | default | meaning | consumed by |
|---|---|---|---|
| `eps_clip` | 0.2 | lower clip delta (ratio floor `1-eps`); `None` in CISPO = unbounded below | the loss |
| `eps_clip_high` | None (=eps_clip) | upper delta; `0.28` = DAPO clip-higher | the loss |
| `use_kl_loss`, `kl_loss_coef` | False, 0.0 | `+ beta·k3(pi‖pi_ref)`; needs `batch.ref_logprobs` | the loss (GRPO only) |
| `grpo_std_normalization` | True | False = Dr. GRPO (drop ÷std in `A_i`) | advantage.py via the controller |
| `loss_agg` | per paper | the reduce, table above | trainer → aggregate.py |
| `use_tis` | False | truncated importance sampling `w_t` (True for async/vLLM) | the loss → tis.py |
| `tis_clip`, `tis_clip_low` | 2.0, 0.0 | the clamp band for `w_t` | tis.py |
| `tis_mode` | "clamp" | "clamp" = truncate (vanilla TIS); "mask" = reject out-of-band (icepop) | tis.py |

## Shared components (the pipeline steps the losses DON'T own)

- **advantage.py** — STEP 2: `A_i` group-relative baseline (+ `degenerate_group_mask`
  for dynamic sampling). Pluggable per-row estimators: `make_batch(advantage_fn=...)`.
- **tis.py** — the engine↔trainer correction `w_t`; orthogonal to every surrogate.
- **aggregate.py** — the reduce; ALL normalization knowledge lives there.

Deeper reading: each `.py` carries the full derivation banner (formula, WHAT
CHANGES vs GRPO, slime grounding). Theory derivations: `notes/` (imported
from the author's rl_notes vault); the vault's annotated `.py` companions
remain in `workspace/rl_notes/`.
