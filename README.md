# miniRL

Minimal, tested post-training laboratory (SFT / RLVR / DPO / agentic RL) on
small open models. Architecture, principles, and status: **[DESIGN.md](DESIGN.md)**
(read it first). Algorithm formula/config quick-reference:
[minirl/algos/README.md](minirl/algos/README.md). Theory derivations:
[notes/](notes/README.md). Subsystem design notes: `docs/`.

## Notation

One notation, used everywhere: the loss-file banners (`minirl/algos/*.py`),
tensor shape comments, `docs/`, and the derivations in `notes/`. Code comments
write `pi_theta` in ASCII; the markdown notes write $\pi_\theta$ — same symbol.
Notes also carry the completion index explicitly ($r_{i,t}$, $y_{i,t}$) where
code comments drop it ($r_t$, $y_t$) because the batch dimension makes it
implicit.

### Indices & sizes (the shape symbols)

| symbol | meaning |
|---|---|
| `i` | completion (sequence) index within a batch |
| `t` | token position |
| `P` | number of prompts in a batch (= number of groups) |
| `G` | group size — completions sampled per prompt (`SamplingParams.n`) |
| `B` | total completions in a batch: `B = P*G`; rows of every `Batch` tensor |
| `b` | rows of ONE microbatch slice under gradient accumulation (`b <= B`) |
| `T` | padded prompt+completion length — the token axis of every trainer-side tensor. One row's own T = `prompt_len + response_len`; a `Batch`'s T = the max over its rows (right-padded) |
| `prompt_len`, `response_len` | one `Trajectory`'s split of its T (properties on `Trajectory`; the response INCLUDES its EOS token) |
| `T_max` | ENGINE-side: padded prompt length of a generation batch — prompts left-padded to the longest (hf_engine.py) |
| `T_gen` | ENGINE-side: the generated-token axis, `<= max_new_tokens`; pad-filled past each row's EOS, trimmed (EOS-inclusive) before becoming a `Trajectory` |
| `max_new_tokens` | the generation budget (`SamplingParams`) — also the value to pass as Dr. GRPO's constant `C` |
| `\|y_i\|` | completion i's real response-token count = `loss_mask[i].sum()` = its `response_len` |
| `V` | vocabulary size (~152k for Qwen3) — why logprobs are gathered per step, never as a `(B, T, V)` buffer (hf_engine.py) |

Shape comments read: `(B, T)` per-token over the batch, `(B,)` per-sequence,
`(P,)` per-group internals (advantage.py only), `(T,)` one `Trajectory`,
`(b, T)` one microbatch, `(B, T, V)` / `(b, T, V)` raw logits.
`loss_mask (B, T) bool` is True on completion (RL) / assistant (SFT) tokens
only — the positions the loss may touch.

Padding sides are load-bearing, not cosmetic: the ENGINE left-pads prompts to
`T_max` (generated tokens must sit flush against their context), while
training batches are right-padded to `T` (make_batch); `Trajectory` tensors
carry no padding at all.

### The four policies

| symbol | what it is | grad? | its logprobs on the realized tokens |
|---|---|---|---|
| `pi_theta` ($\pi_\theta$) | the current policy being trained | the ONLY grad path | `policy_logprobs` — the live forward pass |
| `pi_old` | the policy at the start of this optimizer update | frozen | `Batch.old_logprobs` — trainer's fp32 recompute; the losses fall back to `behavior_logprobs` when it is absent (sync fresh-weights case) |
| `pi_engine` | the rollout engine's policy that actually sampled the tokens (possibly an older weight version, always different numerics) | frozen | `Batch.behavior_logprobs` — "behavior policy" in RL terms; slime calls the same tensor `rollout_log_probs` |
| `pi_ref` | frozen reference model, the KL anchor | frozen | `Batch.ref_logprobs` |

The two-gap picture (tis.py): `pi_engine --(gap 1: TIS w_t)--> pi_old
--(gap 2: PPO clip on r_t)--> pi_theta`. docs/async_training.md writes
`pi_old = learner@v_k` and `pi_theta = learner@v_k+` to emphasize the version
timeline — same objects.

### Per-token and per-sequence quantities

| symbol | definition | shape |
|---|---|---|
| `y_t` | sampled token at position t (notes: $y_{i,t}$) | — |
| `R_i` | terminal (outcome) reward of completion i | `(B,)` |
| `A_i` | group advantage `(R_i - mean_group) / (std_group + eps_std)`; Dr. GRPO drops the `÷std`; ONE scalar per completion, broadcast across its tokens | `(B,)` → `(B, T)` |
| `r_t` | token IS ratio `exp(log pi_theta(y_t) - log pi_old(y_t))` | `(B, T)` |
| `s_i` | GSPO sequence ratio = geometric mean of the token ratios = `exp((1/\|y_i\|) * sum_t log r_t)` | `(B,)`, broadcast to `(B, T)` |
| `rho_i` ($\rho_i$) | the TRUE sequence IS weight `prod_t r_t` — unusable (variance explodes with length); appears only in notes/grpo_to_gspo_derivation.md as what `s_i` replaces | — |
| `w_t` | TIS weight `clamp(exp(log pi_old - log pi_engine), lo, hi)`, always detached | `(B, T)` |
| `d` | the log-ratio fed to k3 (see below) | `(B, T)` |
| `L_t` | per-token loss — the UNREDUCED loss map every loss fn returns, zero outside `loss_mask` | `(B, T)` |
| `L` | the scalar loss after the ONE reduce (`loss_agg`, aggregate.py) | scalar |

### Scalars, operators, and their config fields

| symbol | meaning | config field / default |
|---|---|---|
| `eps` / `eps_hi` | lower/upper clip deltas from 1 — ratio trust window `[1-eps, 1+eps_hi]` | `eps_clip=0.2`, `eps_clip_high=None` (=symmetric). DAPO: `eps_clip_high=0.28`. GSPO: `3e-4`/`4e-4`. CISPO: `eps_clip=None` = no floor |
| `lo`, `hi` | the TIS clamp band for `w_t` | `tis_clip_low=0.0`, `tis_clip=2.0` |
| `beta` ($\beta$) | KL-regularization strength — GRPO's KL penalty coefficient, and the same role as DPO's temperature | `kl_loss_coef=0.0` |
| `C` | Dr. GRPO's CONSTANT reduce denominator (reserved — never the TIS cap) | `loss_agg=<int>`, pass `max_new_tokens` |
| `eps_std` | numerical epsilon in the advantage's `÷(std + eps_std)` — NOT a clip delta | `1e-6` (advantage.py) |
| `sg(.)` | stop-gradient == `.detach()` | — |
| `clip(x, a, b)` / `clamp` | `min(max(x, a), b)`; `a = -inf` (or `None`) = unbounded below | — |
| `k3(d)` | `e^d - d - 1` — the unbiased, non-negative KL estimator (proof: notes/kl_estimators.md) | — |

**k3's two uses** (same estimator, different `d`, different direction):

- KL **penalty** (`use_kl_loss`): `d = log pi_ref - log pi_theta` → estimates
  `KL(pi_theta || pi_ref)` under samples from `pi_theta`.
- `approx_kl` **metric** (every RL loss): `d = log r_t` → estimates
  `KL(pi_old || pi_theta)` — "how far has this minibatch drifted".

### The reduce (`loss_agg` — the ONE place normalization lives)

| value | `L =` | paper |
|---|---|---|
| `"seq_mean"` | `(1/B) * sum_i ( sum_t L_t / \|y_i\| )` | GRPO |
| `"token_mean"` | `sum_i sum_t L_t / sum_i \|y_i\|` | DAPO |
| `int C` | `sum_i sum_t L_t / (B*C)` | Dr. GRPO |

Applied ONCE by the trainer with a minibatch-GLOBAL denominator
(aggregate.py); the loss functions never reduce.

### Local notation inside `notes/` (verbatim vault imports — read with this mapping)

The derivation notes predate this repo and keep their own local symbols where
the underlying concept is PPO-side or statistics-side:

| file | local symbol | means there | caution / repo equivalent |
|---|---|---|---|
| rl_notes vault (cross-refs in grpo.py / advantage.py banners) | `N` | number of completions | = `B` |
| rl_notes vault (same cross-refs) | `L` | padded sequence length | = `T`; vault tensors live on the `(L-1)` PREDICTION axis, ours are position-aligned (`logprobs[:, t]` scores token `t` itself — gather_logprobs) |
| pg_to_ppo_derivation.md | $G_t$ | return (reward-to-go) | NOT group size `G`; in this repo's bandit setting it collapses to the terminal `R_i` |
| pg_to_ppo_derivation.md | $r(\theta)$ | the ratio $\pi_\theta/\pi_{old}$, no token subscript | = `r_t` |
| pg_to_ppo_derivation.md | $A_t$, $V(s)$, $Q(s,a)$, $\delta_t$, $\gamma$, $\lambda$ | per-STEP advantage / value fns / TD residual / GAE knobs | PPO-side machinery; GRPO replaces all of it with the per-completion `A_i` |
| kl_estimators.md | $r$ | likelihood ratio $p/q = \pi_{ref}/\pi$ | NOT the policy ratio `r_t`; its code variable `h = -log r` equals `-d` of the KL penalty |
| grpo_to_gspo_derivation.md | $\sigma^2$ | per-token log-ratio noise variance | not `std_group` |
| dpo_derivation.md | $y_c, y_r$, $\beta$, $\sigma(\cdot)$, $Z(x)$, $\hat r_\theta$ | chosen/rejected, temperature, sigmoid, partition fn, implicit reward | $\beta$ plays the same KL-strength role as `kl_loss_coef` |
