# From GRPO's token ratio to GSPO's sequence ratio — a derivation

Why GSPO replaces $r_t$ with a per-sequence $s_i$, and why its clip range is ~1000× smaller. Short
answer: **the reward is per-SEQUENCE, so the importance weight "should" be per-sequence too — but the
true sequence IS weight is a product of $|y|$ token ratios whose variance explodes with length. GSPO's
fix is the length-normalized GEOMETRIC MEAN of the token ratios: the same information, with variance
that SHRINKS as $1/|y|$ instead of exploding — at the price of no longer being the unbiased IS weight.**

Cross-refs: [`ppo_to_grpo_derivation.md`](ppo_to_grpo_derivation.md) (the surrogate this modifies),
[`pg_to_ppo_derivation.md`](pg_to_ppo_derivation.md) (§ importance sampling),
`minirl/algos/gspo.py` (the implementation — the derivation below is its "THE GSPO MOVE" comment,
expanded).

---

## 0. The mismatch GSPO points at

GRPO scores a per-sequence event (one terminal reward $A_i$ for the whole completion) but corrects
off-policyness with **per-token** weights $r_{i,t}=\pi_\theta(y_{i,t}|\cdot)/\pi_{old}(y_{i,t}|\cdot)$,
and clips each token independently. Granularity mismatch: sequence-level signal, token-level trust
region.

---

## 1. The true sequence IS weight, and why it is unusable

The honest importance weight for a whole completion is the product:
$$
\rho_i=\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}=\prod_{t=1}^{|y_i|} r_{i,t}
\qquad\Longleftrightarrow\qquad
\log\rho_i=\sum_t \log r_{i,t}.
$$
Model each per-token log-ratio as roughly independent noise with variance $\sigma^2$ (post-update
drift + numerics). Then
$$
\mathrm{Var}\big(\log\rho_i\big)\;\approx\;|y_i|\,\sigma^2
\quad\Longrightarrow\quad
\rho_i\ \text{is (log-)normal with spread growing like } e^{\sqrt{|y_i|}\,\sigma}.
$$
For a 2k-token chain-of-thought, $\rho_i$ is astronomically heavy-tailed: almost every sequence would
sit outside ANY clip window $[1-\epsilon,1+\epsilon]$, so a trust region on $\rho_i$ either clips
everything (no learning) or nothing (no protection). This is the same variance-vs-bias wall that
motivates TIS truncation (`tis.py`) — unbiased IS weights are unaffordable at sequence length.

---

## 2. What GRPO actually does about it (the per-token heuristic)

GRPO never forms $\rho_i$; it applies $r_{i,t}$ token-by-token. That is NOT the unbiased sequence
estimator either — it is a heuristic that bounds each token's step separately. Its failure mode is the
mirror image: one outlier token (rare word, numerics spike) clips **itself** out while the other
$|y_i|-1$ tokens of the same completion train on — so the "unit of trust" (a token) disagrees with the
unit of reward (the sequence), and per-token noise still enters the gradient $|y_i|$ times.

---

## 3. The GSPO move: length-normalize the product

Take the geometric mean — the product, deflated by length:
$$
\boxed{\;s_i=\rho_i^{1/|y_i|}=\Big(\prod_t r_{i,t}\Big)^{1/|y_i|}
=\exp\Big(\tfrac{1}{|y_i|}\sum_t\log r_{i,t}\Big)\;}
$$
Same drift information as $\rho_i$, but now the log is a **mean**, not a sum:
$$
\mathrm{Var}\big(\log s_i\big)\approx\frac{\sigma^2}{|y_i|}
\qquad\text{(vs. } |y_i|\sigma^2 \text{ for }\log\rho_i\text{).}
$$
Variance *shrinks* with length instead of exploding. Consequences, in order:

1. $s_i$ concentrates tightly around 1 — which is exactly why GSPO's clip deltas are **tiny**
   ($\epsilon\sim3\!\times\!10^{-4}$, vs. GRPO's $0.2$): the window must be narrow to ever bind.
2. $s_i$ is **biased** as an IS weight ($\mathbb{E}[s_i]\neq\mathbb{E}[\rho_i]$ — Jensen). GSPO
   accepts this deliberately: variance control over unbiasedness, the same trade TIS makes by
   truncating. The clip was already a bias-for-stability device; GSPO just moves it to the right
   granularity.
3. The trust region now accepts/rejects **whole sequences** — matching the granularity of the reward,
   and immunizing a good completion against one outlier token.

---

## 4. The gradient: every token gets a $1/|y_i|$ share

$s_i$ is one number per sequence, broadcast to all its tokens. Differentiate (chain rule through the
mean, using $\nabla\log r_t = \nabla\log\pi_\theta(y_t)$):
$$
\nabla_\theta s_i = s_i\cdot\frac{1}{|y_i|}\sum_t \nabla_\theta\log\pi_\theta(y_{i,t}|\cdot).
$$
So the surrogate $-\min(s_iA_i,\ \mathrm{clip}(s_i)A_i)$ still trains **every token** of an accepted
sequence — each receives an equal $1/|y_i|$ share of the sequence's update — and a clipped sequence
contributes zero for **all** its tokens at once. (In code: the broadcast
`seq_log_ratio.exp().unsqueeze(-1).expand_as(...)` is what routes the gradient back through the mean.)

---

## 5. Side by side

| | GRPO | GSPO |
|---|---|---|
| IS weight | $r_{i,t}$, one per token | $s_i=\exp(\text{mean}_t\log r_{i,t})$, one per sequence |
| relation to true $\rho_i$ | applies its factors separately | $\rho_i^{1/\lvert y_i\rvert}$ (length-normalized) |
| $\mathrm{Var}(\log\cdot)$ | $\sigma^2$ per token, entering $\lvert y_i\rvert$ times | $\sigma^2/\lvert y_i\rvert$, entering once |
| clip granularity / $\epsilon$ | token / $0.2$ | sequence / $\sim3\!\times\!10^{-4}$ |
| outlier token | clips itself, siblings unaffected | averaged away (or the whole seq clips) |
| unit of trust vs unit of reward | mismatched | matched |
| where it matters | short answers: fine | long CoT, MoE routing noise — GSPO's motivating cases |

---

## 6. One-line summary

> The unbiased sequence IS weight is a length-long product with exponentially exploding variance;
> GRPO dodges it with per-token weights that mismatch the per-sequence reward. GSPO takes the product
> and length-normalizes it — a geometric mean whose log-variance shrinks as $1/|y|$ — buying a
> sequence-granular trust region (tiny $\epsilon$, whole-sequence accept/reject) at the cost of IS
> unbiasedness the clip had already spent anyway.
