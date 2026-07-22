# From PPO to GRPO — deriving the group baseline

Why GRPO can delete PPO's value network and still have a valid advantage. Short answer: **LLM RLVR
is a contextual bandit (one terminal reward, resettable prompt), so a Monte-Carlo baseline —
the mean reward of G sibling completions of the SAME prompt — is a legitimate substitute for the
learned $V(s)$; subtracting any per-prompt baseline leaves the policy gradient unbiased.**

Cross-refs: [`pg_to_ppo_derivation.md`](pg_to_ppo_derivation.md) (§ score-function trick, § baselines,
§ the clipped surrogate — everything here builds on it),
`minirl/algos/grpo.py` (the implementation), `minirl/algos/advantage.py` (STEP 2),
`minirl/algos/aggregate.py` (the reduce, §4 below), `minirl/algos/README.md` (config quick-reference).

---

## 0. The setting that makes it possible

For LLM RLVR, one "episode" is: prompt $x$ (state) → completion $y\sim\pi_\theta(\cdot|x)$ (one
mega-action) → one terminal reward $R(x,y)$. No intermediate rewards, no bootstrapping horizon, and —
crucially — the state is **resettable for free**: we can sample as many independent completions of the
same $x$ as we like. PPO's critic exists to estimate $\mathbb{E}_{y}[R\,|\,x]$ from *single* visits;
here we can just… sample $G$ times and average.

---

## 1. Any per-prompt baseline is unbiased

Policy gradient for one prompt (score-function estimator, see pg_to_ppo §1):
$$
\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}\big[R(x,y)\,\nabla_\theta\log\pi_\theta(y|x)\big].
$$
Subtract any baseline $b(x)$ that does not depend on the sampled $y$:
$$
\mathbb{E}_{y}\big[b(x)\,\nabla\log\pi_\theta(y|x)\big]
= b(x)\,\nabla\underbrace{\mathbb{E}_y[1]}_{=\,1} \;=\; b(x)\cdot\nabla 1 \;=\; 0,
$$
using $\mathbb{E}[\nabla\log\pi]=\nabla\!\int\!\pi = \nabla 1 = 0$. So
$$
\boxed{\;\nabla_\theta J=\mathbb{E}\big[(R-b(x))\,\nabla\log\pi_\theta\big]\quad\text{for ANY }b(x)\;}
$$
The baseline changes **variance only**. PPO spends a whole second network learning the
variance-optimal-ish $b(x)=V(x)$; GRPO gets a cheap one for free.

---

## 2. The group mean is a Monte-Carlo critic (with one small bias)

Sample $G$ completions $y_1..y_G$ of the same prompt; set
$$
b(x)=\bar R=\tfrac1G\textstyle\sum_j R_j,\qquad A_i = R_i-\bar R .
$$
$\bar R$ is an unbiased estimate of exactly what PPO's critic tries to learn,
$\mathbb{E}_y[R\,|\,x]$ — no training, no staleness, no critic-loss hyperparameters.

**The fine print**: $\bar R$ *includes* $R_i$ itself, so the baseline is not fully independent of
sample $i$:
$$
A_i=R_i-\tfrac1G\textstyle\sum_j R_j=\big(1-\tfrac1G\big)\Big(R_i-\underbrace{\bar R_{-i}}_{\text{mean of others}}\Big).
$$
Leave-one-out (RLOO) uses $\bar R_{-i}$ and is exactly unbiased; GRPO's version is RLOO scaled by the
constant $(1-\frac1G)$ — a uniform shrinkage, harmless, vanishing as $G$ grows. (This is why RLOO is a
4-line `advantage_fn` away in our code.)

---

## 3. The ÷std — and why Dr. GRPO deletes it

GRPO additionally normalizes per group:
$$
A_i=\frac{R_i-\bar R}{\mathrm{std}(R_{1..G})+\varepsilon}.
$$
Intent: prompt-to-prompt scale invariance (binary rewards, shaped rewards, whatever — advantages come
out $\mathcal O(1)$). The cost (**Dr. GRPO's difficulty-bias critique**): std depends on the sampled
$y$'s, so this is no longer a clean baseline — near-solved or near-impossible prompts (tiny std) get
their advantages **amplified** by the small denominator, over-weighting exactly the prompts with the
least signal. Dr. GRPO: keep $A_i=R_i-\bar R$, drop the division.
In code this is one flag: `grpo_std_normalization=False`.

> Note the degenerate case either way: all $G$ rewards equal $\Rightarrow A_i=0$ for the whole group —
> zero gradient. That is a DATA problem, not a loss problem, and is fixed at collection time
> (DAPO's dynamic sampling; `RolloutConfig(dynamic_sampling=True)`).

---

## 4. Assemble the objective (the surrogate is unchanged from PPO)

Broadcast $A_i$ to every token of completion $i$ (one terminal reward ⇒ uniform per-token credit),
and reuse PPO's clipped surrogate verbatim (derived in pg_to_ppo §3):
$$
L_{i,t}=-\min\!\big(r_{i,t}A_i,\;\mathrm{clip}(r_{i,t},\,1{-}\epsilon,\,1{+}\epsilon_{hi})\,A_i\big),
\qquad r_{i,t}=\tfrac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{old}(y_{i,t}|\cdot)} .
$$
What remains is the **reduce** — and it is a real algorithmic choice, not bookkeeping:
$$
\underbrace{\tfrac1B\textstyle\sum_i\tfrac{1}{|y_i|}\sum_t L_{i,t}}_{\text{seq\_mean (GRPO)}}\qquad
\underbrace{\tfrac{\sum_i\sum_t L_{i,t}}{\sum_i|y_i|}}_{\text{token\_mean (DAPO)}}\qquad
\underbrace{\tfrac{1}{B\cdot C}\textstyle\sum_i\sum_t L_{i,t}}_{\text{const }C\text{ (Dr. GRPO)}}
$$
GRPO's per-sequence mean divides each completion by ITS OWN length — a long wrong answer gets its
penalty diluted (**length bias**, Dr. GRPO's second critique; models drift long-when-wrong). DAPO
weighs tokens equally against the batch's actual token count; Dr. GRPO insists the denominator be a
**constant** $C$ (the generation budget) so the estimator is unbiased w.r.t. sampled lengths — the
choice of $C$ only rescales the lr. One config field: `loss_agg = "seq_mean" | "token_mean" | int`.

---

## 5. Clip-higher (DAPO) in one inequality

With symmetric $\epsilon$, a token at probability $p$ can rise to at most $(1{+}\epsilon)p$ before its
gradient is clipped away: absolute headroom $\epsilon p$ — **tiny exactly when $p$ is tiny**. Rare
exploration tokens ("Wait", "However") get frozen at birth while confident tokens barely feel the
clip; entropy collapses. DAPO raises only the ceiling ($\epsilon_{hi}=0.28>\epsilon=0.2$), leaving the
safety floor alone. That is the whole change: `eps_clip_high=0.28`.

---

## 6. Side by side

| | PPO | GRPO |
|---|---|---|
| baseline $b(x)$ | learned $V(x)$ (second network, own loss, stale) | group mean $\bar R$ (free, fresh, per prompt) |
| unbiasedness | baseline exact by §1 | $(1-\frac1G)$·RLOO (§2); ÷std re-biases (§3) |
| advantage | per-TOKEN (GAE over steps) | per-COMPLETION, broadcast to tokens |
| needs | critic fwd/bwd, GAE, value clip | $G$ samples per prompt |
| surrogate | clipped ratio | **identical** |
| where it lives here | not implemented (see DESIGN non-goals) | `algos/grpo.py` + `advantage.py` + `aggregate.py` |

---

## 7. One-line summary

> Because LLM RLVR is a resettable bandit, "sample $G$ siblings and subtract their mean" is a free,
> (nearly) unbiased Monte-Carlo replacement for PPO's critic; everything else — the clipped
> surrogate — carries over unchanged, and the named variants are two knobs on this construction:
> Dr. GRPO deletes the ÷std and fixes the reduce's denominator, DAPO raises the clip ceiling and
> counts tokens instead of sequences.
