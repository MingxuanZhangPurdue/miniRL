# KL Estimators in RLHF — k1, k2, k3 (with proofs)

Why the "KL" in GRPO/PPO code is a funny expression like `(r−1) − log r` instead of
`Σ p log(p/q)` — and a proof that the k3 estimator is unbiased.

Companion to: [`rl_loss.py`](rl_loss.py) (`approx_kl1/2/3`), [`grpo_loss_explained.py`](grpo_loss_explained.py),
[`ppo_loss_explained.py`](ppo_loss_explained.py). Source idea: http://joschu.net/blog/kl-approx.html

---

## 1. Why we estimate KL instead of computing it

The exact per-token KL is a sum over the **whole vocabulary**:
$$
\mathrm{KL}[\pi\|\pi_{ref}](s)=\sum_{a}\pi(a\mid s)\log\frac{\pi(a\mid s)}{\pi_{ref}(a\mid s)}.
$$
Computing it needs the **full distributions** of both models at every position. In RL we keep only the
log-prob of the **one token that was actually sampled** in the rollout. So we cannot evaluate the sum —
we build a **Monte-Carlo estimator** from that single sample. That is why the code's "KL" looks unusual.

---

## 2. Setup and notation

We sample tokens $x\sim q$ and want the KL to a reference $p$:
$$
q=\pi\ (\text{policy / sampling dist}),\qquad p=\pi_{ref},\qquad
\mathrm{KL}[q\|p]=\mathbb{E}_{x\sim q}\!\left[\log\frac{q(x)}{p(x)}\right].
$$
Define the **likelihood ratio** (reference over policy):
$$
r \equiv \frac{p(x)}{q(x)},\qquad \log r = \log p(x)-\log q(x).
$$
All three estimators are functions of this single sampled $r$. (Note: $-\log r=\log\frac{q}{p}$ is the
per-sample log-ratio of policy over reference.)

---

## 3. The three estimators

| name | formula (per sample $x\sim q$) | unbiased? | sign | variance |
|---|---|---|---|---|
| **k1** | $-\log r$ | ✅ yes | can be **negative** | high |
| **k2** | $\tfrac12(\log r)^2$ | ❌ biased | $\ge 0$ | low |
| **k3** | $(r-1)-\log r$ | ✅ yes | $\ge 0$ | low |

k3 is the sweet spot: **unbiased AND always non-negative AND low variance.** That's why GRPO (and modern
PPO) use it.

Sanity check at $r=1$ (policy $=$ reference): k1 $=0$, k2 $=0$, k3 $=(1-1)-0=0$. All vanish when there is
no divergence. ✓

---

## 4. Proof: k1 is unbiased

Directly from the definition of KL:
$$
\mathbb{E}_{x\sim q}[\,k_1\,]=\mathbb{E}_{q}[-\log r]
=\mathbb{E}_{q}\!\left[\log\frac{q(x)}{p(x)}\right]=\mathrm{KL}[q\|p].\qquad\blacksquare
$$
So k1 is the "obvious" estimator. Its flaws: it is **negative whenever** $p(x)>q(x)$ (the sampled token
was more likely under the reference), which happens often, and it has **high variance**.

---

## 5. Proof: k3 is unbiased  ⟵ the main result

$$
k_3=(r-1)-\log r.
$$
Take the expectation under $x\sim q$ and split:
$$
\mathbb{E}_{q}[\,k_3\,]=\underbrace{\mathbb{E}_{q}[\,r-1\,]}_{(\text{A})}+\underbrace{\mathbb{E}_{q}[-\log r]}_{(\text{B})}.
$$

**Term (A) — the ratio has mean 1, so this vanishes.** This is the key lemma:
$$
\mathbb{E}_{q}[\,r\,]=\mathbb{E}_{x\sim q}\!\left[\frac{p(x)}{q(x)}\right]
=\sum_{x} q(x)\,\frac{p(x)}{q(x)}
=\sum_{x} p(x)=1
\;\;\Longrightarrow\;\; \mathbb{E}_{q}[\,r-1\,]=1-1=0.
$$
(The $q(x)$ cancels; we're left summing the reference distribution to 1. This is just the importance-
sampling identity $\mathbb{E}_q[p/q]=1$.)

**Term (B) — this is exactly the KL** (same as k1):
$$
\mathbb{E}_{q}[-\log r]=\mathbb{E}_{q}\!\left[\log\frac{q}{p}\right]=\mathrm{KL}[q\|p].
$$

**Combine:**
$$
\boxed{\;\mathbb{E}_{q}[\,k_3\,]=0+\mathrm{KL}[q\|p]=\mathrm{KL}[q\|p]\;}\qquad\blacksquare
$$

So k3 is unbiased **for the same reason k1 is** — they share term (B). The extra piece $(r-1)$ is a
**zero-mean correction** that changes the variance and sign behavior *without* changing the expectation.

---

## 6. Proof: k3 is always ≥ 0

Let $f(r)=(r-1)-\log r$ for $r>0$.
$$
f'(r)=1-\tfrac{1}{r}=0 \iff r=1,\qquad f''(r)=\tfrac{1}{r^2}>0\ (\text{convex}),\qquad f(1)=0.
$$
A convex function whose only stationary point is its value-0 minimum is $\ge 0$ everywhere:
$$
(r-1)-\log r \ge 0\quad\forall r>0,\quad\text{equality iff } r=1.\qquad\blacksquare
$$
Equivalently, this is the standard inequality $\log r \le r-1$ (the log lies below its tangent at $r=1$).
So unlike k1, k3 can **never be negative** — it behaves like a proper "distance."

---

## 7. Why k3 has low variance (control variate + Taylor)

**Control-variate view.** Term (A), $\,(r-1)$, has mean 0. Adding *any* multiple of a zero-mean quantity
to an unbiased estimator keeps it unbiased:
$$
\mathbb{E}_q\big[\,(-\log r)+\lambda(r-1)\,\big]=\mathrm{KL}[q\|p]\quad\text{for any }\lambda.
$$
k1 is $\lambda=0$; **k3 is $\lambda=1$**. The term $(r-1)$ is strongly (positively) correlated with
$-\log r$, so adding it **cancels much of the fluctuation** — classic control-variate variance reduction.

**Taylor view (why $\lambda=1$ is special).** Near $r=1$, write $\log r=(r-1)-\tfrac12(r-1)^2+\cdots$. Then
$$
k_3=(r-1)-\log r=(r-1)-\Big[(r-1)-\tfrac12(r-1)^2+\cdots\Big]\approx \tfrac12(r-1)^2.
$$
The **linear term cancels** — leaving a quadratic that is small and always $\ge 0$. The linear term is
exactly what makes k1 swing positive/negative with high variance; k3 kills it. (Note $\tfrac12(r-1)^2\approx
\tfrac12(\log r)^2 = k_2$ near $r=1$ — so k3 ≈ k2 for small drift, but k3 stays *unbiased* globally while k2
does not.)

---

## 8. What about k2?

$k_2=\tfrac12(\log r)^2$ is **biased** (its expectation is not exactly the KL — it matches only to leading
order via the Taylor expansion above), but it is **always $\ge 0$** and **low variance**. It's a cheap,
stable choice when you only need a rough KL signal and don't care about exactness. k3 is usually preferred
because it gets non-negativity *and* unbiasedness simultaneously.

---

## 9. Sign convention & mapping to code ⚠️

Everything above used $r=p/q=\pi_{ref}/\pi$, i.e. $\log r=\log\pi_{ref}-\log\pi$ (**reference − policy**).
The proofs of unbiasedness rely on $\mathbb{E}_q[r]=\mathbb{E}_q[p/q]=1$, which needs this exact orientation.

In code (e.g. `rl_loss.py`) people often store `log_ratio = log_probs - log_probs_ref`
($=\log\pi-\log\pi_{ref}=-\log r$). With that variable $h\equiv-\log r$:
$$
k_1=h,\qquad k_2=\tfrac12 h^2,\qquad k_3=(e^{-h}-1)+h.
$$
So the **canonical unbiased k3** in that convention is `(-log_ratio).exp() ... ` → concretely
`exp(log_ratios_ref_minus_policy) - 1 - (log_ratios_ref_minus_policy)`. If a snippet computes
`(log_ratio.exp() - 1) - log_ratio` with `log_ratio = policy − ref`, double-check the sign: as a **loss
penalty** it still works (it is $\ge 0$ and minimized at $\pi=\pi_{ref}$, since $f$ is convex either way),
but the *value* only equals $\mathrm{KL}[\pi\|\pi_{ref}]$ for the **ref − policy** orientation. Use the
ref−policy sign whenever you log the number or use it as a calibrated penalty.

```python
# canonical, unbiased estimator of KL[pi || pi_ref], samples from pi:
log_r = logp_ref - logp          # = log r = log(pi_ref / pi)   (REF - POLICY)
k1 = -log_r
k2 = 0.5 * log_r**2
k3 = (log_r.exp() - 1) - log_r   # (r - 1) - log r,  >= 0, unbiased, low-variance
```

---

## 10. Practical notes

- **These are algorithm-agnostic.** Same estimators work in PPO (usually folded into the reward) and GRPO
  (added to the loss). Choice of estimator ⟂ choice of algorithm ⟂ reward-vs-loss placement.
- **Per token, then masked-mean.** Compute $k_\bullet$ at each completion token, then average over the
  `action_mask` (ignore prompt/padding) — see `masked_mean`.
- **Default:** **k3**. Non-negative, unbiased, low variance. k1 if you want the simplest/most classic; k2 if
  you want cheap and stable and don't need exactness.

---

## 11. Summary

| | formula | $\mathbb{E}_q[\cdot]$ | $\ge 0$? | variance | use |
|---|---|---|---|---|---|
| **k1** | $-\log r$ | $=\mathrm{KL}$ (unbiased) | no | high | classic / simplest |
| **k2** | $\tfrac12(\log r)^2$ | $\approx\mathrm{KL}$ (biased) | yes | low | cheap, rough |
| **k3** | $(r-1)-\log r$ | $=\mathrm{KL}$ (unbiased) | yes | low | **default** |

> **Core proof in one line:** $\mathbb{E}_q[k_3]=\mathbb{E}_q[r-1]+\mathbb{E}_q[-\log r]=0+\mathrm{KL}[q\|p]$,
> because $\mathbb{E}_q[r]=\mathbb{E}_q[p/q]=\sum p=1$. The added $(r-1)$ is a zero-mean control variate: it
> keeps the estimator unbiased while cancelling the linear (high-variance, sign-flipping) term of $-\log r$,
> and it makes the estimator non-negative ($\log r\le r-1$).
