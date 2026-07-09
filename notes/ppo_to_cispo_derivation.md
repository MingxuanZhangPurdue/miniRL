# From PPO's ratio to CISPO's `log π` — a derivation

Why PPO/GRPO never write an explicit `log π`, yet CISPO does. Short answer: **PPO gets `∇log π`
*for free* through the differentiable ratio; CISPO detaches the ratio, which kills that path, so the
`∇log π` must be re-supplied explicitly by a REINFORCE-style `log π` term.**

Cross-refs: [`cispo_loss_explained.py`](cispo_loss_explained.py),
[`pg_to_ppo_derivation.md`](pg_to_ppo_derivation.md) (§1 score-function trick, §3 surrogate),
[`grpo_loss_explained.py`](grpo_loss_explained.py).

---

## 0. The one identity that explains everything

Log-derivative trick: $\nabla_\theta\pi_\theta=\pi_\theta\nabla_\theta\log\pi_\theta$. Apply it to the
importance ratio $r=\pi_\theta/\pi_{old}$:

$$
\nabla_\theta r=\nabla_\theta\frac{\pi_\theta}{\pi_{old}}=\frac{1}{\pi_{old}}\nabla_\theta\pi_\theta
=\frac{\pi_\theta}{\pi_{old}}\nabla_\theta\log\pi_\theta=r\,\nabla_\theta\log\pi_\theta.
$$

$$
\boxed{\;\nabla_\theta r = r\,\nabla_\theta\log\pi_\theta\;}
$$

**The ratio's gradient *is* the ratio times the score function.** Everything below follows from this.

---

## 1. PPO: `log π` is hidden inside the differentiable ratio

PPO's (unclipped) surrogate loss, with advantage $A$ detached:
$$
L_{\text{PPO}}=-A\,r.
$$
Differentiate, using the boxed identity:
$$
\nabla_\theta L_{\text{PPO}}=-A\,\nabla_\theta r = -A\,r\,\nabla_\theta\log\pi_\theta.
$$
The score function $\nabla\log\pi_\theta$ **appears automatically** — the chain rule through $r$
manufactures $r\,\nabla\log\pi_\theta$. That's why PPO never *types* `log π`: differentiating $r$
produces it. (The clip wraps $r$; when it saturates, $\nabla\,\text{clip}(r)=0$, so that token's
gradient becomes $0$ — the token is dropped from the update.)

---

## 2. CISPO: detaching `r` kills the gradient

CISPO wants to **detach** the ratio and use it as a fixed, clipped *weight* (to bound importance-
sampling variance). But if we naïvely stop-grad it with nothing else:
$$
L=-A\,\underbrace{\text{sg}(r)}_{\text{detached}}
\quad\Longrightarrow\quad
\nabla_\theta L=-A\,\nabla_\theta\,\text{sg}(r)=-A\cdot 0=0.
$$
**Zero gradient.** The $\nabla\log\pi_\theta$ that PPO got for free came *from* differentiating $r$;
detach $r$ and there is no policy-dependent term left for autograd to flow through.

---

## 3. CISPO: re-introduce `log π` explicitly (the REINFORCE surrogate trick)

To get a usable gradient we must add back a differentiable, policy-dependent term — the natural one
is $\log\pi_\theta$:
$$
\boxed{\;L_{\text{CISPO}}=-\,\text{sg}\big(\text{clip}(r)\big)\cdot A\cdot\log\pi_\theta\;}
$$
Now $\text{sg}(\text{clip}(r))$ and $A$ are constants, so autograd differentiates only $\log\pi$:
$$
\nabla_\theta L_{\text{CISPO}}=-\,\text{sg}\big(\text{clip}(r)\big)\cdot A\cdot\nabla_\theta\log\pi_\theta.
$$

This is exactly the **REINFORCE surrogate-loss construction**: to realize a policy gradient
$\mathbb{E}[w\,\nabla\log\pi]$ in autograd, write the pseudo-loss $\mathbb{E}[w\,\log\pi]$ with the
weight $w$ detached, and $\nabla$ gives $\mathbb{E}[w\,\nabla\log\pi]$. Here $w=\text{clip}(r)\cdot A$.

> The "extra `log π`" is **not new signal** — it is the *same* score function PPO was getting
> implicitly from $\nabla r=r\nabla\log\pi$, now written by hand because the gradient path through
> $r$ was cut.

---

## 4. Side by side: same form, different weight

$$
\begin{aligned}
\text{PPO (unclipped):}\quad &\nabla L=-A\cdot \underbrace{r}_{\text{differentiated weight}}\cdot\nabla\log\pi_\theta\\[2pt]
\text{CISPO:}\quad &\nabla L=-A\cdot \underbrace{\text{sg}(\text{clip}(r))}_{\text{detached weight}}\cdot\nabla\log\pi_\theta
\end{aligned}
$$

Both are the **IS-weighted score-function estimator** $-A\cdot(\text{weight})\cdot\nabla\log\pi$. The
only difference is how the weight is treated:

| | weight on $\nabla\log\pi$ | clip behavior |
|---|---|---|
| **PPO / GRPO** | $r$, **differentiated**, clipped | saturated clip $\Rightarrow$ weight's gradient $0$ $\Rightarrow$ **token dropped** |
| **CISPO** | $\text{sg}(\text{clip}(r))$, **detached**, clipped | a fixed bounded number $\Rightarrow$ **always multiplies $\nabla\log\pi$ $\Rightarrow$ token kept** |

- **PPO clipping = trust region**: stop learning from a token that moved too far.
- **CISPO clipping = variance bound**: keep learning from every token, but cap how much a far-moved
  token counts.

At $\theta=\theta_{old}$ (so $r=1$, clip inactive) both reduce to $-A\,\nabla\log\pi_\theta$ — the
vanilla policy gradient. They diverge only as the policy drifts and the clip engages.

---

## 5. One-line summary

> PPO hides $\log\pi$ inside the differentiable ratio ($\nabla r=r\,\nabla\log\pi$). CISPO **detaches**
> the ratio to use it as a pure variance-bounding weight; detaching removes the gradient path, so the
> score function must be **re-supplied explicitly** via a REINFORCE $\log\pi$ term. Result: identical
> gradient *form* $-A\cdot w\cdot\nabla\log\pi$, but the weight $w=\text{sg}(\text{clip}(r))$ is a
> bounded multiplier — so **clipping caps a token's influence instead of zeroing its gradient**, and
> no token is ever dropped from the update.
