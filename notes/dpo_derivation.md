# DPO Derivation — full walkthrough (personal note)

Goal of DPO (Direct Preference Optimization): **skip the reward model + RL loop.** Train directly on
preference pairs `(x, y_chosen, y_rejected)` with a simple classification-style loss, yet provably
optimize the *same* KL-constrained RLHF objective. Reference:
[rlhf-book lec6](https://github.com/natolambert/rlhf-book/blob/main/teach/course/lec6-chap8-dpo.md).

> 🔍 = a step I want to look at closely / likely to discuss. Equations numbered (S#) for reference.

---

## Step 1 — The RLHF objective we're secretly solving

DPO optimizes the standard KL-regularized reward-maximization objective. **Fully explicit form**
(the KL depends on `x`, so it *must* live inside $\mathbb{E}_{x\sim\mathcal{D}}$):

$$
\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\Big[\,\mathbb{E}_{y\sim\pi(\cdot|x)}\big[r(x,y)\big]
\;-\;\beta\,\mathcal{D}_{\text{KL}}\!\big(\pi(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)\,\Big]
\tag{S1}
$$

"Get high reward, but don't drift far from the reference (SFT) model." `β` sets the tradeoff. This is
the *same* objective PPO-RLHF maximizes — DPO just solves it differently.

> ⚠️ **On the notation.** Papers usually write the compact
> $\ \max_\pi \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}[r(x,y)]-\beta\,\mathcal{D}_{\text{KL}}(\pi\|\pi_{\text{ref}})$.
> That's shorthand: $\mathcal{D}_{\text{KL}}(\pi(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))$ is a **per-prompt**
> quantity (a function of $x$), so a well-defined scalar objective **requires** averaging it over
> $x\sim\mathcal{D}$ — i.e. the $\mathbb{E}_x$ above is mandatory, not optional. And the KL is itself an
> $\mathbb{E}_{y\sim\pi}[\log\tfrac{\pi}{\pi_{\text{ref}}}]$. So S1 secretly holds **both** an outer
> $\mathbb{E}_x$ and an inner $\mathbb{E}_y$; the next steps just make them explicit and merge them.

---

## Step 2 — Expand the KL, flip max → min

KL is itself an expectation, $\mathcal{D}_{\text{KL}}(\pi\|\pi_{\text{ref}})=\mathbb{E}_{y\sim\pi}\big[\log\tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\big]$, so (S1) becomes one expectation:

$$
\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot|x)}\Big[\,r(x,y)-\beta\log\tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\,\Big]
\tag{S2a}
$$

Divide by $\beta>0$ (doesn't change the argmax) and flip sign to a **min**:

$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot|x)}\Big[\,\log\tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\tfrac{1}{\beta}r(x,y)\,\Big]
\tag{S2b}
$$

---

## Step 3 — The exp/log trick + the partition function 🔍

**The trick:** rewrite the reward term as a log so it can merge into the existing log:
$$
\tfrac{1}{\beta}r(x,y)=\log e^{\,r(x,y)/\beta}.
$$
Substitute into (S2b) and combine logs:
$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot|x)}\Big[\,\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)\,e^{\,r(x,y)/\beta}}\,\Big]
\tag{S3}
$$

> 🔍 **Why we're stuck here, and why `Z(x)` appears.** We'd love to read (S3) as a KL divergence
> $\mathbb{E}[\log\tfrac{\pi}{q}]$ — because a KL is minimized (to 0) exactly when $\pi=q$, instantly
> giving us the optimal policy. **But the denominator $\pi_{\text{ref}}(y|x)\,e^{r/\beta}$ is NOT a
> probability distribution** — it doesn't sum to 1 over $y$. So it's not a valid $q$ yet.
>
> **Fix:** normalize it. Define the **partition function** (the normalizing constant):
> $$
> Z(x)=\sum_{y}\pi_{\text{ref}}(y|x)\,e^{\,r(x,y)/\beta}.
> \tag{S3-Z}
> $$
> Then $q(y|x)=\tfrac{1}{Z(x)}\pi_{\text{ref}}(y|x)\,e^{r/\beta}$ **is** a valid distribution (sums to 1
> by construction). Note $Z(x)$ depends on $x$ and $\pi_{\text{ref}}$ and $r$ — **but not on $\pi$**.
> It is generally **intractable** (sum over all possible $y$) — that will turn out not to matter.

---

## Step 4 — Complete the KL (factor out `Z`)

Write the denominator as $\pi_{\text{ref}}e^{r/\beta}=Z(x)\cdot q(y|x)$ and split the log:
$$
\log\frac{\pi}{\pi_{\text{ref}}e^{r/\beta}}=\log\frac{\pi}{Z(x)\,q}
=\log\frac{\pi}{q}-\log Z(x).
$$
So (S3) becomes:
$$
\min_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\Big[\,\underbrace{\mathcal{D}_{\text{KL}}\big(\pi(\cdot|x)\,\|\,q(\cdot|x)\big)}_{\ge 0,\ \text{depends on }\pi}\;-\;\underbrace{\log Z(x)}_{\text{constant in }\pi}\,\Big]
\tag{S4}
$$
(The inner $\mathbb{E}_{y\sim\pi(\cdot|x)}[\log\tfrac{\pi}{q}]$ collapsed into the KL, which is why only
$\mathbb{E}_{x\sim\mathcal{D}}$ remains outside; $\log Z(x)$ has no $y$, so the inner expectation leaves
it untouched.)

---

## Step 5 — Gibbs' inequality → the optimal policy 🔍

> 🔍 **Gibbs' inequality:** $\mathcal{D}_{\text{KL}}(\pi\|q)\ge 0$, with equality **iff** $\pi=q$.

In (S4), $\log Z(x)$ is fixed w.r.t. $\pi$, so minimizing the whole thing = minimizing the KL term =
driving it to 0 = setting $\pi=q$. Hence the **closed-form optimal policy**:

$$
\boxed{\ \pi^{*}(y|x)=\tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\big(\tfrac{1}{\beta}r(x,y)\big)\ }
\tag{S5}
$$

Reads as: "reweight the reference model by $e^{r/\beta}$, then renormalize." High-reward responses get
up-weighted; `β` controls how aggressively. (This same formula underlies the PPO target too — it's the
optimal solution of the KL-constrained objective, independent of *how* you reach it.)

---

## Step 6 — Invert: express the reward via the policy 🔍

DPO's pivot: instead of *finding* $\pi^*$ from a known $r$, **solve (S5) for $r$** and treat the policy
as the unknown. Take $\log$ of (S5):
$$
\log\pi^{*}(y|x)=-\log Z(x)+\log\pi_{\text{ref}}(y|x)+\tfrac{1}{\beta}r(x,y).
$$
Rearrange for $r$:
$$
\boxed{\ r^{*}(x,y)=\beta\log\frac{\pi^{*}(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)\ }
\tag{S6}
$$

> 🔍 **Key idea — the "implicit reward."** Any reward function has a corresponding optimal policy
> (S5); inverted, this says the reward is *encoded in* the policy as $\beta\log\tfrac{\pi}{\pi_{\text{ref}}}$
> (plus the $\beta\log Z(x)$ offset). So if we parametrize the *policy* with $\theta$, we are implicitly
> parametrizing a *reward*. The annoying part is the $\beta\log Z(x)$ term — intractable. Watch it die in
> Step 7.

---

## Step 7 — Plug into Bradley–Terry; `Z(x)` cancels 🔍

The Bradley–Terry preference model (what the preference data is assumed to follow):
$$
p^{*}(y_1\succ y_2\mid x)=\frac{\exp r^{*}(x,y_1)}{\exp r^{*}(x,y_1)+\exp r^{*}(x,y_2)}.
\tag{S7a}
$$
Substitute (S6). Each $\exp r^{*}(x,y_i)=\exp\!\big(\beta\log\tfrac{\pi^*(y_i|x)}{\pi_{\text{ref}}(y_i|x)}\big)\cdot \underbrace{\exp(\beta\log Z(x))}_{=\,Z(x)^{\beta}}$.

> 🔍 **The cancellation.** $Z(x)^{\beta}$ is the **same factor** in *every* term — numerator and both
> denominator terms (it depends only on $x$, not on $y_1$ or $y_2$). It factors out and cancels:

$$
p^{*}(y_1\succ y_2\mid x)=\frac{\exp\!\big(\beta\log\tfrac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}\big)}
{\exp\!\big(\beta\log\tfrac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}\big)+\exp\!\big(\beta\log\tfrac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}\big)}
\tag{S7b}
$$

**This is the magic of DPO:** the intractable partition function $Z(x)$ disappears because Bradley–Terry
only cares about reward *differences*, and the offset $\beta\log Z(x)$ is identical for both responses.

---

## Step 8 — Sigmoid form

Divide numerator and denominator of (S7b) by the numerator's exponential, and use
$\sigma(z)=\tfrac{1}{1+e^{-z}}$:

$$
\boxed{\ p^{*}(y_1\succ y_2\mid x)=\sigma\!\Big(\beta\log\tfrac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}-\beta\log\tfrac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}\Big)\ }
\tag{S8}
$$

The preference probability is a sigmoid of the **difference of implicit rewards**
$\hat r_\theta(x,y)=\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.

---

## Step 9 — The DPO loss (max-likelihood on preferences)

Fit $\pi_\theta$ by **maximum likelihood** of the observed preferences = minimize negative log-likelihood
of (S8), with $y_c$ chosen, $y_r$ rejected:

$$
\boxed{\ \mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}})=-\,\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}
\Big[\log\sigma\!\Big(\beta\log\tfrac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)}-\beta\log\tfrac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\Big)\Big]\ }
\tag{S9}
$$

Just a forward pass on the policy + reference (ref frozen), a sigmoid, a log. **No reward model, no
rollouts, no RL loop.** That's the whole payoff.

---

## Step 10 — The gradient & its meaning 🔍

Let $u=\beta\log\tfrac{\pi_\theta(y_c|x)}{\pi_{\text{ref}}(y_c|x)}-\beta\log\tfrac{\pi_\theta(y_r|x)}{\pi_{\text{ref}}(y_r|x)}$ (the implicit-reward margin). Using $\tfrac{d}{dz}\log\sigma(z)=1-\sigma(z)=\sigma(-z)$:

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\,\mathbb{E}\big[(1-\sigma(u))\,\nabla_\theta u\big].
$$
With $\nabla_\theta u=\beta\big(\nabla_\theta\log\pi_\theta(y_c|x)-\nabla_\theta\log\pi_\theta(y_r|x)\big)$ (ref doesn't depend on $\theta$):

$$
\boxed{\ \nabla_\theta\mathcal{L}_{\text{DPO}}=-\,\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_r)-\hat r_\theta(x,y_c)\big)}_{\text{weight: "how wrong" the model is}}\cdot\big[\nabla_\theta\log\pi_\theta(y_c|x)-\nabla_\theta\log\pi_\theta(y_r|x)\big]\Big]\ }
\tag{S10}
$$

**Interpretation:**
- The bracket **pushes up** $\log\pi_\theta(y_c)$ (chosen) and **pushes down** $\log\pi_\theta(y_r)$ (rejected).
- The **weight** $\sigma(\hat r_r-\hat r_c)$ is large when the model currently ranks the *rejected* answer
  *above* the chosen one (i.e. it's wrong) → big update; small when already correct → tiny update. Automatic
  hard-example weighting.

---

## Steps worth a closer look (pick any to expand) 🔍

1. **(S2b)** Why dividing by $\beta$ and flipping sign is legitimate (argmax invariance).
2. **(S3 / S3-Z)** *Why* the partition function is needed — the "denominator isn't a distribution" point.
   The conceptual crux of the whole derivation.
3. **(S4)** The $\pi_{\text{ref}}e^{r/\beta}=Z\cdot q$ factoring and how the KL "appears."
4. **(S5)** Gibbs' inequality — why KL ≥ 0 and = 0 iff equal (and why that gives the global min).
5. **(S6)** The "implicit reward" reinterpretation — the philosophical pivot of DPO.
6. **(S7)** *Why* $Z(x)$ cancels (same offset on both responses; BT sees only differences). Most-asked step.
7. **(S8)** The algebra of turning the softmax-of-two into a sigmoid.
8. **(S10)** The gradient: deriving $\nabla\log\sigma=\sigma(-u)$ and reading the weight term.

> **Open gaps / my questions (fill in together):**
> - [ ]
> - [ ]

---

# Weaknesses of DPO

Source: [rlhf-book lec6-chap8-dpo](https://github.com/natolambert/rlhf-book/blob/main/teach/course/lec6-chap8-dpo.md).

1. **Likelihood displacement.** The loss only optimizes the *margin* between chosen and rejected
   (S8/S9), **not absolute probabilities**. The model can lower the loss by pushing the *rejected*
   log-prob down **faster** than the chosen — so `π(y_c)` can *also fall* as long as `π(y_r)` falls more.
   This can shove probability mass toward **unaddressed, off-distribution** outputs. (The gradient (S10)
   raises `log π(y_c)` and lowers `log π(y_r)`, but nothing pins the absolute level of either.)

2. **Static KL constraint.** DPO steps **directly to the optimal solution** implied by the *fixed* dataset
   and the *fixed* `β`. Online policy-gradient methods instead take steps on **freshly sampled** batches
   with a **per-sample (and even dynamically-adjusted) KL controller** → more adaptive. DPO's `β` is set
   once and its "trust region" is baked into the closed form.

3. **Limited by dataset coverage.** DPO trains on a **fixed, pre-collected** preference set, so it's capped
   by that set's **coverage** — a slightly lower performance ceiling. (Same coverage/pass@k argument as
   the RLVR discussion: offline data can't contain solutions the collectors never produced.)

4. **Lower peak performance vs online RL.** Because PPO/GRPO are **online** they can **explore new regions**
   (sample → verify/score → reinforce), often reaching **higher peak performance** than DPO's offline fit.

5. **Requires quality preference data.** DPO needs preference pairs `(x, y_c, y_r)` collected/generated
   **beforehand** — an extra prerequisite step, and quality-sensitive.

> **Ties to earlier notes:** weaknesses 3–4 are exactly why **DPO is a poor fit for RLVR** (offline, can't
> explore, capped by coverage) — see the "DPO vs RLVR" Q&A below. Weakness 2 connects to the
> KL-placement discussion: DPO's KL is a *static* closed-form constraint, vs PPO's *live* per-step penalty.

---

# My Notes & Q&A

My own running questions and the answers we worked out. (Add more below.)

## Q: Does DPO *have* to use the KL divergence in order to derive it?

**Yes — the KL is load-bearing for the derivation; DPO cannot be derived without it.** But it lives
*implicitly* in the final loss (via the reference model and `β`), not as an explicit `+β·KL` term.

**Why the KL is essential to the derivation:**
- The `−β·KL` term is exactly what produces the closed-form optimal policy
  `π* ∝ π_ref · e^{r/β}` (S5). Drop the KL and the unconstrained `max_π E[r]` has a **degenerate**
  solution — all mass on the single argmax-reward `y` (deterministic). No smooth exponential-tilt form.
- That closed form is what lets us **invert** to the implicit reward `r = β log(π/π_ref) + β log Z` (S6),
- which is what makes `Z(x)` **cancel** in Bradley–Terry (S7).
- Remove the KL → no `π*`, no reparametrization, no `Z` cancellation → **no DPO.**

**Where the KL is in the loss (S9):** there is **no explicit KL term**. It's encoded structurally by
(1) the reference model `π_ref` in every log-ratio `π_θ/π_ref`, and (2) the coefficient `β`.
Minimizing the DPO loss *is* solving the KL-constrained objective — the leash is automatic.

**"Removing" the KL in practice:**
- `β → 0`: the sigmoid margin `β·(…) → 0` → `σ(0)=0.5` → **gradient vanishes**, no learning. `β` is essential.
- Dropping `π_ref` (reference-free): gives a **different method** (SimPO, ORPO, CPO…), not DPO. These
  replace the KL anchor with something else (length-norm, target margin, SFT term) and need it for stability.

**One-liner:** DPO is *inherently* KL-regularized by construction — the KL-constrained objective is what
mathematically generates the `β log(π/π_ref)` reparametrization and the `Z(x)` cancellation. You can't
have "DPO without KL"; the reference model **is** the KL regularization.

Related: see [ppo_to_cispo_derivation.md](ppo_to_cispo_derivation.md) for the analogous "where does the
regularizer live" story in policy-gradient methods.

## Q: <next question>

