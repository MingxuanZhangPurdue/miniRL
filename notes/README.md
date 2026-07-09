# notes/ — theory derivations

Imported verbatim from the author's personal RL study vault (`workspace/rl_notes`),
derivations only — the vault's annotated loss *implementations*
(`*_loss_explained.py`, `rl_loss.py`) are NOT imported because this repo IS
that implementation: `minirl/algos/` carries the same math with the same
notation, tested. Internal links inside these files may point at vault files
that live here as real code instead.

Reading order, and what each note underpins in the repo (the two *_grpo/gspo
notes were written FOR this repo, in the vault's style — the derivation chain
now covers every implemented loss):

1. **[pg_to_ppo_derivation.md](pg_to_ppo_derivation.md)** — policy gradient →
   advantage → PPO clip, and where GAE comes from. Start here. The foundation
   under every surrogate in `algos/`.
2. **[ppo_to_grpo_derivation.md](ppo_to_grpo_derivation.md)** — why the group
   mean is a valid critic replacement (baseline unbiasedness, the (1-1/G)
   RLOO relation), the ÷std and length biases (Dr. GRPO), the three reduces,
   clip-higher (DAPO). Underpins `algos/grpo.py` + `advantage.py` + `aggregate.py`.
3. **[grpo_to_gspo_derivation.md](grpo_to_gspo_derivation.md)** — the true
   sequence IS weight explodes with length; the geometric mean shrinks as
   1/|y|; sequence-granular trust region, tiny eps. Underpins `algos/gspo.py`.
4. **[ppo_to_cispo_derivation.md](ppo_to_cispo_derivation.md)** — why CISPO
   needs an explicit `log π` term (what stop-gradient does to the derivative).
   Underpins `algos/cispo.py`.
5. **[kl_estimators.md](kl_estimators.md)** — k1/k2/k3 KL estimators + the
   k3 unbiasedness proof. Underpins the KL penalty in `algos/grpo.py` and the
   `approx_kl` metric in every loss.
6. **[dpo_derivation.md](dpo_derivation.md)** — the DPO objective from the
   RLHF objective (implicit reward, why β, why a reference model). Will
   underpin `algos/dpo.py` when it lands.

The formula/notation quick-reference for the implemented zoo is
`minirl/algos/README.md`; each loss file's banner carries its own derivation
summary and slime grounding. These notes keep some LOCAL symbols that collide
with the repo's ($G_t$ = return, kl_estimators' $r = \pi_{ref}/\pi$) — the
mapping table lives in the root [README.md § Notation](../README.md#notation).
