# miniRL — session bootstrap

Minimal, tested post-training laboratory (SFT / RLVR / DPO / agentic RL) on
SMALL open models — model-agnostic by construction (everything keys off one
HF `model.name_or_path`); Qwen3-0.6B is the current default/starting point,
with other small families to be tried and supported over time. A study repo
whose miniature architecture deliberately mirrors slime and verl.

**Read [DESIGN.md](DESIGN.md) first** — it is the source of truth: goals,
principles, architecture, file tree with [done] marks, and a Status section
saying exactly what is built and what comes next. Docs-before-code repo:
every subsystem has a design note in `docs/` (sync/async training, precision,
packing, agentic RL); math derivations live in `notes/`; the algorithm
formula/config reference is `minirl/algos/README.md`.

## Environment & commands

- Python: the `mingxuan` conda env, ALWAYS —
  `/Users/mingxuanzhang/miniconda3/envs/mingxuan/bin/python` (torch+MPS,
  transformers, datasets, math-verify). Never create venvs.
- Tests: `<that python> -m pytest tests/ -q` — keep it green; it is fast (~10s).
- This machine is an Apple-silicon Mac: MPS + CPU only. vLLM/FSDP/packing are
  CUDA-path features — designed and documented, exercised on a GPU box later.
- Smoke recipe: `recipes/03_grpo_gsm8k.py` (GRPO on GSM8K, runs on MPS in ~2 min).

## Conventions that are LAW here

1. **DESIGN principle 8**: abstractions must earn their existence — no base
   classes/protocols/registries/string-path plugins; plain callables and the
   file-vs-config rule (an algorithm gets its own file only if its loss BODY
   differs from GRPO's; otherwise it is a named config in `LOSSES`).
2. **Docs before code**: new subsystems get a `docs/*.md` design note first;
   code must match the doc or the doc gets updated.
3. **Algorithm files follow the user's rl_notes annotation style** (banner,
   LOSS formula block + NOTATION legend, WHAT-CHANGES tables, FROZEN/grad
   marks, shape comments on every tensor line) — see any file in
   `minirl/algos/` and the memory note `algo-comment-style`.
4. **Ground claims in sources**: decisions cite slime/verl source or papers;
   when unsure, fetch and read before implementing.
5. Loss functions return UNREDUCED (B, T) maps; the trainer reduces ONCE
   (`loss_agg`); denominators are minibatch-global. Trajectories/batches live
   on CPU; fp32 for logprob math (docs/precision.md).
