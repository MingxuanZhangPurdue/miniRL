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
packing, agentic RL, fast-RL throughput); math derivations live in `notes/`;
the algorithm
formula/config reference is `minirl/algos/README.md`; the repo-wide notation
glossary (symbols, shapes, the four policies, notes/ symbol mapping) is
`README.md` § Notation — new comments and docs MUST use those symbols.

## Environment & commands

TWO dev machines share this repo; detect which one you are on before running
anything (`platform`/`uname`):

- **Apple-silicon Mac** — Python: the `mingxuan` conda env, ALWAYS —
  `/Users/mingxuanzhang/miniconda3/envs/mingxuan/bin/python` (torch+MPS,
  transformers, datasets, math-verify). Never create venvs.
- **Windows 11 desktop, RTX 5070 (12GB)** — THE CUDA box since 2026-07-20.
  Native Windows cannot run the training stack (triton/TE/NCCL are
  Linux-only): everything GPU runs inside the Docker container `minirl-mega`
  (NGC image + mcore 0.18 + bridge 0.5, repo bind-mounted at
  /workspace/miniRL; recipe + validated stack: docs/megatron.md §5a/§6).
  Start Docker Desktop first, then:
  `docker start minirl-mega`, and run things via
  `docker exec -w /workspace/miniRL -e PYTHONPATH=/workspace/miniRL minirl-mega python ...`
  vLLM lives in the SAME container in an isolated venv —
  `/opt/vllm-env/bin/python` (its pinned torch must not replace the NGC
  torch; docs/box_runbook.md §7, incl. the WSL2 no-UVA fallback).
- Tests: `<that python> -m pytest tests/ -q` — keep it green; it is fast
  (~10s Mac CPU, ~15s in the container).
- The Mac is MPS + CPU only. THE trainer is
  Megatron-Core (`minirl/megatron.py`, docs/megatron.md) and is CUDA-box-ONLY
  (megatron-core hard-imports triton; measured 2026-07-20). Local tests drive
  `tests/fake_trainer.py` — the demoted hand-written DDP trainer, now the
  executable spec of the trainer contract (2-process gloo/CPU equivalence —
  docs/ddp.md; FSDP2 history in docs/fsdp2.md). NCCL is validated on the box.
  vLLM RUNS LOCALLY via the vllm-metal plugin in the isolated venv
  `~/.venv-vllm-metal` (user-approved exception to the no-venv rule; repo dev
  stays on mingxuan) — spike findings + weight-update recipe:
  docs/async_tier2.md §8. (Packing was prototyped and rolled back for
  readability — docs/packing.md.)
- Smoke recipes (vLLM-only since 2026-07-20 — HFEngine/StreamAdapter and the
  MPS recipe 03 are REMOVED): `recipes/04_smoke_vllm_cuda.py` (engine
  validation) then `recipes/05_grpo_gsm8k_cuda.py` (GRPO on GSM8K), both
  CUDA-box; on the Mac, vLLM = the vllm-metal venv.

## Conventions that are LAW here

1. **DESIGN principle 8**: abstractions must earn their existence — no base
   classes/protocols/registries/string-path plugins; plain callables and the
   file-vs-config rule (an algorithm gets its own file only if its loss BODY
   differs from GRPO's; otherwise it is a named config in `LOSSES`).
2. **Docs are REFERENCE, not law** (relaxed 2026-07-21; was "docs before
   code"): `docs/*.md` exist to help understanding — consult them for
   context, but do not treat them as binding specs, and do not spend effort
   keeping every doc in lockstep with every change. Update a doc when it
   genuinely helps a future reader; skip the ceremony otherwise.
3. **Algorithm files follow the user's rl_notes annotation style** (banner,
   LOSS formula block + NOTATION legend, WHAT-CHANGES tables, FROZEN/grad
   marks, shape comments on every tensor line) — see any file in
   `minirl/algos/` and the memory note `algo-comment-style`.
4. **Ground claims in sources**: decisions cite slime/verl source or papers;
   when unsure, fetch and read before implementing.
5. Loss functions return UNREDUCED (B, T) maps; the trainer reduces ONCE
   (`loss_agg`); denominators are minibatch-global. Trajectories/batches live
   on CPU; fp32 for logprob math (docs/precision.md).
6. **Every comment in code exists to help a reader understand the code —
   nothing else.** Not for provenance (dates, "measured on...", decisions,
   change history), not for docs/*.md § cross-references, not for pointing
   at tests. All of that lives in docs/ and git history. (Older comments
   predate this rule; don't churn them, but don't imitate them.)
