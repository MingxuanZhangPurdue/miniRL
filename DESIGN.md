# miniRL — Design Document

A minimal, pure-PyTorch post-training laboratory for small open-source LLMs —
**model-agnostic by construction**: nothing in the stack knows the
architecture; everything keys off one HF `model.name_or_path`, and trying a
new family is a checkpoint string plus a test run
(`MINIRL_TEST_MODEL=org/name pytest tests/test_hf_engine.py tests/test_data.py`
— the engine-contract and chat-mask tests are the two model-sensitive spots,
and both carry loud guards). Current default/starting point: **Qwen3-0.6B** —
a boring, standard dense transformer (GQA/RoPE/RMSNorm) with first-class vLLM
support and abundant community RLVR baselines to sanity-check against. More
small families get added by trying them; exotic architectures (e.g. the
hybrid linear-attention Qwen3.5-0.8B) are deliberately not first: in a repo
for learning RL, the model should be the most boring variable. The goal is **understanding, not
throughput**: every stage of modern post-training — SFT, DPO, RLVR, agentic RL,
async RL, on-policy distillation — implemented in small, readable, robust files
that together form a miniature version of production frameworks like
[slime](https://github.com/THUDM/slime) and [verl](https://github.com/verl-project/verl).

## 0. Status (updated 2026-07-14 — keep this section current)

**Built and tested (71 passing tests, all CPU/MPS):** the full RLVR stack —
`HFEngine` (rollouts on any device, exact behavior logprobs) + `VLLMEngine`
(continuous batching; Metal weight-update recipe validated, CUDA branch
awaits the box) + `StreamAdapter` (generate()-only engines speak the
streaming contract; one poll == one round), losses grpo/gspo/cispo/sft as
files + dapo/dr_grpo as named configs (`LOSSES` registry, algos/README.md),
group advantages with pluggable `advantage_fn`, TIS, the three-mode reduce
(`loss_agg`), `make_batch`, **the Megatron trainer** (`minirl/megatron.py`,
built 2026-07-20 per docs/megatron.md — Megatron-Core owns fwd/bwd,
DDP+fp32 grad reduce, bf16+fp32-master optimizer; Megatron-Bridge builds
the model from the HF name and exports HF-named weights for publish; our
losses plug in via the forward_step socket; CUDA-box-only, P1 parity run
still pending. The hand-written trainer it replaced — DDP merged
2026-07-16, ddp.md §7; FSDP2 before that, fsdp2.md — is DEMOTED to
`tests/fake_trainer.py`: the executable spec of the trainer contract that
the local suite drives on CPU), **wandb
metric logging** (`minirl/logging.py`; wandb lives ONLY in recipes; recipe 03
wires it behind `--wandb`), rewards (GSM8K math verifier + level-2 code
sandbox), the data layer (chat templating with assistant-mask, HF prompt
sources, SFT batching), a working GSM8K GRPO recipe
(`recipes/03_grpo_gsm8k.py`, smoke-tested on MPS end to end), and **THE
fully-async controller** (2026-07-14 consolidation, docs/async_tier2.md
§10-§11: `controllers/fully_async.py` — fit_async + collect_groups_dp; k DP
engines via shared dealer/tally with burst-capped deals, one owner thread
per engine, drain-ALL-then-publish, staleness ≤ publish_interval + 1; 1..m
trainer ranks — rank 0 collects + broadcasts + publishes from its own
replicated params, followers only train; 2-rank gloo test vs single-process;
dynamic sampling filters in
`rollout/filtering.py`; `PlacementConfig` + `VLLMEngine(gpu_id=...)` for the
single-node GPU split, slime's layout). Retired the same day: round_based /
streaming controllers, rollout/sampling + rollout/streaming collectors
(every one a k=1 special case of fully_async). REMAINING on-box: real
vllm-metal smoke (EOS parity + weight canary), per-engine GPU pinning spike
(§10a) and NCCL — the CUDA engine smoke PASSED 2026-07-20 (weight canary, EOS
parity, logprob gap; findings: EngineCore fork->spawn, tied-weight dedupe, V1
n>1 fan-out is ours now).

**Decided and documented, not yet built:** **the Megatron P1 box spike
(docs/megatron.md §7 — the P2 code is built and locally green, but the
loss/grad_norm parity vs tests/fake_trainer.py on a frozen batch and the
three integration conventions in minirl/megatron.py's banner are
UNVALIDATED until it runs on CUDA)**, the sync controller
(`controllers/sync.py` — collect -> train -> publish via collect_groups_dp,
no overlap), in-flight weight updates (DEFERRED by decision 2026-07-10 —
docs/async_tier2.md §4), NCCL weight sync (need the CUDA box;
docs/precision.md, DESIGN §6), packing (docs/packing.md — prototyped and
ROLLED BACK for readability, 2026-07-09; build-or-not is an open decision),
agentic runners (docs/agentic_rl.md), RLOO, DPO (notes/dpo_derivation.md
ready), on-policy distillation, eval harness, checkpoint/resume schedule.

**Agreed build order:** SFT recipe → GSM8K RLVR at real scale (GPU) → DPO →
agentic. PPO is a permanent non-goal (§ non-goals).

**Housekeeping:** git repo has a remote (MingxuanZhangPurdue/miniRL) but NO
commits yet — everything is uncommitted working tree.

## 1. Reference reports this repo follows

| Report | What we take from it |
|---|---|
| [Olmo 3](https://arxiv.org/abs/2512.13961) (Ai2) | The canonical open *model flow*: staged post-training pipeline SFT → DPO → RLVR, with every stage reproducible from configs. Our `recipes/` directory mirrors this staging. |
| [GLM-5](https://arxiv.org/abs/2602.15763) (Zhipu) | Async RL infrastructure that decouples generation from training, and async *agentic* RL algorithms for long-horizon interactions. Our async rollout/learner split and staleness-corrected objectives follow this. |
| [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) (Xiaomi) | Multi-Teacher On-Policy Distillation (MOPD): teachers provide dense token-level rewards to a student on the student's own rollouts. We implement a single-teacher "MOPD-lite" (≈ on-policy distillation / GKD) as an advanced recipe. |

Reference codebases and study resources:

- **[verl](https://github.com/verl-project/verl)**: the single-controller idea —
  one driver script owns the dataflow of an algorithm (generate → reward →
  advantage → update), and algorithms are expressed as compositions of
  primitives rather than monolithic trainers. Primary Rosetta Stone target (§9).
- **[slime](https://github.com/THUDM/slime)**: the three-box architecture —
  *training backend* ↔ *data buffer* ↔ *rollout backend* — with weight sync
  between them. We keep the exact same boxes, just in-process/single-node
  PyTorch instead of Megatron + SGLang. Also our model for colocated vs
  disaggregated GPU placement.
- **[SkyRL](https://github.com/novasky-ai/skyrl)** (NovaSky, Berkeley):
  modular full-stack RL framework with a clean generator/trainer/environment
  separation; its environment abstraction (SkyRL-Gym) and agentic multi-turn
  training (SWE-style tasks) are the closest published analogue to our
  `envs/` + `multiturn.py` design — cross-check our `TextEnv` protocol
  against it.
- **[rlhf-book](https://github.com/natolambert/rlhf-book)** (Nathan Lambert,
  *The RLHF Book*): the theory companion to this repo. Each algorithm note in
  `docs/` should cite the relevant chapter; recommended reading in lockstep
  with the roadmap phases (SFT/DPO/policy-gradient chapters before writing
  the corresponding loss).

## 2. Design principles

1. **Pure PyTorch on the training side.** No TRL, no DeepSpeed, no Megatron:
   every loss, trainer, buffer, and sync mechanism is ours. The one deliberate
   exception is the **rollout backend**: `vLLM` is the primary inference engine
   (see §6.0), because rollouts dominate RL wall-clock and because integrating
   a foreign engine — weight sync, logprob mismatch — is itself core RL-infra
   curriculum, not incidental complexity. An `HFEngine` (thin `model.generate`
   wrapper) is kept alongside it as the reference backend so the whole
   pipeline also runs on Apple MPS / CPU. Remaining dependencies:
   `transformers` (model architecture + loading, see principle 2), `datasets`
   (download data), `pyyaml`, optionally `wandb`.
2. **The model comes from HF; the checkpoint is the single source of truth.**
   We use `AutoModelForCausalLM` rather than writing the architecture —
   model internals aren't this repo's curriculum, and HF naming is the lingua
   franca that vLLM's weight loaders already speak, which makes weight sync
   name-mapping-free. Two invariants follow: (a) learner and rollout engine
   each load *independently* from one `model.name_or_path` — model objects
   never cross the boundary, only named tensors; (b) **every checkpoint we
   save is a valid HF checkpoint** (`save_pretrained` + tokenizer), so any
   intermediate artifact loads into the learner, vLLM, or evals with no
   custom format (this is also how Olmo 3 publishes its model flow).
3. **One file, one concept.** Each algorithm is a single file exposing a loss
   function; each infra concern (buffer, weight sync, sampler) is a single file.
   Target: no file over ~400 lines.
4. **Shape comments everywhere.** Every tensor gets a comment like
   `# (B, T, V)` at creation and after any reshape. Loss functions document
   their inputs' shapes and masking semantics in the docstring.
5. **Everything configurable, nothing hidden.** Typed dataclass configs,
   composed from YAML + CLI overrides. An experiment == one YAML file in
   `configs/`. No global state, no magic registries.
6. **Robust minimalism.** Minimal ≠ fragile: correct loss masking, correct
   logprob recomputation, gradient-accumulation-safe metrics, deterministic
   seeding, resumable checkpoints, NaN guards. These are exactly the details
   production frameworks get right and tutorials get wrong.
7. **Single node, multi-GPU as the design target.** The reference setup is one
   node with N GPUs (e.g. 4–8): learner on a DDP process group, rollout
   engines on their own GPUs, real NCCL weight sync between them — the actual
   topology slime/verl manage, minus multi-node. Everything must still degrade
   gracefully to 1 GPU (colocated mode) and to CPU/MPS for smoke tests, so the
   repo stays runnable anywhere.
8. **Abstractions must earn their existence.** No base classes, protocols,
   registries, or generality that a second concrete use hasn't demanded yet —
   and even then, prefer the boring mechanism (duck typing over Protocol, a
   dict over a plugin system, a separate readable file over a configurable
   general function). Applied so far: engines and envs are duck-typed with the
   contract in a docstring; each RL algorithm is its own file rather than
   flags on a shared one. Readability outranks flexibility everywhere.
   Corollary — **no string-path plugins, ever**: slime's `--custom-*-path`
   hooks (rollout fn, rm, advantage, TIS, loss) exist because its users
   configure a packaged CLI from shell scripts; miniRL's users write Python
   recipes against a library they own, so every such hook maps to a plain
   callable argument (e.g. collect_groups' reward_fn / group_filter). Same
   extensibility, no load_function machinery, type-checked.
9. **Concept-compatible with production frameworks.** Component boundaries,
   names, and dataflow deliberately mirror slime and verl, so that after
   miniRL you can open those codebases with zero new mental model — only new
   scale. Wherever miniRL simplifies something (e.g. broadcasting whole weights
   instead of resharding between training and inference layouts), the file
   carries a `# In production:` comment saying exactly what slime/verl do
   instead. See §9 for the full mapping.

## 3. Repository structure

```
miniRL/
├── DESIGN.md                    # this file
├── README.md
├── pyproject.toml
│
├── minirl/
│   ├── config.py                # dataclass configs: CollectConfig + PlacementConfig
│   │                            #   (single-node GPU split, slime's layout — §11)
│   ├── logging.py               # [done] metrics_logger: the on_metrics callback —
│   │                            #   namespacing + derived metrics + iteration-as-step;
│   │                            #   wandb injected by RECIPES, never imported by core
│   │
│   ├── controllers/             # training drivers — exactly TWO by decision (§11)
│   │   ├── fully_async.py       # [done] THE async loop: fit_async + collect_groups_dp —
│   │   │                        #   k DP engines (shared dealer + tally, burst-capped
│   │   │                        #   deals, one owner thread per engine), 1..m trainer
│   │   │                        #   ranks (rank 0 collects + broadcasts + publishes
│   │   │                        #   locally; followers only train — DDP replication),
│   │   │                        #   drain-ALL-then-publish (docs/async_tier2.md
│   │   │                        #   §10-§11, docs/ddp.md). round_based.py /
│   │   │                        #   streaming.py / data_parallel.py retired 2026-07-14 —
│   │   │                        #   both tiers were k=1 special cases of this file
│   │   └── sync.py              # (planned) collect -> train -> publish, no overlap;
│   │                            #   same collector, no pipeline slot — the debugging
│   │                            #   and teaching reference
│   │
│   ├── models/
│   │   ├── hf.py                # load_policy()/load_ref(): AutoModelForCausalLM +
│   │   │                        #   dtype, attn impl, grad ckpt, use_cache=False;
│   │   │                        #   save_checkpoint() always in HF format
│   │   └── value_head.py        # scalar head wrapping the HF model (reward model only —
│   │                            #   no critic: PPO is out of scope, see non-goals)
│   │
│   ├── engine/                  # rollout backends, pluggable (≈ verl's rollout worker)
│   │   │                        # engines are duck-typed on the STREAMING contract:
│   │   │                        #   submit/poll/stash/drain/n_inflight + load_weights
│   │   │                        #   + pad_id; sampling side on rollout/types.SamplingParams
│   │   ├── vllm_engine.py       # [done] PRIMARY backend: low-level LLMEngine step loop
│   │   │                        #   (continuous batching); the streaming interface
│   │   │                        #   only; gpu_id pinning for DP placement (§11);
│   │   │                        #   weight updates via callable RPC + safetensors path
│   │   │                        #   (Metal recipe validated, CUDA branch awaits box;
│   │   │                        #   docs/async_tier2.md §8); Mac: vllm-metal venv
│   │   ├── hf_engine.py         # REFERENCE backend: HFEngine wraps batched
│   │   │                        #   model.generate on its own HF model copy;
│   │   │                        #   returns exact sampling logprobs; runs on
│   │   │                        #   CUDA / MPS / CPU — the local-dev + CI path
│   │   ├── stream_adapter.py    # [done] StreamAdapter: generate()-only engine ->
│   │   │                        #   streaming contract; one poll == one ROUND (the
│   │   │                        #   retired tier 1 as an engine property, §11)
│   │   └── README.md            # when to use which; the logprob-mismatch story
│   │
│   ├── data/                    # HF `datasets` does loading/caching/splits — we only
│   │   │                        #   turn rows into the 3 shapes the repo already defines.
│   │   │                        #   Per-dataset adaptation = a plain row_fn, not a config
│   │   │                        #   schema (principle 8).
│   │   ├── chat.py              # THE single owner of apply_chat_template. encode_prompt
│   │   │                        #   (RL) + encode_conversation (SFT: assistant-only
│   │   │                        #   loss_mask via incremental prefix diffing — the one
│   │   │                        #   hard algorithm here; ≈ slime MultiTurnLossMaskGenerator)
│   │   ├── prompts.py           # HFPromptSource(dataset, tok, row_fn) -> the
│   │   │                        #   (n)->[(ids, meta)] callable collect_groups expects
│   │   ├── sft.py               # HF conversations -> Trajectory(logprobs=0) -> make_batch
│   │   │                        #   (an SFT example IS a Trajectory; reuses RL collation)
│   │   └── preference.py        # chosen/rejected pairs for DPO (with its consumer, later)
│   │
│   ├── algos/                   # pure loss functions — no training loops here.
│   │   │                        # ONE FILE PER PAPER, each readable standalone and
│   │   │                        #   grounded line-by-line against slime; variants are
│   │   │                        #   separate files, not flags. All share one signature:
│   │   │                        #   loss(policy_logprobs, batch, cfg) -> (loss_map (B,T), metrics)
│   │   │                        #   returning UNREDUCED per-token maps. Per-algo configs
│   │   │                        #   (slime CLI names) live next to their loss.
│   │   ├── sft.py               # [done] masked NLL
│   │   ├── grpo.py              # [done] PPO-clip surrogate + optional KL-to-ref
│   │   │                        # DAPO + Dr. GRPO: NAMED CONFIGS of grpo.py in the
│   │   │                        #   LOSSES registry (their loss bodies == GRPO's);
│   │   │                        #   formula table + config reference: algos/README.md
│   │   ├── gspo.py              # [done] sequence-level (geometric-mean) ratio
│   │   ├── cispo.py             # [done] stop-grad clipped IS weight (grad thru clips)
│   │   ├── tis.py               # [done] shared: truncated importance sampling (clamp/icepop)
│   │   ├── aggregate.py         # [done] shared: token- vs sequence-level reduction, ONCE
│   │   ├── advantage.py         # [done] shared: GRPO group norm (Dr.GRPO flag), degenerate mask
│   │   ├── dpo.py               # DPO (needs ref_logprobs in Batch)
│   │   ├── reinforce.py         # REINFORCE / RLOO (leave-one-out baseline; fits the
│   │   │                        #   scalar AdvantageFn tier — batching.py)
│   │   └── distill.py           # on-policy distillation: reverse-KL to teacher (MOPD-lite)
│   │
│   ├── megatron.py              # [built 2026-07-20, box-validation pending] THE trainer:
│   │                            #   Megatron-Core end to end (docs/megatron.md).
│   │                            #   Bridge-built model from the HF name, fused-CE
│   │                            #   logprobs (ONE shift adapter), our losses via the
│   │                            #   forward_step socket, global-denominator rule kept,
│   │                            #   bridge HF export feeds engine publish unchanged.
│   │                            #   CUDA-only (mcore hard-imports triton). Replaced
│   │                            #   train/trainer.py (the hand-written DDP trainer,
│   │                            #   2026-07-16..20 — demoted to tests/fake_trainer.py
│   │                            #   as the executable spec; ddp.md/fsdp2.md history)
│   │
│   ├── rollout/                 # ≈ slime's data buffer + orchestration layer
│   │   ├── types.py             # [done] SamplingParams, Trajectory, Batch (the data contract)
│   │   ├── batching.py          # [done] make_batch (pad + advantages), mini/microbatch slicing
│   │   │                        #   [planned] pack_batch: sequence packing (see §6, packing)
│   │   ├── filtering.py         # [done] group filters (reward_nonzero_std = DAPO dynamic
│   │   │                        #   sampling) + RewardFn/GroupFilter types. Collection
│   │   │                        #   itself lives in controllers/fully_async.py —
│   │   │                        #   sampling.py and streaming.py retired 2026-07-14 (§11)
│   │   ├── buffer.py            # (dropped: tier 2 shipped WITHOUT queues — the engine
│   │   │                        #   stash + held-future join replaced them)
│   │   └── weight_sync.py       # (planned) NCCL learner → engine weight publication;
│   │                            #   today's path is full_state_dict + safetensors file.
│   │                            #   (placement.py dropped: PlacementConfig in config.py
│   │                            #   + VLLMEngine gpu_id cover single-node DP, §11)
│   │
│   ├── rewards/
│   │   ├── math.py              # [done] extract (boxed/last-number) -> normalize ->
│   │   │                        #   exact Fraction equality; math-verify (optional dep)
│   │   │                        #   only for symbolic MATH-style labels
│   │   ├── code.py              # [done] level-2 sandbox: subprocess + rlimit prelude +
│   │   │                        #   group-kill + temp cwd (slime ReTool-shaped);
│   │   │                        #   production = sandbox service via remote_rm pattern
│   │   ├── reward_model.py      # optional learned RM (Bradley-Terry training + scoring)
│   │   └── shaping.py           # length penalties, format bonuses, reward mixing
│   │
│   ├── envs/                    # agentic RL (deferred; full study+design: docs/agentic_rl.md)
│   │   │                        # NO env classes — an agent is a loop, not a component.
│   │   │                        #   Each file is an EPISODE RUNNER: a function that loops
│   │   │                        #   generate -> parse tool call -> execute -> append masked
│   │   │                        #   observation tokens, and returns finished Trajectories.
│   │   │                        #   Plugs into collect_groups as generate_fn; controller
│   │   │                        #   unchanged. (== slime's custom-generate-function pattern)
│   │   ├── python_tool.py       # math-with-interpreter (ReTool-style): first runner
│   │   └── search_qa.py         # retrieval QA (Search-R1-style): second, needs a local index
│   │
│   ├── eval/
│   │   ├── generate_eval.py     # sample-and-score harness
│   │   └── tasks.py             # GSM8K, MATH subset, IFEval-lite, task registry
│   │
│   └── utils/
│       ├── logging.py           # console + wandb/jsonl metrics
│       ├── seeding.py
│       └── misc.py              # NaN guards, timing, memory stats
│
├── recipes/                     # end-to-end runnable experiments (the "model flow")
│   ├── 01_sft.py                # (planned)
│   ├── 02_dpo.py                # (planned)
│   ├── 03_grpo_gsm8k.py         # [done] RLVR on math, Mac/MPS smoke (already async:
│   │                            #   fit_async with StreamAdapter(HFEngine), k=1)
│   ├── 04_smoke_vllm_cuda.py    # [done, needs box] on-box engine validation ladder:
│   │                            #   contract / EOS parity / logprob gap / weight canary
│   ├── 05_grpo_gsm8k_cuda.py    # [done, needs box] k vLLM engines x m DDP ranks via
│   │                            #   torchrun + PlacementConfig (docs/box_runbook.md)
│   ├── 06_agentic_tooluse.py    # (planned) multi-turn tool-use RL
│   └── 07_onpolicy_distill.py   # (planned) MOPD-lite from a larger teacher
│
├── configs/                     # one YAML per experiment
│   ├── sft_qwen06b.yaml
│   ├── dpo_qwen06b.yaml
│   ├── grpo_gsm8k.yaml
│   └── ...
│
├── tests/                       # correctness tests (see §8)
├── docs/                        # one short note per topic. Exist: sync_training.md,
│                                #   async_training.md, precision.md, agentic_rl.md,
│                                #   packing.md, fast_rl.md (continuous batching +
│                                #   in-flight updates: why), async_tier2.md (same:
│                                #   how, file by file); production_gap.md to come
└── notes/                       # THEORY derivations, imported from the author's
                                 #   rl_notes vault (derivations only — the vault's
                                 #   annotated loss implementations ARE minirl/algos/):
                                 #   pg→PPO, KL estimators (k3 proof), PPO→CISPO, DPO
```

## 4. Core data contract

Everything flows through two dataclasses in `rollout/types.py`; getting this
contract right is what makes SFT/DPO/RL/agentic/async all share one trainer.

```python
@dataclass
class Trajectory:
    # A full sampled episode (single- or multi-turn).
    input_ids: Tensor        # (T,)  prompt + all generated/tool tokens
    loss_mask: Tensor        # (T,)  1 where the POLICY produced the token
                             #       (0 on prompt AND tool-output tokens)
    logprobs: Tensor         # (T,)  behavior-policy logprobs at sampling time
    reward: float            # terminal scalar (or per-token via token_rewards)
    version: int             # policy version that generated it (async staleness)
    meta: dict               # prompt id, group id, env info, ...

@dataclass
class Batch:
    # Collated, padded training batch.
    input_ids: Tensor        # (B, T)
    loss_mask: Tensor        # (B, T)
    behavior_logprobs: Tensor# (B, T)  for importance ratios
    advantages: Tensor       # (B, T)  broadcast or per-token
    ...
```

Key invariant, worth a test of its own: **`loss_mask` marks exactly the tokens
the policy is responsible for.** In multi-turn agentic rollouts, tool outputs
are inside `input_ids` but masked out — this single detail is where most
agentic-RL implementations go wrong.

## 5. Algorithm coverage (the curriculum)

Each algorithm = one loss file + one recipe + one doc note. Ordered as a study path:

1. **SFT** — masked NLL. Teaches: chat templates, loss masking, packing.
2. **DPO** — `-log σ(β[log(π_θ(y_c)/π_ref(y_c)) − log(π_θ(y_r)/π_ref(y_r))])`
   on chosen/rejected pairs (y_c, y_r) — notes/dpo_derivation.md eq. (S9).
   Teaches: reference models, implicit reward, why sequence logprobs need
   length awareness. Variants behind flags: IPO, label smoothing (cDPO).
3. **REINFORCE / RLOO** — simplest policy gradient with leave-one-out baseline.
   Teaches: score function estimator, variance reduction, before any clipping.
4. **GRPO family** [implemented] — the workhorse (RLVR à la Olmo 3 /
   DeepSeek-R1): sample G completions per prompt, advantage = group-normalized
   reward, PPO-clip surrogate, optional KL-to-ref penalty. FILE-VS-CONFIG
   RULE: a variant gets its own file only when its loss BODY differs —
   grpo.py, gspo.py (sequence-level ratio), cispo.py (stop-gradient clipped
   IS weight); variants reachable through GRPO's fields are NAMED CONFIGS in
   the LOSSES registry — DAPO (clip-higher + token reduce; dynamic sampling
   via the collector filter, rollout/filtering.py) and Dr. GRPO (no ÷std + constant reduce).
   Formula table, notation, and the full config reference: algos/README.md.
5. **PPO with critic** — [decided 2026-07: NOT implemented] — value-function
   training + GAE is a whole second training loop for an algorithm the RLVR
   field has moved past; we study it by CONTRAST instead (rl_notes
   ppo_loss_explained.py; every grpo.py docstring names what PPO would add).
   The per-token advantage path stays open architecturally (overwrite
   batch.advantages post-collation — batching.py's tier-2 contract).
6. **On-policy distillation (MOPD-lite)** — student samples, teacher scores
   every token; loss = reverse KL to teacher on student's own distribution.
   Teaches: dense rewards, distillation-as-RL (MiMo-V2-Flash's insight).
7. **Async variants of 3–4** — same losses, off-policy data; adds truncated
   importance sampling (TIS) correction using `behavior_logprobs` vs current
   policy, and staleness clipping by `version` (GLM-5 style).

All RL losses share one signature —
`loss(policy_logprobs, batch, cfg) -> (loss, metrics)` — and get logprobs from
one shared, carefully-tested helper (`gather_logprobs(logits, labels)`,
`(B, T, V) → (B, T)`), because *recomputed-logprob mismatches between engine
and learner are the #1 silent RL bug*.

## 6. Sync vs async RL (the infra core)

### 6.0 Rollout backends: vLLM (primary, CUDA) + HF (reference, runs anywhere)

Both controllers talk to a `RolloutEngine` protocol; which backend fills it is
one config field (`rollout.backend: vllm | hf`).

- **`VLLMEngine` (primary)** — used for all real experiments (Phase 3+).
  In-process `vllm.LLM`, `logprobs` requested at sampling time, weights updated
  in place from the learner (per-tensor `load_weights` / collective RPC — the
  same paths TRL and open-instruct use), `sleep()`/`wake_up()` to free KV-cache
  memory in colocated mode. This mirrors verl's vLLM rollout worker and plays
  the role SGLang plays for slime.
- **`HFEngine` (reference)** — batched `model.generate` on the *same* HF
  module class as the learner, returning exact sampling logprobs. Slow but
  runs on CUDA, Apple MPS, and CPU — the local-dev and CI path (develop the
  whole pipeline on a Mac, run real experiments on the GPU box), and the
  trusted oracle when debugging vLLM integration: since it shares the
  learner's module, its logprob gap vs the learner is near zero by
  construction, isolating vLLM-specific numerics.
  (MPS notes, recorded in `hf_engine.py`: prefer fp32/fp16 over bf16;
  set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Logprob gotcha, verified empirically:
  use `generate(..., output_logits=True)` for behavior logprobs — `output_scores`
  returns post-temperature/top-p logits and inflates the engine↔learner gap
  from ~1e-5 to ~0.8 nats/token.)

**The logprob-mismatch experiment (first-class, not a footnote).** vLLM's
kernels, paged attention, and bf16 numerics produce sampling logprobs that
differ slightly from the learner's recomputation of the *same* tokens — so
even "on-policy" data is mildly off-policy. This mismatch is the historical
motivation for truncated importance sampling in production pipelines. Because
we have both backends, we can study it directly rather than take it on faith:

1. Measure per-token `|logπ_vllm − logπ_learner|` and `|logπ_hf − logπ_learner|`
   distributions (logged continuously as `engine_learner_kl` — a standing
   dashboard metric, with an alert threshold, since drift here silently
   corrupts every RL loss).
2. Ablation recipe: GRPO with ratios computed against sampling logprobs vs
   recomputed logprobs, with and without TIS — reproduce the divergence, then
   the fix.

### Synchronous (baseline, `rollout/controller.py`)

```
loop:
  engine.load_weights(policy)                  # weights always fresh
  trajs  = engine.generate(prompts, G per prompt)
  trajs  = reward_fn(trajs)
  batch  = make_batch(trajs, advantage_fn)
  for _ in range(ppo_epochs): trainer.step(batch)
```

Simple, on-policy, but the GPU alternates between generation and training —
exactly the inefficiency GLM-5's async infra removes.

### Asynchronous (`controllers/fully_async.py`)

Built in two tiers, mirroring slime (full study + design: docs/async_training.md):

- **Tier 1 — one-step-off pipelining (the basic async trainer, built first)**:
  slime's train_async.py shape — launch rollout k+1 as a background future,
  train on rollout k, and JOIN the in-flight generation before every weight
  publish so weights never change mid-generation. One thread, one future, no
  buffer; staleness is structurally bounded at update_weights_interval.
  Requires recomputing old_logprobs at update start (three-policies rule);
  TIS absorbs the version + numerics gap.
- **Tier 2 — fully async worker pool (later, for agentic/variable-length
  episodes)**: the diagram below — persistent workers, bounded queue,
  abort-and-requeue (or partial-rollout resume) on weight updates; slime's
  fully_async_rollout / GLM-5 style.

```
┌──────────────┐  trajectories   ┌────────────┐   batches   ┌─────────────┐
│ Rollout      │ ───────────────▶│  Buffer    │────────────▶│  Learner    │
│ workers (×N) │                 │ (queue +   │             │ (train loop)│
│ own engine + │ ◀───────────────│  staleness │◀────────────│             │
│ env loop     │  weights v_k    │  filter)   │  publish    └─────────────┘
└──────────────┘                 └────────────┘  weights
```

- **GPU placement** (`placement.py`): the reference topology on an N-GPU node
  is *disaggregated* — e.g. with 4 GPUs, learner as a 3-rank DDP group on
  GPUs 0–2, one rollout engine on GPU 3 (ratios configurable). *Colocated* mode
  (learner and engine share every GPU, alternating phases — verl's hybrid
  engine) is the 1-GPU fallback and an ablation axis: measure the throughput
  crossover between the two, which is precisely the trade-off slime exposes as
  its two deployment modes.
- **Rollout workers**: separate processes (`torch.multiprocessing`), each owns
  a `RolloutEngine` on its assigned GPU + env instances, pulls prompts, pushes
  finished `Trajectory`s tagged with the weight `version` they were sampled
  under. Long agentic episodes never block the learner (GLM-5's key motivation).
- **Buffer**: bounded queue with a staleness policy — drop or down-weight
  trajectories older than `max_version_lag`.
- **Weight sync** (`weight_sync.py`): per tensor, in HF naming — the learner
  already holds every parameter in full (DDP replicates), NCCL-broadcasts it
  over a learner+engines process group (the same mechanism slime uses to push
  Megatron weights into SGLang), and `VLLMEngine` feeds it into vLLM's
  `load_weights`, which maps HF-named tensors into vLLM's internal fused
  layout. Broadcast-then-remap is a miniature of
  the resharding problem verl solves between FSDP and vLLM formats; because
  the learner is the HF module, no name translation is ever needed. Streaming
  per tensor (bucketed) avoids a full-model memory spike. Gotchas handled
  here: tied embeddings (don't sync `lm_head` separately when
  `tie_word_embeddings=True`). The `HFEngine` path is a plain
  `load_state_dict` — no remapping; CPU/MPS falls back to shared-memory
  tensors.
  Workers pick up new weights between episodes — mid-episode weight switching
  is off by default but available as a config flag, since it's a real
  design axis in GLM-5's async agentic RL.
- **Off-policy correction**: the same two-gap decomposition as tier-1 async —
  TIS weight `w_t = clamp(exp(logπ_old − logπ_engine), lo, hi)` on the pg
  term, plus PPO clipping of `r_t = exp(logπ_θ − logπ_old)` (the two-gap
  picture in `algos/tis.py`). Ablation configs let you show *why* this is
  needed (train with/without and watch it diverge).

The sync controller is a degenerate case of the async one (1 worker,
`max_version_lag = 0`) — but we keep both files because the sync one should be
readable in five minutes.

### Sequence packing (design only — build-or-not is an open decision)

Full implementation design — file-by-file changes, worked example, testing
strategy: **docs/packing.md**. A prototype was built and rolled back
(2026-07-09): correct (packed==padded pinned by tests before removal), but it
threaded an optional second layout through Batch/trainer/aggregate/gspo and
made the core files harder to read — readability wins here.

RL batches have brutal length variance, so padded (B, T) rectangles are
mostly pad — slime doesn't even have a padded path: every Megatron microbatch
is ONE concatenated `[1, T_padded]` row with `cu_seqlens` boundaries
(`PackedSeqParams`, THD layout), filled to a token budget
(`--use-dynamic-batch-size --max-tokens-per-gpu`) instead of a fixed sequence
count. Design points for our `pack_batch`, learned from their code:

- **Packing is attention BOUNDARIES, not an attention mask.** A 2D mask
  cannot express block-diagonality; naive packing with `attention_mask=1`
  lets sequences attend across the seam — silent contamination, no crash.
  Real mechanisms: varlen flash-attention fed `cu_seqlens` (Megatron/slime),
  or HF's equivalent — `position_ids` that reset to 0 per packed sequence
  under FlashAttention-2. We use the HF form.
- **CUDA-only**, like vLLM: no varlen kernels on MPS. The padded (B, T) path
  stays as the Mac/reference format; packing is a drop-in alternative to
  make_batch behind one config flag (`train.pack: true`).
- Elementwise losses (grpo/dapo/cispo/sft) work on a packed row unchanged.
  What needs segment-aware variants: sequence-mode aggregation and GSPO's
  per-sequence ratio (cu_seqlens segment sums instead of `.sum(-1)` per row).
- Seam positions (first token of each packed sequence) get garbage logprobs
  from the previous sequence's last token — always harmless because prompt
  tokens are loss-masked, but it gets its own test.
- The batch unit becomes a token budget (`max_tokens_per_microbatch`), which
  also evens out step times; the motivating metric is `frac_padding` logged
  by the padded path.

## 7. Agentic RL

Built entirely on the same `Trajectory` contract. Full study (slime's agent
stack) + minimal design live in docs/agentic_rl.md; deferred until SFT, the
RLVR recipe, and DPO are done. The essentials:

- **No env abstraction** — after studying slime (which has none either: agents
  are `--custom-generate-function-path` functions), the earlier gym-flavored
  `TextEnv` idea is dropped. An agent is a while loop around generate; the
  "agent runner" is that loop + a tool-call parser + an executor + trajectory
  bookkeeping (preserve sampled token ids and logprobs, loss-mask everything
  the model didn't author, step G episodes per prompt in a batch).
- First runner: `envs/python_tool.py`, ReTool-style math-with-interpreter on
  GSM8K/MATH — every ingredient already exists (math verifier as reward, the
  code sandbox as the tool, fit_async unchanged); prerequisites are only
  `SamplingParams.stop` and stdout capture in run_python.
- Runner ladder for later: text-protocol tools (ours) → formal JSON function
  calling → wrapping existing agents via protocol adapters (slime's
  AnthropicAdapter tier, for Claude-Code-style harnesses).
- Combined with the async controller, this reproduces the GLM-5 setting:
  long-horizon, variable-length episodes, learner never idle.

## 8. Configuration, logging, testing

**Config** (`config.py`): nested dataclasses (`ModelCfg`, `DataCfg`, `OptimCfg`,
`AlgoCfg`, `RolloutCfg`, `AsyncCfg`), loaded from YAML with dot-path CLI
overrides (`python recipes/03_grpo_gsm8k.py --cfg configs/grpo_gsm8k.yaml
optim.lr=2e-6 algo.kl_coef=0.0`). Every run dumps its resolved config next to
its checkpoints.

**Metrics** (log per step, these are the RL debugging dashboard):
reward mean/std, advantage stats, policy entropy, approx-KL to behavior and to
ref, clip fraction, ratio max, response length, `engine_learner_kl` (§6.0 —
the engine/trainer logprob gap), buffer staleness histogram (async),
tokens/sec for engine and learner separately.

**Tests** (`tests/`) — correctness over coverage:
- `gather_logprobs` against `F.cross_entropy` (and: logits upcast to fp32
  before `log_softmax` — bf16 logprobs are noisy enough to corrupt ratios).
- HFEngine sampling logprobs == learner recomputed logprobs (fp32, greedy).
- Checkpoint round-trip: `save_checkpoint` output loads via both
  `AutoModelForCausalLM.from_pretrained` and `vllm.LLM`.
- VLLMEngine vs learner logprob gap stays under a documented tolerance on a
  fixed prompt set (guards against silent regressions after vLLM upgrades);
  weight sync round-trip: learner → vLLM `load_weights` → generation changes.
- Loss-mask invariants on multi-turn trajectories (no learning on tool outputs).
- DPO loss against a hand-computed tiny example; GRPO advantage vs hand calc.
- Async smoke test: 2 workers + learner on a 2-layer random model, versions
  advance, no deadlock.

## 9. Rosetta Stone: miniRL ↔ slime ↔ verl

The point of this repo is that finishing it removes the learning curve for the
real frameworks. This table is the contract; keep it updated as code lands.

| Concept | miniRL | slime | verl |
|---|---|---|---|
| Overall dataflow owner | `rollout/controller.py` — a plain `fit()` loop that calls generate → reward → advantage → update | `train.py` driver over Ray actors | **single-controller**: `RayPPOTrainer.fit()` driving worker groups |
| Data protocol between stages | `rollout/types.py` (`Trajectory`, `Batch`) | `Sample` dataclass flowing through the buffer | `DataProto` (tensordict + meta) passed between workers |
| Rollout backend | duck-typed engines: `VLLMEngine` (primary, CUDA) or `HFEngine` (reference, any device) | SGLang server(s) behind a router | vLLM/SGLang rollout workers inside `ActorRolloutRefWorker` |
| Training backend | `minirl/megatron.py` (Megatron-Core, DP-only config) | Megatron-LM actor | FSDP / Megatron actor workers |
| Actor / rollout placement | `rollout/placement.py`: **colocated vs disaggregated** GPU assignment on one node | **colocated vs disaggregated** modes on GPU groups | **hybrid engine**: actor & rollout share GPUs, offload/reload between phases |
| Weight sync learner → sampler | `rollout/weight_sync.py`: versioned NCCL broadcast (shm fallback on CPU) | bucketed NCCL broadcast (disaggregated) or CUDA IPC (colocated) | `sync_model_weights` / resharding between FSDP and vLLM formats |
| Data buffer | `rollout/buffer.py` (bounded queue + staleness filter) | Data Buffer module (also the custom-data/partial-rollout hook point) | replay/experience handling inside the trainer loop |
| Async / off-policy RL | `controllers/fully_async.py`, `version` tag + TIS correction | asynchronous training mode; partial rollouts | one-step-off async mode; agent-loop workers |
| Reward plumbing | `rewards/` fns taking `Trajectory` → float | custom reward via `--custom-rm-path` | `RewardManager` (rule-based fns or RM worker) |
| Advantage estimators | `algos/advantage.py` (GAE, group-norm, RLOO) | selected per algorithm in training scripts | `core_algos.py` advantage estimator registry (`grpo`, `gae`, `rloo`, ...) |
| Algorithm zoo | `algos/*.py` one file per loss | PPO/GRPO variants via CLI flags | `ppo`, `grpo`, `dapo`, `rloo`, ... trainer configs |
| Agentic / multi-turn | `envs/` + `multiturn.py` interaction loop | custom rollout function (`--rollout-function-path`) | agent loop / multi-turn rollout with tool calling |
| Config system | dataclasses + YAML + dot-path CLI overrides | argparse mega-flags (Megatron style) | Hydra/OmegaConf YAML trees |
| Parallelism ceiling | DDP learner + N rollout engines, single node | 3D parallelism (TP/PP/DP) multi-node via Megatron | 3D parallelism via Megatron or FSDP, Ray multi-node |

Reading this table bottom-up also tells you exactly what miniRL *chose not to
build* (Ray orchestration, paged attention, resharding between training and
inference layouts) — each of those gets a short explainer in `docs/production_gap.md`
so the jump to slime/verl is documented, not just implied.

**Post-miniRL reading path** (also goes in `docs/production_gap.md`):
1. verl: `RayPPOTrainer.fit()` — you already know this loop; now read how
   `DataProto` moves between resource pools.
2. verl: `ActorRolloutRefWorker` — the hybrid engine; miniRL's two-weight-copies
   trick is the degenerate single-GPU version of this.
3. slime: the buffer module and weight-sync path (NCCL broadcast buckets) —
   miniRL's `weight_sync.py` is a shared-memory stand-in for exactly this code.
4. slime: custom rollout functions — miniRL's `envs/multiturn.py` plays this role.

## 10. Roadmap

(The original phase plan, kept as the target sequence. ACTUAL current state
and the agreed build order live in §0 Status — trust §0 where they disagree;
e.g. async landed before SFT, sync-controller and PPO rows are superseded.)

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Skeleton: config, logging, `models/hf.py`, `HFEngine` | Can chat via `engine.generate` on a Mac; sampling logprobs match learner recomputation; HF-format checkpoint round-trips |
| 0.5 | `VLLMEngine` behind the same protocol | Same prompts through both backends; logprob-gap distributions measured and documented |
| 1 | SFT | Loss curve sane; eval win on a held-out instruct set |
| 2 | DPO | Implicit-reward margins increase; beats SFT on preference eval |
| 3 | Sync GRPO on GSM8K (RLVR), vLLM rollouts | GSM8K accuracy climbs meaningfully from SFT baseline |
| 4 | RLOO + ablation configs | Reproduce known qualitative results (e.g. clip-higher → longer stable runs); logprob-mismatch/TIS ablation (§6.0) |
| 5 | Async infra | Disaggregated async GRPO on N GPUs matches sync final reward with higher throughput; staleness ablation; colocated-vs-disaggregated throughput comparison |
| 5.5 | Sequence packing (CUDA) | SFT first, then RL: packed loss == padded loss on the same data (equivalence test); tokens/sec gain ≈ measured `frac_padding`; seam-position test |
| 6 | Agentic RL | Tool-use task trained end-to-end, multi-turn masking verified |
| 7 | On-policy distillation | Student recovers most of teacher's GSM8K gain at 0.6B |
| 8 | Docs pass | Each algo has a `docs/` note: math, shapes, pitfalls, paper links; `docs/production_gap.md` maps every simplification to its slime/verl counterpart |

## 11. Explicit non-goals

- No writing model architectures — `AutoModelForCausalLM` is the model; the
  curriculum is post-training, not transformer internals.
- No optimizer choices — AdamW (fp32 states), full stop. Every serious
  post-training recipe uses it; swapping optimizers is not this repo's
  curriculum.
- No PPO / critic training — value-function learning + GAE is a second
  training loop for an algorithm modern RLVR has moved past; the GRPO family
  + RLOO cover the curriculum. GAE-style per-token advantages remain
  architecturally possible (tier-2 batch.advantages overwrite) but unplanned.
- No multi-language code judging — Python datasets only (which is where open
  RLVR research lives at this scale). C++/competitive-programming judging
  (compile + stdin/stdout diff) and repo-test-suite rewards are documented
  paths, not code; multi-language at scale = a sandbox service (SandboxFusion)
  behind the remote_rm pattern.
- No *building* serving-grade inference (paged attention, speculative decoding,
  CUDA graphs) — that's what the vLLM dependency is for; our `HFEngine` stays
  a thin `model.generate` wrapper. If you later want to *learn* inference
  internals, writing a KV-cache sampler from scratch is a natural side quest,
  but it is not on this repo's critical path.
- No 3D parallelism (TP/PP) / MoE / multi-node — a DDP learner group plus
  dedicated rollout GPUs on one node is the ceiling.
- No training models past ~4B; the lab animal is a 0.6–1B base model.
- No prompt-engineering products; rewards are verifiable or learned, not vibes.
