# TMax (allenai) — terminal-agent RL training, repo study

Repo: `workspace/tmax` (paper arXiv:2606.23321). Trains "tmax" terminal-using
agents with a fork of open-instruct (`training/open-instruct/`). All citations
below are `path:line` relative to the tmax repo root.

Questions answered (same spirit as [agentic_rl.md](agentic_rl.md)):
1. Which Qwen checkpoints + original or modified chat template?
2. Verifier: exactly how rollout rewards are computed.
3. High-level pipeline + sandboxes for training vs. evaluation.

---

## 1. Base models & chat template

### Checkpoints

RL scripts (`training/open-instruct/scripts/tmax/RL/*.sh`, all run `grpo_fast.py`):

| Run | `--model_name_or_path` | cite |
|---|---|---|
| 2B / 4B / 9B | `hamishivi/Qwen3.5-{2B,4B,9B}` (re-uploads, NOT `Qwen/...`) | `RL/qwen35_4b.sh:55` |
| 27B | `Qwen/Qwen3.6-27B` (stock) | `RL/qwen36_27b.sh:49` |
| Qwen3-8B | `hamishivi/sft_qwen3_8b_our_sft_cleaned_func` (an SFT ckpt) | `RL/qwen3_8b.sh:55` |

Key point: **the Qwen3.5/3.6 RL runs start directly from instruct models — no
SFT stage in the released recipe.** Only the older Qwen3-8B run is
RL-on-top-of-SFT. SFT scripts exist separately (`scripts/tmax/SFT/`, datasets
`allenai/tmax-sft{,-big}`) on `Qwen/Qwen3-8B` and `hamishivi/Qwen3.5-9B`, but
the tmax RL scripts are not chained onto them.

### Chat template: stock *code path*, modified *checkpoint*

- No tmax script passes `--chat_template_name`; with `chat_template_name=None`
  the tokenizer falls back to **the model repo's own Jinja template verbatim**
  (`open_instruct/dataset_transformation.py:846-854`: `tokenizer.chat_template =
  AutoTokenizer.from_pretrained(...).chat_template`). None of the fork's
  `CHAT_TEMPLATES` dict entries (`dataset_transformation.py:159-648`) are used.
- BUT the `hamishivi/Qwen3.5-*` checkpoints themselves carry a **custom
  "interleaved reasoning" chat template** — SFT script comment: *"We use a
  version of Qwen 3.5 with an interleaved reasoning chat template"*
  (`scripts/tmax/SFT/sft_qwen35_9b_small.sh:3`). Stock Qwen strips `<think>`
  from history; this variant keeps per-turn reasoning across the multi-turn
  trajectory. Confirmed in eval trajectories: every assistant turn ends with a
  bare `</think>` (opening tag prefilled by template) across ~45 turns
  (`evaluation_assets/daytona/tmax-9b/.../trajectory.json`).
- The matching rollout-side parser `vllm_qwen3_xml` prefills
  `<|im_start|>assistant\n<think>\n` on **every** turn and feeds tool results
  back as a `user` turn wrapped in `<tool_response>`
  (`open_instruct/environments/tools/parsers.py:260-266`); the Qwen3-8B run
  uses `vllm_hermes` (plain assistant header, `tool` role) instead.
- Template inspection tooling: `scripts/tokenizers/render_chat_template_examples.py`
  (samples carry `reasoning_content`).

### Qwen3.5 fixes in the fork (numerical, not tokenizer-vocab)

- GatedDeltaNet **packing patch**: linear-attention layers otherwise leak conv
  + recurrent state across packed sequence boundaries → wrong logprobs,
  inflated KL (`open_instruct/qwen3_5_packing_patch.py:1-8`; applied in Ray
  workers, `grpo_fast.py:250` — main-process-only application left
  vllm-vs-local logprob diff at ~0.21 vs ~0.02, `CHANGELOG.md:7`).
- **fp32 lm_head in vLLM** to kill bf16 logprob rounding
  (`open_instruct/vllm_utils.py:145-167`, `--lm_head_fp32 true` in all RL scripts).
- Qwen3.5 template crashes on system/tool-only prefixes → deferred label
  masking in SFT (`dataset_transformation.py:1158-1207`); guard stripping stray
  pad tokens from qwen input_ids (`dataset_transformation.py:1501-1505`).

---

## 2. Verifier: reward for rollouts

**One line: sparse terminal binary 0/1 = "did the task's pytest suite pass",
executed inside the task's own container at submit time. No format reward, no
length penalty, no partial credit.**

Chain (training-time, env `--tools swerl_vanillux_sandbox`):

1. Agent has a single `bash` tool; when a command's output contains the submit
   marker `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, the episode ends and tests
   run (`open_instruct/environments/swerl_vanillux_sandbox.py:463-471`).
2. Tests are uploaded to `/tests` **only at submit time** — deferred from reset
   so the agent can't read the verifier ("prevent peeking",
   `swerl_vanillux_sandbox.py:474`).
3. Env runs `bash /tests/test.sh` (timeout floored to 600s,
   `swerl_vanillux_sandbox.py:135,501`). Generated `test.sh` runs
   `pytest test_final_state.py` and writes `1` or `0` to
   `/logs/verifier/reward.txt`; reward comes ONLY from that file, exit code is
   ignored (`scripts/data/convert_tmax_tasks.py:50-66`,
   `rl_data/scripts/analyze/convert_to_harbor.py:252-268`). Whole-suite pass =
   1, any failing test = 0.
4. `_parse_reward` cats `reward.txt`, clamps to [0,1], returns 0.0 if missing
   (`swerl_vanillux_sandbox.py:516-528`). Zero-reward terminal cases: OOM kill,
   never submitting within `--max_steps 64`, env-reset failure, tool timeout —
   all indistinguishable from a wrong answer.
5. Dataset rows use `dataset: "passthrough"` → `PassthroughVerifier` returns
   0.0; the trajectory score is the env reward via `LastRewardAggregator`
   (last turn's reward) (`open_instruct/ground_truth_utils.py:1029-1061,
   1385-1402`). `--verification_reward 1.0` only sets `max_possible_score` for
   the solved-rate metric.
6. Shaping knobs all exist but are **off** in tmax runs: r1-format reward,
   non-stop penalty, concave (Box–Cox) length penalty, truncation masking
   (`open_instruct/data_loader.py:585-670`).

### Worked examples (real checked-in trials)

The verifier never reads the trajectory text — it inspects the **final
container state** left behind by the agent's commands. Trajectory = multi-turn
tool loop: `system` + task instruction (`user`), then repeating
`assistant(<think>…</think> + one bash tool call)` / `tool(output +
exit_code)`, ending with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

**Pass (reward 1)** — `evaluation_assets/daytona/tmax-9b/terminal-bench-2-0/fix-git__XLSnB9t/`:
instruction: *"I just made some changes to my personal site and checked out
master, but now I can't find those changes… merge them into master."* The
tmax-9b agent runs ~36 bash calls (`git status` → `git reflog` finds the
dangling commit `650dba4 "Move to Stanford"` → `git merge` → resolves the
`about.md` conflict by hand with a heredoc → verifies → submits). Then the
verifier runs pytest on the task's tests, which just assert file contents of
the merged working tree:
`PASSED test_about_file / PASSED test_layout_file → 2 passed` →
`verifier/reward.txt` = `1`.

**Genuine fail (reward 0)** — `.../bn-fit-modify__eRYde5i/`: pytest ran 9
tests; 8 passed, 1 failed
(`test_intervened__data_structure`: set of learned DAG edges != expected set
`{("U","M"),("U","D"),…}`) → `1 failed, 8 passed` → reward `0`.
**All-or-nothing: 8/9 correct still scores 0** — no partial credit.

**Infra fail (also reward 0)** — `.../adaptive-rejection-sampler__NUp6a7r/`:
`test.sh`'s own setup broke (apt GPG error → `curl: command not found` →
`uvx: command not found`), pytest never executed → reward `0`. This is the
"verifier-infra failure ≡ task failure" confound in the flesh.

**What generated tmax tests assert** (corpus on HF; spec in
`rl_data/generator/completion_test_gen.py:34-58`): a pytest file
`test_final_state.py` that checks the post-task OS state at **absolute paths**
using the privileged `truth` from `task.json` — instructed to *derive*
expected values rather than copy constants, use invariants over brittle
full-file equality, stdlib+pytest only (legacy) or a per-verifier-kind
allow-list (`metric_threshold` → numpy/torch/sklearn…, `multi_protocol` →
requests, `completion_test_gen.py:75-100`). Tests see truth; the agent never
does (tests upload only at submit).

Training-time mapping: same contract per rollout; the group is 32 such
rollouts on the *same task*, score ∈ {0,1} each, advantage = score − group
mean.

### Score → advantage → loss

- **Advantage**: group-based, `--advantage_normalization_type centered` =
  reward minus group mean, **no ÷std** (Dr.GRPO/RLOO-style), group = 32
  rollouts/prompt × 8 prompts (`data_loader.py:876-906`; script
  `RL/qwen35_4b.sh:52-53,89`). Binary reward ⇒ advantage is `1-p̂` / `-p̂` with
  p̂ = group pass rate. One scalar per trajectory, broadcast to all response
  tokens. Zero-std groups (all-pass or all-fail) are dropped; `--active_sampling`
  backfills (`data_loader.py:1181`).
- **DPPO** (`--loss_fn dppo`, arXiv:2602.04879) is a *policy-loss* change, not
  a reward change: unclipped `-A·ratio`, with a hard **trust-region mask** that
  zeroes tokens already outside a divergence ball δ AND moving further away
  (A>0 & ratio>1, or A<0 & ratio<1); moves back toward the rollout policy are
  never masked (`open_instruct/grpo_utils.py:846-853, 653-694`). tmax:
  TV divergence, δ=0.1, ratio anchored to vLLM rollout logprobs
  (`--use_vllm_logprobs true`; `RL/qwen35_4b.sh:90-92`).

---

## 3. Pipeline + sandboxes

### End-to-end flow

```
rl_data: generate_tasks ──► generate_solutions ──► analyze ──► upload_to_hf
 (gemini-3.1-pro-preview     (agent pass@k,          (pass@k,     (HF dataset;
  samples orthogonal axes;    gemini-3-flash,         balance)     passing SFT
  builds Apptainer image      Apptainer)                           trajectories
  = executability filter)                                          → tmax-sft)
                                    │
        prebuilt Docker images (content-hashed, build_tmax_images.py)
                                    ▼
 RL: grpo_fast.py + swerl_vanillux_sandbox over allenai/tmax-15k-open-instruct
                                    ▼
 Eval: Harbor (`harbor run`) + Vanillux2Agent on TB-2.0 / TB-Lite / SWE-bench,
       model served by vLLM
```

- Tasks are a **single independent draw from orthogonal axes** (domain, skill,
  persona, fixture kind, task/command complexity, verifier kind) — no
  teacher-validation stage; only an *executability* check (image builds, tests
  run) (`rl_data/README.md:11-49`, axes in `rl_data/generator/task_template_gen.py`).
- **Difficulty**: design-time via complexity axes + graded verifiers; post-hoc
  tiering from strong-model rollouts (Frontier <40% … Core 60–80%,
  `rl_data/scripts/analyze/classify_difficulty.py:5-13`); train-time **soft
  filtering** — zero-variance groups give no gradient and are dropped
  (README's stated substitute for validation, `rl_data/README.md:33-39`).
- Generated `truth`/`test_final_state.py` are LLM text never executed against a
  reference solution — acknowledged drift risk (`rl_data/generate_tasks.py:3-11`).
- Agent (train = eval): "vanillux" harness — mini-SWE-agent-derived prompts,
  **bash as the only tool**, submit marker, format-error recovery, head/tail
  output truncation. Eval version `Vanillux2Agent/agent.py`; RL env
  `swerl_vanillux_sandbox.py` reimplements the same harness so train/eval match.

### Sandboxes — training

- Env code supports two backends: **Docker-API** and **Apptainer**
  (`open_instruct/environments/backends.py:162,570`). On the actual cluster the
  Docker backend talks to **rootless Podman** via sharded unix sockets
  (`scripts/docker/docker_login.sh`, `SWERL_PODMAN_DOCKER_HOSTS`;
  `environments/pool.py:23-25` round-robins `--pool_size 512` sandbox actors
  across shards). Apptainer is the daemonless HPC alternative.
- Task envs are **prebuilt, content-hashed Docker images**: Apptainer
  `container.def` → Dockerfile → DockerHub, tag written back into the HF
  dataset's `env_config.image` (`scripts/data/build_tmax_images.py:198-252,
  624-641`); each task must name its image (`image.txt` / `env_config.image`,
  no default — `swerl_vanillux_sandbox.py:264-289`). `requests` is baked into
  images because `multi_protocol` verifiers import it (commit `621fcfe4`,
  `build_tmax_images.py:197-203`). Scale plumbing: registry pull-through
  mirror + in-process image prewarm actor (`data_loader.py:128-166`).

### Sandboxes — evaluation

- Eval runs through **Harbor** (external framework, pip dep). Two env backends:
  - `--env docker` on Beaker = the same rootless-Podman trick, plus a pile of
    documented Harbor/compose podman-compat patches
    (`scripts/beaker/run_eval_in_job.sh:131-214`, why-notes in
    `scripts/beaker/README.md:158-187`).
  - `--env daytona` = **Daytona cloud sandboxes** — this is what the shipped
    results used: every checked-in config under `evaluation_assets/daytona/`
    has `"environment": {"type": "daytona"}`, agent `Vanillux2Agent`, model
    served by local vLLM, `n_attempts: 5`.
- Datasets: `terminal-bench@2.0` (default, `beaker_configs/launch_eval.sh:44`),
  `openthoughts-tblite@2.0`, SWE-bench 100-task subset
  (`scripts/swebench100_tasks.txt`); training corpus decontaminated against
  TB-2.0/TB-Lite via 13-gram overlap
  (`rl_data/scripts/decontamination/run_decontamination.sh:81-86`).
- Eval-side verifier contract is identical to training: `tests/test.sh` →
  `reward.txt` (per-trial `verifier/reward.txt` visible in
  `evaluation_assets/daytona/...`).

---

## Takeaways for miniRL

- **Reward minimalism works here**: pure terminal 0/1 from a hermetic pytest
  verifier + centered group advantage; all shaping (format/length) left off.
  Difficulty control moved *upstream* into data generation instead of into the
  reward.
- **Sandbox trick worth remembering**: defer uploading the verifier into the
  container until the submit marker fires — reward-hacking-by-reading-tests is
  structurally impossible.
- **Chat-template lesson**: nothing in the training code changes the template;
  the modification ships *inside the checkpoint* (interleaved per-turn
  reasoning kept in history), and the rollout parser must prefill/parse to
  match (`vllm_qwen3_xml`).
- **Infra failure = reward 0** (missing reward.txt, OOM, reset failure) — a
  known confound the repo accepts; verifier-infra errors are indistinguishable
  from task failure.
