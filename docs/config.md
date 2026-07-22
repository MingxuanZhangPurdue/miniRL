# Config: per-component, slime-named

Status: **built 2026-07-20.** miniRL splits configuration by COMPONENT, one
frozen dataclass each, and reuses slime's argument NAMES so a slime recipe
translates field-for-field. Grounded in slime's `utils/arguments.py` groups
(Rollout&Sampling, Data&Dataset, Reward) and the qwen3-4B example.

## The components

    MegatronTrainConfig   minirl/megatron.py   optimizer, precision, packing, the DDP knobs
    GRPOConfig (+algos)   minirl/algos/*       the loss (eps_clip, kl, advantage estimator)
    PlacementConfig       minirl/config.py     single-node GPU split (train vs rollout)
    RolloutConfig         minirl/config.py     generation + collection  (THIS doc)
    DataConfig            minirl/config.py     where prompts come from   (THIS doc)
    reward                a plain callable     Trajectory -> float       (THIS doc)

Loss and train configs already existed and stay put; this note covers the
rollout side, which previously lived as two split objects (`SamplingParams`
+ `CollectConfig`) plus ad-hoc `HFPromptSource` args.

## RolloutConfig — merges CollectConfig + the sampling knobs

The unlock: `CollectConfig.group_size` and `SamplingParams.n` were always the
SAME number (one request IS one group) — slime unifies them as
`n_samples_per_prompt`. Merging kills that duplication and gives one object
for the whole generate-and-collect step.

    field                    slime flag                     was
    rollout_batch_size       --rollout-batch-size           CollectConfig.target_groups
    n_samples_per_prompt     --n-samples-per-prompt          group_size AND sampling.n (unified)
    rollout_temperature      --rollout-temperature           sampling.temperature
    rollout_top_p            --rollout-top-p                 sampling.top_p
    rollout_top_k            --rollout-top-k                 (new; -1 = off)
    rollout_max_response_len --rollout-max-response-len      sampling.max_new_tokens
    dynamic_sampling         --dynamic-sampling-*            CollectConfig.strategy == "filter"
    over_sampling_rounds     (budget; cf. over-sampling)      CollectConfig.max_rounds

`SamplingParams` STAYS as the low-level engine wire-type — it mirrors
`vllm.SamplingParams` field names (`temperature`, `n`, `max_tokens`) on
purpose, and the engines consume it in submit/poll. `RolloutConfig` is the
experiment-knob layer; `RolloutConfig.sampling_params()` derives the wire
object. So slime names live in the config, vLLM names stay at the wire, and
one helper bridges them.

## DataConfig — the dataset, slime's Data&Dataset group

    field                slime flag              default
    prompt_data          --prompt-data           (required; the dataset id/path, recorded)
    input_key            --input-key             "input"
    label_key            --label-key             None
    apply_chat_template  --apply-chat-template   True
    rollout_max_prompt_len  --rollout-max-prompt-len   None (rows over it are
                            DROPPED at load, never truncated — a mangled
                            problem statement makes its reward gradient noise)
    enable_thinking      --apply-chat-template-kwargs '{"enable_thinking": ...}'   False
    rollout_shuffle      --rollout-shuffle       True
    rollout_seed         --rollout-seed          0

enable_thinking lives HERE (not per-call-site) so training and eval prompts
can never disagree on the template regime — the recipes thread
data_cfg.enable_thinking into make_eval_prompts too.

`input_key`/`label_key` drive a GENERIC row adapter: row -> a single user
message from `row[input_key]`, and `row[label_key]` into a FIXED meta key
`"label"`. So every reward reads `traj.meta["label"]`, decoupled from the
dataset's column name. Datasets needing real transformation still pass a
custom `row_fn` (the escape hatch) — e.g. `gsm8k_row`, which `.strip()`s the
question. This adopts slime's column-mapping names, superseding prompts.py's
earlier row_fn-only stance (a tiny function per dataset); the row_fn path
survives as the fallback, so nothing is lost.

The recipe still LOADS the dataset (it knows the split/config
idiosyncrasies — `load_dataset("openai/gsm8k", "main", split="train")`);
`DataConfig` records the identifier and supplies the row-adaptation knobs.

## Reward — a plain callable, not a config field

RLVR only: no reward model, ever. The reward is a `reward_fn(traj) -> float`
passed straight to `fit_async` (already its contract). This is the
principle-8-native form of slime's escape hatch: slime is CLI-driven, so its
reward is a STRING — either `--rm-type deepscaler` (a built-in verifier
selected from a closed `if/elif`) or `--custom-rm-path pkg.mod:fn` (a string
import path, `load_function`). miniRL recipes are Python, so they pass the
callable itself: no `rm_type` enum, no `load_function`, maximum flexibility.

Every slime `rm_type` (`deepscaler`, `dapo`, `math`, ...) is just a name for
a math answer-checker — `minirl/rewards/math.py::make_math_reward_fn` already
IS one (extract boxed answer, grade by exact/normalized/`math-verify`
equivalence). So `reward_fn = make_math_reward_fn(tokenizer)` covers the
common case; `code_reward` or any custom callable covers the rest.

The one thing dropped vs slime: the reward choice is not in the config dump.
Mitigation — the recipe logs `reward_fn.__qualname__` into the run config, so
every run still records its reward by name; the recipe itself (git-tracked)
is the experiment definition.

## EvalConfig — periodic benchmark eval (minirl/eval.py)

    field                     slime flag                    default
    eval_interval             --eval-interval               None (never)
    eval_before_train         (inverse of --skip-eval-before-train)  True
    n_samples_per_eval_prompt --n-samples-per-eval-prompt   1
    eval_temperature          --eval-temperature            0.0 (greedy)
    eval_top_p / eval_top_k   --eval-top-p / --eval-top-k   1.0 / -1
    eval_max_response_len     --eval-max-response-len       1024

Same split as everywhere: the NUMERIC knobs are config; WHICH benchmarks
(datasets + rewards) are code — a list of `EvalSet(name, prompts,
reward_fn)` handed to fit_async, the analog of slime's `--eval-config`
YAML `datasets:` section. Eval generates through the SAME rollout engines
in the post-publish quiescent window (all engines idle at exactly the
published version — hence eval_interval must be a multiple of
publish_interval), plus an untrained baseline before iteration 1. Metrics:
eval/{name}/reward_mean, response_len_mean, truncated_ratio, t_eval.

## Name-alignment left for a follow-up (train/loss configs)

Cheap slime renames NOT done here, to keep this change scoped to rollout:
`MegatronTrainConfig.minibatch_size` -> `global_batch_size`, `max_grad_norm`
-> `clip_grad`, `adam_betas` -> `adam_beta1`/`adam_beta2`. Eval args
(`--eval-prompt-data`, ...) wait for the eval harness.
