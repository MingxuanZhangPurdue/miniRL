# Agentic RL: study + minimal design (deferred until SFT / RLVR / DPO are done)

Research notes from slime's agent stack (docs/en/get_started/agent.md,
slime/agent/*, examples/search-r1 + retool + coding_agent_rl, 2026-07) and the
design for miniRL's minimal version. Doc-before-code, per repo convention.

## 1. The demystification: an agent is a loop, not a component

There is no "agent" object anywhere in slime — and there will be none here.
At the mechanical level an agent is a while loop around a generate call, and
the AGENT RUNNER is that loop plus a parser, an executor, and trajectory
bookkeeping:

```python
context = system_prompt_describing_tools + user_question      # tokens
for turn in range(max_turns):
    segment = engine.generate(context, stop=[CLOSE_FENCE])    # model reasons, maybe calls
    context += segment                                        # loss_mask=True  (policy tokens)
    call = parse_tool_call(segment)                           # e.g. a ```python fence
    if call is None:                                          # no call -> final answer
        break
    obs = execute(call)                                       # sandbox / search / API
    context += tokenize(f"```output\n{obs}\n```")             # loss_mask=False (env tokens)
```

"Reasoning" is just generated tokens (trained on, no machinery); "tool calls"
are parse+execute; "the answer" is the turn with no call. ReAct, ReTool,
ToRL, Search-R1 are all this loop with different parse rules and tools.

What an RL training runner must do that normal agent frameworks don't —
and why we write ~120 lines instead of importing one:
  - preserve SAMPLED TOKEN IDS + per-token logprobs for every model segment
    (frameworks speak strings; retokenizing responses corrupts the training
    target — slime's adapters exist specifically to avoid this),
  - maintain the loss_mask boundary between policy and environment tokens,
  - step G episodes per prompt as a batch each turn (early finishers drop out),
  - stitch everything into ONE flat Trajectory per episode.
Our Trajectory contract (types.py) was designed for exactly this.

## 2. How slime does it (findings)

- **No env abstraction.** The whole mechanism is `--custom-generate-function-path`:
  one function runs the agent workflow per sample and returns Sample objects
  with tokens / loss_mask / response_length / reward filled (or reward deferred
  to --custom-rm-path). Their doc's golden rule, verbatim: the workflow "may
  speak in strings, chat messages, tool calls" but "the training target should
  stay token based."
- **Fan-out**: one rollout may return list[Sample] (subagent segments,
  context-compaction pieces) sharing a rollout_id — kept together for loss
  aggregation. Noted for later; not minimal.
- **Protocol adapters** (slime.agent.adapters.AnthropicAdapter / OpenAIAdapter):
  for training EXISTING agents (Claude Code, OpenAI SDK) without rewriting
  them — the adapter impersonates the model API the agent client talks to,
  serves each call from SGLang with input_ids + return_logprob=True, and
  exports the token segments as trainable trajectory pieces. The agent IS the
  runner; the adapter is a wiretap on its model calls.
- **Serving concerns** that only bite at scale: session-affinity routing
  (same episode -> same worker for prefix-cache hits), PD disaggregation for
  the long-tail multi-turn latency. Out of scope; listed for production_gap.
- Examples ladder: retool (python-interpreter math, homegrown subprocess
  sandbox), search-r1 (local retrieval HTTP server + exact-match reward),
  coding_agent_rl (per-sample sandbox, agent edits code, clean sandbox runs
  tests -> reward).

## 3. The runner ladder (place any agentic setup on it)

1. **Text protocol** (minimal, ours): tool calls are text conventions
   (```python fences, <search> tags); parse by string match + stop strings.
2. **Formal function calling**: tools as JSON schemas via the chat template's
   `tools=`, model emits structured tool_call blocks, a real parser decodes.
   Same loop, sturdier syntax.
3. **Wrap an existing agent** via a protocol adapter (slime tier). Heavy;
   only worth it to train Claude-Code-style harnesses.

## 4. Datasets (HF-first)

| Family | HF datasets | Env needed | Weight |
|---|---|---|---|
| Math + Python interpreter (ReTool/ToRL) | openai/gsm8k, DigitalLearningGmbH/MATH-lighteval, BytedTsinghua-SIA/DAPO-Math-17k | our code sandbox (built) | feather |
| Search / multi-hop QA (Search-R1) | hotpotqa/hotpot_qa, PeterJinGo/nq_hotpotqa_train, corpus RUC-NLPIR/FlashRAG_datasets | retrieval server over a wiki index (GBs) | medium |
| SWE agents | princeton-nlp/SWE-bench, SWE-Gym/SWE-Gym, R2E-Gym/R2E-Gym | Docker per instance + repo test suites | heavy |
| API/user-sim (tau-bench) | GitHub, not HF-native | LLM-simulated user | heavy |

## 5. Minimal miniRL version: math-with-interpreter (envs/python_tool.py)

Chosen because every ingredient except the runner already exists: dataset
(GSM8K/MATH), reward (rewards/math.py — the boxed final answer is still the
reward; tools don't change it), tool executor (rewards/code.py's sandbox),
controller (fit_async unchanged — the episode runner plugs into
collect_groups as generate_fn, returning G trajectories per prompt; reward_fn
stays make_math_reward_fn since labels ride in traj.meta).

The loop of §1 with: system prompt announcing the ```python tool; stop string
at the closing fence; sandbox executes the block with stdout captured;
`\n```output\n{stdout}\n```\n` appended as masked tokens; episode ends on
no-code-block / EOS / max_turns / token budget; segments stitched into one
flat Trajectory (policy tokens: mask True + engine logprobs; tool tokens:
mask False + logprob 0; version = engine version at episode start).

Prerequisites (small, engine-portable):
  1. SamplingParams.stop: list[str] | None — HF generate has stop_strings=,
     vLLM has stop=; contract stays duck-typed.
  2. rewards/code.py run_python: capture and return stdout (tool obs needs it).

Tests when built: loss_mask never True on tool-output tokens (the §4-of-types
invariant, now multi-turn); stitched logprobs match per-segment engine
reports; an episode with a deliberate tool error still terminates; batched
episodes with early finishers produce correct per-episode trajectories.

## 6. Sequencing

Deferred by decision (2026-07): build order is SFT -> RLVR recipe -> DPO,
then this. Nothing in the trainer/controller/losses needs to change for it —
the runner is purely additive.

## 7. Qwen3 template findings (probed empirically, 2026-07-13)

The default model's SHIPPED template already covers the agentic surface — no
custom template work needed (schema stays OpenAI/HF messages+tools; the
template renders it):

- **Tools = Hermes format, built in**: schemas injected into the system block
  in `<tools>` tags (a system turn is SYNTHESIZED if absent); calls expected
  as `<tool_call>{json}</tool_call>`; results rendered in `<tool_response>`
  tags — folded into a **user** turn, not a distinct tool role, so
  assistant-only masking already treats tool output as context.
- **Reasoning**: `enable_thinking` template variable; False pre-injects an
  empty `<think></think>` into the generation prompt (the skip mechanism).

**The trap (drives a runner design decision):** re-rendering a conversation
STRIPS `<think>` blocks from all PREVIOUS assistant turns — thinking survives
only in the current one. Generated-turn tokens != re-rendered-history tokens,
so multi-turn trajectories fork:

1. re-render per turn (Qwen's inference convention, saves context) — the
   trajectory is NOT one contiguous token sequence; logprobs/masks need
   per-turn segments; retokenization hazard class;
2. accumulate raw generated token ids (what engines + the Trajectory
   contract already do) — contiguous (T,) trajectory, but old think blocks
   stay in context, spending budget and diverging from inference convention.

The runner must pick one EXPLICITLY (leaning 2 for training correctness —
it keeps the per-token logprob/mask story identical to single-turn — with 1
as the deploy-time convention); decide when envs/ gets built.
