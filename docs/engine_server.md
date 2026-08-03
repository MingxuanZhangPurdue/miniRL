# Engine servers — miniRL's process topology as a scale model of slime

Drafted 2026-08-03 after the first multi-GPU pod session. Status: DESIGN,
not built. Self-contained: read with `../slime` open, nothing else needed.

## 0. The principle this design serves

miniRL is a **miniSlime**: same load-bearing ideas, radically less code,
still a complete working system. The test for every component here is:
(a) a reader can point at its slime counterpart and say "same idea", and
(b) whatever slime has that we dropped, we dropped because it serves
SCALE (70B models, multi-node fleets), never because it serves the idea.
This note applies that test to the one place where miniRL's shape still
diverges from slime's: process topology.

## 1. Slime's topology (what we are modeling)

slime/train.py is a ~100-line driver that owns the loop:

    pgs             = create_placement_groups(args)          # GPU allocation (Ray)
    rollout_manager = create_rollout_manager(args, pgs)      # sglang SERVERS inside
    actor_model     = create_training_models(args, pgs, ...) # Megatron ranks (Ray actors)
    for rollout_id in range(...):
        data = rollout_manager.generate(rollout_id)          # separate processes generate
        actor_model.train(rollout_id, data)                  # separate processes train
        actor_model.update_weights()                         # trainer -> engines sync

The structural facts that matter:

  1. **Engines are server processes**, never objects inside a training
     rank. The trainer and the engine never share an interpreter, an env,
     or a CUDA context.
  2. **The driver talks to both sides through handles** (Ray remotes /
     HTTP). Control flow is readable at the top; processes are the
     boundary of responsibility.
  3. **Weight sync is an explicit, versioned protocol** with pluggable
     transports (disk / distributed broadcast / tensor IPC).
  4. Ray exists to do placement, spawning, and health for a FLEET. That
     is its entire justification.

## 2. Today's miniRL divergence, and what it cost

miniRL currently constructs `VLLMEngine` objects INSIDE torchrun rank 0.
The controller loop (`fit_async`) is slime-shaped, but the topology is
not — trainer and engines share one process, one env, one fate. The
2026-08-03 pod session priced that divergence in blood:

  - vLLM's engine child inherited torchrun's rendezvous env and hung
    forever (fixed by scrubbing 14 variables around construction — an
    airlock hidden in a constructor);
  - one venv had to hold vLLM's torch AND Megatron/TE simultaneously —
    the entire broken-TE-wheel saga exists only because of this;
  - an engine fault kills the training job;
  - vLLM's spawned child re-executes the recipe's `__main__` (Python
    spawn semantics) — a standing landmine.

Every one of these is the colocation tax. Slime does not pay it; neither
should the scale model, because the topology IS one of the core ideas.

## 3. The design: slime's topology at 1/100th the machinery

Component-by-component mapping. Right column states what was dropped and
why the drop loses scale only, never the idea.

    slime                        miniRL                       what the minimization drops
    -----                        ------                       ---------------------------
    train.py driver process      fit_async on rank 0          Ray owns slime's processes, so its
                                                              driver must live outside them; without
                                                              Ray, rank 0 doubles as driver. Loop
                                                              shape identical: collect -> train ->
                                                              publish.
    Ray placement groups         gpu ids in PlacementConfig;  placement for ONE node is a list of
                                 CUDA_VISIBLE_DEVICES in the  integers.
                                 server's Popen env
    RolloutManager +             k engine_server.py           sglang router load-balances a fleet;
    sglang servers + router      processes, one per rollout   our pull-based dealer in the collector
                                 GPU; EngineProxy handles     threads already balances k local
                                 in the controller            engines. vLLM LLMEngine instead of
                                                              sglang: we own the step loop.
    RayTrainGroup /              torchrun ranks, unchanged    Ray actors exist so slime can spawn
    train_actor.py                                            ranks across nodes; torchrun does the
                                                              same for one node in one line.
    update_weights protocol      drain -> safetensors file -> disk variant only. slime's NCCL
    (disk/distributed/tensor)    server reloads via vLLM's    broadcast path moves 70B in seconds;
                                 native reload_weights        a file moves 0.6B in seconds. Same
                                                              protocol: quiesce, transfer, restamp
                                                              version.
    Buffer actor                 prompt_source closure +      an actor because slime's producers and
                                 in-memory lists              consumers live in different processes
                                                              on different nodes; ours share the
                                                              controller.
    onload/offload               (nothing)                    slime time-shares GPUs between train
    choreography                                              and rollout; we space-share (2+2).
                                                              Simpler, and it teaches the same
                                                              boundary.

New code, in full:

    minirl/engine_server.py  (~90 loc)  main(): argv -> VLLMEngine, then a
                                        zmq REP loop: one request = one
                                        contract method call (pickled)
    minirl/engine_proxy.py   (~70 loc)  EngineProxy: the same contract as
                                        methods -> REQ round trips; Popen
                                        launcher + readiness ping; context
                                        manager for teardown

The controller does not change. `train_async.py` and `eval.py` already
consume engines through a seven-member duck-typed contract — they cannot
tell a proxy from an object, which is the whole point of having had the
contract.

## 4. The contract on the wire

    member          request                       reply
    ------          -------                       -----
    submit          (prompt_ids, params, meta)    gid: str
    poll            ()                            list[list[Trajectory]]
    stash           (group,)                      ok
    drain           ()                            ok
    n_inflight      ()                            int
    pad_id, version cached at connect             int
    load_weights    (weights_dir, version)        ok, after reload
    shutdown        ()                            ok; server exits

Semantics identical to in-process: poll() advances exactly one engine
step server-side. The RPC hop (~100µs, ipc:// unix socket, pickle) is
noise against a 10-50ms engine step. Localhost-only trust by
construction (0700 socket dir); tcp:// is a door for a second node that
does not exist.

load_weights splits along the process boundary without changing the
protocol: today the engine object writes the safetensors file and points
vLLM's native `reload_weights` worker RPC at it. Tomorrow the PROXY
writes the file (the trainer owns the tensors), sends the path; the
server asserts idleness and reloads. Drain-then-publish quiescence,
storage dedup for tied weights, version restamp: all unchanged.

Threading: one owner thread per engine at any moment (collector, then
eval worker, then the publishing main thread — phases never overlap). A
`threading.Lock` in the proxy makes that assumption explicit, since zmq
REQ sockets tolerate no concurrent use.

## 5. Lifecycle

  - **Launch**: the recipe Popens one server per rollout GPU FIRST —
    engines warm up while Megatron loads (colocation forced the opposite
    order; that ordering law dies). argv: model, socket path, seed,
    max_model_len. env: an explicit whitelist dict — CUDA_VISIBLE_DEVICES
    for placement, HF cache, VLLM_* knobs, and nothing inherited. The 14-
    variable torchrun airlock becomes one visible `env={...}`.
  - **Readiness**: proxy pings until ack with a deadline (first-run
    engine compile takes minutes); recipes fail loud past it.
  - **Health**: every RPC carries a generous timeout; expiry raises and
    the run dies loudly — same blast radius as an in-process crash today,
    now with a named culprit. Respawn/elasticity: door, not plan
    (fail-loud serves a study repo).
  - **Teardown**: proxies are context managers — `shutdown`, then
    terminate+wait as backstop.
  - **Per-role envs — the pod-session prize**: the server's interpreter
    is argv[0] of the Popen. Trainer venv: megatron/bridge/TE, no vllm.
    Engine venv: vllm, no megatron. Each is trivial to build alone; every
    hard problem in the 2026-08-03 runbook was their intersection.

## 6. What the migration deletes

  - The torchrun-identity scrub and CUDA_VISIBLE_DEVICES save/restore in
    `VLLMEngine.__init__` (superseded by the Popen whitelist).
  - The trainer-before-engines construction law.
  - The one-venv-for-two-stacks requirement (and most of the runbook §4
    pin gymnastics with it).
  - The spawn re-import landmine (the server's `__main__` is 90 lines).

One-path rule: recipes and controllers speak proxies ONLY; constructing
`VLLMEngine` in-process is for `engine_server.py` and nothing else.
Recipe 04 validates the real transport (one server, one GPU).

## 7. Testing (CPU, Mac suite, no vllm import)

Controller-side tests keep in-process `FakeStreamEngine` — the dealer,
filter, and staleness logic are transport-blind. One new file,
`tests/test_engine_rpc.py` (~60 loc): the server dispatch loop wrapped
around a FakeStreamEngine in a thread, a real EngineProxy in the test —
round-trip fidelity for every contract member (tensor-intact
Trajectories, stash ordering, drain quiescence, version restamp via a
load_weights path call).

## 8. Migration ladder (next pod session)

    1. rpc test green locally                          [Mac, CPU]
    2. recipe 04 through one server                    [pod, 1 GPU]
    3. recipe 09 smoke: 1 trainer rank + 1 server      [pod, 2 GPU]
    4. recipe 09 2+2 — the deferred 10-iteration run   [pod, 4 GPU]
    5. delete the airlock code                         [after 4]

## 9. Doors deliberately left shut

  - NCCL/CUDA-IPC weight sync — revisit when publish time is visible in
    the iteration metrics, not before.
  - tcp:// multi-node engines, auth — no second node exists.
  - Router/cache-aware balancing — the dealer already balances k local
    engines; a router solves fleet problems.
  - Background-stepping server (poll() returning without stepping) —
    changes semantics; today's determinism is worth more than the
    latency it would hide.
