# RunPod setup runbook — unified Megatron+vLLM env (validated 2026-08-02)

The first environment where the trainer and the engines coexist in ONE
process/venv — required by recipes 05/09 (rank 0 constructs `VLLMEngine`
and `MegatronTrainer` in the same interpreter). Everything below was
executed and verified on a 4x A100-SXM4-80GB pod; the exact failure modes
and their fixes are recorded because they WILL recur.

## 1. Pod choice

- 4x A100 80GB (SXM), Secure Cloud. Filter hosts by **CUDA >= 13.0**
  (torch 2.11+cu130 needs driver >= 580; ours: 580.126.16).
- Any recent RunPod Ubuntu 24.04 pytorch template works. Ours shipped
  git/tmux/uv preinstalled and nvcc 12.8 — **nvcc is not actually needed**:
  TE now ships prebuilt `transformer-engine-cu13` wheels, nothing compiles.
- Container disk >= 50GB. `/workspace` is a shared network FS — keep the
  venv and HF cache on local disk (`/root`) for speed.
- Set nothing else in the template; WANDB key travels as a file (§3).

## 2. Connect

Add the Mac's `~/.ssh/id_ed25519.pub` under RunPod Settings -> SSH Public
Keys BEFORE creating the pod. Two SSH paths exist:

- **Proxy** (`ssh <pod-id>@ssh.runpod.io`): interactive-PTY only. Plain
  `ssh host 'cmd'` fails with "Your SSH client doesn't support PTY"; no
  scp/rsync. Scriptable only by piping stdin into a forced PTY:
  `printf 'cmd; exit\n' | ssh -tt <pod-id>@ssh.runpod.io`.
- **Direct TCP** (use this): the pod's real sshd. IP/port are in the web
  UI ("SSH over exposed TCP") or inside the pod as `$RUNPOD_PUBLIC_IP` /
  `$RUNPOD_TCP_PORT_22`. Full exec channel + scp/rsync:

      ssh  -i ~/.ssh/id_ed25519 -p <port> root@<ip> 'nvidia-smi'
      scp  -i ~/.ssh/id_ed25519 -P <port> file root@<ip>:/root/

## 3. Secrets

`WANDB_API_KEY=...` lives in `.env` at the repo root (gitignored — the
`# Environments` block covers it). Ship it and source it at launch:

    scp -i ~/.ssh/id_ed25519 -P <port> .env root@<ip>:/root/miniRL/.env
    # in the run command:  set -a; . /root/miniRL/.env; set +a

## 4. The env recipe (the validated sequence, in this exact order)

    cd /root && git clone https://github.com/MingxuanZhangPurdue/miniRL.git
    cd miniRL && uv venv --python 3.12 && source .venv/bin/activate

    # 1. vLLM FIRST — its pinned torch becomes THE torch.
    uv pip install vllm==0.25.1                 # -> torch 2.11.0+cu130

    # 2. Megatron stack on top (P1-validated pins).
    uv pip install megatron-core==0.18.0
    uv pip install --no-deps megatron-bridge==0.5.0
    uv pip install omegaconf "hydra-core<=1.3.2" accelerate peft diffusers \
        qwen-vl-utils timm mistral-common wandb "transformers==5.8.1" \
        datasets math-verify pytest nvidia-modelopt

    # 3. TE — prebuilt cu13 wheels, no compile, no nvcc needed.
    uv pip install "transformer-engine[pytorch]==2.16.1"

    # 4. LAST, always: TE's wheel needs a newer cuBLAS 13.x than torch
    #    pins, and modelopt DOWNGRADES it back if installed after.
    uv pip install -U nvidia-cublas             # 13.1 -> 13.6 here

Gotchas, each hit for real on 2026-08-02:

- `megatron-bridge` plain-installed backtracks mcore to 0.13 and drags in
  ~100 packages — `--no-deps` is load-bearing (slime does the same).
- The bridge ALSO needs `nvidia-modelopt` (AutoBridge imports it). The NGC
  container preinstalled it, so megatron.md §6's hand-dep list misses it.
- TE import order: `import transformer_engine` BEFORE torch fails
  (`libcublas.so.13: cannot open`) — torch's import is what preloads the
  pip NVIDIA libs. Every minirl entry point imports torch first; only
  bare-TE smoke tests trip this.
- TE 2.16.1 vs torch's cublas pin: `undefined symbol:
  cublasLtGroupedMatrixLayoutInit_internal` means cuBLAS is too old —
  that's step 4. `nvidia-cublas-cu13` is a DEPRECATED tombstone package;
  the real package is `nvidia-cublas`.
- vLLM wants transformers 5.14+, bridge pins 5.8.1 — 5.8.1 wins, vLLM
  0.25.1 accepts it (uv resolves without conflict).
- Apex is absent (NGC had it): mcore falls back to Torch norms/optimizer
  helpers with a UserWarning. Accepted — TE provides the fused paths that
  matter.

## 5. Verify (all passed 2026-08-02)

    python - <<'EOF'
    import torch
    print(torch.cuda.device_count())            # 4
    x = torch.randn(1024,1024, device=0, dtype=torch.bfloat16)
    print((x@x).float().norm().item())          # cublas 13.6 kernel runs
    import transformer_engine.pytorch           # after torch!
    import megatron.core, vllm
    from megatron.bridge import AutoBridge
    print("FULL STACK OK")
    EOF
    python -m pytest tests/ -q

pytest: 91/92 green. `test_dp_burst_cap_prevents_hoarding` is FLAKY on the
128-core pod (1 pass / 2 fail over three runs) — thread-timing assumption,
pre-existing, not an env issue; tracked separately.

Pre-warm HF artifacts (all public, no token):

    python -c "from huggingface_hub import snapshot_download; from datasets import load_dataset; \
    snapshot_download('Qwen/Qwen3-0.6B'); \
    load_dataset('allenai/Dolci-RL-Zero-Code-7B', split='train'); \
    load_dataset('google-research-datasets/mbpp', 'full', split='test')"

Quirk seen on this pod: GPU 0 reported ~16GB used with zero processes
(stale host accounting). Harmless on 80GB cards; re-check with
`nvidia-smi --query-compute-apps=pid --format=csv` if memory gets tight.

## 6. Not yet run (next session picks up here)

The P3 ladder, in order: recipe 04 (engine smoke) -> 08 `--bf16` then
`--packed` (Megatron+TE parity in THIS env) -> 09 one-step Dolci smoke
(1+1, `--num-rollout 1 --rollout-batch-size 8 --eval-interval 0`) ->
09 full 2+2 with `--wandb`. Long runs go in tmux.

## 7. Reflection — make setup a one-liner from the repo

This session rediscovered the whole stack by hand from docs + trial and
error. The pins and the ORDER above are law, but they live in prose; the
repo should carry them executably. Two options, smallest first:

1. **`scripts/setup_pod.sh`** — exactly §4 as an idempotent script,
   checked in. Zero abstraction, one `curl | bash` away after clone.
2. **`pyproject.toml` with a `pod` extra + `[tool.uv]`
   `override-dependencies`** to neuter megatron-bridge's dep tree
   declaratively (uv honors overrides where pip's `--no-deps` is
   all-or-nothing). Setup becomes `uv venv && uv pip install -e ".[pod]"
   && uv pip install -U nvidia-cublas`. More moving parts, but the pins
   become machine-checked instead of doc prose.

Recommendation: do (1) now — it earns its existence the day a pod dies
mid-experiment; consider (2) only if the stack stabilizes across several
pods. Either way the §4 gotcha list stays here as the WHY.
