"""VLLMEngine — continuous-batching rollout backend (tier 2, docs/async_tier2.md).

Wraps vLLM's low-level LLMEngine (add_request / step) instead of the blocking
llm.generate(), because owning the step loop is what streaming collection
needs: finished GROUPS come back incrementally (poll) while vLLM's scheduler
keeps every slot busy (continuous batching is vLLM's core design — we
implement none of it, we only keep it fed).

One interface: the streaming contract — submit() / poll() / stash() /
drain() / n_inflight / load_weights / pad_id — consumed by
controllers/fully_async.py (collect_groups_dp drives poll() directly).
(A blocking tier-1 generate() existed until 2026-07-16; it retired with the
round-based controller — generate()-style engines now enter via
engine/stream_adapter.py instead.)

IN-FLIGHT UPDATES ARE DEFERRED (decision 2026-07-10, docs/async_tier2.md §4):
load_weights REQUIRES a quiescent engine (drain-then-publish, asserted), so
every completion is generated under exactly ONE weight version — the tier-1
invariant survives and Trajectory.version stays a scalar.

Platform notes (spike findings, docs/async_tier2.md §8):
  - vllm-metal (Mac): weight updates need BOTH the MLX tree load AND direct
    per-layer `self_attn._inner` assignments — `load_weights` alone silently
    skips attention (the paged wrapper hides it from the parameter tree).
    Callable RPC needs VLLM_ALLOW_INSECURE_SERIALIZATION=1.
  - CUDA vLLM: the torch branch below follows the standard RLHF pattern
    (model.load_weights(name->tensor iterator)); UNVALIDATED until the box.
  - EOS parity with HFEngine (response INCLUDES its eos token) must be
    verified on the first real run — vLLM configs differ on stop-token
    inclusion. gsm8k smoke vs HFEngine is the check.

vLLM imports live inside methods so this module imports cleanly in
environments without vLLM (the repo test env); instantiating needs the vLLM
env (Mac: ~/.venv-vllm-metal).
"""

import os
import tempfile

import torch
from torch import Tensor

from minirl.rollout.types import SamplingParams, Trajectory


class VLLMEngine:
    """Continuous-batching rollout engine (contract: SamplingParams docstring)."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        seed: int = 0,  # DP fleets: give each engine a different seed (uncorrelated sampling)
        gpu_id: int | None = None,  # pin to ONE GPU (DP placement, docs/async_tier2.md §11)
    ):
        from transformers import AutoTokenizer
        from vllm import EngineArgs, LLMEngine

        self.model_name_or_path = model_name_or_path
        # GPU pinning by env (§10(a)): the V1 EngineCore SUBPROCESS spawned
        # inside from_engine_args inherits CUDA_VISIBLE_DEVICES; the parent's
        # value is restored right after. Two caveats until the on-box spike:
        # init the learner/torch.cuda BEFORE any engine (CUDA reads the env
        # once at context creation), and construct engines sequentially (the
        # mutation is process-global).
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            self.engine = LLMEngine.from_engine_args(
                EngineArgs(model=model_name_or_path, dtype=dtype, max_model_len=max_model_len, seed=seed)
            )
        finally:
            if gpu_id is not None:
                if old_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd
        tok = AutoTokenizer.from_pretrained(model_name_or_path)
        eos = tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else int(eos)
        self.version = 0

        self._pending: dict[str, dict] = {}  # request_id -> {prompt_ids, meta, version}
        self._stash: list[list[Trajectory]] = []  # finished groups awaiting a poll (drain surplus)
        self._next_id = 0

    # ---------------- tier-2 streaming interface ----------------

    @property
    def n_inflight(self) -> int:
        """Groups submitted but not yet delivered (stash NOT counted — those
        are already finished work waiting to be picked up)."""
        return len(self._pending)

    def submit(self, prompt_ids: Tensor, params: SamplingParams, meta: dict | None = None) -> str:
        """Queue ONE prompt for a whole group: one vLLM request with n=G.

        The request finishes atomically when its slowest member does — no
        loss, since GRPO cannot reward/filter a group before all G siblings
        exist (docs/async_tier2.md §2). vLLM starts prefilling it on the next
        step, in whatever slots are free (continuous batching).
        """
        from vllm import SamplingParams as VSP
        from vllm.inputs import TokensPrompt

        rid = f"req-{self._next_id}"
        self._next_id += 1
        self.engine.add_request(
            rid,
            TokensPrompt(prompt_token_ids=[int(t) for t in prompt_ids]),
            VSP(
                n=params.n,
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_new_tokens,
                logprobs=0,  # attach the SAMPLED token's logprob, computed at sampling time
            ),
        )
        self._pending[rid] = {
            "prompt_ids": prompt_ids.cpu(),
            "meta": dict(meta or {}),
            "version": self.version,  # the ONLY version this group will ever see (drain-then-publish)
        }
        return rid

    def poll(self) -> list[list[Trajectory]]:
        """Advance the engine one step; return every group that finished.

        Stashed groups (drain surplus from a previous publish) are delivered
        first, without consuming a step. Returns [] when a step produced no
        finishers; callers loop on n_inflight, not on poll's emptiness.
        """
        out, self._stash = self._stash, []
        if self._pending:
            for req_out in self.engine.step():
                if req_out.finished:
                    out.append(self._to_group(req_out))
        return out

    def stash(self, group: list[Trajectory]) -> None:
        """Hand a finished-but-unconsumed group back; next poll returns it first.

        Used by the collector for surplus survivors and by drain — work is
        never thrown away, it is just consumed by the NEXT collection.
        """
        self._stash.append(group)

    def drain(self) -> None:
        """Run the engine until nothing is in flight; finishers go to the stash.

        The pre-publish quiescence step: leftovers complete under the OLD
        weights (their submit-time version — single-version completions hold)
        and are consumed by the next collection, at most one publish stale.
        """
        while self._pending:
            for group in self.poll():
                self.stash(group)

    # ---------------- weight updates (drain-then-publish ONLY) ----------------

    def load_weights(self, named_tensors, version: int) -> None:
        """Publish learner weights into the RUNNING engine. Engine must be idle.

        Path-based: the state dict is written to a safetensors file and the
        WORKER loads it — no tensor serialization over RPC. The worker-side
        function handles both platforms; the Metal branch implements the
        spike's full recipe (§8): MLX tree load + per-layer `_inner`
        attention assignments (load_weights alone would silently skip
        attention — the franken-policy trap).
        """
        assert not self._pending, "load_weights during in-flight generation (drain first)"
        from safetensors.torch import save_file

        tensors = {k: v.detach().cpu().contiguous() for k, v in named_tensors}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "learner.safetensors")
            save_file(tensors, path)
            self.engine.collective_rpc(_apply_weights_from_file, args=(path,))
        self.version = version

    def _to_group(self, req_out) -> list[Trajectory]:
        """One finished RequestOutput -> G Trajectories (CPU, by contract)."""
        req = self._pending.pop(req_out.request_id)
        prompt: Tensor = req["prompt_ids"]  # (T_prompt,)
        n_p = prompt.numel()
        group = []
        for comp in req_out.outputs:
            ids = torch.tensor(list(comp.token_ids), dtype=torch.long)  # (n_gen,)
            n_g = ids.numel()
            if comp.logprobs:  # list[dict[token_id -> Logprob]], sampled token always present
                lps = torch.tensor(
                    [comp.logprobs[t][int(ids[t])].logprob for t in range(n_g)], dtype=torch.float32
                )
            else:
                lps = torch.zeros(n_g)
            group.append(
                Trajectory(
                    input_ids=torch.cat([prompt, ids]),  # one row's (T,): prompt_len + response_len
                    loss_mask=torch.cat(
                        [torch.zeros(n_p, dtype=torch.bool), torch.ones(n_g, dtype=torch.bool)]
                    ),
                    logprobs=torch.cat([torch.zeros(n_p), lps]),
                    version=req["version"],
                    meta=dict(req["meta"]),
                )
            )
        return group


def _apply_weights_from_file(worker, path: str) -> str:
    """Worker-side weight load (runs INSIDE the engine worker via collective_rpc).

    Defined at module level for readability but shipped by VALUE through
    vLLM's callable-RPC (cloudpickle), so the worker env does not need minirl
    importable. Metal branch = the validated spike recipe (docs/async_tier2.md
    §8); torch branch = standard vLLM RLHF pattern, unvalidated until the box.
    """
    model = worker.model_runner.model
    try:
        import mlx.core as mx  # vllm-metal worker → the Metal recipe
    except ImportError:
        from safetensors.torch import load_file  # CUDA vLLM worker

        model.load_weights(list(load_file(path).items()))
        return "torch"

    weights = mx.load(path)
    model.load_weights(list(weights.items()), strict=False)  # tree params: embed/norms/MLP
    # Attention lives behind SDPAPagedAttentionWrapper._inner — invisible to
    # the parameter tree, read LIVE by the Metal kernel each forward. Assign
    # directly, per layer. (Spike-validated; _inner is a private plugin
    # attribute — the canary test must fail loud if the layout changes.)
    layers = getattr(getattr(model, "model", model), "layers", [])
    n_assigned = 0
    for i, layer in enumerate(layers):
        inner = getattr(getattr(layer, "self_attn", None), "_inner", None)
        if inner is None:
            continue
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            key = f"model.layers.{i}.self_attn.{name}.weight"
            mod = getattr(inner, name, None)
            if mod is not None and key in weights:
                mod.weight = weights[key].astype(mod.weight.dtype)
                n_assigned += 1
    assert n_assigned > 0 or not layers, (
        "no attention weights assigned — plugin layout changed? (docs/async_tier2.md §8 canary)"
    )
    return f"metal:{n_assigned}"
