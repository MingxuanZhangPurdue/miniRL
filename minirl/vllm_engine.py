"""VLLMEngine — continuous-batching rollout backend (tier 2).

Wraps vLLM's low-level LLMEngine (add_request / step) instead of the blocking
llm.generate(), because owning the step loop is what streaming collection
needs: finished GROUPS come back incrementally (poll) while vLLM's scheduler
keeps every slot busy (continuous batching is vLLM's core design — we
implement none of it, we only keep it fed).

One interface: the streaming contract — submit() / poll() / stash() /
drain() / n_inflight / load_weights / pad_id — consumed by
train_async.py (collect_groups_dp drives poll() directly).
(A blocking tier-1 generate() existed until 2026-07-16; it retired with the
round-based controller. The generate()-engine on-ramp — HFEngine +
StreamAdapter — retired 2026-07-20 with the vLLM-only decision.)

IN-FLIGHT UPDATES ARE DEFERRED (decision 2026-07-10):
load_weights REQUIRES a quiescent engine (drain-then-publish, asserted), so
every completion is generated under exactly ONE weight version — the tier-1
invariant survives and Trajectory.version stays a scalar.

Platform notes:
  - Weight publish uses vLLM's NATIVE `reload_weights` worker RPC (no
    custom worker code; see load_weights).
  - EOS parity (response INCLUDES its eos token — the loss-mask convention
    the trainer assumes, rollout/types.py) must be verified on the first
    real run — vLLM configs differ on stop-token inclusion.
    recipes/04_smoke_vllm_cuda.py is the check.
  - WSL2 (Docker Desktop GPU passthrough): no UVA, so vLLM's V2 model
    runner cannot init — __init__ detects WSL and defaults the V1 runner
    (VLLM_USE_V2_MODEL_RUNNER=0). Real Linux keeps V2.

Imports are file-level: this module only imports where vLLM is installed
(the box's /opt/vllm-env) — exactly like minirl/megatron.py only imports
where megatron does.
"""

import os
import platform
import tempfile

import torch
from safetensors.torch import save_file
from torch import Tensor
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine
from vllm import SamplingParams as VSP
from vllm.inputs import TokensPrompt

from minirl.rollout.types import SamplingParams, Trajectory


class VLLMEngine:
    """Continuous-batching rollout engine (contract: SamplingParams docstring)."""

    def __init__(
        self,
        model_name_or_path: str,
        gpu_id: int | None = None,  # OURS: pin to ONE GPU (DP placement)
        **engine_kwargs,  # everything else forwards to vLLM's EngineArgs
        #   untouched (seed per engine for uncorrelated sampling,
        #   gpu_memory_utilization when the GPU is shared, ...) — every vLLM
        #   knob is reachable without this class naming it. Two defaults
        #   applied if absent: dtype="bfloat16", max_model_len=4096.
    ):

        # vLLM launches EngineCore as a subprocess via FORK by default, and a
        # forked child cannot re-initialize CUDA. Our ordering rule (§11)
        # GUARANTEES the parent has a live CUDA context before engines exist
        # (learner first), so fork always dies — force spawn (found on the
        # A100 box, 2026-07-20). Safe: recipes guard __main__, and the
        # CUDA_VISIBLE_DEVICES pinning below inherits through spawn the same.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # WSL2's GPU passthrough has no UVA (unified addressing), which
        # vLLM's V2 model runner requires at init — fall back to the V1
        # runner there. Real Linux boxes keep the default.
        if "microsoft" in platform.uname().release.lower():
            os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
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
        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs.setdefault("max_model_len", 4096)
        try:
            self.engine = LLMEngine.from_engine_args(
                EngineArgs(model=model_name_or_path, **engine_kwargs)
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

    @property
    def n_inflight(self) -> int:
        """Groups submitted but not yet delivered (stash NOT counted — those
        are already finished work waiting to be picked up)."""
        return len(self._pending)

    def submit(self, prompt_ids: Tensor, params: SamplingParams, meta: dict | None = None) -> str:
        """Queue ONE prompt for a whole group: G CHILD requests with n=1 each.

        The fan-out is OURS, not vLLM's: V1 implements n>1 in its high-level
        entrypoints (LLM/AsyncLLM parallel sampling), NOT in LLMEngine —
        add_request with n=G returns a single completion (found on the A100
        box, 2026-07-20). Owning the fan-out also removes any dependence on
        version-fragile n semantics. The group still finishes atomically when
        its slowest child does — no loss, since GRPO cannot reward/filter a
        group before all G siblings exist. Children
        prefill on the next step, in whatever slots are free.
        """
        gid = f"req-{self._next_id}"
        self._next_id += 1
        group = {
            "prompt_ids": prompt_ids.cpu(),
            "meta": dict(meta or {}),
            "version": self.version,  # the ONLY version this group will ever see (drain-then-publish)
            "waiting": set(),  # child ids still generating
            "done": [],  # finished children's Trajectories
        }
        self._pending[gid] = group
        token_ids = [int(t) for t in prompt_ids]
        for j in range(params.n):
            cid = f"{gid}/{j}"
            group["waiting"].add(cid)
            self.engine.add_request(
                cid,
                TokensPrompt(prompt_token_ids=token_ids),
                VSP(
                    n=1,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    max_tokens=params.max_new_tokens,
                    logprobs=0,  # attach the SAMPLED token's logprob, computed at sampling time
                    # Token ids are the single source of truth: trajectories
                    # carry ids only, and rewards decode the response slice
                    # themselves — nothing reads RequestOutput.text, so skip
                    # the engine's incremental detokenization. (Safe because
                    # we stop on token ids, never on stop STRINGS, which
                    # would need the running text.)
                    detokenize=False,
                ),
            )
        return gid

    def poll(self) -> list[list[Trajectory]]:
        """Advance the engine one step; return every group that finished.

        Stashed groups (drain surplus from a previous publish) are delivered
        first, without consuming a step. Returns [] when a step produced no
        finishers; callers loop on n_inflight, not on poll's emptiness.
        """
        out, self._stash = self._stash, []
        if self._pending:
            for req_out in self.engine.step():
                if not req_out.finished:
                    continue
                gid = req_out.request_id.rsplit("/", 1)[0]
                group = self._pending[gid]
                group["waiting"].discard(req_out.request_id)
                group["done"].append(self._to_traj(req_out, group))
                if not group["waiting"]:  # slowest child just finished
                    self._pending.pop(gid)
                    out.append(group["done"])
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

        Path-based and 100% vLLM-native: the state dict is written to a
        safetensors file and the WORKER loads it — no tensor serialization
        over RPC. The worker's `reload_weights` RPC (v1/worker/gpu_worker.py)
        `weights_path=` mode points its own model loader at our directory:
        it globs the *.safetensors, streams them through model.load_weights,
        and warns if any expected weight was missing.
        """
        assert not self._pending, "load_weights during in-flight generation (drain first)"
        # Ship each STORAGE once: tied weights (Qwen: lm_head <- embed_tokens)
        # are aliases, and safetensors refuses aliased tensors. Loaders
        # re-tie on their side: vLLM skips lm_head for tied configs.
        tensors: dict[str, Tensor] = {}
        seen: set[int] = set()
        for k, v in named_tensors:
            ptr = v.untyped_storage().data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            tensors[k] = v.detach().cpu().contiguous()
        with tempfile.TemporaryDirectory() as td:
            save_file(tensors, os.path.join(td, "learner.safetensors"))
            self.engine.collective_rpc("reload_weights", kwargs={"weights_path": td})
        self.version = version

    def _to_traj(self, req_out, group: dict) -> Trajectory:
        """One finished CHILD RequestOutput (n=1) -> one Trajectory (CPU)."""
        prompt: Tensor = group["prompt_ids"]  # (T_prompt,)
        n_p = prompt.numel()
        (comp,) = req_out.outputs  # n=1 child: exactly one completion
        ids = torch.tensor(list(comp.token_ids), dtype=torch.long)  # (n_gen,)
        n_g = ids.numel()
        if comp.logprobs:  # list[dict[token_id -> Logprob]], sampled token always present
            lps = torch.tensor(
                [comp.logprobs[t][int(ids[t])].logprob for t in range(n_g)], dtype=torch.float32
            )
        else:
            lps = torch.zeros(n_g)
        return Trajectory(
            input_ids=torch.cat([prompt, ids]),  # one row's (T,): prompt_len + response_len
            loss_mask=torch.cat(
                [torch.zeros(n_p, dtype=torch.bool), torch.ones(n_g, dtype=torch.bool)]
            ),
            logprobs=torch.cat([torch.zeros(n_p), lps]),
            version=group["version"],
            meta=dict(group["meta"]),
        )
