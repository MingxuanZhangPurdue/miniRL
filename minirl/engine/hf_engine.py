"""HFEngine — reference rollout backend wrapping transformers' generate().

Slow but runs on CUDA / Apple MPS / CPU: the local-dev, CI, and debugging
path. Because it shares the learner's HF module class, its logprob gap vs the
learner is ~0 by construction — the trusted oracle when debugging VLLMEngine.
# In production: this box is a vLLM/SGLang server (slime's rollout backend,
# verl's ActorRolloutRefWorker); the interface below is what stays the same.

MPS notes: prefer fp32/fp16 (bf16 support is spotty); if an op is missing,
run with PYTORCH_ENABLE_MPS_FALLBACK=1.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from minirl.rollout.types import SamplingParams, Trajectory


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFEngine:
    """Reference rollout engine (contract: see SamplingParams in rollout/types.py)."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_batch_size: int = 8,  # generation micro-batch; bounds memory
    ):
        self.device = device or _pick_device()
        # fp32 off-CUDA: exact logprobs matter more than speed on the dev path.
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.max_batch_size = max_batch_size
        self.version = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        # eos may be a single id or a list (e.g. Qwen: <|im_end|> + <|endoftext|>).
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos_ids = torch.tensor([eos] if isinstance(eos, int) else list(eos))
        self.pad_id = self.tokenizer.pad_token_id or int(self.eos_ids[0])

    @torch.no_grad()
    def generate(self, prompt_ids: list[Tensor], params: SamplingParams) -> list[Trajectory]:
        # Expand to B*n prompt copies, grouped by prompt, then micro-batch.
        expanded = [p for p in prompt_ids for _ in range(params.n)]
        trajectories: list[Trajectory] = []
        for start in range(0, len(expanded), self.max_batch_size):
            trajectories += self._generate_batch(expanded[start : start + self.max_batch_size], params)
        return trajectories

    def _generate_batch(self, prompts: list[Tensor], params: SamplingParams) -> list[Trajectory]:
        b = len(prompts)
        lens = [int(p.numel()) for p in prompts]
        t_max = max(lens)

        # LEFT-pad: decoder-only generation must have prompts flush against the
        # generated tokens; right-padding would put pads inside the context.
        input_ids = torch.full((b, t_max), self.pad_id, dtype=torch.long)  # (B, T_max)
        attention_mask = torch.zeros((b, t_max), dtype=torch.long)  # (B, T_max)
        for i, (p, n) in enumerate(zip(prompts, lens)):
            input_ids[i, t_max - n :] = p
            attention_mask[i, t_max - n :] = 1

        greedy = params.temperature == 0.0
        out = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            do_sample=not greedy,
            **({} if greedy else {"temperature": params.temperature, "top_p": params.top_p}),
            max_new_tokens=params.max_new_tokens,
            pad_token_id=self.pad_id,
            return_dict_in_generate=True,
            output_logits=True,  # RAW logits — output_scores is post temperature/top-p
            # and would inflate the engine<->learner gap from ~1e-5 to ~0.8 nats/token.
        )
        gen = out.sequences[:, t_max:].cpu()  # (B, T_gen) — pad-filled past each row's eos

        # Behavior logprobs, one step at a time to avoid a (B, T_gen, V) buffer
        # (V is ~250k for Qwen3.5). out.logits: tuple of T_gen tensors, (B, V).
        step_lps = [
            F.log_softmax(step.float(), dim=-1).gather(-1, gen[:, t : t + 1].to(self.device)).cpu()
            for t, step in enumerate(out.logits)
        ]  # T_gen x (B, 1)
        gen_logprobs = torch.cat(step_lps, dim=1)  # (B, T_gen)

        trajectories = []
        for i in range(b):
            # Keep tokens up to AND including the first eos: the policy produced
            # eos, so it is trained on (loss_mask True). After it: pad garbage.
            hits = torch.isin(gen[i], self.eos_ids).nonzero()
            n_gen = int(hits[0]) + 1 if len(hits) else gen.shape[1]
            trajectories.append(
                Trajectory(
                    input_ids=torch.cat([prompts[i].cpu(), gen[i, :n_gen]]),  # (T_i + n_gen,)
                    loss_mask=torch.cat(
                        [torch.zeros(lens[i], dtype=torch.bool), torch.ones(n_gen, dtype=torch.bool)]
                    ),
                    logprobs=torch.cat([torch.zeros(lens[i]), gen_logprobs[i, :n_gen]]),
                    version=self.version,
                )
            )
        return trajectories

    def load_weights(self, named_tensors, version: int) -> None:
        state_dict = dict(named_tensors)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # Tied embeddings: a learner sd may legitimately omit the tied lm_head.
        real_missing = [k for k in missing if "lm_head" not in k]
        assert not real_missing and not unexpected, (
            f"weight sync mismatch: missing {real_missing}, unexpected {unexpected}"
        )
        self.version = version
