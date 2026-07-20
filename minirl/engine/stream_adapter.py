"""StreamAdapter — gives a generate()-only engine the streaming interface.

The two-controller consolidation left ONE training
loop, and it speaks the streaming contract (submit/poll/stash/drain/
n_inflight/load_weights/pad_id). HFEngine speaks the old two-method contract
(generate/load_weights). This adapter bridges them WITHOUT touching HFEngine:

    one poll() == one ROUND: everything submitted since the last poll
    generates as a single blocking engine.generate() call, and every group
    finishes at once.

Under collect_groups_dp's top_up -> poll loop, that degenerates continuous
batching back into exactly the retired round-based collection: top_up
submits a wave, the next poll generates the whole wave, dropped groups'
replacements ride the following wave. Tier 1 is a property of the ENGINE,
not a controller file.

What this costs vs a real streaming engine (why VLLMEngine exists): no
incremental finishers (the batch's slowest member gates everything — the
classic "two waits") and refills only land between rounds.
Correctness is identical; throughput is what you give up.
"""

from torch import Tensor

from minirl.rollout.types import SamplingParams, Trajectory


class StreamAdapter:
    """Duck-typed streaming engine wrapping a generate()-engine (HFEngine)."""

    def __init__(self, engine):
        self.engine = engine  # needs: generate(list[Tensor], params) grouped by prompt,
        #                              load_weights(named, version), pad_id, version
        self._pending: list[tuple[Tensor, SamplingParams, dict]] = []
        self._stash: list[list[Trajectory]] = []

    @property
    def pad_id(self) -> int:
        return self.engine.pad_id

    @property
    def version(self) -> int:
        return self.engine.version

    @property
    def n_inflight(self) -> int:
        return len(self._pending)

    def submit(self, prompt_ids: Tensor, params: SamplingParams, meta: dict | None = None) -> None:
        self._pending.append((prompt_ids, params, dict(meta or {})))

    def poll(self) -> list[list[Trajectory]]:
        """Stash first (mirroring VLLMEngine); else generate EVERY pending
        prompt as one blocking round and return all its groups."""
        out, self._stash = self._stash, []
        if self._pending:
            pending, self._pending = self._pending, []
            params = pending[0][1]
            assert all(p == params for _, p, _ in pending), "mixed SamplingParams in one round"
            trajs = self.engine.generate([ids for ids, _, _ in pending], params)
            for i, (_, _, meta) in enumerate(pending):  # generate() is grouped by prompt
                group = trajs[i * params.n : (i + 1) * params.n]
                for t in group:
                    t.meta.update(meta)  # HFEngine doesn't carry meta; the adapter does
                out.append(group)
        return out

    def stash(self, group: list[Trajectory]) -> None:
        self._stash.append(group)

    def drain(self) -> None:
        while self._pending:
            for group in self.poll():
                self.stash(group)

    def load_weights(self, named_tensors, version: int) -> None:
        assert not self._pending, "load_weights with a pending round (drain first)"
        self.engine.load_weights(named_tensors, version)
