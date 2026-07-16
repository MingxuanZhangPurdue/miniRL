"""DDP equivalence tests (docs/ddp.md §6) — 2 CPU processes over gloo.

The invariance standard, same as microbatching: DISTRIBUTED AND
SINGLE-PROCESS MUST PRODUCE THE SAME MATH ON IDENTICAL DATA. One spawn runs
all three loss_agg modes (amortizes process startup); the parent computes the
single-process references and compares parameters.

(The FSDP2-era pytree trap does not apply here: DDP hooks PARAMETERS, not
forward outputs, so a SimpleNamespace-returning test model is fine.)
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as tdist
import torch.multiprocessing as mp
from torch import nn

from minirl.algos import GRPOConfig, grpo_loss
from minirl.rollout.batching import make_batch
from minirl.rollout.types import Trajectory
from minirl.train import TrainConfig, Trainer

VOCAB = 61
WORLD = 2
LOSS_AGGS = ["seq_mean", "token_mean", 10]


class TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))  # (B, T, V)


def make_trajs(b: int = 4, group_size: int = 2) -> list[Trajectory]:
    """Variable lengths, alternating rewards within groups (as in test_trainer)."""
    torch.manual_seed(0)
    trajs = []
    for i in range(b):
        n = 2 + 3 + i
        mask = torch.cat([torch.zeros(2, dtype=torch.bool), torch.ones(3 + i, dtype=torch.bool)])
        trajs.append(
            Trajectory(
                input_ids=torch.randint(0, VOCAB, (n,)),
                loss_mask=mask,
                logprobs=torch.where(mask, torch.randn(n).abs().neg(), torch.zeros(n)),
                reward=float(i % 2),
                meta={"group_id": i // group_size},
            )
        )
    return trajs


def _reference_state_dict(loss_agg) -> dict[str, torch.Tensor]:
    """Single-process Trainer, one fit_batch — the ground truth."""
    torch.manual_seed(42)
    trainer = Trainer(
        TinyLM(),
        grpo_loss,
        GRPOConfig(loss_agg=loss_agg),
        TrainConfig(lr=1e-2, minibatch_size=8, micro_batch_size=8),
    )
    trainer.fit_batch(make_batch(make_trajs(), pad_id=0)[0])
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def _worker(rank: int, port: int, out_dir: str) -> None:
    """Runs in each spawned process: all three loss_agg modes + the extras."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD)
    )
    tdist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from minirl.train.distributed import DistTrainer, full_state_dict

        results: dict = {}
        for loss_agg in LOSS_AGGS:
            torch.manual_seed(42)  # SAME init as the single-process reference
            trainer = DistTrainer(
                TinyLM(),
                grpo_loss,
                GRPOConfig(loss_agg=loss_agg),
                # micro_batch_size=1: each rank runs 2 microbatches -> exercises
                # no_sync local accumulation with all-reduce-on-last (docs/ddp.md §3)
                TrainConfig(lr=1e-2, minibatch_size=8, micro_batch_size=1),
            )
            trainer.fit_batch(make_batch(make_trajs(), pad_id=0)[0])
            results[str(loss_agg)] = full_state_dict(trainer.model)

        # divisibility assert fires loud on ragged batches (5 rows, world 2)
        try:
            trainer.step(make_batch(make_trajs(5), pad_id=0)[0])
            results["ragged_asserted"] = False
        except AssertionError:
            results["ragged_asserted"] = True

        if rank == 0:
            torch.save(results, os.path.join(out_dir, "dist_results.pt"))
        tdist.barrier()
    finally:
        tdist.destroy_process_group()


def test_two_rank_ddp_matches_single_process(tmp_path):
    port = 29511  # fixed local port; gloo binds 127.0.0.1
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=WORLD, join=True)
    results = torch.load(tmp_path / "dist_results.pt")

    for loss_agg in LOSS_AGGS:
        ref = _reference_state_dict(loss_agg)
        dist_sd = results[str(loss_agg)]
        assert set(dist_sd) == set(ref)  # full_state_dict round-trips every param name
        for k in ref:
            assert torch.allclose(dist_sd[k], ref[k], atol=1e-6), (
                f"loss_agg={loss_agg}: parameter {k} diverged between 2-rank DDP "
                "and single-process training"
            )
    assert results["ragged_asserted"], "B % world != 0 must fail loud (docs/ddp.md §1)"
