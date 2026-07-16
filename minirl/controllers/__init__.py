"""Training drivers — exactly TWO by decision (docs/async_tier2.md §11):

  fully_async.py  fit_async   THE training loop: k DP engines (streaming
                              interface; HFEngine joins via
                              engine/stream_adapter.py — one poll == one
                              round), 1..m trainer ranks (ONE Trainer class —
                              DDP auto-engages under torchrun; rank 0
                              collects + broadcasts + publishes, followers
                              only train), drain-ALL-then-publish, staleness
                              bound publish_interval + 1.
  sync.py         (planned)   collect -> train -> publish, no overlap; the
                              debugging/teaching reference. Same collector
                              (collect_groups_dp — continuous batching applies
                              to sync training too), no pipeline slot.

Retired 2026-07-14: round_based.py and streaming.py — both were k=1 special
cases of fully_async (round_based survives as StreamAdapter's semantics;
streaming.py's walkthrough comments moved into fully_async.py). History and
rationale: docs/async_training.md banner, docs/async_tier2.md §11.
"""

from minirl.controllers.fully_async import collect_groups_dp, fit_async

__all__ = ["collect_groups_dp", "fit_async"]
