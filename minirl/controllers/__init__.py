"""Training drivers — exactly TWO by decision:

  fully_async.py  fit_async   THE training loop: k DP engines (the streaming
                              interface — VLLMEngine, vLLM-only since
                              2026-07-20), 1..m trainer ranks (ONE trainer —
                              DDP auto-engages under torchrun; rank 0
                              collects + broadcasts + publishes, followers
                              only train), drain-ALL-then-publish, staleness
                              bound publish_interval + 1.
  sync.py         (planned)   collect -> train -> publish, no overlap; the
                              debugging/teaching reference. Same collector
                              (collect_groups_dp — continuous batching applies
                              to sync training too), no pipeline slot.

Retired 2026-07-14: round_based.py and streaming.py — both were k=1 special
cases of fully_async (streaming.py's walkthrough comments moved into
fully_async.py). Retired 2026-07-20 with the vLLM-only decision: HFEngine
and its StreamAdapter (the generate()-engine on-ramp).
"""

from minirl.controllers.fully_async import collect_groups_dp, fit_async

__all__ = ["collect_groups_dp", "fit_async"]
