"""Training drivers — one file per COLLECTION STRATEGY, same pipeline skeleton.

Every controller is the same loop (collect rollout k+1 on a worker thread
while training on rollout k; held-future join before any weight publish);
they differ only in how the batch is collected and what the engine must
support:

  round_based.py  fit_async         tier 1: round-based collect_groups; needs
                                    only engine.generate() — HFEngine's home,
                                    the readable slime train_async.py mirror
                                    (docs/async_training.md)
  streaming.py    fit_async_stream  tier 2: continuous batching via
                                    collect_groups_stream; needs the streaming
                                    engine interface (submit/poll/stash/drain
                                    — VLLMEngine); drain-then-publish
                                    (docs/async_tier2.md)
  data_parallel.py (planned)        tier 2 x N engines: shared dealer + tally
                                    (docs/async_tier2.md §10); expected to
                                    make streaming.py its N=1 degenerate case
                                    — decide merge-or-keep when it lands

There is no sync controller anywhere: sync training is fit_async with the
future resolved eagerly (a degenerate case, not a file — DESIGN principle 8).
"""

from minirl.controllers.round_based import fit_async
from minirl.controllers.streaming import fit_async_stream

__all__ = ["fit_async", "fit_async_stream"]
