"""Benchmark eval through the rollout engines — no separate eval model.

An EvalSet is a FIXED, finite list of prompts plus a reward: eval never
shuffles, never epochs, and scores every prompt every time, so curves are
comparable across evals. run_eval is called by the controller in the
post-publish quiescent window (all engines idle, weights exactly the
just-published version); it deals each set's prompts across the k engines,
generates with the eval sampling params, scores, and returns one flat
metrics dict namespaced eval/{set}/{metric}.

Engines may hold STASHED leftover training groups from the pre-publish
drain. Those belong to the NEXT collection, not to eval: run_eval pulls
them out via the public poll/stash contract before submitting eval prompts
and puts them back after, so eval can never consume (or be polluted by)
training work.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from minirl.config import EvalConfig
from minirl.data.chat import encode_prompt
from minirl.rollout.types import SamplingParams, Trajectory


@dataclass(frozen=True)
class EvalSet:
    """One benchmark: a name, fixed prompts, and how to score a trajectory."""

    name: str
    prompts: list  # [(prompt_ids (T,), meta)] — meta carries the reward's label
    reward_fn: Callable[[Trajectory], float]
    sampling: SamplingParams | None = None  # None -> EvalConfig's params


def make_eval_prompts(dataset, tokenizer, row_fn, enable_thinking: bool = False,
                      limit: int | None = None) -> list:
    """Tokenize a whole eval split into the fixed [(ids, meta)] list.

    row_fn is the same adapter shape the training prompt source uses
    (data/prompts.py); limit caps the set for cheap periodic evals.
    """
    n = len(dataset) if limit is None else min(limit, len(dataset))
    out = []
    for i in range(n):
        messages, meta = row_fn(dataset[i])
        out.append((encode_prompt(tokenizer, messages, enable_thinking), meta))
    return out


def _eval_one_engine(engine, jobs: list, sampling: SamplingParams) -> list[Trajectory]:
    """Generate this engine's share of one set (runs on the engine's own
    thread — same one-owner rule as collection). Returns all trajectories."""
    assert engine.n_inflight == 0, "eval on a busy engine — publish/drain first"
    held = engine.poll()  # leftover TRAINING groups out of the way (idle poll == stash)
    for ids, meta in jobs:
        engine.submit(ids, sampling, meta)
    done: list[Trajectory] = []
    groups_left = len(jobs)
    while groups_left:
        for group in engine.poll():
            done.extend(group)
            groups_left -= 1
    for group in held:  # give the leftovers back for the next collection
        engine.stash(group)
    return done


def run_eval(engines: list, eval_sets: list[EvalSet], cfg: EvalConfig) -> dict:
    """Score every set on the freshly published weights -> flat metrics dict."""
    metrics: dict = {}
    t0 = time.perf_counter()
    for s in eval_sets:
        sampling = s.sampling if s.sampling is not None else cfg.sampling_params()
        with ThreadPoolExecutor(max_workers=len(engines)) as pool:  # one owner thread per engine
            futures = [
                pool.submit(_eval_one_engine, e, s.prompts[i :: len(engines)], sampling)
                for i, e in enumerate(engines)
            ]
            trajs = [t for f in futures for t in f.result()]
        rewards = [s.reward_fn(t) for t in trajs]
        lens = [t.response_len for t in trajs]
        metrics[f"eval/{s.name}/reward_mean"] = sum(rewards) / len(rewards)
        metrics[f"eval/{s.name}/response_len_mean"] = sum(lens) / len(lens)
        metrics[f"eval/{s.name}/truncated_ratio"] = (  # hit the cap = never finished
            sum(n >= sampling.max_new_tokens for n in lens) / len(lens)
        )
    metrics["t_eval"] = time.perf_counter() - t0
    return metrics
