"""Metrics logging — builds the on_metrics callback; wandb stays OUT of core.

The integration seam is the controllers' `on_metrics: Callable[[dict], None]`:
controllers/trainer emit flat dicts of raw facts and never know where numbers
go. This module is the RECIPE-layer glue that turns those dicts into a wandb
stream (or a console line). Core modules never import this file, this file
imports wandb never — the recipe hands in a live `run` object (anything with
`.log(dict, step=int)`), constructed by `wandb.init(...)` in the recipe.

Namespacing (flat controller keys -> wandb UI groups; UNKNOWN keys pass
through un-prefixed, so new losses' metrics land automatically — no registry):

    rollout/  reward_mean reward_std frac_degenerate_groups groups
              groups_generated groups_dropped rounds submitted polls
              leftover_inflight response_tokens frac_padding batch_size max_len
    train/    loss grad_norm lr approx_kl clip_frac ratio_max kl_ref
              tis_mean tis_clip_frac nll ppl
    time/     t_generate t_train t_iter (+ derived tokens_per_sec)
    async/    staleness

Derived here, not in the controllers (they emit raw facts only):
    time/tokens_per_sec  = response_tokens / t_generate
    rollout/drop_rate    = groups_dropped / groups_generated  (dataset
                           saturation — the dynamic-sampling health signal)

Step semantics: everything is logged at step=m["iteration"], so training
curves and any future eval results (logged as `eval/...` at the same step by
the recipe / eval harness) share one x-axis.
"""

from typing import Callable

_NAMESPACE = {
    "rollout": (
        "reward_mean", "reward_std", "frac_degenerate_groups", "groups",
        "groups_generated", "groups_dropped", "rounds", "submitted", "polls",
        "leftover_inflight", "response_tokens", "frac_padding", "batch_size", "max_len",
    ),
    "train": (
        "loss", "grad_norm", "lr", "approx_kl", "clip_frac", "ratio_max",
        "kl_ref", "tis_mean", "tis_clip_frac", "nll", "ppl",
    ),
    "time": ("t_generate", "t_train", "t_iter", "t_eval"),
    "async": ("staleness",),
}
_KEY_TO_GROUP = {key: group for group, keys in _NAMESPACE.items() for key in keys}


def namespace_metrics(m: dict) -> dict:
    """Flat controller dict -> wandb-grouped dict, plus the derived metrics.

    `iteration` is dropped from the payload (it becomes the wandb step);
    unrecognized keys pass through bare rather than erroring — forward
    compatibility for new losses/collectors without touching this table.
    """
    out = {}
    for k, v in m.items():
        if k == "iteration":
            continue
        group = _KEY_TO_GROUP.get(k)
        out[f"{group}/{k}" if group else k] = v
    if m.get("t_generate") and "response_tokens" in m:
        out["time/tokens_per_sec"] = m["response_tokens"] / m["t_generate"]
    if m.get("groups_generated"):
        out["rollout/drop_rate"] = m.get("groups_dropped", 0) / m["groups_generated"]
    return out


def metrics_logger(run=None, echo: bool = True) -> Callable[[dict], None]:
    """Build an on_metrics callback for the controllers (or an SFT loop).

    run: a wandb Run — or ANY object with .log(dict, step=int|None) (tests
         use a fake) — or None for console-only.
    echo: also print the one-line summary (the smoke-run view).
    """

    def log(m: dict) -> None:
        if echo:
            parts = [f"iter {m['iteration']}"] if "iteration" in m else []
            for label, key, fmt in (
                ("reward", "reward_mean", ".3f"), ("loss", "loss", "+.4f"),
                ("kl", "approx_kl", ".4f"), ("clip", "clip_frac", ".3f"),
                ("tis", "tis_mean", ".3f"), ("stale", "staleness", "d"),
                ("t_gen", "t_generate", ".1f"), ("t_train", "t_train", ".1f"),
            ):
                if key in m:
                    parts.append(f"{label}={m[key]:{fmt}}")
            for key in m:  # eval scores (already namespaced eval/{set}/...)
                if key.startswith("eval/") and key.endswith("/reward_mean"):
                    parts.append(f"{key.split('/')[1]}={m[key]:.3f}")
            print("  ".join(parts), flush=True)
        if run is not None:
            run.log(namespace_metrics(m), step=m.get("iteration"))

    return log
