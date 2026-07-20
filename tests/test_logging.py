"""metrics_logger tests — a FakeRun (duck-typed .log) stands in for wandb,
so the suite never needs the wandb package or a network."""

from minirl.logging import metrics_logger, namespace_metrics


class FakeRun:
    def __init__(self):
        self.calls: list[tuple[dict, int | None]] = []

    def log(self, payload: dict, step=None):
        self.calls.append((payload, step))


CONTROLLER_DICT = {  # a representative fit_async_stream emission
    "iteration": 7, "staleness": 1, "t_train": 2.0, "t_iter": 5.0, "t_generate": 4.0,
    "groups_generated": 10, "groups_dropped": 2, "groups": 8, "submitted": 10,
    "polls": 40, "leftover_inflight": 1, "batch_size": 16, "max_len": 90,
    "response_tokens": 800, "frac_padding": 0.1, "reward_mean": 0.5, "reward_std": 0.5,
    "frac_degenerate_groups": 0.0, "loss": -0.02, "grad_norm": 0.7, "lr": 1e-6,
    "approx_kl": 0.001, "clip_frac": 0.05, "ratio_max": 1.2, "tis_mean": 0.998,
    "tis_clip_frac": 0.0, "brand_new_metric": 42.0,  # future loss metric, unmapped
}


def test_namespacing_and_derived_metrics():
    out = namespace_metrics(CONTROLLER_DICT)
    assert out["rollout/reward_mean"] == 0.5 and out["train/loss"] == -0.02
    assert out["time/t_generate"] == 4.0 and out["async/staleness"] == 1
    assert out["time/tokens_per_sec"] == 800 / 4.0  # derived
    assert out["rollout/drop_rate"] == 2 / 10  # derived saturation signal
    assert out["brand_new_metric"] == 42.0  # unknown keys pass through BARE
    assert "iteration" not in out  # it is the step, not a metric


def test_logger_routes_to_run_with_iteration_as_step(capsys):
    run = FakeRun()
    metrics_logger(run)(CONTROLLER_DICT)
    (payload, step), = run.calls
    assert step == 7 and payload["rollout/reward_mean"] == 0.5
    line = capsys.readouterr().out  # echo line prints alongside
    assert "iter 7" in line and "reward=0.500" in line and "stale=1" in line


def test_logger_without_run_is_console_only(capsys):
    metrics_logger(None)(CONTROLLER_DICT)  # must not raise
    assert "reward=0.500" in capsys.readouterr().out


def test_echo_false_is_silent(capsys):
    run = FakeRun()
    metrics_logger(run, echo=False)(CONTROLLER_DICT)
    assert capsys.readouterr().out == "" and len(run.calls) == 1


def test_missing_keys_are_skipped_not_fatal():
    run = FakeRun()
    metrics_logger(run)({"loss": 1.0, "nll": 2.0})  # SFT-style dict: no iteration
    (payload, step), = run.calls
    assert step is None and payload == {"train/loss": 1.0, "train/nll": 2.0}
