"""Engine RPC round-trip tests — CPU only, no vLLM, real unix sockets.

The transport must be invisible: whatever an engine returns in-process
must come back identical through the proxy (pickle + socket). serve()
runs in a thread around the FakeStreamEngine (the executable spec of the
streaming contract); an EngineProxy drives it exactly the way
train_async/eval drive a real engine. Plus the two safety properties the
split exists for: the child-env whitelist strips torchrun's identity, and
the file publish ships each storage once (tied weights).
"""

import os
import tempfile
import threading

import torch
from safetensors.torch import load_file

from minirl.engine_proxy import EngineProxy, _server_env, save_named_tensors
from minirl.engine_server import serve
from tests.test_train_async import SAMPLING, FakeStreamEngine


class ServerFake(FakeStreamEngine):
    """FakeStreamEngine + the server half of the file-based publish."""

    def load_weights_path(self, weights_dir: str, version: int) -> None:
        assert self.n_inflight == 0, "weights changed with requests in flight — drain first"
        (f,) = [f for f in os.listdir(weights_dir) if f.endswith(".safetensors")]
        self.received = load_file(os.path.join(weights_dir, f))
        self.version = version
        self.published.append(version)


def _serve_pair(engine):
    """serve(engine) on a background thread; return (proxy, thread)."""
    sock = os.path.join(tempfile.mkdtemp(prefix="minirl-rpc-"), "e.sock")
    th = threading.Thread(target=serve, args=(engine, sock), daemon=True)
    th.start()
    return EngineProxy(sock, ready_timeout=10.0, rpc_timeout=10.0), th


def test_round_trip_contract():
    fake = ServerFake(finish_after=2)
    proxy, th = _serve_pair(fake)

    # hello caches the connect-time constants
    proxy.wait_ready()
    assert proxy.pad_id == fake.pad_id
    assert proxy.version == -1  # FakeStreamEngine: nothing published yet

    # publish v0 through the real file path (submit asserts version >= 0)
    w = torch.randn(4, 3)
    proxy.load_weights([("a", w), ("a_tied_alias", w), ("b", torch.zeros(2))], version=0)
    assert proxy.version == 0 and fake.version == 0
    assert set(fake.received) == {"a", "b"}  # the alias shipped ONCE
    assert torch.equal(fake.received["a"], w)

    # submit -> poll: trajectories come back tensor-identical
    prompt = torch.tensor([7, 8, 9])
    gid = proxy.submit(prompt, SAMPLING, {"tag": "rpc"})
    assert isinstance(gid, str) and proxy.n_inflight == 1
    groups = []
    while proxy.n_inflight:
        groups += proxy.poll()
    (group,) = groups
    assert len(group) == SAMPLING.n
    for t in group:
        assert torch.equal(t.input_ids[:3], prompt)
        assert t.version == 0 and t.meta["tag"] == "rpc"
        assert t.loss_mask.dtype == torch.bool and t.logprobs.dtype == torch.float32

    # stash hands the group back; the next poll returns it FIRST
    proxy.stash(group)
    polled = proxy.poll()
    assert len(polled) == 1 and torch.equal(polled[0][0].input_ids, group[0].input_ids)

    # drain leaves nothing in flight (finishers go to the stash)
    proxy.submit(prompt, SAMPLING, None)
    proxy.drain()
    assert proxy.n_inflight == 0
    assert proxy.poll()  # the drained group, from the stash

    # shutdown ends the serve loop
    proxy.close()
    th.join(timeout=5)
    assert not th.is_alive()


def test_server_errors_surface_with_traceback():
    proxy, th = _serve_pair(ServerFake(finish_after=1))
    try:
        proxy.submit(torch.tensor([1]), SAMPLING, None)  # version still -1: server asserts
        raise AssertionError("expected the server-side assert to surface")
    except RuntimeError as e:
        assert "submit before first publish" in str(e)  # the remote traceback travels back
    finally:
        proxy.close()
        th.join(timeout=5)


def test_server_env_whitelist(monkeypatch):
    # torchrun identity + secrets must NOT reach the engine process;
    # HF/vLLM knobs and the basics must.
    for k, v in {"RANK": "0", "WORLD_SIZE": "2", "MASTER_ADDR": "10.0.0.1",
                 "TORCHELASTIC_USE_AGENT_STORE": "True", "LOCAL_RANK": "0",
                 "WANDB_API_KEY": "secret", "HF_HOME": "/data/hf",
                 "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}.items():
        monkeypatch.setenv(k, v)
    env = _server_env(3)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    for leaked in ("RANK", "WORLD_SIZE", "MASTER_ADDR",
                   "TORCHELASTIC_USE_AGENT_STORE", "LOCAL_RANK", "WANDB_API_KEY"):
        assert leaked not in env
    assert env["HF_HOME"] == "/data/hf"
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert "PATH" in env and "HOME" in env


def test_save_named_tensors_dedups_storages(tmp_path):
    w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = str(tmp_path / "w.safetensors")
    save_named_tensors([("emb", w), ("lm_head", w), ("bias", torch.ones(2))], path)
    loaded = load_file(path)
    assert set(loaded) == {"emb", "bias"}  # second alias of the same storage skipped
    assert torch.equal(loaded["emb"], w)
