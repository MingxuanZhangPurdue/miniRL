"""EngineProxy + launcher — the controller's handles to engine-server processes.

launch_engine_servers() Popens one engine_server (engine_server.py) per
rollout GPU and yields proxies that satisfy the streaming-engine contract —
train_async/eval cannot tell a proxy from an in-process engine. Two
properties do all the isolation work:

  - The child env is an explicit WHITELIST. torchrun advertises its
    rendezvous (RANK, MASTER_*, TORCHELASTIC_*) to every descendant, and an
    engine that inherits that identity tries to JOIN the trainer's store
    instead of hosting its own — and hangs forever. Enumerating what the
    child may see makes the leak impossible, and CUDA_VISIBLE_DEVICES
    becomes the entire placement mechanism.
  - The server's interpreter is a parameter (MINIRL_ENGINE_PYTHON), so the
    engine venv and trainer venv can be DIFFERENT pythons: the trainer
    process never imports vllm, the engine never imports megatron.

Weight publish stays file-based end to end: the proxy writes ONE
safetensors file and ships the PATH; the server reloads it through vLLM's
native worker RPC. Same protocol as in-process, one extra path-sized
message.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from multiprocessing.connection import Client

from safetensors.torch import save_file


def save_named_tensors(named_tensors, path: str) -> None:
    """Write a learner state dict as ONE safetensors file, each STORAGE once.

    Tied weights (Qwen: lm_head <- embed_tokens) are aliases, and
    safetensors refuses aliased tensors; loaders re-tie on their side
    (vLLM skips lm_head for tied configs).
    """
    tensors, seen = {}, set()
    for k, v in named_tensors:
        ptr = v.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        tensors[k] = v.detach().cpu().contiguous()
    save_file(tensors, path)


_KEEP = ("HOME", "PATH", "TMPDIR", "USER")
_KEEP_PREFIXES = ("HF_", "HUGGINGFACE", "VLLM_", "TORCHINDUCTOR", "CUDA_CACHE")


def _server_env(gpu_id: int | str) -> dict[str, str]:
    """The whitelist: what an engine server is allowed to inherit."""
    env = {k: v for k, v in os.environ.items()
           if k in _KEEP or k.startswith(_KEEP_PREFIXES)}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # placement IS the env: born seeing one GPU
    env["PYTHONUNBUFFERED"] = "1"  # server log lines land as they happen
    return env


class EngineProxy:
    """The engine contract over one unix-socket connection.

    Every phase drives an engine from ONE thread at a time (its collector,
    then an eval worker, then the publishing main thread); the lock makes
    that assumption explicit — the connection carries one call in flight.
    """

    def __init__(self, socket_path: str, *, proc=None, log_path: str | None = None,
                 ready_timeout: float = 900.0, rpc_timeout: float = 900.0):
        self._sock_path = socket_path
        self._proc = proc
        self._log_path = log_path
        self._ready_timeout = ready_timeout
        self._rpc_timeout = rpc_timeout
        self._conn = None
        self._lock = threading.Lock()
        self._pad_id: int | None = None
        self._version: int | None = None

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until the server's socket accepts (bind is its LAST init step)."""
        if self._conn is not None:
            return
        deadline = time.monotonic() + (timeout if timeout is not None else self._ready_timeout)
        while self._conn is None:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"engine server exited rc={self._proc.returncode} during startup"
                    + (f" — see {self._log_path}" if self._log_path else ""))
            try:
                self._conn = Client(self._sock_path, family="AF_UNIX")
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"engine server not ready after {self._ready_timeout:.0f}s"
                        + (f" — see {self._log_path}" if self._log_path else ""))
                time.sleep(1.0)
        hello = self._call("hello")
        self._pad_id, self._version = hello["pad_id"], hello["version"]

    def _call(self, method: str, *args):
        if self._conn is None:
            self.wait_ready()
        with self._lock:
            self._conn.send((method, args))
            if not self._conn.poll(self._rpc_timeout):
                raise RuntimeError(
                    f"engine server: no reply to {method!r} within {self._rpc_timeout:.0f}s"
                    + (f" — see {self._log_path}" if self._log_path else ""))
            status, out = self._conn.recv()
        if status == "err":
            raise RuntimeError(f"engine server raised in {method!r}:\n{out}")
        return out

    # ---- the streaming contract, one round trip per member ----

    def submit(self, prompt_ids, params, meta: dict | None = None) -> str:
        return self._call("submit", prompt_ids, params, meta)

    def poll(self):
        return self._call("poll")

    def stash(self, group) -> None:
        self._call("stash", group)

    def drain(self) -> None:
        self._call("drain")

    @property
    def n_inflight(self) -> int:
        return self._call("n_inflight")

    @property
    def pad_id(self) -> int:
        if self._pad_id is None:
            self.wait_ready()
        return self._pad_id

    @property
    def version(self) -> int:
        if self._version is None:
            self.wait_ready()
        return self._version

    def load_weights(self, named_tensors, version: int) -> None:
        """File-based publish: write once here, ship the PATH, server reloads."""
        with tempfile.TemporaryDirectory(prefix="minirl-pub-") as td:
            save_named_tensors(named_tensors, os.path.join(td, "learner.safetensors"))
            self._call("load_weights_path", td, version)
        self._version = version

    def close(self) -> None:
        """Polite shutdown if connected; the launcher's terminate() is the backstop."""
        if self._conn is None:
            return
        try:
            self._call("shutdown")
        except Exception:
            pass
        self._conn.close()
        self._conn = None


@contextmanager
def launch_engine_servers(model: str, gpu_ids, *, seeds=None, python: str | None = None,
                          ready_timeout: float = 900.0, rpc_timeout: float = 900.0,
                          **engine_kwargs):
    """Popen one engine server per GPU; yield their proxies; clean up on exit.

    `python` (or $MINIRL_ENGINE_PYTHON) selects the ENGINE venv's
    interpreter — the trainer's venv never needs vllm installed. Extra
    kwargs (max_model_len, gpu_memory_utilization, ...) forward to
    VLLMEngine via one JSON argv. Proxies connect lazily: construction
    returns immediately so engines warm up while the trainer builds; call
    wait_ready() (or any method) to block on readiness.
    """
    python = python or os.environ.get("MINIRL_ENGINE_PYTHON") or sys.executable
    sock_dir = tempfile.mkdtemp(prefix="minirl-eng-")  # /tmp-short: AF_UNIX paths are length-capped
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    procs, logs, proxies = [], [], []
    try:
        for i, gpu in enumerate(gpu_ids):
            sock = os.path.join(sock_dir, f"e{i}.sock")
            log = open(os.path.join(sock_dir, f"e{i}.log"), "w")
            cmd = [python, "-m", "minirl.engine_server", model, sock,
                   "--seed", str(seeds[i] if seeds is not None else gpu)]
            if engine_kwargs:
                cmd += ["--engine-kwargs", json.dumps(engine_kwargs)]
            env = _server_env(gpu)
            env["PYTHONPATH"] = repo_root
            procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
            logs.append(log)
            proxies.append(EngineProxy(sock, proc=procs[-1], log_path=log.name,
                                       ready_timeout=ready_timeout, rpc_timeout=rpc_timeout))
        yield proxies
    finally:
        for proxy in proxies:
            proxy.close()
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for log in logs:
            log.close()
        shutil.rmtree(sock_dir, ignore_errors=True)
