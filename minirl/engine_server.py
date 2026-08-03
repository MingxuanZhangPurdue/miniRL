"""Engine server — a rollout engine in its OWN process, the contract on a socket.

Trainer and engines must not share a process: a shared process means a
shared environment (torchrun's rendezvous identity leaks into the engine's
children), a shared venv (vLLM's torch pin vs the Megatron stack), and a
shared fate on any crash. So recipes launch THIS module once per rollout
GPU — `python -m minirl.engine_server MODEL SOCKET ...`, with
CUDA_VISIBLE_DEVICES pinned by the launcher's env whitelist — and the
controller drives it through EngineProxy (engine_proxy.py), which satisfies
the same submit/poll/stash/drain/n_inflight/load_weights contract the
in-process engine did.

serve() is deliberately dumb: ONE client, one pickled (method, args)
request at a time, replies ("ok", result) or ("err", traceback). Binding
the socket is the LAST construction step, so "connectable" == "engine
ready" — the proxy's readiness probe needs no extra handshake.
"""

import argparse
import json
import traceback
from multiprocessing.connection import Listener


def serve(engine, socket_path: str) -> None:
    """Answer contract calls on a unix socket until shutdown or disconnect."""
    with Listener(socket_path, family="AF_UNIX") as listener:
        with listener.accept() as conn:
            while True:
                try:
                    method, args = conn.recv()
                except EOFError:  # controller closed its end: nothing left to serve
                    return
                if method == "shutdown":
                    conn.send(("ok", None))
                    return
                if method == "hello":  # connect-time constants in one round trip
                    conn.send(("ok", {"pad_id": engine.pad_id, "version": engine.version}))
                    continue
                try:
                    attr = getattr(engine, method)
                    out = attr(*args) if callable(attr) else attr  # n_inflight is a property
                    conn.send(("ok", out))
                except Exception:
                    conn.send(("err", traceback.format_exc()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("socket")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-kwargs", default="{}",
                    help="JSON dict forwarded to VLLMEngine (max_model_len, "
                         "gpu_memory_utilization, ...)")
    args = ap.parse_args()

    from minirl.vllm_engine import VLLMEngine  # vllm exists only in THIS process's env

    serve(VLLMEngine(args.model, seed=args.seed, **json.loads(args.engine_kwargs)), args.socket)


if __name__ == "__main__":
    main()
