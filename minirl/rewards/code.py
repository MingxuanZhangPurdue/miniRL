"""Code-execution reward with a level-2 sandbox (DESIGN: rewards/).

The reward is simple: extract the model's ```python fence, append the task's
test asserts, run it; reward 1.0 iff the process exits cleanly.

The sandbox exists because we execute MODEL-GENERATED code millions of times
at temperature 1 — the tail of that distribution contains infinite loops,
memory bombs, and stray file writes by pure chance, and any one of them can
take down a training run. The threat model here is ACCIDENTS, not attackers:

  - wall-clock timeout + process-GROUP kill (children may spawn children)
  - self-imposed rlimits (CPU seconds, address space, file size, no cores),
    set by a prelude INSIDE the child — thread-safe (no preexec_fn, which can
    deadlock when called from our rollout worker thread) and readable
  - throwaway temp cwd (stray writes land nowhere), minimal env,
    python -I (isolated mode: no site-packages injection, no cwd on sys.path)

This is the same call slime made for its own tool-use RL: their ReTool
example ships exactly this shape (subprocess + tempdir + timeout + memory
caps + a concurrency semaphore), no external service.
# In production: a sandbox SERVICE on separate machines (SandboxFusion is the
# verl-native standard; Piston/E2B likewise) reached via HTTP — slime's
# remote_rm pattern. Reached from here by swapping reward_fn for a POST;
# nothing else changes. macOS caveat: RLIMIT_AS is only loosely enforced by
# the kernel — the timeout is the real backstop on Mac.
"""

import os
import re
import signal
import subprocess
import sys
import tempfile

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Runs first inside the child: the child confines ITSELF before any model
# code executes. rlimits cannot be raised back once lowered.
_PRELUDE = """\
import resource
for limit, value in [
    ("RLIMIT_CPU", {cpu_s}),          # cpu-seconds, catches busy loops
    ("RLIMIT_AS", {mem_bytes}),       # address space (loose on macOS)
    ("RLIMIT_FSIZE", 10_000_000),     # max bytes any written file can reach
    ("RLIMIT_CORE", 0),               # no core dumps
]:
    try:
        resource.setrlimit(getattr(resource, limit), (value, value))
    except (ValueError, OSError):
        pass
"""


def extract_code(text: str) -> str | None:
    """The LAST ```python fence in the response (models often draft then fix)."""
    blocks = _FENCE_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def run_python(code: str, timeout_s: float = 10.0, memory_mb: int = 1024) -> tuple[bool, str]:
    """Execute code in a sandboxed child. Returns (exit ok, stderr tail)."""
    program = _PRELUDE.format(cpu_s=int(timeout_s) + 1, mem_bytes=memory_mb * 2**20) + code
    with tempfile.TemporaryDirectory() as cwd:
        path = os.path.join(cwd, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        proc = subprocess.Popen(
            [sys.executable, "-I", path],  # -I: isolated (no cwd/user site on sys.path)
            cwd=cwd,
            env={"PATH": os.defpath},  # no tokens/keys from our environment
            stdout=subprocess.DEVNULL,  # model prints freely; we only care about exit code
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group -> we can kill descendants too
            text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group, not just the child
            proc.communicate()
            return False, "timeout"
        return proc.returncode == 0, (stderr or "")[-500:]


def code_reward(response_text: str, tests: str, timeout_s: float = 10.0) -> float:
    """1.0 iff the response's code block passes the task's asserts.

    `tests` is the dataset's test snippet (MBPP/HumanEval style), e.g.
    "assert add(2, 3) == 5" — appended after the model's code so the asserts
    see its definitions. No fence -> 0 (format is part of the reward spec,
    same as math's boxed answer).
    """
    code = extract_code(response_text)
    if code is None:
        return 0.0
    ok, _ = run_python(code + "\n\n" + tests, timeout_s=timeout_s)
    return float(ok)
