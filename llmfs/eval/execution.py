"""Run untrusted generated code with a timeout. Stage 20 §3, Stage 22 §4.

Model-written code is untrusted: it can loop forever, allocate all memory,
delete files. Each program runs in a separate `python -I` process, in an
empty temporary directory, with CPU-time, memory and file-size limits and a
wall-clock timeout. This keeps a runaway completion from hurting the run.

It is NOT a security boundary against deliberately malicious code (no network
or filesystem isolation beyond the temp dir). For a model trained on public
Python by us, that is an acceptable risk on a personal machine; a public
service would need a container or a VM.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def _limits(mem_mb: int):
    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))   # 1 MB files max
        os.setsid()                                                  # own process group
    return set_limits


SENTINEL = "__LLMFS_TESTS_COMPLETED__"


def run_program(src: str, timeout: float = 5.0, mem_mb: int = 512) -> tuple[bool, str]:
    """True if the program runs to its last line within the timeout.

    Exit status 0 is not enough: generated code can call os._exit(0) or
    sys.exit(0) before the asserts run. So a sentinel line is printed AFTER
    the program, and must appear on stdout (docs/20 §2).
    """
    # EXERCISE (module 10): write this function.
    # Write src plus a final print of SENTINEL into a temporary directory and run it
    # with sys.executable -I, the timeout, preexec_fn=_limits(mem_mb) and a minimal env.
    # Pass only if the exit code is 0 AND the sentinel is on stdout. Return (passed,
    # reason).
    raise NotImplementedError("run_program: your code here")


def run_many(programs: list[str], timeout: float = 5.0, workers: int = 8) -> list[tuple[bool, str]]:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda s: run_program(s, timeout), programs))
