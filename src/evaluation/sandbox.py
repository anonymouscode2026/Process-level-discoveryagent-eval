"""
Sandbox execution for untrusted solution code.

run_code_in_sandbox() writes code to a temp file and spawns it via a
caller-supplied runner command.  On POSIX it applies a virtual-memory cap
via resource.setrlimit to prevent runaway allocations.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


def run_code_in_sandbox(
    code: str,
    runner_cmd: list[str],
    timeout: int = 60,
    memory_mb: int = 2048,
    workdir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute code in a subprocess with optional memory and time limits.

    The code string is written to a temporary .py file.  The runner command
    receives the path to that file appended as its last argument, e.g.:
        runner_cmd = ["python", "evaluator.py"]
        → subprocess runs: python evaluator.py /tmp/solution_XXXX.py

    Args:
        code:       Python source code to execute.
        runner_cmd: Command list; the solution file path is appended.
        timeout:    Wall-clock timeout in seconds.
        memory_mb:  Maximum virtual-memory in MiB (POSIX only; ignored on Windows).
        workdir:    Working directory for the subprocess. If None, uses a temp dir.

    Returns:
        dict with keys:
            "stdout"        — captured stdout string
            "stderr"        — captured stderr string
            "returncode"    — integer exit code (None if timed out)
            "timed_out"     — True if the process exceeded timeout
            "solution_path" — absolute path of the temp solution file (str)
    """
    # Determine working directory
    use_workdir = workdir or tempfile.mkdtemp(prefix="sandbox_")

    # Write solution code to a temp file in workdir
    solution_fd, solution_path = tempfile.mkstemp(
        suffix=".py", prefix="solution_", dir=use_workdir
    )
    try:
        with os.fdopen(solution_fd, "w", encoding="utf-8") as fh:
            fh.write(code)

        cmd = list(runner_cmd) + [solution_path]

        # preexec_fn sets a virtual-memory limit on POSIX systems
        preexec = None
        if sys.platform != "win32":
            import resource

            mem_bytes = memory_mb * 1024 * 1024

            def _set_limits():
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except (ValueError, resource.error):
                    pass  # ignore if the limit is too strict for the OS

            preexec = _set_limits

        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=use_workdir,
                preexec_fn=preexec,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = None
            timed_out = True

        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "timed_out": timed_out,
            "solution_path": solution_path,
        }
    except Exception:
        # Clean up solution file on unexpected errors; normally callers handle cleanup
        try:
            os.unlink(solution_path)
        except OSError:
            pass
        raise
