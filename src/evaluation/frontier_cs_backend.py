"""
Backend helpers for evaluating Frontier-CS problems.

Provides lazy sys.path injection and a module-level singleton cache so
SingleEvaluator is only constructed once per (backend, base_dir) pair.

Also installs a one-time monkey-patch on
`frontier_cs.runner.research_docker.ResearchDockerRunner._run_docker` so that
GPU containers are launched with `--gpus device=N` (per-process round-robin
assignment) instead of `--gpus all`. Without this, every parallel pilot
worker's container would expose all 4 GPUs and default to the same physical
GPU 0, leading to memory contention. See `gpu_pinning.py` for the assignment
policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_PKG_ROOT = Path(__file__).resolve().parents[2]

# Frontier-CS is an external repo not shipped with this package. Clone it and
# point FRONTIER_CS_ROOT at its root (containing `src/`), or accept the default
# external/ subdir convention.
_DEFAULT_BASE_DIR = Path(
    os.environ.get("FRONTIER_CS_ROOT", str(_PKG_ROOT / "external/Frontier-CS"))
)
_FRONTIER_CS_SRC = _DEFAULT_BASE_DIR / "src"

# Module-level singleton cache keyed by (backend, base_dir)
_EVALUATOR_CACHE: dict[tuple, object] = {}

# Idempotent monkey-patch flag
_DOCKER_PATCH_INSTALLED = False


def _inject_path() -> None:
    """Lazily inject Frontier-CS src into sys.path."""
    src = str(_FRONTIER_CS_SRC)
    if src not in sys.path:
        if not _FRONTIER_CS_SRC.exists():
            raise RuntimeError(
                f"Frontier-CS src not found at attempted path: {_FRONTIER_CS_SRC}"
            )
        sys.path.insert(0, src)


def _patch_research_docker_for_gpu_pinning() -> None:
    """
    Monkey-patch ResearchDockerRunner._run_docker so GPU containers use
    `--gpus device=N` (round-robin per-process) instead of `--gpus all`.

    Idempotent across calls within the same process. The patch only changes
    the GPU flag; all other docker arguments are constructed identically to
    upstream.
    """
    global _DOCKER_PATCH_INSTALLED
    if _DOCKER_PATCH_INSTALLED:
        return
    _inject_path()
    from frontier_cs.runner import research_docker as _rd  # type: ignore

    # Only patch once per process
    if getattr(_rd.ResearchDockerRunner._run_docker, "_omc_gpu_pinned", False):
        _DOCKER_PATCH_INSTALLED = True
        return

    def _patched_run_docker(self, workspace, docker_config, needs_gpu, timeout, uv_project=None):
        # Lazy import so the patch doesn't fail if gpu_pinning is somehow missing
        try:
            from src.evaluation.gpu_pinning import docker_gpu_arg
            gpu_value = docker_gpu_arg()
        except Exception:
            gpu_value = "all"

        # Give the container a unique name so we can force-kill it on timeout.
        # The original `timeout --foreground Ns docker run` wrapper does NOT
        # reliably kill the inner process when the agent's solution code
        # contains an infinite loop — Docker's signal propagation through
        # bash → python is unreliable. We replace it with subprocess.run's
        # own timeout + an explicit `docker kill` fallback.
        import uuid as _uuid
        container_name = f"omc_eval_{_uuid.uuid4().hex[:12]}"

        cmd = ["docker", "run", "--rm", "--name", container_name]
        if needs_gpu:
            cmd.extend(["--gpus", gpu_value])
        if docker_config.dind:
            cmd.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])
        cmd.extend(["-v", f"{workspace}:/workspace:ro"])
        if self.datasets_dir.exists():
            cmd.extend(["-v", f"{self.datasets_dir}:/datasets:ro"])
        cmd.extend(["-w", "/work"])
        cmd.append(docker_config.image)
        run_script = self._get_run_script(uv_project=uv_project, dind=docker_config.dind)
        cmd.extend(["bash", "-c", run_script])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout if timeout else None,
            )
            logs = result.stdout + "\n" + result.stderr
            return result, logs
        except subprocess.TimeoutExpired as exc:
            # subprocess timeout fired — force-kill the container directly.
            # `docker kill` sends SIGKILL to PID 1 in the container, which
            # the daemon enforces via the cgroup, so the inner python loop
            # cannot ignore it.
            try:
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except Exception:
                pass
            # Synthesize a CompletedProcess-like return so the upstream
            # caller can check returncode == 124 (timeout exit code) just
            # like with the old `timeout` wrapper.
            class _TimeoutResult:
                def __init__(self, stdout, stderr):
                    self.returncode = 124
                    self.stdout = stdout or ""
                    self.stderr = stderr or ""
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr += f"\n[omc] container {container_name} killed after {timeout}s timeout\n"
            result = _TimeoutResult(stdout, stderr)
            return result, stdout + "\n" + stderr

    _patched_run_docker._omc_gpu_pinned = True  # type: ignore[attr-defined]
    _rd.ResearchDockerRunner._run_docker = _patched_run_docker
    _DOCKER_PATCH_INSTALLED = True


def get_frontier_cs_evaluator(
    backend: str = "docker",
    base_dir: Path = _DEFAULT_BASE_DIR,
) -> object:
    """Return a cached SingleEvaluator instance for the given backend and base_dir."""
    _inject_path()
    _patch_research_docker_for_gpu_pinning()
    from frontier_cs.single_evaluator import SingleEvaluator  # type: ignore

    key = (backend, str(base_dir))
    if key not in _EVALUATOR_CACHE:
        _EVALUATOR_CACHE[key] = SingleEvaluator(
            backend=backend,
            base_dir=base_dir,
            register_cleanup=False,
        )
    return _EVALUATOR_CACHE[key]


def _result_to_dict(result: object) -> dict:
    """Convert an EvaluationResult to the adapter dict schema."""
    # EvaluationResult fields: problem_id, score, score_unbounded, status,
    # message, logs, duration_seconds, metadata
    # status is an EvaluationStatus enum; .success property = (status == SUCCESS)
    score = getattr(result, "score", None)
    success = bool(getattr(result, "success", False))
    message = getattr(result, "message", None)
    logs = getattr(result, "logs", None)
    duration = getattr(result, "duration_seconds", None)
    metadata = getattr(result, "metadata", {}) or {}
    status_val = None
    status_obj = getattr(result, "status", None)
    if status_obj is not None:
        status_val = getattr(status_obj, "value", str(status_obj))

    # Strip per-case `output` payload. The judge returns the agent's full
    # submission string for every test case, which for long-output problems
    # (grammar_fuzzing, LCS-style algo problems) can reach 10 MB per case ×
    # multiple cases per step × 150 steps per run, blowing trajectory files
    # up to tens of gigabytes. We keep the length as a signal and drop the
    # content — the code is already stored in `generation.code` upstream,
    # so the output is recoverable by rerunning.
    cases = metadata.get("cases")
    if isinstance(cases, list):
        stripped = []
        for c in cases:
            if isinstance(c, dict) and "output" in c:
                c = dict(c)
                c["output_len"] = len(c["output"]) if c["output"] is not None else 0
                del c["output"]
            stripped.append(c)
        metadata = dict(metadata)
        metadata["cases"] = stripped

    # Truncate docker container stdout/stderr. Research-track evaluators can
    # emit tens of MB per step (observed 60–75 MB/step for grammar_fuzzing
    # once the fuzzer starts printing every generated SQL statement), which
    # compounds across 150 steps to multi-GB trajectories. We keep the head
    # + tail and a length signal; the agent's code + score are unaffected.
    _LOG_HEAD = 4096
    _LOG_TAIL = 4096
    logs_full_len = len(logs) if isinstance(logs, str) else 0
    logs_truncated = False
    if isinstance(logs, str) and logs_full_len > _LOG_HEAD + _LOG_TAIL + 128:
        logs = (
            logs[:_LOG_HEAD]
            + f"\n\n...[{logs_full_len - _LOG_HEAD - _LOG_TAIL} bytes truncated]...\n\n"
            + logs[-_LOG_TAIL:]
        )
        logs_truncated = True

    return {
        "score": float(score) if score is not None else 0.0,
        "success": success,
        "error": message if not success else None,
        "details": {
            "status": status_val,
            "logs": logs,
            "logs_full_len": logs_full_len,
            "logs_truncated": logs_truncated,
            "duration_seconds": duration,
            **metadata,
        },
    }


def evaluate_research(problem_id: str, code: str) -> dict:
    """
    Evaluate a research-track problem via Docker.

    Args:
        problem_id: Research problem identifier (e.g. "cant_be_late/high_availability_loose_deadline_large_overhead")
        code: Python solution code

    Returns:
        dict with keys: score (float), success (bool), error (Optional[str]), details (dict)
    """
    try:
        evaluator = get_frontier_cs_evaluator(backend="docker")
        result = evaluator.evaluate("research", problem_id, code, backend="docker")
        return _result_to_dict(result)
    except Exception as exc:
        return {
            "score": 0.0,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "details": {},
        }


def evaluate_algorithmic(problem_id: str, code: str) -> dict:
    """
    Evaluate an algorithmic-track problem via Docker judge.

    Args:
        problem_id: Algorithmic problem identifier (e.g. "0")
        code: C++ solution code

    Returns:
        dict with keys: score (float), success (bool), error (Optional[str]), details (dict)
    """
    try:
        evaluator = get_frontier_cs_evaluator(backend="docker")
        result = evaluator.evaluate("algorithmic", problem_id, code, backend="docker")
        return _result_to_dict(result)
    except Exception as exc:
        return {
            "score": 0.0,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "details": {},
        }
