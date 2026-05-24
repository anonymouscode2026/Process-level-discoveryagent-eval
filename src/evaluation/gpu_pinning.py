"""
GPU device pinning for parallel discovery-agent workers.

When N parallel workers each launch GPU subprocesses (or Docker containers
with --gpus all), they all default to physical GPU 0 unless we pin them.
This module provides a deterministic per-process assignment so 6 workers
spread evenly across e.g. 4 A100s.

Two layers handled:

1. Subprocess + torch (SkyDiscover gpu_mode tasks):
   `task_adapter._evaluate_skydiscover` builds the subprocess env via
   `gpu_env_dict()` so the child process sees only the assigned GPU as
   `cuda:0` via the standard CUDA_VISIBLE_DEVICES mechanism.

2. Docker + nvidia-container-toolkit (Frontier-CS research tasks):
   `frontier_cs_backend.py` monkey-patches `ResearchDockerRunner._run_docker`
   to use `--gpus device=N` (in place of `--gpus all`). The container then
   sees only one physical GPU, mapped to its internal `cuda:0`.

Per-process assignment is cached after first call, so the same worker always
uses the same GPU for the entire run. Assignment is `os.getpid() % N` by
default; set the env var `OMC_GPU_PIN_INDEX` to force a specific index, or
`OMC_DISABLE_GPU_PINNING=1` to disable pinning entirely.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

# Caches (per-process)
_GPU_COUNT_CACHE: Optional[int] = None
_PROCESS_GPU_INDEX: Optional[int] = None

# Transient override for the current eval — set by gpu_eval_lock_any when a
# dynamic GPU is picked because the statically-pinned one was busy. Reset to
# None after the eval. Respected by current_gpu_index() → gpu_env_dict() and
# docker_gpu_arg() so the subprocess runs on the GPU we actually hold the
# lock for, not the pinned one.
_OVERRIDE_GPU_INDEX: Optional[int] = None

# Default location for the cross-process counter file. Override with the
# OMC_GPU_PIN_COUNTER env var (e.g. one counter per pilot timestamp).
_DEFAULT_COUNTER_PATH = "/tmp/omc_gpu_pin_counter"

# Default directory for per-GPU cross-process evaluation locks. Shared by
# ALL pilot processes on the host so that multiple concurrent pilots (e.g.
# one Claude + one DeepSeek) serialize GPU evaluations per physical GPU.
# Override with OMC_GPU_EVAL_LOCK_DIR.
_DEFAULT_EVAL_LOCK_DIR = "/tmp/omc_gpu_eval_locks"

# Username running the eval pipeline. The `_externally_busy_gpus` helper uses
# this to distinguish our own GPU processes from third-party processes when
# detecting contention on a shared host. Defaults to empty (treat all owners
# as external) — override with the EVAL_USER env var on multi-tenant hosts.
_EVAL_USER = os.environ.get("EVAL_USER", "")


def _disabled() -> bool:
    return os.environ.get("OMC_DISABLE_GPU_PINNING", "").lower() in {"1", "true", "yes"}


def visible_gpu_count() -> int:
    """
    Return the number of GPUs visible via `nvidia-smi`. Cached after first call.

    Returns 0 if nvidia-smi is missing, errors, or reports no devices.
    """
    global _GPU_COUNT_CACHE
    if _GPU_COUNT_CACHE is not None:
        return _GPU_COUNT_CACHE
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
            _GPU_COUNT_CACHE = len(lines)
        else:
            _GPU_COUNT_CACHE = 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _GPU_COUNT_CACHE = 0
    return _GPU_COUNT_CACHE


def _claim_counter_slot() -> Optional[int]:
    """
    Atomically claim the next slot from a file-backed cross-process counter.

    Returns the integer slot index (0-based) or None on any failure.
    Uses fcntl.flock so multiple worker processes hitting the same file
    serialize cleanly. The counter is monotonic; modulo by GPU count is
    applied by the caller.
    """
    counter_path = Path(os.environ.get("OMC_GPU_PIN_COUNTER", _DEFAULT_COUNTER_PATH))
    try:
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        # Open for read+write, create if missing
        fd = os.open(counter_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                content = f.read().strip()
                slot = int(content) if content else 0
                f.seek(0)
                f.truncate()
                f.write(str(slot + 1))
                f.flush()
                os.fsync(f.fileno())
                return slot
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        return None


def assign_gpu_index() -> int:
    """
    Return the GPU index this PROCESS should use (deterministic round-robin).

    Cached after first call so the same worker always returns the same index.
    Resolution order:
      1. `OMC_GPU_PIN_INDEX` env var if set (explicit per-process override).
      2. File-backed atomic counter at `OMC_GPU_PIN_COUNTER` (default
         /tmp/omc_gpu_pin_counter), modulo visible GPU count. This guarantees
         uniform distribution across N workers regardless of OS pid layout.
      3. `os.getpid() % visible_gpu_count()` (fallback if the counter file
         cannot be opened, e.g. read-only filesystem).
      4. 0 if no GPUs are visible.
    """
    global _PROCESS_GPU_INDEX
    if _PROCESS_GPU_INDEX is not None:
        return _PROCESS_GPU_INDEX

    override = os.environ.get("OMC_GPU_PIN_INDEX")
    if override is not None:
        try:
            _PROCESS_GPU_INDEX = int(override)
            return _PROCESS_GPU_INDEX
        except ValueError:
            pass

    n = visible_gpu_count()
    if n <= 0:
        _PROCESS_GPU_INDEX = 0
        return _PROCESS_GPU_INDEX

    slot = _claim_counter_slot()
    if slot is not None:
        _PROCESS_GPU_INDEX = slot % n
    else:
        # Fallback: pid-based modulo
        _PROCESS_GPU_INDEX = os.getpid() % n
    return _PROCESS_GPU_INDEX


def current_gpu_index() -> int:
    """
    Return the GPU index to use for the current eval call.

    Resolution:
      1. If `_OVERRIDE_GPU_INDEX` is set (by gpu_eval_lock_any because we
         dynamically picked a free GPU instead of the statically-pinned
         one), return it.
      2. Otherwise, return the cached per-process assignment.

    Use this — NOT assign_gpu_index() — wherever a subprocess or container
    needs a GPU index: it ensures the env matches the GPU we hold the lock
    for, not the one we were pinned to at init.
    """
    if _OVERRIDE_GPU_INDEX is not None:
        return _OVERRIDE_GPU_INDEX
    return assign_gpu_index()


def set_override_gpu_index(idx: Optional[int]) -> None:
    """Set or clear the transient GPU override for this process."""
    global _OVERRIDE_GPU_INDEX
    _OVERRIDE_GPU_INDEX = idx


def gpu_env_dict(base: Optional[dict] = None) -> dict:
    """
    Return an env dict with `CUDA_VISIBLE_DEVICES` set to the GPU we are
    about to evaluate on (override if set, else pinned).
    """
    env = dict(base) if base is not None else os.environ.copy()
    if _disabled():
        return env
    if visible_gpu_count() <= 0:
        return env
    env["CUDA_VISIBLE_DEVICES"] = str(current_gpu_index())
    return env


def docker_gpu_arg() -> str:
    """
    Return the value to pass after `--gpus` when launching a Docker container,
    honoring the transient override.
    """
    if _disabled():
        return "all"
    if visible_gpu_count() <= 0:
        return "all"
    return f"device={current_gpu_index()}"


def _externally_busy_gpus(threshold_mb: int = 5000) -> set:
    """
    Return the set of GPU indices currently used by EXTERNAL (non-EVAL_USER)
    processes for >= threshold_mb of memory, EXCLUDING our own Frontier-CS
    Docker evaluator (which runs as root inside the container with the
    distinctive `evaluator.py --solution-path /work/execution_env` cmdline).

    Used by `gpu_eval_lock_any` to skip GPUs being shared with other users
    so timing-sensitive evaluators (gemm/cross_entropy/trimul/...) don't
    silently get a polluted score from SM contention.

    Cached result is invalidated after _BUSY_TTL_SEC so repeated lock
    acquisitions in the same eval don't spam nvidia-smi.
    """
    global _BUSY_CACHE, _BUSY_CACHE_TS
    now = time.monotonic()
    if _BUSY_CACHE is not None and (now - _BUSY_CACHE_TS) < _BUSY_TTL_SEC:
        return _BUSY_CACHE

    busy: set = set()
    try:
        import subprocess
        # Map gpu_uuid -> idx
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        uuid_to_idx = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = int(parts[0])

        # Query running compute apps
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
            try:
                pid = int(pid_s); mem = int(mem_s)
            except ValueError:
                continue
            if mem < threshold_mb:
                continue
            gpu_idx = uuid_to_idx.get(uuid)
            if gpu_idx is None:
                continue
            # Resolve owner + cmdline
            try:
                ps = subprocess.run(
                    ["ps", "-o", "user=,args=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=2,
                ).stdout.strip()
            except Exception:
                continue
            if not ps:
                continue
            owner, _, cmdline = ps.partition(" ")
            if _EVAL_USER and owner == _EVAL_USER:
                continue
            # Our own Frontier-CS Docker eval runs as root with this signature
            if "evaluator.py --solution-path /work/execution_env" in cmdline:
                continue
            busy.add(gpu_idx)
    except Exception:
        # If anything fails (nvidia-smi missing, timeout, etc.) just return
        # empty set — we'd rather risk contention than block all locks.
        busy = set()

    _BUSY_CACHE = busy
    _BUSY_CACHE_TS = now
    return busy


# Cache for _externally_busy_gpus (refreshed every _BUSY_TTL_SEC).
_BUSY_CACHE: Optional[set] = None
_BUSY_CACHE_TS: float = 0.0
_BUSY_TTL_SEC: float = 5.0


@contextlib.contextmanager
def gpu_eval_lock_any(n_gpus: int, pinned: int):
    """
    Cross-process GPU eval lock with DYNAMIC GPU selection.

    Tries LOCK_EX|LOCK_NB on each of the n_gpus lock files in shuffled order.
    The first lock acquired determines which physical GPU this eval runs on.
    If ALL are held, falls back to blocking on `pinned` (the worker's
    originally-pinned GPU).

    Yields the picked GPU index and sets the module-level override so that
    subsequent gpu_env_dict() / docker_gpu_arg() calls in this process
    reflect the picked GPU, not the pinned one.

    This breaks the worker-pin ↔ job-schedule aliasing: with stride-4 GPU
    task interleave AND mod-4 pinning, the pure pinned-GPU lock sends every
    GPU task to GPU 0, leaving GPUs 1-3 idle. With any-GPU selection, the
    first worker to need a GPU grabs whichever is free — full 4× utilization.
    """
    import random
    if n_gpus <= 0:
        yield -1
        return
    lock_dir = Path(os.environ.get("OMC_GPU_EVAL_LOCK_DIR", _DEFAULT_EVAL_LOCK_DIR))
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield -1
        return
    order = list(range(n_gpus))
    random.shuffle(order)

    # Skip GPUs currently used by external (non-EVAL_USER, non-our-Docker-eval)
    # processes — their compute steals SMs and silently corrupts timing-based
    # evaluator scores (gemm_optimization, cross_entropy, trimul, etc.).
    busy_external = _externally_busy_gpus()

    # Pass 1: non-blocking scan, take the first CLEAN free GPU.
    for idx in order:
        if idx in busy_external:
            continue
        lock_path = lock_dir / f"gpu_{idx}.lock"
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        # Got a free CLEAN GPU.
        set_override_gpu_index(idx)
        try:
            yield idx
        finally:
            set_override_gpu_index(None)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return

    # Pass 1b: all CLEAN GPUs busy by our own workers — try the externally-
    # contaminated ones too rather than blocking. (Better to score-contaminate
    # one eval than to deadlock the whole pilot.)
    for idx in order:
        if idx not in busy_external:
            continue
        lock_path = lock_dir / f"gpu_{idx}.lock"
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        set_override_gpu_index(idx)
        try:
            yield idx
        finally:
            set_override_gpu_index(None)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return

    # Pass 2: every GPU lock held by another of our workers — block on pinned.
    lock_path = lock_dir / f"gpu_{pinned}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    set_override_gpu_index(pinned)
    try:
        yield pinned
    finally:
        set_override_gpu_index(None)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextlib.contextmanager
def gpu_eval_lock(gpu_index: int):
    """
    Cross-process exclusive lock for GPU `gpu_index` evaluations.

    Uses fcntl.flock on a per-GPU lock file under OMC_GPU_EVAL_LOCK_DIR
    (default /tmp/omc_gpu_eval_locks). All processes on the same host that
    enter this CM for the same index will serialize — including processes
    in different ProcessPoolExecutor pools / different pilot invocations.
    This protects timing-based evaluators whose score is corrupted by
    concurrent GPU workloads on the same physical device.

    A no-op (yields immediately) if `gpu_index < 0` or locking fails.
    """
    if gpu_index < 0:
        yield
        return
    lock_dir = Path(os.environ.get("OMC_GPU_EVAL_LOCK_DIR", _DEFAULT_EVAL_LOCK_DIR))
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield
        return
    lock_path = lock_dir / f"gpu_{gpu_index}.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        yield
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def describe() -> dict:
    """Return a small dict describing the pinning state for logging/debug."""
    return {
        "disabled": _disabled(),
        "visible_gpus": visible_gpu_count(),
        "assigned_index": assign_gpu_index(),
        "pid": os.getpid(),
    }
