"""
Parallel pilot runner for the mentor-report experiment matrix.

Runs a fixed (task × level × runs) matrix of experiments in parallel using
ProcessPoolExecutor. Each worker is an independent Python process that
imports run_experiment and writes its trajectory JSON to the output dir.

Per-run stdout/stderr is redirected to a log file in --log-dir. Incremental
save is already enabled in DiscoveryAgent._record_step so crashed runs
retain partial trajectories.

Usage:
    PYTHONPATH=. python scripts/run.py \\
        --workers 4 --steps 100 --runs-per-cell 2 \\
        --out-dir data/trajectories \\
        --log-dir logs \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


# Per-worker global set by _init_worker. Integer index of this worker's
# pinned physical GPU, used to select the cross-process flock file so
# evaluations on the same GPU serialize. -1 means no GPU gating.
_MY_GPU_IDX: int = -1


def _assign_api_pool_slot(pool_size: int) -> int:
    """
    Cross-process atomic round-robin for API key / endpoint slots.

    Separate counter file from the GPU counter so the two round-robins
    don't interfere. Returns 0..pool_size-1 (deterministic, uniform
    distribution across workers).
    """
    import fcntl
    counter_path = Path(os.environ.get(
        "OMC_API_POOL_COUNTER",
        "/tmp/omc_api_pool_counter",
    ))
    try:
        counter_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(counter_path, os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(fd, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                content = f.read().strip()
                slot = int(content) if content else 0
                f.seek(0); f.truncate()
                f.write(str(slot + 1))
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return slot % max(pool_size, 1)
    except Exception:
        return os.getpid() % max(pool_size, 1)


def _apply_deepseek_pool_slot() -> None:
    """
    If DEEPSEEK_POOL_SIZE>1, pick a slot (1-based in the env vars) and copy
    DEEPSEEK_API_KEY_<k> / DEEPSEEK_BASE_URL_<k> onto the base names that
    LLMClient reads. No-op if pool is disabled.
    """
    try:
        pool_size = int(os.environ.get("DEEPSEEK_POOL_SIZE", "0"))
    except ValueError:
        pool_size = 0
    if pool_size < 2:
        return
    slot = _assign_api_pool_slot(pool_size)
    key = os.environ.get(f"DEEPSEEK_API_KEY_{slot+1}")
    url = os.environ.get(f"DEEPSEEK_BASE_URL_{slot+1}")
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
    if url:
        os.environ["DEEPSEEK_BASE_URL"] = url
    os.environ["OMC_DEEPSEEK_ASSIGNED_SLOT"] = str(slot + 1)


def _apply_gpt_oss_pool_slot() -> None:
    """
    If GPT_OSS_POOL_SIZE>1, pick a slot and copy GPT_OSS_API_KEY_<k> /
    GPT_OSS_BASE_URL_<k> onto the base names that LLMClient reads. Mirrors
    _apply_deepseek_pool_slot — same shape, different env-var prefix.
    Used to round-robin gpt-oss-120b across two Azure AI Inference endpoints.
    """
    try:
        pool_size = int(os.environ.get("GPT_OSS_POOL_SIZE", "0"))
    except ValueError:
        pool_size = 0
    if pool_size < 2:
        return
    slot = _assign_api_pool_slot(pool_size)
    key = os.environ.get(f"GPT_OSS_API_KEY_{slot+1}")
    url = os.environ.get(f"GPT_OSS_BASE_URL_{slot+1}")
    if key:
        os.environ["GPT_OSS_API_KEY"] = key
    if url:
        os.environ["GPT_OSS_BASE_URL"] = url
    os.environ["OMC_GPT_OSS_ASSIGNED_SLOT"] = str(slot + 1)


def _apply_kimi_pool_slot() -> None:
    """
    If KIMI_POOL_SIZE>1, pick a slot and copy KIMI_API_KEY_<k> /
    KIMI_BASE_URL_<k> onto the base names that LLMClient reads. Mirrors
    _apply_deepseek_pool_slot — equal-weight round-robin across the two
    Azure AI Inference endpoints (same hosts as gpt-oss but different
    deployment name).
    """
    try:
        pool_size = int(os.environ.get("KIMI_POOL_SIZE", "0"))
    except ValueError:
        pool_size = 0
    if pool_size < 2:
        return
    slot = _assign_api_pool_slot(pool_size)
    key = os.environ.get(f"KIMI_API_KEY_{slot+1}")
    url = os.environ.get(f"KIMI_BASE_URL_{slot+1}")
    if key:
        os.environ["KIMI_API_KEY"] = key
    if url:
        os.environ["KIMI_BASE_URL"] = url
    os.environ["OMC_KIMI_ASSIGNED_SLOT"] = str(slot + 1)


def _apply_gpt54_pool_slot() -> None:
    """
    GPT-5.4 can be deployed across multiple Azure OpenAI endpoints with
    potentially unequal quotas. Workers are distributed by weight:
    GPT54_POOL_WEIGHTS=`20,30,10` (for example) defines the per-slot bucket
    sizes; we pick a 0..(sum-1) idx via the same fcntl counter and bucket it
    into a 1-based slot. Then copy GPT54_BASE_URL_<k> / GPT54_API_KEY_<k>
    onto the base names LLMClient reads.
    """
    try:
        pool_size = int(os.environ.get("GPT54_POOL_SIZE", "0"))
    except ValueError:
        pool_size = 0
    if pool_size < 2:
        return
    weights_str = os.environ.get("GPT54_POOL_WEIGHTS", "")
    try:
        weights = [int(w) for w in weights_str.split(",") if w.strip()]
    except ValueError:
        weights = []
    if not weights or len(weights) != pool_size:
        # Fall back to even split if weights missing or malformed.
        weights = [1] * pool_size
    total = sum(weights)
    idx = _assign_api_pool_slot(total)
    cum = 0
    slot = 0
    for i, w in enumerate(weights):
        cum += w
        if idx < cum:
            slot = i
            break
    key = os.environ.get(f"GPT54_API_KEY_{slot+1}")
    url = os.environ.get(f"GPT54_BASE_URL_{slot+1}")
    if key:
        os.environ["GPT54_API_KEY"] = key
    if url:
        os.environ["GPT54_BASE_URL"] = url
    os.environ["OMC_GPT54_ASSIGNED_SLOT"] = str(slot + 1)


def _init_worker(n_gpu_locks: int) -> None:
    """
    ProcessPoolExecutor initializer.

    Receives the number of physical GPUs to gate on (0 disables GPU gating).
    Asks gpu_pinning which GPU this worker is pinned to and stores the
    index, then monkey-patches TaskAdapter.evaluate so any task with
    requires_gpu=True takes a cross-process flock on that GPU's lock file
    (see src.evaluation.gpu_pinning.gpu_eval_lock). Because the lock lives
    on the filesystem (shared under OMC_GPU_EVAL_LOCK_DIR), workers across
    DIFFERENT pilot processes — e.g. a concurrent Claude pilot and DeepSeek
    pilot — also serialize per-GPU, protecting timing-based scores.

    Also assigns the worker a DeepSeek API-pool slot if DEEPSEEK_POOL_SIZE>1.
    """
    # API pool first so LLMClient instances built lazily inside this worker
    # see the slot-local endpoint/key, not the parent's.
    _apply_deepseek_pool_slot()
    _apply_gpt_oss_pool_slot()
    _apply_gpt54_pool_slot()
    _apply_kimi_pool_slot()
    global _MY_GPU_IDX

    if n_gpu_locks <= 0:
        _MY_GPU_IDX = -1
        return

    # Resolve this worker's pinned GPU index. assign_gpu_index() is the
    # cross-process atomic counter that gpu_pinning uses, and it caches the
    # result so subsequent calls (including from gpu_env_dict during eval)
    # return the same value.
    try:
        from src.evaluation.gpu_pinning import assign_gpu_index
        _MY_GPU_IDX = assign_gpu_index() % n_gpu_locks
    except Exception:
        _MY_GPU_IDX = 0

    from src.evaluation.task_adapter import TaskAdapter
    from src.evaluation.gpu_pinning import gpu_eval_lock_any
    _orig_evaluate = TaskAdapter.evaluate
    _n_gpu_locks = n_gpu_locks  # capture for closure

    def _gated_evaluate(self, code):
        if getattr(self, "requires_gpu", False) and _MY_GPU_IDX >= 0:
            # Dynamic GPU selection: pick the first free GPU (LOCK_NB scan),
            # fall back to blocking on pinned if all busy. This breaks the
            # mod-N aliasing between stride-N schedule and mod-N pinning
            # (which pushed every GPU task to GPU 0).
            with gpu_eval_lock_any(_n_gpu_locks, _MY_GPU_IDX):
                return _orig_evaluate(self, code)
        return _orig_evaluate(self, code)

    TaskAdapter.evaluate = _gated_evaluate  # type: ignore[method-assign]


# The experiment matrix is fixed here — change it to modify the pilot.
DEFAULT_TASKS = [
    "signal_processing",   # math / signal — fast, ~0.499 baseline verified
    "txn_scheduling",      # systems — fast, ~2702 baseline verified
    "cloudcast",           # systems — fast, evaluator verified
    "llm_sql",             # cs_research — fast, evaluator verified
]
DEFAULT_LEVELS = ["L0", "L1"]
DEFAULT_MODEL = "claude-opus-4-6"


def run_one_job(
    task: str,
    model: str,
    level: str,
    step_budget: int,
    run_id: str,
    out_dir: str,
    log_dir: str,
    l0_rewrite_every: int,
) -> dict[str, Any]:
    """
    Worker entry point — runs a single experiment cell.

    Redirects stdout/stderr to a per-run log file so parent process
    aggregated output stays clean.

    `out_dir` and `log_dir` are the model-level roots (e.g.
    data/trajectories/<model> and logs/<model>). This worker nests a
    per-task subdirectory under each, so trajectories land at
    <out_dir>/<task>/<run_id>.json.
    """
    task_out_dir = Path(out_dir) / task
    task_log_dir = Path(log_dir) / task
    task_out_dir.mkdir(parents=True, exist_ok=True)
    task_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_log_dir / f"{run_id}.log"

    # Redirect BOTH stdout and stderr to the log file, line-buffered
    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stdout = log_fh
    sys.stderr = log_fh

    t0 = time.time()
    print(f"[worker pid={os.getpid()}] starting {task}/{run_id}")
    print(f"  task={task}  model={model}  level={level}  steps={step_budget}")

    try:
        from src.runner import run_experiment
        traj = run_experiment(
            task_name=task,
            model_name=model,
            agency_level=level,
            step_budget=step_budget,
            run_id=run_id,
            output_dir=str(task_out_dir),
            l0_rewrite_every=l0_rewrite_every,
        )
        elapsed = time.time() - t0
        best = float(traj.best_score) if traj.best_score != float("-inf") else None
        print(f"[worker pid={os.getpid()}] DONE {task}/{run_id} in {elapsed:.0f}s  "
              f"steps={len(traj.steps)}  best_score={best}")
        return {
            "run_id": run_id,
            "task": task,
            "level": level,
            "success": True,
            "steps_completed": len(traj.steps),
            "best_score": best,
            "tokens_used": traj.tokens_used,
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        traceback.print_exc()
        elapsed = time.time() - t0
        return {
            "run_id": run_id,
            "task": task,
            "level": level,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": elapsed,
        }
    finally:
        log_fh.close()


def _maybe_skip(task: str, run_id: str, out_dir: str, resume: bool) -> bool:
    """
    Skip only when the trajectory is *truly finished*. A trajectory is
    considered finished iff it has `finished_at` set (agent exited its
    run loop normally) OR it reached the declared step_budget.

    Partial trajectories — from worker crashes, tmux kills, or API rate-
    limit abandonment — are NOT skipped, because re-running doesn't
    resume them; we need a fresh run to reach the full budget.
    """
    if not resume:
        return False
    out_json = Path(out_dir) / task / f"{run_id}.json"
    if not out_json.exists():
        return False
    try:
        data = json.loads(out_json.read_text())
        steps = data.get("steps") or []
        nsteps = len(steps)
        step_budget = int((data.get("metadata") or {}).get("step_budget", 0))
        finished = bool(data.get("finished_at"))
        # A trajectory is only "done" if it finished cleanly AND reached the
        # declared step budget. `finished_at` alone is insufficient because
        # the agent's try/finally can set it even when an outer SIGKILL
        # interrupts the run mid-way, leaving a trajectory that looks
        # finished but only has e.g. 37/150 steps (observed after rate-limit
        # kills). Require both signals for a true skip.
        if finished and step_budget and nsteps >= step_budget:
            print(f"  [skip] {task}/{run_id}  ({nsteps}/{step_budget} steps, finished={finished})")
            return True
        # L0/L0f early-exit (consecutive_failures >= threshold): a real agent
        # behavior we want to preserve, NOT a partial run. The agent stamps
        # metadata["early_exit_reason"] when this fires.
        early_exit_reason = (data.get("metadata") or {}).get("early_exit_reason")
        if finished and early_exit_reason:
            print(f"  [skip-early-exit] {task}/{run_id}  ({nsteps}/{step_budget or '?'} steps, "
                  f"reason={early_exit_reason})")
            return True
        # Partial, not skipping — we will overwrite it on the fresh run.
        print(f"  [redo] {task}/{run_id}  ({nsteps}/{step_budget or '?'} steps, finished={finished})")
    except Exception:
        print(f"  [retry] {task}/{run_id}  (corrupt file)")
    return False


def build_jobs(
    tasks: list[str],
    levels: list[str],
    model: str,
    runs_per_cell: int,
    step_budget: int,
    out_dir: str,
    log_dir: str,
    l0_rewrite_every: int,
    resume: bool,
) -> list[tuple]:
    """Enumerate the full (task × level × run) matrix in declaration order."""
    jobs = []
    skipped = 0
    for task in tasks:
        for level in levels:
            for r in range(1, runs_per_cell + 1):
                run_id = f"{level}__run{r}"
                if _maybe_skip(task, run_id, out_dir, resume):
                    skipped += 1
                    continue
                jobs.append((task, model, level, step_budget, run_id, out_dir, log_dir, l0_rewrite_every))
    print(f"  => {len(jobs)} jobs to run, {skipped} skipped")
    return jobs


def build_jobs_from_order(
    order_file: str,
    model: str,
    step_budget: int,
    out_dir: str,
    log_dir: str,
    l0_rewrite_every: int,
    resume: bool,
) -> list[tuple]:
    """
    Read an explicit run order from a text file. One run per line, format:

        <task_name> <level> <run_index>

    Lines starting with '#' or blank lines are ignored. Existing finished
    runs are skipped iff resume=True. The order in the file is preserved
    so the worker pool pulls jobs in the requested sequence — this is how
    we keep GPU-bound runs spread out instead of stacking on the same GPU.
    """
    jobs = []
    skipped = 0
    with open(order_file) as fh:
        for raw in fh:
            # Drop any inline comment, then trim
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"  [warn] malformed line in {order_file}: {raw!r}")
                continue
            task, level, run_str = parts
            try:
                r = int(run_str)
            except ValueError:
                print(f"  [warn] non-int run index in {order_file}: {raw!r}")
                continue
            run_id = f"{level}__run{r}"
            if _maybe_skip(task, run_id, out_dir, resume):
                skipped += 1
                continue
            jobs.append((task, model, level, step_budget, run_id, out_dir, log_dir, l0_rewrite_every))
    print(f"  => {len(jobs)} jobs to run from {order_file}, {skipped} skipped")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--steps", type=int, default=100,
                        help="step_budget per run")
    parser.add_argument("--runs-per-cell", type=int, default=2,
                        help="Runs per (task, level) cell")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="Task names from configs/tasks.yaml")
    parser.add_argument("--levels", nargs="+", default=DEFAULT_LEVELS,
                        help="Agency levels to run")
    parser.add_argument("--l0-rewrite-every", type=int, default=5,
                        help="L0 forced REWRITE cadence")
    parser.add_argument("--out-dir", required=True,
                        help="Directory for trajectory JSON files")
    parser.add_argument("--log-dir", required=True,
                        help="Directory for per-run log files + summary")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output JSON already exists")
    parser.add_argument("--job-order", default=None,
                        help="Optional path to a text file with explicit "
                             "(task level run_idx) order, one per line. "
                             "Overrides --tasks/--levels/--runs-per-cell "
                             "for the actual job set.")
    parser.add_argument("--gpu-semaphore", type=int, default=4,
                        help="Max concurrent GPU evaluations across all "
                             "workers (default: 4 = number of A100s on this "
                             "machine). Set to 0 to disable gating.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Discovery-agent parallel pilot driver")
    print("=" * 70)
    print(f"  model:        {args.model}")
    print(f"  tasks:        {args.tasks}")
    print(f"  levels:       {args.levels}")
    print(f"  runs/cell:    {args.runs_per_cell}")
    print(f"  step_budget:  {args.steps}")
    print(f"  workers:      {args.workers}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  log_dir:      {args.log_dir}")
    print(f"  job_order:    {args.job_order or '(cartesian product)'}")
    print(f"  gpu_sem:      {args.gpu_semaphore}")
    print()

    if args.job_order:
        jobs = build_jobs_from_order(
            order_file=args.job_order,
            model=args.model,
            step_budget=args.steps,
            out_dir=args.out_dir,
            log_dir=args.log_dir,
            l0_rewrite_every=args.l0_rewrite_every,
            resume=args.resume,
        )
    else:
        jobs = build_jobs(
            tasks=args.tasks,
            levels=args.levels,
            model=args.model,
            runs_per_cell=args.runs_per_cell,
            step_budget=args.steps,
            out_dir=args.out_dir,
            log_dir=args.log_dir,
            l0_rewrite_every=args.l0_rewrite_every,
            resume=args.resume,
        )

    if not jobs:
        print("Nothing to run.")
        return 0

    print()
    print(f"=== Launching {len(jobs)} jobs with {args.workers} parallel workers ===")
    print()

    results: list[dict[str, Any]] = []
    wall_t0 = time.time()

    # Per-GPU cross-process file locks. We resolve how many physical GPUs
    # to gate on and pass the COUNT to each worker; workers then use a
    # flock-based lock (see gpu_pinning.gpu_eval_lock) on their pinned
    # GPU index. Because the lock lives on the filesystem under
    # OMC_GPU_EVAL_LOCK_DIR, it serializes GPU evals across *any* pilots
    # running on the host — e.g. a concurrent Claude pilot and DeepSeek
    # pilot share the same per-GPU lock, which is required for the timing-
    # based GPU task scores to stay uncontaminated.
    n_gpu_locks = 0
    if args.gpu_semaphore and args.gpu_semaphore > 0:
        try:
            from src.evaluation.gpu_pinning import visible_gpu_count
            # Upper-bounded by --gpu-semaphore. Pass --gpu-semaphore=1 to
            # gate all GPU evals through a single lock (useful when only
            # one physical GPU is actually available/clean to us, e.g.
            # others occupy the rest).
            n_gpu_locks = min(visible_gpu_count() or args.gpu_semaphore,
                              args.gpu_semaphore)
        except Exception:
            n_gpu_locks = args.gpu_semaphore
        lock_dir = os.environ.get("OMC_GPU_EVAL_LOCK_DIR", "/tmp/omc_gpu_eval_locks")
        print(f"  GPU locks:    {n_gpu_locks} × flock (shared dir: {lock_dir})")

    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(n_gpu_locks,),
        ) as pool:
            futures = {
                pool.submit(run_one_job, *job): f"{job[0]}/{job[4]}"  # task/run_id
                for job in jobs
            }
            completed = 0
            for fut in as_completed(futures):
                run_id = futures[fut]
                completed += 1
                try:
                    r = fut.result()
                    results.append(r)
                    if r["success"]:
                        bs = r.get("best_score")
                        bs_str = f"{bs:.4f}" if isinstance(bs, (int, float)) else str(bs)
                        print(f"  [{completed:2d}/{len(jobs)}] OK   {run_id:55s}  "
                              f"steps={r['steps_completed']:3d}  best={bs_str}  "
                              f"elapsed={r['elapsed_sec']:.0f}s  tokens={r.get('tokens_used',0):,}")
                    else:
                        print(f"  [{completed:2d}/{len(jobs)}] FAIL {run_id:55s}  "
                              f"error={r.get('error')[:80]}  elapsed={r['elapsed_sec']:.0f}s")
                except Exception as exc:
                    print(f"  [{completed:2d}/{len(jobs)}] EXC  {run_id}: {exc}")
                    results.append({
                        "run_id": run_id, "success": False,
                        "error": f"unhandled: {exc}",
                    })
    except KeyboardInterrupt:
        print("\n!!! KeyboardInterrupt — shutting down pool (current runs may complete) !!!")

    wall_elapsed = time.time() - wall_t0
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print()
    print("=" * 70)
    print(f"Done: {ok} ok, {fail} failed, {len(jobs)-len(results)} unaccounted for")
    print(f"Wall time: {wall_elapsed/60:.1f} min  ({wall_elapsed/3600:.2f} h)")
    print("=" * 70)

    # Save summary JSON
    summary_path = Path(args.log_dir) / "summary.json"
    summary = {
        "model": args.model,
        "tasks": args.tasks,
        "levels": args.levels,
        "runs_per_cell": args.runs_per_cell,
        "step_budget": args.steps,
        "workers": args.workers,
        "wall_time_sec": wall_elapsed,
        "n_jobs": len(jobs),
        "n_ok": ok,
        "n_fail": fail,
        "results": results,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written to: {summary_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
