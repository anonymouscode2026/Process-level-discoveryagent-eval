"""
Unified subprocess harness for invoking SkyDiscover evaluators.

Background
----------
SkyDiscover has TWO evaluator styles and our pre-fix adapter only handled one:

  (A) "wrapper" style — the evaluator.py defines ``def evaluate(path): ...``
      at module top level and ends with ``if __name__ == "__main__":
      from wrapper import run; run(evaluate)``. Running ``python evaluator.py
      <path>`` calls ``evaluate()`` via the wrapper, which redirects
      evaluator-internal prints to stderr and emits clean JSON on stdout.
      Used by: signal_processing, matmul, hexagon_packing, heilbronn_convex,
      circle_packing_rect, minimizing_max_min_dist, erdos_min_overlap,
      ADRS/cloudcast, ADRS/llm_sql, ADRS/txn_scheduling.

  (B) "library" style — the evaluator.py is a thin module that only
      manipulates sys.path and re-exports ``evaluate`` from ``shared_eval``;
      it has NO ``__main__`` block. Running ``python evaluator.py <path>``
      only runs the top-level imports and exits with empty stdout, so the
      pre-fix adapter thought the run "succeeded" with score 0.
      Used by: gpu_mode/vecadd, gpu_mode/grayscale, gpu_mode/trimul.

This harness unifies both styles by always calling ``evaluate()`` in Python,
normalizing the return value to a plain dict, and printing it as JSON on the
last line of stdout so the adapter's parser can pick it up reliably.

Usage (from the adapter subprocess.run call):
    python src/evaluation/_skydiscover_harness.py <task_dir> <solution_path>

The harness:
  1. Inserts <task_dir> at sys.path[0] so ``from evaluator import evaluate``
     finds the local module.
  2. Imports ``evaluate`` from the evaluator module.
  3. Calls ``evaluate(solution_path)`` with stdout/stderr captured so the
     evaluator's own logging doesn't pollute our JSON line.
  4. Normalizes the return value: handles plain dicts (wrapper style) AND
     ``EvaluationResult`` dataclass instances (library style).
  5. Writes captured prints to stderr and the normalized JSON to stdout.

Exit code is always 0 on a reachable evaluate() — it's up to the adapter to
inspect the JSON's ``combined_score`` / ``success`` fields. A non-zero exit
indicates an import-time or infrastructure failure.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback


def _normalize_result(result) -> dict:
    """Convert any evaluator return value into a plain dict with ``combined_score``."""
    if isinstance(result, dict):
        return dict(result)

    # EvaluationResult-style dataclass (gpu_mode / shared_eval).
    if hasattr(result, "metrics") or hasattr(result, "artifacts"):
        out = {}
        metrics = getattr(result, "metrics", None)
        if metrics:
            out.update(dict(metrics))
        artifacts = getattr(result, "artifacts", None)
        if artifacts:
            # Artifacts can include non-numeric data (error strings, stack
            # traces). Keep them under a nested key for debuggability.
            try:
                out["artifacts"] = dict(artifacts)
            except Exception:
                out["artifacts"] = {"_repr": repr(artifacts)}
        # Some EvaluationResult dataclasses expose additional named fields.
        for attr in ("score", "combined_score", "status", "error", "duration_seconds"):
            val = getattr(result, attr, None)
            if val is not None and attr not in out:
                out[attr] = val
        if "combined_score" not in out and "score" in out:
            out["combined_score"] = out["score"]
        return out

    # Fallback: wrap whatever it was
    return {"combined_score": 0.0, "_raw_repr": repr(result)[:500]}


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: _skydiscover_harness.py <task_dir> <solution_path>\n"
        )
        return 2

    task_dir = sys.argv[1]
    solution_path = sys.argv[2]

    if not os.path.isdir(task_dir):
        sys.stderr.write(f"task_dir does not exist: {task_dir}\n")
        return 3

    # Put the task dir first on sys.path so ``from evaluator import evaluate``
    # picks up the local evaluator.py (not some shadowed module name).
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)

    try:
        from evaluator import evaluate  # type: ignore
    except Exception as exc:
        tb = traceback.format_exc()
        sys.stderr.write(f"harness: import evaluator failed:\n{tb}\n")
        # Emit a JSON so the adapter parser sees a structured failure
        print(json.dumps({
            "combined_score": 0.0,
            "success": False,
            "error": f"import evaluator failed: {type(exc).__name__}: {exc}",
            "failure_stage": "import",
        }))
        return 4

    # Capture any prints made during evaluate() so they don't contaminate
    # the JSON line we emit at the end.
    capture_buf = io.StringIO()
    result = None
    exc_info = None
    try:
        with contextlib.redirect_stdout(capture_buf), contextlib.redirect_stderr(capture_buf):
            result = evaluate(solution_path)
    except BaseException as exc:  # include KeyboardInterrupt / SystemExit
        exc_info = (type(exc).__name__, str(exc), traceback.format_exc())

    captured = capture_buf.getvalue()
    if captured:
        # Trim to last 8 KB to bound our own log noise
        sys.stderr.write(captured[-8192:])

    if exc_info is not None:
        cls_name, msg, tb = exc_info
        payload = {
            "combined_score": 0.0,
            "success": False,
            "error": f"evaluate() raised {cls_name}: {msg}",
            "traceback": tb[-4000:],
            "failure_stage": "evaluate",
        }
        print(json.dumps(payload, default=str))
        return 0  # still exit 0; adapter will read success=False

    normalized = _normalize_result(result)
    if "combined_score" not in normalized:
        normalized["combined_score"] = 0.0
    print(json.dumps(normalized, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
