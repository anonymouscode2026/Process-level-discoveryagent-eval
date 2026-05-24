"""
TaskAdapter — unified interface for evaluating solution code against a benchmark task.

Supports:
  - source=="skydiscover": invokes the benchmark's evaluator.py via a small harness
  - source=="frontier_cs_research": evaluates via Frontier-CS Docker runner (research track)
  - source=="frontier_cs_algorithmic": evaluates via Frontier-CS judge server (algorithmic track)
  - source=="frontier_cs": raises NotImplementedError
  - source=="algotune": raises NotImplementedError

Usage:
    from src.evaluation.task_adapter import TaskAdapter
    adapter = TaskAdapter(task_config)
    result = adapter.evaluate(code_string)
    # result = {"score": float, "success": bool, "error": Optional[str], "details": dict}
"""

from __future__ import annotations

import json
import os
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

_PKG_ROOT = Path(__file__).resolve().parents[2]


# Root of the SkyDiscover benchmark tree. SkyDiscover is an external repo not
# shipped with this package; clone it and point SKYDISCOVER_ROOT at its
# `benchmarks/` dir (or pass an explicit `skydiscover_root` to TaskAdapter).
_SKYDISCOVER_ROOT = Path(
    os.environ.get("SKYDISCOVER_ROOT", str(_PKG_ROOT / "external/skydiscover/benchmarks"))
)

# Root of the Frontier-CS repo (external — same pattern).
_FRONTIER_CS_ROOT = Path(
    os.environ.get("FRONTIER_CS_ROOT", str(_PKG_ROOT / "external/Frontier-CS"))
)


class TaskAdapter:
    """
    Wraps a task config entry and provides evaluate() / get_task_prompt() / get_initial_code().

    Args:
        task_config: A single task dict as loaded from configs/tasks.yaml (one list entry).
        skydiscover_root: Override the default SkyDiscover benchmarks root path.
    """

    def __init__(
        self,
        task_config: dict[str, Any],
        skydiscover_root: Optional[str] = None,
    ) -> None:
        self.name: str = task_config["name"]
        self.source: str = task_config["source"]
        self.task_id: str = task_config["task_id"]
        self.description: str = task_config.get("description", "")
        self.score_range: list[float] = task_config.get("score_range", [0.0, 1.0])
        self.evaluator_path: str = task_config.get("evaluator_path", "")
        self.domain: str = task_config.get("domain", "")
        self.requires_gpu: bool = bool(task_config.get("requires_gpu", False))
        self.expected_approaches: list[str] = task_config.get("expected_approaches", [])

        # Language the evaluator expects for submitted code. Frontier-CS
        # algorithmic problems are judged as C++; everything else in our
        # current registry is Python. Used by agent prompt templates so the
        # generation prompt asks for the right language.
        if self.source == "frontier_cs_algorithmic":
            self.code_language: str = "cpp"
        else:
            self.code_language = task_config.get("code_language", "python")

        self._skydiscover_root = Path(skydiscover_root) if skydiscover_root else _SKYDISCOVER_ROOT

        # Cache problem directory for frontier_cs_* sources
        self._frontier_cs_problem_dir: Optional[Path] = None
        if self.source == "frontier_cs_research":
            self._frontier_cs_problem_dir = (
                _FRONTIER_CS_ROOT / "research" / "problems" / self.task_id
            )
        elif self.source == "frontier_cs_algorithmic":
            self._frontier_cs_problem_dir = (
                _FRONTIER_CS_ROOT / "algorithmic" / "problems" / self.task_id
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, code: str) -> dict[str, Any]:
        """
        Evaluate a solution code string against the task.

        Returns:
            dict with keys:
                "score"   — float score (0.0 on failure)
                "success" — bool, True if evaluation ran without errors
                "error"   — Optional[str] error message
                "details" — dict with raw evaluator output and metrics
        """
        if self.source == "skydiscover":
            return self._evaluate_skydiscover(code)
        elif self.source == "frontier_cs_research":
            from src.evaluation.frontier_cs_backend import evaluate_research
            return evaluate_research(self.task_id, code)
        elif self.source == "frontier_cs_algorithmic":
            from src.evaluation.frontier_cs_backend import evaluate_algorithmic
            return evaluate_algorithmic(self.task_id, code)
        elif self.source == "frontier_cs":
            # Frontier-CS evaluator integration (requires external repo)
            raise NotImplementedError(
                "frontier_cs evaluation not yet implemented. "
                "Planned: use frontier_cs.SingleEvaluator from the frontier-cs-eval benchmark."
            )
        elif self.source == "algotune":
            # AlgoTune evaluator integration (requires external repo)
            raise NotImplementedError(
                "algotune evaluation not yet implemented."
            )
        else:
            raise ValueError(f"Unknown task source: {self.source!r}")

    def _skydiscover_problem_dir(self) -> Path:
        """
        Resolve the actual SkyDiscover problem directory for this task.

        Some tasks have README.md / config.yaml / initial_program.py directly
        under ``self.task_id`` (e.g. ``math/signal_processing``). Others nest
        them in a numbered difficulty-variant subdir (e.g.
        ``math/hexagon_packing/12/``, where the parent also contains ``11/``).
        Without resolving to the CORRECT subdir we'd either miss the files
        entirely (empty prompt) or — worse — silently point at the wrong
        difficulty variant and give the agent instructions that don't match
        what the evaluator will run.

        Probe order:
          1. **Derive from ``self.evaluator_path``** if set. For path
             ``math/hexagon_packing/12/evaluator/evaluator.py`` the problem
             directory is the parent of the evaluator's parent:
             ``.../math/hexagon_packing/12/``. This is the ONLY source of
             truth for which difficulty variant this task uses, because
             ``task_id`` alone can be ambiguous.
          2. Otherwise, try ``self.task_id`` itself if it contains any
             standard marker file.
          3. Otherwise, iterate immediate subdirectories looking for a marker
             (fallback for tasks without an evaluator_path set).
          4. Final fallback: the original ``task_id`` directory.
        """
        markers = ("initial_program.py", "README.md", "config.yaml", "reference.py")

        # 1. Derive from evaluator_path — the authoritative source.
        # Two layouts are in use:
        #   (a) <task>/<N>/evaluator/evaluator.py  (nested math: problem dir
        #       is parent.parent, e.g. hexagon_packing/12/)
        #   (b) <task>/evaluator.py                (flat gpu_mode: problem
        #       dir is parent, e.g. gpu_mode/vecadd/)
        # Try the DEEPER candidate first (parent), so gpu_mode tasks don't
        # accidentally resolve to the domain-level gpu_mode/README.md.
        if self.evaluator_path:
            eval_file = self._skydiscover_root / self.evaluator_path
            for candidate in (eval_file.parent, eval_file.parent.parent):
                try:
                    if candidate.exists() and any((candidate / m).exists() for m in markers):
                        return candidate
                except OSError:
                    continue

        base = self._skydiscover_root / self.task_id
        if not base.exists():
            return base

        # 2. task_id directory has markers itself
        if any((base / m).exists() for m in markers):
            return base

        # 3. iterate immediate subdirs for a marker
        try:
            for child in sorted(base.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    if any((child / m).exists() for m in markers):
                        return child
        except OSError:
            pass

        # 4. fallback
        return base

    def get_task_prompt(self) -> str:
        """
        Return a task description string for use as agent context.

        Returns the benchmark's README/config description AND, when available,
        the initial_program.py (or equivalent seed code) so the agent knows
        the expected function signature / interface the evaluator will call.
        Falls back to self.description if nothing else is present.
        """
        description: str = ""
        if self.source == "skydiscover":
            benchmark_dir = self._skydiscover_problem_dir()
            # Try README.md first
            readme = benchmark_dir / "README.md"
            if readme.exists():
                description = readme.read_text(encoding="utf-8")
            else:
                # Try config.yaml description
                config_yaml = benchmark_dir / "config.yaml"
                if config_yaml.exists():
                    try:
                        import yaml
                        with config_yaml.open() as fh:
                            cfg = yaml.safe_load(fh)
                        description = cfg.get("prompt", {}).get("system_message", "")
                    except Exception:
                        pass
        elif self.source == "frontier_cs_research":
            problem_dir = self._frontier_cs_problem_dir
            if problem_dir is not None:
                for name in ("readme", "README", "README.md", "problem.md"):
                    candidate = problem_dir / name
                    if candidate.exists():
                        description = candidate.read_text(encoding="utf-8")
                        break
        elif self.source == "frontier_cs_algorithmic":
            problem_dir = self._frontier_cs_problem_dir
            if problem_dir is not None:
                for name in ("statement.txt", "statement.md"):
                    candidate = problem_dir / name
                    if candidate.exists():
                        description = candidate.read_text(encoding="utf-8")
                        break

        if not description:
            description = self.description

        # Append the initial/seed program so the agent knows the expected
        # function signature and interface the evaluator will invoke.
        initial_code = self.get_initial_code()
        if initial_code:
            description = (
                description
                + "\n\n## Reference initial program (defines the interface the evaluator expects)\n"
                + "```python\n"
                + initial_code
                + "\n```\n"
                + "The evaluator imports your solution as a Python module and calls the "
                + "function(s) defined in this initial program. Your submission must preserve "
                + "the same top-level function name(s) and signature(s)."
            )
        return description

    def get_initial_code(self) -> Optional[str]:
        """
        Return the initial/seed program for this task, if available.

        Reads initial_program.py from the benchmark directory.
        Returns None if the file does not exist.
        """
        if self.source == "skydiscover":
            benchmark_dir = self._skydiscover_problem_dir()
            initial = benchmark_dir / "initial_program.py"
            if initial.exists():
                return initial.read_text(encoding="utf-8")
        elif self.source == "frontier_cs_research":
            problem_dir = self._frontier_cs_problem_dir
            if problem_dir is not None and problem_dir.exists():
                import glob as _glob
                for pattern in ("initial_*.py", "seed*.py", "baseline*.py"):
                    matches = sorted(_glob.glob(str(problem_dir / pattern)))
                    if matches:
                        return Path(matches[0]).read_text(encoding="utf-8")
        elif self.source == "frontier_cs_algorithmic":
            return "#include <bits/stdc++.h>\nusing namespace std;\nint main(){ return 0; }\n"
        return None

    # ------------------------------------------------------------------
    # SkyDiscover evaluation
    # ------------------------------------------------------------------

    def _evaluate_skydiscover(self, code: str) -> dict[str, Any]:
        """
        Evaluate code against a SkyDiscover benchmark.

        Copies code to a temp solution file, then invokes the benchmark's
        evaluator.py via a small inline harness:

            from evaluator import evaluate; print(json.dumps(evaluate(solution_path)))

        Falls back to using the current Python interpreter if the benchmark's
        uv environment is missing.
        """
        benchmark_dir = self._skydiscover_root / self.task_id
        evaluator_file = self._skydiscover_root / self.evaluator_path

        if not evaluator_file.exists():
            return {
                "score": 0.0,
                "success": False,
                "error": f"Evaluator not found: {evaluator_file}",
                "details": {},
            }

        evaluator_dir = evaluator_file.parent

        # Write solution to a temp file
        with tempfile.NamedTemporaryFile(
            suffix=".py", prefix="solution_", dir=str(evaluator_dir),
            delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            solution_path = tmp.name

        try:
            # Determine Python interpreter: prefer uv venv if present
            python_exe = self._resolve_python(benchmark_dir)

            # Use our unified harness (handles both wrapper-style and
            # library-style evaluator.py layouts). See
            # _skydiscover_harness.py for the rationale — invoking the
            # evaluator directly misses library-style tasks (gpu_mode).
            harness_path = Path(__file__).resolve().parent / "_skydiscover_harness.py"
            cmd = [python_exe, str(harness_path), str(evaluator_dir), solution_path]

            # GPU pinning: for tasks that use a GPU, give the subprocess
            # CUDA_VISIBLE_DEVICES = the per-process round-robin assignment so
            # parallel pilot workers don't all stack onto GPU 0.
            sub_env: Optional[dict] = None
            if self.requires_gpu:
                try:
                    from src.evaluation.gpu_pinning import gpu_env_dict

                    sub_env = gpu_env_dict()
                except Exception:
                    sub_env = None  # fall back to inheriting parent env

            try:
                # 600s per evaluation call. Heavy tasks (heilbronn_convex's
                # scipy optimization, matmul's JAX JIT, circle_packing_rect)
                # can exceed 300s on the first step. Most tasks finish in
                # under 30s; this is a worst-case ceiling.
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(evaluator_dir),
                    env=sub_env,
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()

                if proc.returncode != 0 and not stdout:
                    return {
                        "score": 0.0,
                        "success": False,
                        "error": f"Evaluator exited {proc.returncode}: {stderr}",
                        "details": {"stdout": stdout, "stderr": stderr},
                    }

                # Parse JSON output from evaluator
                score, raw = self._parse_evaluator_output(stdout)
                return {
                    "score": score,
                    "success": True,
                    "error": None,
                    "details": raw,
                }

            except subprocess.TimeoutExpired:
                return {
                    "score": 0.0,
                    "success": False,
                    "error": "Evaluator timed out after 600s",
                    "details": {},
                }

        finally:
            try:
                os.unlink(solution_path)
            except OSError:
                pass

    @staticmethod
    def _resolve_python(benchmark_dir: Path) -> str:
        """
        Return the Python executable to use for running the evaluator.

        Prefers the uv-managed venv inside the benchmark dir (.venv/bin/python).
        Falls back to sys.executable (current interpreter).
        """
        venv_python = benchmark_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        # Also check the skydiscover repo root for a shared venv
        repo_venv = benchmark_dir.parents[2] / ".venv" / "bin" / "python"
        if repo_venv.exists():
            return str(repo_venv)
        return sys.executable

    @staticmethod
    def _parse_evaluator_output(stdout: str) -> tuple[float, dict[str, Any]]:
        """
        Parse the evaluator's JSON stdout and extract a scalar score.

        Looks for "combined_score" first, then falls back to the first
        numeric value found in the dict.

        Returns:
            (score: float, raw_dict: dict)
        """
        if not stdout:
            return 0.0, {}

        # Find the last JSON object in stdout (evaluators may print log lines first)
        raw: dict[str, Any] = {}
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    raw = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if not raw:
            return 0.0, {"raw_stdout": stdout}

        # Extract score: prefer combined_score, then first numeric value
        if "combined_score" in raw:
            return float(raw["combined_score"]), raw

        # Try metrics sub-dict
        metrics = raw.get("metrics", {})
        if "combined_score" in metrics:
            return float(metrics["combined_score"]), raw

        # Fall back to first numeric value in the top-level dict
        for v in raw.values():
            try:
                return float(v), raw
            except (TypeError, ValueError):
                continue

        return 0.0, raw
