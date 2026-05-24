"""
Level0fAgent — fixed-schedule agent with single-step environment feedback.

Sits between L0 (zero context) and L1 (history window + LLM-chosen mode) on
exactly ONE axis: whether the agent sees what the evaluator said about its
most recent attempt. Holds the other L0 properties constant:

  - No multi-step history. Only the IMMEDIATELY PREVIOUS step is surfaced.
  - No meta-reasoning. REFINE vs REWRITE is still a fixed external schedule,
    NOT model-chosen. DecisionStep is synthesized exactly like L0, with
    action_type set to "REFINE_FORCED" / "REWRITE_FORCED".

Purpose: isolate the marginal value of environment feedback from the marginal
value of history accumulation and meta-reasoning. Comparing:
  - L0  vs L0f: effect of seeing "my last attempt produced score X / errored
                with message Y" without any other change.
  - L0f vs L1:  effect of history window + LLM-chosen mode on top of env
                feedback.

The feedback block injected into context_text contains:
  - the action_type the last step ran (REFINE_FORCED / REWRITE_FORCED)
  - its score, score_delta, and is_new_best flag
  - evaluation success/failure and (if failed) the error message
  - a truncated view of eval_result.details so metric-level hints (e.g.
    matmul "decomposition does not match") reach the model

Nothing else — no earlier history, no summary, no plateau detection.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.agent.base import DiscoveryAgent, truncate_error_hybrid
from src.trajectory.schemas import DecisionStep, Trajectory, TrajectoryStep

_MAX_CONSECUTIVE_FAILURES = 5


class Level0fAgent(DiscoveryAgent):
    """
    Fixed-schedule agent with one-step environment feedback.

    Args:
        rewrite_every: REWRITE cadence (same semantics as Level0Agent).
    """

    def __init__(self, *args, rewrite_every: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if rewrite_every < 1:
            raise ValueError(f"rewrite_every must be >= 1, got {rewrite_every}")
        self.rewrite_every = rewrite_every
        self.trajectory.metadata["l0f_rewrite_every"] = rewrite_every

    def run(self) -> Trajectory:
        consecutive_failures = 0

        while (
            self._budget_remaining() > 0
            and consecutive_failures < _MAX_CONSECUTIVE_FAILURES
        ):
            step_number = len(self.trajectory.steps) + 1

            is_rewrite = (step_number % self.rewrite_every == 0)
            if not self.trajectory.best_code:
                is_rewrite = True
            mode = "REWRITE" if is_rewrite else "REFINE"
            forced_action = f"{mode}_FORCED"

            decision = DecisionStep(
                level="L0f",
                prompt_tokens=0,
                reasoning=(
                    f"L0f fixed schedule (rewrite_every={self.rewrite_every}): "
                    f"step {step_number} is {mode} (external, not model-chosen)."
                ),
                action_type=forced_action,
            )

            context_text = _build_context(
                step_number=step_number,
                best_score=self.trajectory.best_score,
                mode=mode,
                last_step=self.trajectory.steps[-1] if self.trajectory.steps else None,
            )

            generation = self._generation_query(mode, context_text)
            eval_result = self.task.evaluate(generation.code)

            if not eval_result.get("success", False):
                prev_best = self.trajectory.best_score
                if prev_best == float("-inf"):
                    prev_best = 0.0
                failure_result = {
                    "score": prev_best,
                    "success": False,
                    "error": eval_result.get("error", "unknown error"),
                    "details": eval_result.get("details", {}),
                    "execution_success": False,
                }
                self._record_step(
                    decision=decision,
                    generation=generation,
                    eval_result=failure_result,
                    prompt_summary=f"L0f {forced_action}: last-step env feedback",
                )
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                self._record_step(
                    decision=decision,
                    generation=generation,
                    eval_result=eval_result,
                    prompt_summary=f"L0f {forced_action}: last-step env feedback",
                )

        # Stamp metadata so post-hoc analysis and --resume can distinguish
        # "agent gave up" from infra/budget exhaustion.
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self.trajectory.metadata["early_exit_reason"] = "consecutive_failures"
            self.trajectory.metadata["early_exit_threshold"] = _MAX_CONSECUTIVE_FAILURES
        return self.trajectory


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    step_number: int,
    best_score: float,
    mode: str,
    last_step: Optional[TrajectoryStep],
) -> str:
    """
    Return the per-step context string.

    At step 1 there is no previous step, so we fall back to L0-style context.
    From step 2 onward we inject a compact single-step feedback block.
    """
    if best_score == float("-inf"):
        score_str = "no solution evaluated yet"
    else:
        score_str = f"{best_score:.6f}"

    header = (
        f"(L0f step {step_number} | mode={mode} (fixed schedule) | "
        f"best_score={score_str})"
    )

    if last_step is None:
        return header + "\n\nNo previous step — this is the first attempt."

    prev = last_step
    prev_mode = prev.decision.action_type
    prev_score = f"{prev.score:.6f}" if prev.score is not None else "?"
    prev_delta = (
        f"{prev.score_delta:+.6f}" if prev.score_delta is not None else "+?"
    )
    prev_new_best = "yes" if prev.is_new_best else "no"

    eval_result = prev.eval_result or {}
    success = eval_result.get("success", False)
    error_msg = eval_result.get("error") or ""
    details = eval_result.get("details") or {}

    lines = [
        header,
        "",
        "Previous step feedback (from the evaluator):",
        f"  step {prev.step_number}: {prev_mode}",
        f"  score={prev_score}  delta={prev_delta}  new_best={prev_new_best}",
        f"  status: {'success' if success else 'failure'}",
    ]
    if not success and error_msg:
        lines.append(f"  error: {truncate_error_hybrid(error_msg)}")

    if details:
        try:
            details_str = json.dumps(details, default=str)
        except Exception:
            details_str = repr(details)
        if len(details_str) > 1500:
            details_str = details_str[:1500] + " …(truncated)"
        lines.append(f"  details: {details_str}")

    return "\n".join(lines)
