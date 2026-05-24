"""
Level0Agent — zero-context, fixed-schedule agent.

Level 0 is intentionally simpler than L1+ in TWO ways:
  1. No memory / no history in the prompt — each step sees only the task
     description, current best score, and (for REFINE) the current best code.
  2. No model-chosen mode — REFINE vs REWRITE is decided by an EXTERNAL
     fixed schedule (every Nth step is REWRITE, the rest are REFINE).

Because the model never makes a meta-decision at L0, the two-step decision
query is skipped entirely: we synthesize a DecisionStep with action_type
set to "REFINE_FORCED" or "REWRITE_FORCED" (distinguishable from the
model-chosen "REFINE"/"REWRITE" used at L1 and above) and perform only the
generation query.

The rewrite cadence is a hyperparameter (`rewrite_every`, default 5 →
4:1 REFINE:REWRITE ratio). Smaller values rewrite more often; setting it
to a very large value effectively disables REWRITE.
"""

from __future__ import annotations

from typing import Optional

from src.agent.base import DiscoveryAgent
from src.trajectory.schemas import DecisionStep, Trajectory

_MAX_CONSECUTIVE_FAILURES = 5


class Level0Agent(DiscoveryAgent):
    """
    Simplest agent: each step sees only the task prompt and best score.

    REFINE/REWRITE is chosen by a fixed external schedule, NOT by the model.
    Step numbers are 1-indexed: step N is REWRITE iff (N % rewrite_every == 0).

    Args:
        rewrite_every: REWRITE cadence. With default 5, every 5th step is
                       REWRITE (steps 5, 10, 15, ...) and the rest are
                       REFINE. A 4:1 REFINE:REWRITE ratio.
    """

    def __init__(self, *args, rewrite_every: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if rewrite_every < 1:
            raise ValueError(f"rewrite_every must be >= 1, got {rewrite_every}")
        self.rewrite_every = rewrite_every
        # Stash the schedule in metadata for reproducibility
        self.trajectory.metadata["l0_rewrite_every"] = rewrite_every

    def run(self) -> Trajectory:
        consecutive_failures = 0

        while (
            self._budget_remaining() > 0
            and consecutive_failures < _MAX_CONSECUTIVE_FAILURES
        ):
            step_number = len(self.trajectory.steps) + 1

            # Fixed external schedule: every rewrite_every-th step is REWRITE
            is_rewrite = (step_number % self.rewrite_every == 0)
            # Safety override: if no best code is available yet, REFINE is
            # pointless — force REWRITE so the agent produces a solution.
            if not self.trajectory.best_code:
                is_rewrite = True

            mode = "REWRITE" if is_rewrite else "REFINE"
            forced_action = f"{mode}_FORCED"

            # Synthesize a DecisionStep — no LLM call at L0 for the decision.
            decision = DecisionStep(
                level="L0",
                prompt_tokens=0,
                reasoning=(
                    f"L0 fixed schedule (rewrite_every={self.rewrite_every}): "
                    f"step {step_number} is {mode} (external, not model-chosen)."
                ),
                action_type=forced_action,
            )

            # Minimal per-step context. The generation prompt itself (in
            # base.py) already injects best_score; for REFINE it injects
            # best_code; for REWRITE it injects neither code nor history.
            context_text = _build_context(
                step_number=step_number,
                best_score=self.trajectory.best_score,
                mode=mode,
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
                    prompt_summary=f"L0 {forced_action}: task+score only",
                )
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                self._record_step(
                    decision=decision,
                    generation=generation,
                    eval_result=eval_result,
                    prompt_summary=f"L0 {forced_action}: task+score only",
                )

        # Stamp metadata so post-hoc analysis and --resume can distinguish
        # "agent gave up" (a real behavior we want to preserve) from "ran out
        # of step_budget" or "infrastructure crashed mid-run".
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self.trajectory.metadata["early_exit_reason"] = "consecutive_failures"
            self.trajectory.metadata["early_exit_threshold"] = _MAX_CONSECUTIVE_FAILURES
        return self.trajectory


def _build_context(step_number: int, best_score: float, mode: str) -> str:
    if best_score == float("-inf"):
        score_str = "no solution evaluated yet"
    else:
        score_str = f"{best_score:.6f}"
    return (
        f"(L0 step {step_number} | mode={mode} (fixed schedule) | "
        f"best_score={score_str})"
    )
