"""
Level1Agent — history-aware agent.

Context includes formatted recent step history (last history_window steps).
REWRITE path: summarizes prior approaches.
"""

from __future__ import annotations

from src.agent.base import DiscoveryAgent, _eval_feedback_suffix
from src.trajectory.schemas import Trajectory

_MAX_CONSECUTIVE_FAILURES = 5


class Level1Agent(DiscoveryAgent):
    """
    Iterative agent with step history in context.

    Args:
        history_window: Number of recent steps to include in context.
    """

    def __init__(self, *args, history_window: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history_window = history_window

    def run(self) -> Trajectory:
        consecutive_failures = 0

        while self._budget_remaining() > 0 and consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
            context_text = self._build_context()
            decision = self._decision_query(context_text)

            # For REWRITE, use summarized approach context
            if decision.action_type == "REWRITE":
                gen_context = self._summarize_attempts()
            else:
                gen_context = context_text

            generation = self._generation_query(decision.action_type, gen_context)
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
                    prompt_summary=f"L1: history_window={self.history_window}",
                )
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                self._record_step(
                    decision=decision,
                    generation=generation,
                    eval_result=eval_result,
                    prompt_summary=f"L1: history_window={self.history_window}",
                )

        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self.trajectory.metadata["early_exit_reason"] = "consecutive_failures"
            self.trajectory.metadata["early_exit_threshold"] = _MAX_CONSECUTIVE_FAILURES

        return self.trajectory

    def _build_context(self) -> str:
        """Build context with recent step history."""
        steps = self.trajectory.steps
        if not steps:
            return "No steps completed yet."

        recent = steps[-self.history_window:]
        lines = []
        for s in recent:
            mode = s.decision.action_type
            edit_type = (s.eval_result or {}).get("edit_type", mode)
            delta_str = f"{s.score_delta:+.4f}" if s.score_delta is not None else "+?"
            line = f"Step {s.step_number}: [{mode}] {edit_type} → score {s.score:.6f} ({delta_str})"
            # Append evaluator feedback so the agent knows WHY a step
            # failed or produced a degenerate/zero score.
            line += _eval_feedback_suffix(s)
            lines.append(line)

        return "Recent steps:\n" + "\n".join(lines)

    def _summarize_attempts(self) -> str:
        """
        Summarize prior approaches for REWRITE context.

        Groups consecutive steps by strategy_label (if set) or by mode.
        Returns a compact list of prior approaches.
        """
        steps = self.trajectory.steps
        if not steps:
            return "No prior attempts."

        # Group consecutive steps
        groups = []
        cur_label = None
        cur_scores = []

        for s in steps:
            label = None
            if s.eval_result:
                label = s.eval_result.get("strategy_label")
            if not label and s.belief_state:
                label = s.belief_state.get("strategy_label")
            if not label:
                label = s.decision.action_type or "unknown"

            if label == cur_label:
                cur_scores.append(s.score)
            else:
                if cur_label is not None:
                    groups.append((cur_label, cur_scores))
                cur_label = label
                cur_scores = [s.score]

        if cur_label is not None:
            groups.append((cur_label, cur_scores))

        lines = ["Prior approaches tried:"]
        for i, (label, scores) in enumerate(groups, 1):
            best = max(scores)
            lines.append(f"  {i}. {label} ({len(scores)} step(s), best score: {best:.6f})")

        lines.append("\nFor this REWRITE, use a substantially different approach.")
        return "\n".join(lines)
