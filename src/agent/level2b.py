"""
Level2bAgent — LLM-maintained belief state agent.

The agent maintains a free-form natural-language belief_state_text that
is updated by a SEPARATE LLM call (not the decision or generation call).
This tests whether the LLM can produce useful self-diagnosis from raw
trajectory data, WITHOUT receiving any pre-computed analysis (no plateau
detection, no approach clustering, no strategy labels).

Update triggers (any of):
  - Every update_interval steps (default 5)
  - Score jump > 1 std of recent deltas
  - 3 consecutive execution failures
  - 5+ steps with no improvement (plateau)

The belief state text is injected into both decision and generation prompts
as: "Your current understanding of the search progress:\n{belief_state_text}"
"""

from __future__ import annotations

import json
import statistics

from src.agent.base import DiscoveryAgent, _eval_feedback_suffix, truncate_error_hybrid
from src.trajectory.schemas import Trajectory

_MAX_CONSECUTIVE_FAILURES = 5

BELIEF_UPDATE_PROMPT = """\
You are maintaining a running analysis of your optimization search progress.

Task: {task_description}

Your previous analysis:
{previous_belief_state}

New steps since your last analysis:
{steps_data}

Current state: {state_summary}
{compression_note}

Update your analysis. Cover these points with SPECIFIC details (algorithm names, score numbers, concrete observations — not generic statements):

1. APPROACHES TRIED: What algorithms/strategies have you used? What scores did each achieve? Be specific about what made each approach work or fail.

2. CURRENT SITUATION: Where does the score stand? Is progress stalling? What does the score trajectory suggest?

3. BOTTLENECKS: What specific problems are blocking improvement? (e.g., "timeout suggests O(n^3) complexity is too slow", "score stuck at 0.72 suggests the greedy heuristic can't escape local optima")

4. NEXT DIRECTIONS: What should be tried next and why? Rank by expected impact. Be concrete (e.g., "switch from SA to LP relaxation" not "try a different approach").

5. DEAD ENDS: What approaches have definitively failed and should NOT be retried? Why?

Keep your analysis under 1500 words. Be honest about uncertainty."""


class Level2bAgent(DiscoveryAgent):
    """
    Agent with LLM-maintained belief state updated by a separate LLM call.

    Args:
        update_interval: How often (in steps) to trigger a belief state update.
    """

    def __init__(self, *args, update_interval: int = 5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.update_interval = update_interval
        self.belief_state_text: str = ""
        self._steps_since_update: int = 0
        self._consecutive_failures: int = 0
        self._last_update_step: int = 0
        self.trajectory.metadata["l2b_update_interval"] = update_interval

    def run(self) -> Trajectory:
        consecutive_failures = 0

        while self._budget_remaining() > 0 and consecutive_failures < _MAX_CONSECUTIVE_FAILURES:
            # Check if belief state needs updating
            if self._should_update_belief():
                self._update_belief_state()

            context_text = self._build_context()
            decision = self._decision_query(context_text)

            if decision.action_type == "REWRITE":
                gen_context = self._build_rewrite_context()
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
                    prompt_summary=f"L2b: belief_state+history, update_interval={self.update_interval}",
                )
                self._consecutive_failures += 1
                consecutive_failures += 1
            else:
                self._consecutive_failures = 0
                consecutive_failures = 0
                self._record_step(
                    decision=decision,
                    generation=generation,
                    eval_result=eval_result,
                    prompt_summary=f"L2b: belief_state+history, update_interval={self.update_interval}",
                )

            self._steps_since_update += 1

        # Mirror L0 / L0f: stamp early_exit_reason when the consecutive-failure
        # threshold trips so resume can [skip-early-exit] instead of [redo].
        # For L2b, the typical trigger is the evaluator timing out 5 runs in a
        # row (the LLM keeps producing slow / hanging code), not LLM failure.
        if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self.trajectory.metadata["early_exit_reason"] = "consecutive_failures"
            self.trajectory.metadata["early_exit_threshold"] = _MAX_CONSECUTIVE_FAILURES

        return self.trajectory

    def _should_update_belief(self) -> bool:
        """Trigger belief update on: interval, score jump, 3 failures, or plateau."""
        if not self.trajectory.steps:
            return False

        # (a) Regular interval
        if self._steps_since_update >= self.update_interval:
            return True

        # (b) Score jump > 1 std of recent deltas
        if len(self.trajectory.steps) >= 3:
            recent_deltas = [
                abs(s.score_delta)
                for s in self.trajectory.steps[-10:]
                if s.score_delta is not None
            ]
            if recent_deltas:
                std = statistics.stdev(recent_deltas) if len(recent_deltas) > 1 else 0.0
                last_delta = abs(self.trajectory.steps[-1].score_delta or 0)
                if std > 0 and last_delta > std:
                    return True

        # (c) 3 consecutive execution failures
        if self._consecutive_failures >= 3:
            return True

        # (d) 5+ steps with no improvement (plateau)
        if len(self.trajectory.steps) >= 5:
            recent = self.trajectory.steps[-5:]
            if all(not s.is_new_best for s in recent):
                # Only trigger if we haven't updated since the plateau started
                if self._steps_since_update >= 5:
                    return True

        return False

    def _update_belief_state(self) -> None:
        """
        Call the LLM to produce/update the belief state from raw trajectory data.
        Uses the MAIN agent LLM client (Opus), not the analyst.
        """
        # Gather raw step data since last update
        steps_since = self.trajectory.steps[self._last_update_step:]

        # Build the raw data block for each step
        step_blocks = []
        for s in steps_since:
            # Include strategy_label from analyst if available
            strategy = (s.eval_result or {}).get("strategy_label", "")
            strategy_str = f", strategy={strategy}" if strategy else ""
            block_lines = [f"Step {s.step_number}: mode={s.decision.action_type}{strategy_str}"]

            # Code: for REWRITE show first 250 lines, for REFINE show diff-like summary
            code = s.generation.code or ""
            if "REWRITE" in s.decision.action_type:
                code_lines = code.splitlines()[:250]
                code_preview = "\n".join(code_lines)
                if len(code.splitlines()) > 250:
                    code_preview += f"\n... ({len(code.splitlines())} lines total, truncated)"
                block_lines.append(f"Code (new solution):\n```\n{code_preview}\n```")
            else:
                # For REFINE, show a brief summary
                code_lines = code.splitlines()
                block_lines.append(
                    f"Code after refinement: {len(code_lines)} lines, {len(code)} chars"
                )
                # Show first and last 10 lines for context
                if len(code_lines) > 20:
                    preview = "\n".join(code_lines[:10]) + "\n...\n" + "\n".join(code_lines[-10:])
                else:
                    preview = code
                block_lines.append(f"```\n{preview}\n```")

            # Score info
            delta_str = f"{s.score_delta:+.4f}" if s.score_delta is not None else "N/A"
            block_lines.append(
                f"Score: {s.score:.6f} (delta: {delta_str}), new_best: {s.is_new_best}"
            )

            # Error info
            er = s.eval_result or {}
            if not er.get("success", True):
                err_msg = truncate_error_hybrid(er.get("error") or "")
                block_lines.append(f"FAILED: {err_msg}")
                details = er.get("details", {})
                if details:
                    try:
                        det_str = json.dumps(details, default=str)[:1500]
                    except Exception:
                        det_str = str(details)[:1500]
                    block_lines.append(f"Details: {det_str}")
            elif s.score <= 0:
                details = er.get("details", {})
                if details:
                    try:
                        det_str = json.dumps(details, default=str)[:1500]
                    except Exception:
                        det_str = str(details)[:1500]
                    block_lines.append(f"Details (score<=0): {det_str}")

            step_blocks.append("\n".join(block_lines))

        steps_text = "\n\n---\n\n".join(step_blocks)

        # Current state summary
        best = self.trajectory.best_score
        best_str = f"{best:.6f}" if best != float("-inf") else "none"
        state_summary = (
            f"Steps completed: {len(self.trajectory.steps)}\n"
            f"Best score so far: {best_str}\n"
            f"Budget remaining: {self._budget_remaining()} of {self.step_budget} steps"
        )

        # Build the update prompt
        compression_note = ""
        if len(self.belief_state_text.split()) > 1500:
            compression_note = (
                "\nIMPORTANT: Your previous belief state has grown too long. "
                "Compress and focus on currently relevant information; older "
                "context can be summarized in one sentence."
            )

        prompt = BELIEF_UPDATE_PROMPT.format(
            task_description=getattr(self.task, "description", "")[:300],
            previous_belief_state=self.belief_state_text
            or "(No previous belief state — this is the first update.)",
            steps_data=steps_text,
            state_summary=state_summary,
            compression_note=compression_note,
        )

        response = self.llm_client.complete(prompt, max_tokens=3000)
        self._tokens_used += response["prompt_tokens"] + response["completion_tokens"]
        self.trajectory.tokens_used = self._tokens_used

        self.belief_state_text = response["text"].strip()

        # Record in history for post-hoc analysis
        self.trajectory.belief_state_history.append(self.belief_state_text)

        # Reset counters
        self._steps_since_update = 0
        self._last_update_step = len(self.trajectory.steps)
        self._consecutive_failures = 0

    def _build_context(self) -> str:
        """Inject belief state + last 5 steps with error feedback."""
        parts = []
        if self.belief_state_text:
            parts.append(
                f"Your current understanding of the search progress:\n{self.belief_state_text}"
            )
        else:
            parts.append("No belief state yet — this is an early step.")

        # Last 5 steps summary (with error feedback, like L1)
        steps = self.trajectory.steps
        if steps:
            recent = steps[-5:]
            history_lines = ["Recent steps:"]
            for s in recent:
                delta_str = f"{s.score_delta:+.4f}" if s.score_delta is not None else "+?"
                line = (
                    f"  Step {s.step_number}: [{s.decision.action_type}] "
                    f"→ score {s.score:.6f} ({delta_str})"
                )
                line += _eval_feedback_suffix(s)
                history_lines.append(line)
            parts.append("\n".join(history_lines))

        return "\n\n".join(parts)

    def _build_rewrite_context(self) -> str:
        """For REWRITE mode, emphasize what hasn't worked."""
        ctx = self._build_context()
        ctx += "\n\nFor REWRITE: use a substantially different strategy from what your analysis identifies as tried or failed."
        return ctx
