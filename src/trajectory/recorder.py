"""
TrajectoryRecorder — thin wrapper around Trajectory that appends steps.

Usage (worker-2 agents will use this):
    from src.trajectory.recorder import TrajectoryRecorder
    from src.trajectory.schemas import Trajectory, DecisionStep, GenerationStep

    traj = Trajectory(run_id=..., task_id=..., model=..., level=..., started_at=...)
    rec = TrajectoryRecorder(traj)
    rec.record_step(decision, generation, eval_result)
    traj.save("data/trajectories/run_001.json")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.trajectory.schemas import (
    DecisionStep,
    GenerationStep,
    Trajectory,
    TrajectoryStep,
)


class TrajectoryRecorder:
    """
    Wraps a Trajectory and provides a single method to append a completed step.

    The recorder is intentionally stateless beyond the trajectory it wraps:
    all persistence is delegated to Trajectory.save().
    """

    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory

    def record_step(
        self,
        decision: DecisionStep,
        generation: GenerationStep,
        eval_result: dict[str, Any],
        prompt_summary: str = "",
        belief_state: Optional[dict[str, Any]] = None,
    ) -> TrajectoryStep:
        """
        Compute derived fields and append a new TrajectoryStep to the trajectory.

        Args:
            decision:       The decision/reasoning step.
            generation:     The code generation step.
            eval_result:    Raw evaluator output dict.  Must contain a "score" key
                            (float) OR a "combined_score" key; falls back to 0.0.
            prompt_summary: Short human-readable description of the prompt context.
            belief_state:   Optional snapshot of the agent's belief state dict.

        Returns:
            The appended TrajectoryStep (also stored in trajectory.steps).
        """
        # Extract score from eval_result (handle both key conventions)
        score: float = float(
            eval_result.get("score",
            eval_result.get("combined_score", 0.0))
        )

        # Compute score_delta relative to current best
        score_delta = score - self.trajectory.best_score

        # Determine if this is a new best.
        # First step is always a new best (so best_code gets seeded) as long
        # as the eval did not explicitly fail.
        first_step = len(self.trajectory.steps) == 0
        eval_failed = isinstance(eval_result, dict) and eval_result.get("success") is False
        if first_step and not eval_failed:
            is_new_best = True
        else:
            is_new_best = score > self.trajectory.best_score

        # Update trajectory bests if improved
        if is_new_best:
            self.trajectory.best_score = score
            self.trajectory.best_code = generation.code

        # Accumulate token counts
        self.trajectory.total_prompt_tokens += decision.prompt_tokens
        self.trajectory.total_completion_tokens += generation.completion_tokens

        step = TrajectoryStep(
            step_number=len(self.trajectory.steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision=decision,
            generation=generation,
            score=score,
            score_delta=score_delta,
            is_new_best=is_new_best,
            eval_result=eval_result,
            prompt_summary=prompt_summary,
            belief_state=belief_state,
        )
        self.trajectory.steps.append(step)
        return step
