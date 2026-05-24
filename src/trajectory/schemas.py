"""
Trajectory data schemas for discovery-behavioral-analysis.

These dataclasses are the stable contract between all workers:
- worker-1 (this file): defines the schemas
- worker-2: uses TrajectoryRecorder to record agent steps
- worker-3: reads saved Trajectory JSON for metrics analysis

Serialization: use dataclasses.asdict() for full nested dict conversion.
Persistence: Trajectory.save(path) / Trajectory.load(path) for JSON round-trips.
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class DecisionStep:
    """
    Records the agent's reasoning / decision at one step.

    Fields:
        level:          Agent level that generated this step (L0/L1//L2b).
        prompt_tokens:  Number of tokens in the prompt sent to the LLM.
        reasoning:      Free-text reasoning or plan produced by the agent.
        action_type:    What the agent decided to do (e.g. "generate", "reflect",
                        "hypothesis", "verify", "meta_reason").
        belief_summary: Optional concise snapshot of the agent's belief state at
                        the time of this decision.
    """
    level: str
    prompt_tokens: int
    reasoning: str
    action_type: str
    belief_summary: Optional[str] = None
    completion_tokens: int = 0
    raw_response: str = ""


@dataclass
class GenerationStep:
    """
    Records a code generation (or edit) produced by the LLM.

    Fields:
        model:             Model ID used for this generation.
        completion_tokens: Tokens in the LLM's completion.
        code:              The generated / modified code string.
        diff:              Optional unified-diff string relative to the previous best.
        temperature:       Sampling temperature used.
        stop_reason:       How the generation ended ("end_turn", "max_tokens", etc.).
    """
    model: str
    completion_tokens: int
    code: str
    diff: Optional[str] = None
    temperature: float = 1.0
    stop_reason: str = "end_turn"
    prompt_tokens: int = 0
    raw_response: str = ""


@dataclass
class TrajectoryStep:
    """
    One step in a trajectory: a paired (decision, generation) with evaluation outcome.

    Fields:
        step_number:    1-based index within the trajectory.
        timestamp:      ISO-8601 UTC timestamp of when the step was recorded.
        decision:       The DecisionStep for this step.
        generation:     The GenerationStep for this step.
        score:          Evaluation score returned by the task evaluator.
        score_delta:    score - previous_best_score (positive = improvement).
        is_new_best:    True if this step set a new best score.
        eval_result:    Full evaluation result dict (stdout/stderr/returncode etc.).
        prompt_summary: Optional short description of what was in the prompt context.
        belief_state:   Optional raw belief-state dict snapshot (provider-specific).
    """
    step_number: int
    timestamp: str
    decision: DecisionStep
    generation: GenerationStep
    score: float
    score_delta: float
    is_new_best: bool
    eval_result: dict[str, Any]
    prompt_summary: str = ""
    belief_state: Optional[dict[str, Any]] = None


@dataclass
class Trajectory:
    """
    Complete record of one agent run on one task.

    Fields:
        run_id:       Unique identifier for this run (e.g. UUID or timestamp slug).
        task_id:      Task identifier as in configs/tasks.yaml (e.g. "math/circle_packing").
        model:        Model name as in configs/models.yaml (e.g. "claude-opus-4-6").
        level:        Agent level (L0/L1//L2b).
        started_at:   ISO-8601 UTC timestamp when the run started.
        step_budget:  Maximum number of steps allowed for this run (default 150).
        finished_at:  ISO-8601 UTC timestamp when the run ended (set on save or finish).
        steps:        Ordered list of TrajectoryStep records.
        best_score:   Best score achieved so far across all steps.
        best_code:    Code string that produced best_score.
        tokens_used:  Cumulative token count across all LLM calls (for cost tracking).
        total_prompt_tokens:     Cumulative prompt tokens across all steps.
        total_completion_tokens: Cumulative completion tokens across all steps.
        metadata:     Arbitrary run metadata (seed, config overrides, etc.).
    """
    run_id: str
    task_id: str
    model: str
    level: str
    started_at: str
    step_budget: int = 150
    finished_at: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    best_score: float = 0.0
    best_code: str = ""
    tokens_used: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    belief_state_history: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a fully nested plain dict (suitable for JSON serialization)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """
        Serialize the trajectory to a JSON file.

        Stamps finished_at with current UTC time if not already set.
        Creates parent directories as needed.
        """
        if not self.finished_at:
            self.finished_at = datetime.now(timezone.utc).isoformat()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        """
        Deserialize a Trajectory from a JSON file.

        Fully reconstructs nested dataclasses (TrajectoryStep, DecisionStep,
        GenerationStep) so callers receive typed objects, not plain dicts.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Recursively rebuild nested dataclasses from a plain dict."""
        steps_raw = data.pop("steps", [])
        steps = [cls._step_from_dict(s) for s in steps_raw]
        belief_state_history = data.pop("belief_state_history", [])
        return cls(steps=steps, belief_state_history=belief_state_history, **data)

    @staticmethod
    def _step_from_dict(data: dict[str, Any]) -> TrajectoryStep:
        decision_raw = data.pop("decision")
        generation_raw = data.pop("generation")
        decision = DecisionStep(**decision_raw)
        generation = GenerationStep(**generation_raw)
        return TrajectoryStep(decision=decision, generation=generation, **data)


# ---------------------------------------------------------------------------
# Self-test (python -m src.trajectory.schemas)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    dec = DecisionStep(
        level="L1",
        prompt_tokens=512,
        reasoning="Try a hexagonal layout.",
        action_type="generate",
        belief_summary="No prior best yet.",
    )
    gen = GenerationStep(
        model="claude-opus-4-6",
        completion_tokens=1024,
        code="def run_packing(): pass",
        diff=None,
        temperature=1.0,
        stop_reason="end_turn",
    )
    step = TrajectoryStep(
        step_number=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=dec,
        generation=gen,
        score=1.23,
        score_delta=1.23,
        is_new_best=True,
        eval_result={"returncode": 0},
    )
    traj = Trajectory(
        run_id="test-001",
        task_id="math/circle_packing",
        model="claude-opus-4-6",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        step_budget=150,
        steps=[step],
        best_score=1.23,
        best_code="def run_packing(): pass",
        tokens_used=1536,
        total_prompt_tokens=512,
        total_completion_tokens=1024,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        traj.save(tmp_path)
        loaded = Trajectory.load(tmp_path)
        assert isinstance(loaded, Trajectory), "load() must return Trajectory"
        assert isinstance(loaded.steps[0], TrajectoryStep), "step must be TrajectoryStep"
        assert isinstance(loaded.steps[0].decision, DecisionStep), "decision must be DecisionStep"
        assert isinstance(loaded.steps[0].generation, GenerationStep), "generation must be GenerationStep"
        assert loaded.best_score == 1.23
        assert loaded.step_budget == 150
        print("schemas self-test passed")
    finally:
        os.unlink(tmp_path)
