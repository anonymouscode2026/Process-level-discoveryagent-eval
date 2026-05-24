"""
Tests for trajectory schema round-trip serialization.
"""

from __future__ import annotations

import tempfile
import os
from datetime import datetime, timezone

import pytest

from src.trajectory.schemas import (
    Trajectory,
    TrajectoryStep,
    DecisionStep,
    GenerationStep,
)


def make_step(n: int, score: float = 1.0, is_new_best: bool = False) -> TrajectoryStep:
    decision = DecisionStep(
        level="L1",
        prompt_tokens=100 + n * 10,
        reasoning=f"Reasoning for step {n}",
        action_type="generate",
        belief_summary=f"Belief at step {n}",
    )
    generation = GenerationStep(
        model="claude-opus-4-6",
        completion_tokens=200 + n * 20,
        code=f"def solve_{n}(): return {n}",
        diff=f"--- a\n+++ b\n@@ line {n} @@",
        temperature=0.8,
        stop_reason="end_turn",
    )
    return TrajectoryStep(
        step_number=n,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        generation=generation,
        score=score,
        score_delta=score - (score - 0.5),
        is_new_best=is_new_best,
        eval_result={"returncode": 0, "stdout": f"output_{n}", "stderr": ""},
        prompt_summary=f"Prompt summary for step {n}",
        belief_state={"key": f"value_{n}"},
    )


def make_trajectory() -> Trajectory:
    steps = [
        make_step(1, score=1.0, is_new_best=True),
        make_step(2, score=2.5, is_new_best=True),
        make_step(3, score=2.3, is_new_best=False),
    ]
    return Trajectory(
        run_id="test-run-001",
        task_id="math/circle_packing",
        model="claude-opus-4-6",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=2.5,
        best_code="def solve_2(): return 2",
        total_prompt_tokens=330,
        total_completion_tokens=660,
        metadata={"seed": 42, "config": "default"},
    )


def assert_steps_equal(a: TrajectoryStep, b: TrajectoryStep) -> None:
    assert a.step_number == b.step_number
    assert a.score == b.score
    assert a.score_delta == b.score_delta
    assert a.is_new_best == b.is_new_best
    assert a.prompt_summary == b.prompt_summary

    # Decision
    assert a.decision.level == b.decision.level
    assert a.decision.prompt_tokens == b.decision.prompt_tokens
    assert a.decision.reasoning == b.decision.reasoning
    assert a.decision.action_type == b.decision.action_type
    assert a.decision.belief_summary == b.decision.belief_summary

    # Generation
    assert a.generation.model == b.generation.model
    assert a.generation.completion_tokens == b.generation.completion_tokens
    assert a.generation.code == b.generation.code
    assert a.generation.diff == b.generation.diff
    assert a.generation.temperature == b.generation.temperature
    assert a.generation.stop_reason == b.generation.stop_reason

    # eval_result
    assert a.eval_result == b.eval_result
    # belief_state
    assert a.belief_state == b.belief_state


def test_round_trip_save_load():
    """Trajectory saves and loads with all fields preserved."""
    traj = make_trajectory()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        traj.save(tmp_path)
        loaded = Trajectory.load(tmp_path)

        assert isinstance(loaded, Trajectory)
        assert loaded.run_id == traj.run_id
        assert loaded.task_id == traj.task_id
        assert loaded.model == traj.model
        assert loaded.level == traj.level
        assert loaded.best_score == traj.best_score
        assert loaded.best_code == traj.best_code
        assert loaded.total_prompt_tokens == traj.total_prompt_tokens
        assert loaded.total_completion_tokens == traj.total_completion_tokens
        assert loaded.metadata == traj.metadata
        assert len(loaded.steps) == 3

        for orig, loaded_step in zip(traj.steps, loaded.steps):
            assert isinstance(loaded_step, TrajectoryStep)
            assert isinstance(loaded_step.decision, DecisionStep)
            assert isinstance(loaded_step.generation, GenerationStep)
            assert_steps_equal(orig, loaded_step)
    finally:
        os.unlink(tmp_path)


def test_empty_trajectory_round_trip():
    """Empty trajectory (no steps) saves and loads cleanly."""
    traj = Trajectory(
        run_id="empty-run",
        task_id="test/empty",
        model="claude-haiku-4-5-20251001",
        level="L0",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        traj.save(tmp_path)
        loaded = Trajectory.load(tmp_path)
        assert loaded.run_id == "empty-run"
        assert loaded.steps == []
        assert loaded.best_score == 0.0
    finally:
        os.unlink(tmp_path)


def test_to_dict_nested():
    """to_dict produces a plain dict with nested dicts (not dataclasses)."""
    traj = make_trajectory()
    d = traj.to_dict()

    assert isinstance(d, dict)
    assert "steps" in d
    assert isinstance(d["steps"], list)
    assert len(d["steps"]) == 3

    step0 = d["steps"][0]
    assert isinstance(step0, dict)
    assert isinstance(step0["decision"], dict)
    assert isinstance(step0["generation"], dict)
    assert step0["decision"]["level"] == "L1"
    assert step0["generation"]["code"] == "def solve_1(): return 1"
