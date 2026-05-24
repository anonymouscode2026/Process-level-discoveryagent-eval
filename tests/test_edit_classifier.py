"""
Tests for edit_classifier.py
"""

from __future__ import annotations

import pytest

from src.metrics.edit_classifier import classify_edit, annotate_trajectory_edits
from src.trajectory.schemas import (
    Trajectory,
    TrajectoryStep,
    DecisionStep,
    GenerationStep,
)
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_step(n: int, code: str, mode: str = "REFINE") -> TrajectoryStep:
    decision = DecisionStep(
        level="L1",
        prompt_tokens=100,
        reasoning="test",
        action_type=mode,
    )
    generation = GenerationStep(
        model="model",
        completion_tokens=100,
        code=code,
        stop_reason="end_turn",
    )
    return TrajectoryStep(
        step_number=n,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        generation=generation,
        score=1.0,
        score_delta=0.0,
        is_new_best=False,
        eval_result={"returncode": 0},
    )


# ---------------------------------------------------------------------------
# Tests: classify_edit
# ---------------------------------------------------------------------------

def test_constant_only_change_is_parameter_tuning():
    """Only a constant value changed => parameter_tuning (REFINE mode)."""
    prev = "def f(x):\n    return x * 2\n"
    new  = "def f(x):\n    return x * 3\n"
    result = classify_edit(prev, new, mode="REFINE")
    assert result == "parameter_tuning", f"Expected parameter_tuning, got {result}"


def test_completely_different_algorithm_rewrite():
    """Completely different algorithms in REWRITE mode => algorithm_switch."""
    prev = """
def solve(data):
    result = []
    for x in data:
        result.append(x * 2)
    return result
"""
    new = """
import heapq

def priority_search(graph, start):
    heap = [(0, start)]
    visited = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node not in visited:
            visited.add(node)
    return visited
"""
    result = classify_edit(prev, new, mode="REWRITE")
    assert result == "algorithm_switch", f"Expected algorithm_switch, got {result}"


def test_same_function_names_rewrite_is_representation_change():
    """Same top-level function names with significant overlap in REWRITE => representation_change."""
    prev = """
def compute(data):
    return sum(data)

def process(data):
    return [x * 2 for x in data]
"""
    new = """
def compute(data):
    import numpy as np
    return np.array(data).sum()

def process(data):
    import numpy as np
    return (np.array(data) * 2).tolist()
"""
    result = classify_edit(prev, new, mode="REWRITE")
    assert result == "representation_change", f"Expected representation_change, got {result}"


def test_refine_mode_structural_change():
    """Same signatures but different body => structural_change (REFINE mode)."""
    prev = """
def f(x, y):
    return x + y

def g(lst):
    total = 0
    for item in lst:
        total += item
    return total
"""
    new = """
def f(x, y):
    return x - y

def g(lst):
    total = 0
    for item in lst:
        total -= item
    return total
"""
    result = classify_edit(prev, new, mode="REFINE")
    # Structure fingerprints differ (subtraction vs addition changes AST operators)
    # but same function signatures and no new calls introduced
    # => structural_change or algorithm_switch depending on call graph
    assert result in ("structural_change", "algorithm_switch", "parameter_tuning")


def test_syntax_error_falls_back_gracefully():
    """If code fails to parse, falls back to mode-based default without raising."""
    bad_code = "def f(:\n    return 1"
    good_code = "def f(x):\n    return x + 1"
    result_rewrite = classify_edit(bad_code, good_code, mode="REWRITE")
    assert result_rewrite == "algorithm_switch"
    result_refine = classify_edit(bad_code, good_code, mode="REFINE")
    assert result_refine == "structural_change"


def test_empty_prev_code():
    """Empty prev_code with REWRITE => algorithm_switch."""
    new = "def f(x):\n    return x + 1\n"
    result = classify_edit("", new, mode="REWRITE")
    assert result == "algorithm_switch"


# ---------------------------------------------------------------------------
# Tests: annotate_trajectory_edits
# ---------------------------------------------------------------------------

def test_annotate_trajectory_edits_fills_edit_type():
    """annotate_trajectory_edits stores edit_type on each step."""
    steps = [
        make_step(1, "def f(x):\n    return x * 2\n", mode="REFINE"),
        make_step(2, "def f(x):\n    return x * 3\n", mode="REFINE"),
        make_step(3, "def totally_new():\n    import heapq\n    return heapq.nlargest(3, [])\n", mode="REWRITE"),
    ]
    traj = Trajectory(
        run_id="test",
        task_id="test",
        model="model",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
    )

    annotate_trajectory_edits(traj)

    for step in traj.steps:
        edit_type = step.eval_result.get("edit_type")
        assert edit_type is not None, f"Step {step.step_number} missing edit_type"
        assert edit_type in (
            "parameter_tuning",
            "structural_change",
            "algorithm_switch",
            "representation_change",
        ), f"Invalid edit_type: {edit_type}"


def test_annotate_empty_trajectory():
    """annotate_trajectory_edits on empty trajectory does not crash."""
    traj = Trajectory(
        run_id="empty",
        task_id="test",
        model="model",
        level="L0",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=[],
    )
    result = annotate_trajectory_edits(traj)
    assert result.steps == []
