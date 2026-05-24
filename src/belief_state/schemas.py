"""
BeliefState — snapshot of the agent's belief about the task state.

Computed deterministically from a Trajectory by build_system_belief_state().
Used by Level2bAgent to inform context construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailedApproach:
    """
    Records an approach that was tried and abandoned without being resumed.

    Fields:
        name:                 Label identifying the approach (strategy_label or action_type).
        best_score_achieved:  Best score seen during this approach's run.
        why_abandoned:        Heuristic reason string (e.g. "plateaued for 5 steps").
        steps_spent:          Number of steps spent on this approach.
    """
    name: str
    best_score_achieved: float
    why_abandoned: str
    steps_spent: int


@dataclass
class BeliefState:
    """
    Deterministic snapshot of agent progress extracted from trajectory data.

    Fields:
        steps_completed:      Number of steps completed so far.
        best_score:           Best score achieved across all steps.
        approaches_tried:     List of approach dicts, each with keys:
                              label (str), count (int), best_score (float).
        failed_approaches:    List of FailedApproach records for approaches that were
                              tried, abandoned, and never resumed.
        current_plateau:      Dict describing whether agent is in a plateau:
                              {"in_plateau": bool, "length": int, "since_step": int}.
        score_trajectory:     List of scores in order (one per step).
        edit_history_summary: Short text summary of edit_type counts over last 4 steps.
        budget_remaining_pct: Fraction of token budget remaining (0.0–1.0).
    """
    steps_completed: int
    best_score: float
    approaches_tried: list[dict[str, Any]]
    current_plateau: dict[str, Any]
    score_trajectory: list[float]
    edit_history_summary: str
    budget_remaining_pct: float
    failed_approaches: list[FailedApproach] = field(default_factory=list)
