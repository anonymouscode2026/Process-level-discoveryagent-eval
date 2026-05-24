"""
Cross-trajectory variance metrics for discovery agent trajectories.

compute_cell_variance(trajectories) -> dict
Given multiple trajectories from the same model x task x level cell (different seeds),
computes variance/similarity metrics across trajectories.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schemas import Trajectory


def _detect_plateau_steps(steps, epsilon: float = 1e-6, min_length: int = 3) -> set[int]:
    """
    Return set of step indices (0-based) that are part of a plateau.
    A plateau is a run of >= min_length consecutive steps where improvement
    in running best < epsilon.
    """
    n = len(steps)
    if n < min_length:
        return set()

    scores = [s.score for s in steps]
    running_best = []
    cur_best = float("-inf")
    for sc in scores:
        if sc > cur_best:
            cur_best = sc
        running_best.append(cur_best)

    improvements = [0.0]
    for i in range(1, n):
        improvements.append(running_best[i] - running_best[i - 1])

    plateau_indices: set[int] = set()
    i = 0
    while i < n:
        if improvements[i] < epsilon:
            run_start = i
            while i < n and improvements[i] < epsilon:
                i += 1
            run_length = i - run_start
            if run_length >= min_length:
                for j in range(run_start, run_start + run_length):
                    plateau_indices.add(j)
        else:
            i += 1

    return plateau_indices


def _jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def compute_cell_variance(trajectories: list["Trajectory"]) -> dict:
    """
    Compute cross-trajectory variance metrics for a cell (same model x task x level).

    Args:
        trajectories: List of Trajectory objects from the same cell (different seeds).

    Returns:
        Dict with keys:
        - breakthrough_timing_variance: std of step index where largest score_delta > 0 occurs
        - plateau_pattern_similarity: avg pairwise Jaccard of plateau step-index sets
        - final_score_variance: std of best_score across trajectories
        - final_score_cv: coefficient of variation (std/mean) of best_score
        - n_trajectories: number of trajectories used
    """
    if not trajectories:
        return {
            "breakthrough_timing_variance": None,
            "plateau_pattern_similarity": None,
            "final_score_variance": None,
            "final_score_cv": None,
            "n_trajectories": 0,
        }

    n = len(trajectories)

    # --- breakthrough timing variance ---
    breakthrough_indices = []
    for traj in trajectories:
        steps = traj.steps if traj.steps else []
        if not steps:
            continue
        best_delta_idx = None
        best_delta_val = 0.0
        for i, s in enumerate(steps):
            delta = s.score_delta if s.score_delta is not None else 0.0
            if delta > best_delta_val:
                best_delta_val = delta
                best_delta_idx = i
        if best_delta_idx is not None:
            breakthrough_indices.append(float(best_delta_idx))

    breakthrough_timing_variance = _std(breakthrough_indices) if len(breakthrough_indices) >= 2 else None

    # --- plateau pattern similarity (avg pairwise Jaccard) ---
    plateau_sets = []
    for traj in trajectories:
        steps = traj.steps if traj.steps else []
        plateau_sets.append(_detect_plateau_steps(steps))

    if n >= 2:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(_jaccard(plateau_sets[i], plateau_sets[j]))
        plateau_pattern_similarity = sum(pairs) / len(pairs) if pairs else None
    else:
        plateau_pattern_similarity = None

    # --- final score variance and CV ---
    best_scores = []
    for traj in trajectories:
        steps = traj.steps if traj.steps else []
        if steps:
            # Use running best of step scores
            cur_best = float("-inf")
            for s in steps:
                if s.score > cur_best:
                    cur_best = s.score
            best_scores.append(cur_best)
        elif hasattr(traj, "best_score") and traj.best_score is not None:
            best_scores.append(float(traj.best_score))

    if len(best_scores) >= 2:
        final_score_variance = _std(best_scores)
        mean_score = sum(best_scores) / len(best_scores)
        final_score_cv = final_score_variance / mean_score if mean_score != 0 else None
    else:
        final_score_variance = None
        final_score_cv = None

    return {
        "breakthrough_timing_variance": breakthrough_timing_variance,
        "plateau_pattern_similarity": plateau_pattern_similarity,
        "final_score_variance": final_score_variance,
        "final_score_cv": final_score_cv,
        "n_trajectories": n,
    }
