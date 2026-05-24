"""
build_system_belief_state — deterministic belief state extractor.

Computes a BeliefState from a Trajectory without any LLM calls.
"""
# Belief state v2 (with failed_approaches) — pilot runs before 2026-04-08 used v1 without this field

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from src.trajectory.schemas import Trajectory
from src.belief_state.schemas import BeliefState, FailedApproach


def build_system_belief_state(
    trajectory: Trajectory,
    step_budget: Optional[int] = None,
) -> BeliefState:
    """
    Compute a deterministic BeliefState from a Trajectory.

    Args:
        trajectory:  The trajectory to summarize.
        step_budget: Step budget for remaining-pct computation. If None,
                     uses trajectory.step_budget.

    Returns:
        A fully populated BeliefState.
    """
    steps = trajectory.steps
    steps_completed = len(steps)
    best_score = trajectory.best_score

    # --- score trajectory ---
    score_trajectory = [s.score for s in steps]

    # --- plateau detection: >= 3 consecutive steps where |score - running_best| < 1e-6 ---
    current_plateau = _detect_plateau(steps)

    # --- approach clustering ---
    approaches_tried = _cluster_approaches(steps)

    # --- edit history summary (last 4 steps) ---
    edit_history_summary = _edit_history_summary(steps)

    # --- budget remaining (step-based) ---
    if step_budget is None:
        step_budget = trajectory.step_budget
    if step_budget is None or step_budget <= 0:
        budget_remaining_pct = 0.0
    else:
        budget_remaining_pct = max(0.0, (step_budget - steps_completed) / step_budget)

    # --- failed approaches ---
    failed_approaches = _compute_failed_approaches(steps)

    return BeliefState(
        steps_completed=steps_completed,
        best_score=best_score,
        approaches_tried=approaches_tried,
        failed_approaches=failed_approaches,
        current_plateau=current_plateau,
        score_trajectory=score_trajectory,
        edit_history_summary=edit_history_summary,
        budget_remaining_pct=budget_remaining_pct,
    )


def _detect_plateau(steps: list) -> dict[str, Any]:
    """
    Detect whether the agent is in a score plateau.

    Plateau: >= 3 consecutive steps where |score - running_best_at_that_point| < 1e-6.
    Returns dict with in_plateau, length, since_step (1-based).
    """
    if len(steps) < 3:
        return {"in_plateau": False, "length": 0, "since_step": 0}

    # Walk backwards counting consecutive non-improving steps
    # A step is "non-improving" if score <= best_before_this_step + 1e-6
    # We track the running best up to each step
    running_best = float("-inf")
    bests = []
    for s in steps:
        if s.score > running_best:
            running_best = s.score
        bests.append(running_best)

    # From the end, count how many consecutive steps have score within 1e-6 of bests[i-1]
    plateau_length = 0
    for i in range(len(steps) - 1, -1, -1):
        # best before step i is bests[i-1] if i > 0 else -inf
        best_before = bests[i - 1] if i > 0 else float("-inf")
        if abs(steps[i].score - best_before) < 1e-6 or steps[i].score <= best_before + 1e-6:
            plateau_length += 1
        else:
            break

    in_plateau = plateau_length >= 3
    since_step = (len(steps) - plateau_length + 1) if in_plateau else 0
    return {
        "in_plateau": in_plateau,
        "length": plateau_length if in_plateau else 0,
        "since_step": since_step,
    }


def _cluster_approaches(steps: list) -> list[dict[str, Any]]:
    """
    Group consecutive steps by strategy_label (from eval_result) or by mode_choice.

    Each group becomes an "approach" dict with:
        label (str), count (int), best_score (float), last_score (float),
        first_step (int, 1-based), last_step (int, 1-based).
    """
    if not steps:
        return []

    def _label(step) -> str:
        # Try strategy_label in eval_result or belief_state
        if step.eval_result:
            lbl = step.eval_result.get("strategy_label")
            if lbl:
                return str(lbl)
        if step.belief_state:
            lbl = step.belief_state.get("strategy_label")
            if lbl:
                return str(lbl)
        # Fall back to mode_choice from decision
        return step.decision.action_type or "unknown"

    # Group consecutive runs with same label
    approaches = []
    cur_label = None
    cur_count = 0
    cur_best = float("-inf")
    cur_last = float("-inf")
    cur_first_step = 1

    for i, step in enumerate(steps):
        lbl = _label(step)
        step_num = i + 1
        if lbl == cur_label:
            cur_count += 1
            if step.score > cur_best:
                cur_best = step.score
            cur_last = step.score
        else:
            if cur_label is not None:
                approaches.append({
                    "label": cur_label,
                    "count": cur_count,
                    "best_score": cur_best,
                    "last_score": cur_last,
                    "first_step": cur_first_step,
                    "last_step": step_num - 1,
                })
            cur_label = lbl
            cur_count = 1
            cur_best = step.score
            cur_last = step.score
            cur_first_step = step_num

    if cur_label is not None:
        approaches.append({
            "label": cur_label,
            "count": cur_count,
            "best_score": cur_best,
            "last_score": cur_last,
            "first_step": cur_first_step,
            "last_step": len(steps),
        })

    return approaches


def _compute_failed_approaches(steps: list) -> list[FailedApproach]:
    """
    Identify approaches that were tried, abandoned, and never resumed.

    An approach is "failed" if:
      (a) it was tried and subsequently a DIFFERENT approach was tried, AND
      (b) the agent did NOT return to it later in the trajectory.

    Returns list sorted by steps_spent descending.
    """
    if not steps:
        return []

    def _label(step) -> str:
        if step.eval_result:
            lbl = step.eval_result.get("strategy_label")
            if lbl:
                return str(lbl)
        if step.belief_state:
            lbl = step.belief_state.get("strategy_label")
            if lbl:
                return str(lbl)
        return step.decision.action_type or "unknown"

    # Build list of (label, step) pairs grouped into consecutive runs
    runs: list[tuple[str, list]] = []
    cur_label = None
    cur_steps: list = []
    for step in steps:
        lbl = _label(step)
        if lbl == cur_label:
            cur_steps.append(step)
        else:
            if cur_label is not None:
                runs.append((cur_label, cur_steps))
            cur_label = lbl
            cur_steps = [step]
    if cur_label is not None:
        runs.append((cur_label, cur_steps))

    # The last run is still active — only consider runs[:-1] as potentially abandoned
    if len(runs) <= 1:
        return []

    # Collect labels that appear in later runs (to detect "came back to")
    # For run i to be "failed", its label must NOT appear in any run after run i
    failed: list[FailedApproach] = []
    for i, (label, run_steps) in enumerate(runs[:-1]):
        # Check if this label reappears in any later run
        later_labels = {lbl for lbl, _ in runs[i + 1:]}
        if label in later_labels:
            continue  # agent came back to it — not failed

        # Determine why abandoned
        scores = [s.score for s in run_steps]
        best_score = max(scores) if scores else 0.0
        steps_spent = len(run_steps)

        # Check for implementation failures — collect truncated error messages
        failed_steps = [
            s for s in run_steps
            if (s.eval_result or {}).get("execution_success") is False
            or (s.eval_result or {}).get("success") is False
        ]
        failure_count = len(failed_steps)

        if failure_count > 0:
            # Collect distinct error messages (truncated) from failed steps
            error_snippets: list[str] = []
            seen_errors: set[str] = set()
            for s in failed_steps:
                err = (s.eval_result or {}).get("error") or ""
                if err:
                    snippet = err[:120].replace("\n", " ").strip()
                    if snippet and snippet not in seen_errors:
                        seen_errors.add(snippet)
                        error_snippets.append(snippet)
            why = f"implementation failures ({failure_count} of {steps_spent} steps failed)"
            if error_snippets:
                why += "; errors: " + "; ".join(error_snippets[:2])
        elif len(scores) >= 2 and (scores[-1] - scores[0]) < 0:
            why = f"regressed from best {best_score:.4f} to {scores[-1]:.4f}"
        elif len(scores) >= 3:
            # Check if last min(steps_spent, 5) scores improved less than epsilon
            window = scores[-min(steps_spent, 5):]
            improvement = max(window) - window[0] if window else 0.0
            if improvement < 1e-6:
                why = f"plateaued at score {best_score:.4f} for {len(window)} steps without improvement"
            else:
                why = "superseded by better approach"
        else:
            why = "superseded by better approach"

        failed.append(FailedApproach(
            name=label,
            best_score_achieved=best_score,
            why_abandoned=why,
            steps_spent=steps_spent,
        ))

    # Sort by steps_spent descending (most-invested first)
    failed.sort(key=lambda fa: fa.steps_spent, reverse=True)
    return failed


def _edit_history_summary(steps: list) -> str:
    """
    Summarize edit_type counts over the last 4 steps.

    Looks for "edit_type" in step.eval_result or step.belief_state.
    Returns a short string like "refine×2, rewrite×1, unknown×1".
    """
    last_steps = steps[-4:] if len(steps) >= 4 else steps
    counts: Counter = Counter()
    for step in last_steps:
        edit_type = None
        if step.eval_result:
            edit_type = step.eval_result.get("edit_type")
        if not edit_type and step.belief_state:
            edit_type = step.belief_state.get("edit_type")
        if not edit_type:
            # Fall back to decision action_type
            edit_type = step.decision.action_type or "unknown"
        counts[edit_type] += 1

    if not counts:
        return "no steps"

    parts = [f"{k}x{v}" for k, v in counts.most_common()]
    return ", ".join(parts)
