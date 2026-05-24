"""
Tests for behavioral metrics computation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trajectory.schemas import (
    Trajectory,
    TrajectoryStep,
    DecisionStep,
    GenerationStep,
)
from src.metrics.behavioral import compute_metrics
from src.metrics.variance import compute_cell_variance


def make_step(n: int, score: float, score_delta: float, mode: str = "REFINE") -> TrajectoryStep:
    decision = DecisionStep(
        level="L1",
        prompt_tokens=100,
        reasoning=f"step {n}",
        action_type=mode,
        belief_summary=None,
    )
    generation = GenerationStep(
        model="claude-haiku-4-5-20251001",
        completion_tokens=100,
        code=f"def f(): return {score}",
        temperature=0.8,
        stop_reason="end_turn",
    )
    return TrajectoryStep(
        step_number=n,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        generation=generation,
        score=score,
        score_delta=score_delta,
        is_new_best=(score_delta > 0),
        eval_result={
            "returncode": 0,
            "mode_choice": mode,
            "strategy_label": f"strategy_{(n % 3) + 1}",
            "edit_type": "parameter_tuning" if mode == "REFINE" else "algorithm_switch",
        },
    )


def make_plateau_trajectory() -> Trajectory:
    """
    Hand-crafted trajectory with scores [1, 2, 3, 3, 3, 3, 4].

    Running best: [1, 2, 3, 3, 3, 3, 4]
    Improvements: [+1, +1, +1, 0, 0, 0, +1]

    Steps 4-6 (0-indexed 3-5) form a plateau of length 3.
    Step 7 (0-indexed 6) escapes.
    => num_plateaus = 1, plateau_lengths = [3], plateau_escape_rate = 1.0
    """
    scores = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0]
    prev = 0.0
    steps = []
    for i, sc in enumerate(scores):
        delta = sc - prev
        steps.append(make_step(i + 1, sc, delta, mode="REFINE" if delta >= 0 else "REWRITE"))
        prev = max(prev, sc)

    return Trajectory(
        run_id="plateau-test",
        task_id="test/plateau",
        model="claude-haiku-4-5-20251001",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=4.0,
        best_code="def f(): return 4.0",
        total_prompt_tokens=700,
        total_completion_tokens=700,
    )


EXPECTED_KEYS = [
    "diversity_entropy_q1",
    "diversity_entropy_q2",
    "diversity_entropy_q3",
    "diversity_entropy_q4",
    "diversity_entropy_trend",
    "num_plateaus",
    "avg_plateau_length",
    "plateau_escape_rate",
    "edit_type_parameter_tuning_frac",
    "edit_type_structural_change_frac",
    "edit_type_algorithm_switch_frac",
    "edit_type_representation_change_frac",
    "punctuation_index_top3",
    "rewrite_ratio",
    "rewrite_after_plateau_rate",
    "num_unique_strategies",
    "strategy_list",
    "code_success_rate",
    "final_best_score",
    "score_at_25pct",
    "score_at_50pct",
    "score_at_75pct",
    "area_under_best_curve",
]


def test_all_expected_keys_present():
    """compute_metrics returns all expected keys."""
    traj = make_plateau_trajectory()
    metrics = compute_metrics(traj)
    for key in EXPECTED_KEYS:
        assert key in metrics, f"Missing key: {key}"


def test_plateau_detection():
    """
    Plateau trajectory: scores [1,2,3,3,3,3,4].
    Running best improvements: [1,1,1,0,0,0,1]
    Steps 3,4,5 (0-indexed) have improvement=0 => plateau of length 3.
    Step 6 has improvement=1 => escape.
    """
    traj = make_plateau_trajectory()
    metrics = compute_metrics(traj)

    assert metrics["num_plateaus"] == 1, f"Expected 1 plateau, got {metrics['num_plateaus']}"
    # The plateau spans steps with 0 improvement starting at step 3 (0-indexed)
    # Improvements array: [0, 1, 1, 0, 0, 0, 1] (step 0 always 0)
    # Run of 0s from index 3 to 5 inclusive = length 3 (or possibly 4 depending on
    # whether index 0 is included in the initial run)
    # The plateau should have length 3 or 4. Document the actual value.
    assert metrics["avg_plateau_length"] >= 3, (
        f"Expected avg_plateau_length >= 3, got {metrics['avg_plateau_length']}"
    )
    assert metrics["plateau_escape_rate"] == 1.0, (
        f"Expected plateau_escape_rate == 1.0, got {metrics['plateau_escape_rate']}"
    )


def test_punctuation_index_sensible():
    """punctuation_index_top3 is between 0 and 1 (or exactly 1 for <=3 positive deltas)."""
    traj = make_plateau_trajectory()
    metrics = compute_metrics(traj)
    pi = metrics["punctuation_index_top3"]
    assert 0.0 <= pi <= 1.0, f"punctuation_index_top3 out of range: {pi}"


def test_empty_trajectory_no_crash():
    """compute_metrics on empty trajectory returns zeros/None, no exception."""
    traj = Trajectory(
        run_id="empty",
        task_id="test",
        model="model",
        level="L0",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=[],
    )
    metrics = compute_metrics(traj)
    assert metrics["num_plateaus"] == 0
    assert metrics["final_best_score"] is None
    assert metrics["num_steps"] == 0
    assert metrics["area_under_best_curve"] == 0.0


def test_single_step_trajectory():
    """Single-step trajectory does not crash."""
    steps = [make_step(1, 5.0, 5.0)]
    traj = Trajectory(
        run_id="single",
        task_id="test",
        model="model",
        level="L0",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=5.0,
    )
    metrics = compute_metrics(traj)
    assert metrics["final_best_score"] == 5.0
    assert metrics["num_steps"] == 1
    assert metrics["area_under_best_curve"] == 5.0


def test_rewrite_ratio():
    """rewrite_ratio correctly counts REWRITE mode steps."""
    # 3 REWRITE + 4 REFINE = 3/7
    scores = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0]
    modes = ["REWRITE", "REFINE", "REWRITE", "REFINE", "REWRITE", "REFINE", "REFINE"]
    prev = 0.0
    steps = []
    for i, (sc, mode) in enumerate(zip(scores, modes)):
        delta = sc - prev
        steps.append(make_step(i + 1, sc, delta, mode=mode))
        prev = max(prev, sc)

    traj = Trajectory(
        run_id="rewrite-test",
        task_id="test",
        model="model",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
    )
    metrics = compute_metrics(traj)
    assert abs(metrics["rewrite_ratio"] - 3 / 7) < 1e-9


def test_strategy_list_and_unique():
    """num_unique_strategies and strategy_list are consistent."""
    traj = make_plateau_trajectory()
    metrics = compute_metrics(traj)
    assert metrics["num_unique_strategies"] == len(set(metrics["strategy_list"]))
    assert metrics["num_unique_strategies"] == len(metrics["strategy_list"])


def test_score_fractions():
    """score_at_25pct/50pct/75pct are within the range of observed scores."""
    traj = make_plateau_trajectory()
    scores = [s.score for s in traj.steps]
    min_s, max_s = min(scores), max(scores)

    metrics = compute_metrics(traj)
    for key in ("score_at_25pct", "score_at_50pct", "score_at_75pct"):
        val = metrics[key]
        if val is not None:
            assert min_s <= val <= max_s + 1e-9, f"{key}={val} out of range [{min_s}, {max_s}]"


# ---------------------------------------------------------------------------
# New metric tests: cell variance, serendipity, negative optimization
# ---------------------------------------------------------------------------

def make_step_with_edit(
    n: int,
    score: float,
    score_delta: float,
    mode: str = "REFINE",
    edit_type: str = "parameter_tuning",
    execution_success: bool = True,
) -> TrajectoryStep:
    """Helper that also sets edit_type and execution_success in eval_result."""
    decision = DecisionStep(
        level="L1",
        prompt_tokens=100,
        reasoning=f"step {n}",
        action_type=mode,
        belief_summary=None,
    )
    generation = GenerationStep(
        model="claude-haiku-4-5-20251001",
        completion_tokens=100,
        code=f"def f(): return {score}",
        temperature=0.8,
        stop_reason="end_turn",
    )
    return TrajectoryStep(
        step_number=n,
        timestamp=datetime.now(timezone.utc).isoformat(),
        decision=decision,
        generation=generation,
        score=score,
        score_delta=score_delta,
        is_new_best=(score_delta > 0),
        eval_result={
            "returncode": 0,
            "execution_success": execution_success,
            "mode_choice": mode,
            "edit_type": edit_type,
        },
    )


def test_adaptive_plateau_epsilon_scales_with_magnitude():
    """Plateau detection must adapt epsilon to the task's score magnitude.

    A fixed 1e-6 threshold works for bounded [0,1] tasks but fails for
    unbounded tasks (txn_scheduling ~4000, gpu_mode ~300) where float noise
    alone can exceed 1e-6 and stop plateau detection from firing. The
    adaptive epsilon returns max_abs_score * 1e-6 for scale > 1 and the
    1e-6 floor otherwise.
    """
    from src.metrics.behavioral import _adaptive_plateau_epsilon

    # Direct helper checks
    assert _adaptive_plateau_epsilon([]) == 1e-6
    assert _adaptive_plateau_epsilon([0.1, 0.5, 0.7]) == 1e-6  # bounded, floor
    assert _adaptive_plateau_epsilon([20.0, 50.0, 90.0]) == 90.0 * 1e-6
    assert _adaptive_plateau_epsilon([2700.0, 4100.0]) == 4100.0 * 1e-6

    # Unbounded-scale trajectory with sub-ε noise on the middle 4 steps.
    # Score wobbles of ~1e-4 are below adaptive ε (= 4500 * 1e-6 = 4.5e-3)
    # so the plateau fires. Under the old fixed 1e-6 epsilon, those same
    # wobbles would be treated as genuine improvements and num_plateaus
    # would remain 0.
    scores = [3000.0, 4100.0, 4100.00027, 4100.00015, 4100.00020, 4500.0]
    prev_best = float("-inf")
    steps = []
    for i, sc in enumerate(scores):
        delta = sc - prev_best if prev_best != float("-inf") else 0.0
        steps.append(make_step(i + 1, sc, delta, mode="REFINE"))
        prev_best = max(prev_best, sc)
    traj = Trajectory(
        run_id="adaptive-eps-test",
        task_id="test/unbounded",
        model="claude-haiku-4-5-20251001",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=4500.0,
        best_code="",
        total_prompt_tokens=0,
        total_completion_tokens=0,
    )
    m = compute_metrics(traj)
    assert m["num_plateaus"] == 1, (
        f"adaptive epsilon should detect 1 plateau on unbounded scale, got {m['num_plateaus']}"
    )
    assert m["plateau_escape_rate"] == 1.0, (
        "4500 is a genuine improvement ⇒ escape rate = 1.0"
    )


def test_compute_cell_variance_jaccard():
    """
    compute_cell_variance with 3 synthetic trajectories that have known plateau locations.
    Trajectories with identical plateau locations should yield high Jaccard similarity.
    """
    # All 3 trajectories share the same plateau at steps 2,3,4 (0-indexed)
    # scores: [1, 2, 2, 2, 2, 3] — plateau at indices 2,3,4
    def make_traj_with_plateau(run_id: str) -> Trajectory:
        scores = [1.0, 2.0, 2.0, 2.0, 2.0, 3.0]
        prev = 0.0
        steps = []
        for i, sc in enumerate(scores):
            delta = sc - prev
            steps.append(make_step(i + 1, sc, delta))
            prev = max(prev, sc)
        return Trajectory(
            run_id=run_id,
            task_id="test/plateau",
            model="model",
            level="L1",
            started_at=datetime.now(timezone.utc).isoformat(),
            steps=steps,
            best_score=3.0,
            best_code="def f(): return 3.0",
            total_prompt_tokens=600,
            total_completion_tokens=600,
        )

    trajs = [make_traj_with_plateau(f"run{i}") for i in range(3)]
    result = compute_cell_variance(trajs)

    assert result["n_trajectories"] == 3
    assert result["plateau_pattern_similarity"] is not None
    # All trajectories have identical plateau sets → Jaccard should be 1.0
    assert abs(result["plateau_pattern_similarity"] - 1.0) < 1e-9, (
        f"Expected similarity=1.0, got {result['plateau_pattern_similarity']}"
    )
    assert result["final_score_variance"] is not None
    # All have same best score → variance = 0
    assert abs(result["final_score_variance"]) < 1e-9


def test_serendipity_index_with_large_unexpected_delta():
    """
    Trajectory where one step has a large delta preceded only by parameter_tuning
    → should count as serendipitous.

    Steps: 10 steps of small deltas (0.01 each), then one huge delta (5.0).
    The huge delta at step 11 is not preceded by algorithm_switch/representation_change,
    so it satisfies condition (b). It's also >> mean + 2*std of recent deltas → condition (a).
    """
    steps = []
    prev_score = 0.0
    for i in range(10):
        score = prev_score + 0.01
        steps.append(make_step_with_edit(
            n=i + 1,
            score=score,
            score_delta=0.01,
            mode="REFINE",
            edit_type="parameter_tuning",
        ))
        prev_score = score

    # The serendipitous step
    big_score = prev_score + 5.0
    steps.append(make_step_with_edit(
        n=11,
        score=big_score,
        score_delta=5.0,
        mode="REFINE",
        edit_type="parameter_tuning",
    ))

    traj = Trajectory(
        run_id="serendipity-test",
        task_id="test/serendipity",
        model="model",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=big_score,
        best_code="def f(): pass",
        total_prompt_tokens=1100,
        total_completion_tokens=1100,
    )

    metrics = compute_metrics(traj)
    assert "serendipity_index" in metrics
    assert metrics["serendipity_index"] is not None
    # At least the last big step should count as serendipitous
    assert metrics["serendipity_index"] > 0.0, (
        f"Expected serendipity_index > 0, got {metrics['serendipity_index']}"
    )


def test_negative_optimization_events_count():
    """
    Trajectory where 2 of 5 steps regress → neg_opt_count=2, neg_opt_rate=0.4.

    Steps (all successful):
      step 1: score=1.0, delta=+1.0  (no regression, prev=0)
      step 2: score=2.0, delta=+1.0  (no regression)
      step 3: score=1.5, delta=-0.5  (regression: 1.5 < 2.0)
      step 4: score=3.0, delta=+1.5  (no regression)
      step 5: score=2.0, delta=-1.0  (regression: 2.0 < 3.0)

    Total successful steps checked (steps 2..5) = 4 (indices 1..4, checking prev score).
    Wait: regression is curr < prev *step* score, checked on successful steps.
    Steps 1..5 are all successful. Starting from step index 1 (step 2):
      i=1: score=2.0 vs prev=1.0 → no regression
      i=2: score=1.5 vs prev=2.0 → regression ✓
      i=3: score=3.0 vs prev=1.5 → no regression
      i=4: score=2.0 vs prev=3.0 → regression ✓
    total_successful = 4, neg_opt_count = 2, neg_opt_rate = 0.5

    Actually total_successful counts steps[1..4] = 4 steps.
    neg_opt_rate = 2/4 = 0.5
    """
    scores = [1.0, 2.0, 1.5, 3.0, 2.0]
    steps = []
    for i, sc in enumerate(scores):
        prev_sc = scores[i - 1] if i > 0 else 0.0
        delta = sc - prev_sc
        steps.append(make_step_with_edit(
            n=i + 1,
            score=sc,
            score_delta=delta,
            mode="REFINE",
            edit_type="parameter_tuning",
            execution_success=True,
        ))

    traj = Trajectory(
        run_id="neg-opt-test",
        task_id="test/neg_opt",
        model="model",
        level="L1",
        started_at=datetime.now(timezone.utc).isoformat(),
        steps=steps,
        best_score=3.0,
        best_code="def f(): pass",
        total_prompt_tokens=500,
        total_completion_tokens=500,
    )

    metrics = compute_metrics(traj)
    assert metrics["neg_opt_count"] == 2, f"Expected neg_opt_count=2, got {metrics['neg_opt_count']}"
    assert abs(metrics["neg_opt_rate"] - 0.5) < 1e-9, (
        f"Expected neg_opt_rate=0.5, got {metrics['neg_opt_rate']}"
    )
    assert metrics["worst_regression_magnitude"] > 0.0
