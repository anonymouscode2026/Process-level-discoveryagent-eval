"""
Behavioral metrics computation for discovery agent trajectories.

compute_metrics(trajectory) -> dict
Computes all behavioral metrics from an annotated Trajectory object.
"""

from __future__ import annotations

import math
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schemas import Trajectory

from src.metrics.strategy_analyst import get_step_strategy_label
from src.metrics.edit_classifier import get_step_edit_type


def _shannon_entropy(labels: list[str]) -> float:
    """Compute Shannon entropy (base 2) of a list of labels."""
    if not labels:
        return 0.0
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _linear_slope(values: list[float]) -> float:
    """Compute linear regression slope of a list of values (x = 0,1,2,...)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _adaptive_plateau_epsilon(scores: list[float]) -> float:
    """
    Return a plateau-detection epsilon scaled to the trajectory's score magnitude.

    A fixed 1e-6 threshold works for tasks whose evaluator returns values in
    roughly [0, 1] (e.g. signal_processing combined_score, heilbronn_convex
    ratio-to-SOTA), but is WAY too tight for unbounded tasks whose scores
    live on a larger scale:

      - txn_scheduling returns ``1e6 / (1 + makespan)``, typical range ~2700-4200
      - gpu_mode tasks return ``SCORE_SCALE / geom_mean_us``, typical range ~30-3000
      - Frontier-CS research tasks return scores in [0, 100]

    With a fixed 1e-6 threshold, ANY float wobble on these tasks is treated as
    a genuine improvement, so plateau detection never fires and
    ``num_plateaus``, ``avg_plateau_length``, ``plateau_escape_rate``, and
    ``rewrite_after_plateau_rate`` all become degenerate.

    Policy: ``epsilon = max(1e-6, max_abs_score * 1e-6)``. The 1e-6 floor
    preserves original behavior for bounded tasks (scale <= 1). For larger
    scales, epsilon scales proportionally — an improvement must be at least
    1 part per million of the task's observed magnitude to count as "real".

    This uses the scores of the passed-in list, not ``trajectory.best_score``.
    That is, it adapts to the specific run's scale, not a global constant.
    """
    if not scores:
        return 1e-6
    scale = max(abs(s) for s in scores)
    if scale <= 1.0:
        return 1e-6
    return scale * 1e-6


def _quarter_labels(strategy_labels: list[Optional[str]]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split strategy labels into 4 quarters by step index."""
    n = len(strategy_labels)
    if n == 0:
        return [], [], [], []
    q_size = n / 4
    quarters = [[], [], [], []]
    for i, label in enumerate(strategy_labels):
        if label is None:
            continue
        q_idx = min(int(i / q_size), 3)
        quarters[q_idx].append(label)
    return tuple(quarters)


def serendipity_index(trajectory: "Trajectory") -> dict[str, Any]:
    """
    Compute the serendipity index for a trajectory.

    An improvement at step i is "unexpected" if:
    (a) score_delta[i] > mean(recent_deltas) + 2*std(recent_deltas),
        where recent_deltas = last 10 positive deltas before step i.
        Skip condition (a) if fewer than 3 recent positive deltas exist.
    (b) The past 2 steps (i-1, i-2) did NOT have edit_type in
        {"algorithm_switch", "representation_change"}.
        Skip condition (b) if all edit_type annotations are None.

    Returns dict with:
    - serendipity_index: count_unexpected / count_positive_deltas (0.0 if no positives)
    - serendipity_annotations_available: bool (False if all edit_types are None)
    """
    steps = trajectory.steps if trajectory.steps else []
    edit_types = [get_step_edit_type(s) for s in steps]
    annotations_available = any(et is not None for et in edit_types)

    count_positive = 0
    count_unexpected = 0

    recent_positive_deltas: list[float] = []

    for i, s in enumerate(steps):
        delta = s.score_delta if s.score_delta is not None else 0.0
        if delta <= 0:
            continue

        count_positive += 1

        # Condition (a): check if delta is surprisingly large
        cond_a = False
        if len(recent_positive_deltas) >= 3:
            mean_r = sum(recent_positive_deltas) / len(recent_positive_deltas)
            var_r = sum((x - mean_r) ** 2 for x in recent_positive_deltas) / len(recent_positive_deltas)
            std_r = math.sqrt(var_r)
            cond_a = delta > mean_r + 2 * std_r
        else:
            # Not enough baseline — skip this check (treat as not unexpectedly large)
            cond_a = False

        # Condition (b): past 2 steps did NOT use a deliberate strategy shift
        if annotations_available:
            deliberate_types = {"algorithm_switch", "representation_change"}
            past_2_deliberate = False
            for j in [i - 1, i - 2]:
                if 0 <= j < len(steps):
                    et = edit_types[j]
                    if et in deliberate_types:
                        past_2_deliberate = True
                        break
            cond_b = not past_2_deliberate
        else:
            cond_b = True  # skip condition (b)

        if cond_a and cond_b:
            count_unexpected += 1

        # Update recent positive deltas (keep last 10)
        recent_positive_deltas.append(delta)
        if len(recent_positive_deltas) > 10:
            recent_positive_deltas.pop(0)

    si = count_unexpected / count_positive if count_positive > 0 else 0.0

    return {
        "serendipity_index": si,
        "serendipity_annotations_available": annotations_available,
    }


def negative_optimization_events(trajectory: "Trajectory", threshold: float = 0.01) -> dict[str, Any]:
    """
    Detect steps where the agent committed to a worse solution.

    Only counts steps where eval succeeded (execution_success=True or returncode==0).
    A regression is when score < previous step's score.

    Returns dict with:
    - neg_opt_count: number of regressing successful steps
    - neg_opt_count_refine: subset where mode was REFINE/REFINE_FORCED
    - neg_opt_count_rewrite: subset where mode was REWRITE/REWRITE_FORCED
    - neg_opt_rate: neg_opt_count / total_successful_steps
    - worst_regression_magnitude: largest single-step score drop
    - neg_opt_after_plateau_rate: fraction of post-plateau steps that cause regression
    """
    steps = trajectory.steps if trajectory.steps else []
    n = len(steps)

    def is_successful(step) -> bool:
        er = step.eval_result
        if er and isinstance(er, dict):
            if er.get("execution_success") is True:
                return True
            if er.get("returncode") == 0:
                return True
            if er.get("success") is True:
                return True
        return False

    def get_mode(step) -> str:
        mc = None
        if step.eval_result and isinstance(step.eval_result, dict):
            mc = step.eval_result.get("mode_choice")
        if not mc and step.decision:
            act = (step.decision.action_type or "").upper()
            if "REWRITE" in act:
                mc = "REWRITE"
            elif "REFINE" in act:
                mc = "REFINE"
        return (mc or "").upper()

    # Identify plateau-ending step indices
    plateau_end_indices: set[int] = set()
    if n >= 3:
        scores = [s.score for s in steps]
        running_best = []
        cur_best = float("-inf")
        for sc in scores:
            if sc > cur_best:
                cur_best = sc
            running_best.append(cur_best)

        # Adaptive epsilon — scales with the trajectory's own score magnitude
        # so plateau detection works on both bounded ([0,1]) and unbounded
        # (txn_scheduling, gpu_mode) tasks.
        epsilon = _adaptive_plateau_epsilon(scores)

        improvements = [0.0]
        for i in range(1, n):
            improvements.append(running_best[i] - running_best[i - 1])

        i = 0
        while i < n:
            if improvements[i] < epsilon:
                run_start = i
                while i < n and improvements[i] < epsilon:
                    i += 1
                run_length = i - run_start
                if run_length >= 3 and i < n:
                    plateau_end_indices.add(i)  # step immediately after plateau
            else:
                i += 1

    total_successful = 0
    neg_opt_count = 0
    neg_opt_count_refine = 0
    neg_opt_count_rewrite = 0
    worst_regression = 0.0

    # Post-plateau regression tracking
    post_plateau_total = 0
    post_plateau_regressions = 0

    for i in range(1, n):
        step = steps[i]
        prev_step = steps[i - 1]

        if not is_successful(step):
            continue

        total_successful += 1
        prev_score = prev_step.score
        curr_score = step.score

        is_regression = curr_score < prev_score

        if is_regression:
            neg_opt_count += 1
            mode = get_mode(step)
            if "REFINE" in mode:
                neg_opt_count_refine += 1
            elif "REWRITE" in mode:
                neg_opt_count_rewrite += 1

            drop = prev_score - curr_score
            if drop > worst_regression:
                worst_regression = drop

        if i in plateau_end_indices:
            post_plateau_total += 1
            if is_regression:
                post_plateau_regressions += 1

    neg_opt_rate = neg_opt_count / total_successful if total_successful > 0 else 0.0
    worst_regression_magnitude = worst_regression if neg_opt_count > 0 else 0.0
    neg_opt_after_plateau_rate = (
        post_plateau_regressions / post_plateau_total if post_plateau_total > 0 else 0.0
    )

    return {
        "neg_opt_count": neg_opt_count,
        "neg_opt_count_refine": neg_opt_count_refine,
        "neg_opt_count_rewrite": neg_opt_count_rewrite,
        "neg_opt_rate": neg_opt_rate,
        "worst_regression_magnitude": worst_regression_magnitude,
        "neg_opt_after_plateau_rate": neg_opt_after_plateau_rate,
    }


def compute_metrics(trajectory: "Trajectory") -> dict[str, Any]:
    """
    Compute all behavioral metrics from a (possibly annotated) Trajectory.

    Handles empty/partial trajectories gracefully — returns 0.0 or None for
    undefined values.

    Returns a flat dict with all metric keys.
    """
    steps = trajectory.steps if trajectory.steps else []
    n = len(steps)

    # Extract per-step data
    scores = [s.score for s in steps]
    score_deltas = [s.score_delta for s in steps]
    strategy_labels = [get_step_strategy_label(s) for s in steps]
    edit_types = [get_step_edit_type(s) for s in steps]

    # Extract mode choices from eval_result or decision action_type
    mode_choices = []
    for s in steps:
        mc = None
        if s.eval_result and isinstance(s.eval_result, dict):
            mc = s.eval_result.get("mode_choice")
        if not mc and s.decision:
            act = (s.decision.action_type or "").upper()
            if "REWRITE" in act:
                mc = "REWRITE"
            elif "REFINE" in act:
                mc = "REFINE"
        mode_choices.append(mc)

    # Running best scores
    running_best = []
    cur_best = float("-inf")
    for sc in scores:
        if sc > cur_best:
            cur_best = sc
        running_best.append(cur_best)

    # ------------------------------------------------------------------
    # Diversity entropy per quarter
    # ------------------------------------------------------------------
    q1_labels, q2_labels, q3_labels, q4_labels = _quarter_labels(strategy_labels)
    diversity_entropy_q1 = _shannon_entropy(q1_labels)
    diversity_entropy_q2 = _shannon_entropy(q2_labels)
    diversity_entropy_q3 = _shannon_entropy(q3_labels)
    diversity_entropy_q4 = _shannon_entropy(q4_labels)
    diversity_entropy_trend = _linear_slope(
        [diversity_entropy_q1, diversity_entropy_q2, diversity_entropy_q3, diversity_entropy_q4]
    )

    # ------------------------------------------------------------------
    # Plateau detection
    # ------------------------------------------------------------------
    # Plateau = run of >= 3 consecutive steps where improvement in running_best
    # is below an ADAPTIVE epsilon scaled to the trajectory's score magnitude.
    # See _adaptive_plateau_epsilon for the rationale: a fixed 1e-6 floor is
    # fine for bounded tasks but breaks on unbounded scales (txn_scheduling,
    # gpu_mode) where noise and meaningful improvement both far exceed 1e-6.
    # Escape = step immediately following the plateau with improvement >= epsilon.

    num_plateaus = 0
    plateau_lengths = []
    plateau_escapes = 0

    if n >= 3:
        epsilon = _adaptive_plateau_epsilon(scores)

        # Compute improvement per step
        improvements = [0.0]
        for i in range(1, n):
            improvements.append(running_best[i] - running_best[i - 1])

        i = 0
        while i < n:
            # Check if current step starts a plateau (improvement < epsilon)
            if improvements[i] < epsilon:
                run_start = i
                while i < n and improvements[i] < epsilon:
                    i += 1
                run_length = i - run_start
                if run_length >= 3:
                    num_plateaus += 1
                    plateau_lengths.append(run_length)
                    # Check if escape follows
                    if i < n and improvements[i] >= epsilon:
                        plateau_escapes += 1
            else:
                i += 1

    avg_plateau_length = (
        sum(plateau_lengths) / len(plateau_lengths) if plateau_lengths else 0.0
    )
    plateau_escape_rate = (
        plateau_escapes / num_plateaus if num_plateaus > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Edit type fractions
    # ------------------------------------------------------------------
    edit_type_names = [
        "parameter_tuning",
        "structural_change",
        "algorithm_switch",
        "representation_change",
    ]
    valid_edit_types = [et for et in edit_types if et is not None]
    total_edits = len(valid_edit_types)
    edit_fracs = {}
    for et in edit_type_names:
        key = f"edit_type_{et}_frac"
        if total_edits > 0:
            edit_fracs[key] = sum(1 for x in valid_edit_types if x == et) / total_edits
        else:
            edit_fracs[key] = 0.0

    # ------------------------------------------------------------------
    # Punctuation index: sum(top-3 positive score_deltas) / sum(all positive)
    # ------------------------------------------------------------------
    positive_deltas = sorted(
        [d for d in score_deltas if d is not None and d > 0], reverse=True
    )
    sum_positive = sum(positive_deltas)
    sum_top3 = sum(positive_deltas[:3])
    punctuation_index_top3 = sum_top3 / sum_positive if sum_positive > 0 else 0.0

    # ------------------------------------------------------------------
    # Rewrite ratio
    # ------------------------------------------------------------------
    rewrite_ratio = (
        sum(1 for mc in mode_choices if mc and mc.upper() == "REWRITE") / n
        if n > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Rewrite after plateau rate
    # ------------------------------------------------------------------
    rewrite_after_plateau_rate = 0.0
    if num_plateaus > 0 and n >= 3:
        # Must use the SAME adaptive epsilon as the plateau detection block
        # above, otherwise plateau boundaries would mismatch between the two
        # loops and rewrite_after_plateau_rate could exceed 1.0.
        epsilon = _adaptive_plateau_epsilon(scores)

        improvements = [0.0]
        for i in range(1, n):
            improvements.append(running_best[i] - running_best[i - 1])

        rewrites_after_plateau = 0
        i = 0
        while i < n:
            if improvements[i] < epsilon:
                run_start = i
                while i < n and improvements[i] < epsilon:
                    i += 1
                run_length = i - run_start
                if run_length >= 3:
                    # Check if next step was REWRITE
                    if i < n:
                        mc = mode_choices[i]
                        if mc and mc.upper() == "REWRITE":
                            rewrites_after_plateau += 1
            else:
                i += 1
        rewrite_after_plateau_rate = rewrites_after_plateau / num_plateaus

    # ------------------------------------------------------------------
    # Strategy stats
    # ------------------------------------------------------------------
    valid_labels = [lb for lb in strategy_labels if lb is not None]
    num_unique_strategies = len(set(valid_labels))
    strategy_list = sorted(set(valid_labels))

    # ------------------------------------------------------------------
    # Code success rate (steps where eval returned success)
    # ------------------------------------------------------------------
    success_count = 0
    for s in steps:
        er = s.eval_result
        if er and isinstance(er, dict):
            if er.get("returncode") == 0 or er.get("success") is True:
                success_count += 1
    code_success_rate = success_count / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Score snapshots
    # ------------------------------------------------------------------
    final_best_score = running_best[-1] if running_best else None

    def score_at_fraction(frac: float) -> Optional[float]:
        if n == 0:
            return None
        # Use step fraction: running-best at step = floor(frac * n)
        idx = min(int(math.floor(frac * n)), n - 1)
        idx = max(0, idx)
        return running_best[idx]

    score_at_25pct = score_at_fraction(0.25)
    score_at_50pct = score_at_fraction(0.50)
    score_at_75pct = score_at_fraction(0.75)

    # ------------------------------------------------------------------
    # Area under best curve (trapezoidal, x = step index)
    # ------------------------------------------------------------------
    if n >= 2:
        area_under_best_curve = sum(
            (running_best[i] + running_best[i + 1]) / 2.0
            for i in range(n - 1)
        )
    elif n == 1:
        area_under_best_curve = running_best[0]
    else:
        area_under_best_curve = 0.0

    # ------------------------------------------------------------------
    # Serendipity index
    # ------------------------------------------------------------------
    serendipity_result = serendipity_index(trajectory)

    # ------------------------------------------------------------------
    # Negative optimization events
    # ------------------------------------------------------------------
    neg_opt_result = negative_optimization_events(trajectory)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    metrics: dict[str, Any] = {
        "diversity_entropy_q1": diversity_entropy_q1,
        "diversity_entropy_q2": diversity_entropy_q2,
        "diversity_entropy_q3": diversity_entropy_q3,
        "diversity_entropy_q4": diversity_entropy_q4,
        "diversity_entropy_trend": diversity_entropy_trend,
        "num_plateaus": num_plateaus,
        "avg_plateau_length": avg_plateau_length,
        "plateau_escape_rate": plateau_escape_rate,
        **edit_fracs,
        "punctuation_index_top3": punctuation_index_top3,
        "rewrite_ratio": rewrite_ratio,
        "rewrite_after_plateau_rate": rewrite_after_plateau_rate,
        "num_unique_strategies": num_unique_strategies,
        "strategy_list": strategy_list,
        "code_success_rate": code_success_rate,
        "final_best_score": final_best_score,
        "score_at_25pct": score_at_25pct,
        "score_at_50pct": score_at_50pct,
        "score_at_75pct": score_at_75pct,
        "area_under_best_curve": area_under_best_curve,
        # Serendipity
        **serendipity_result,
        # Negative optimization
        **neg_opt_result,
        # Metadata from trajectory
        "run_id": trajectory.run_id,
        "task_id": trajectory.task_id,
        "model_name": trajectory.model,
        "level": trajectory.level,
        "num_steps": n,
    }

    return metrics


def analyze_belief_state_evolution(trajectory) -> dict:
    """
    Analyze how the LLM's self-maintained belief state evolves over a trajectory.

    Only meaningful for L2b trajectories that have belief_state_history.

    Returns dict with:
        n_updates: int — how many times belief was updated
        avg_length_words: float — average belief state length in words
        avg_length_chars: float — average belief state length in chars
        length_trend: str — "growing", "shrinking", "stable"
        self_correction_count: int — times belief explicitly contradicts prior
        self_correction_phrases: list[str] — extracted correction phrases
        consecutive_similarity: list[float] — bag-of-words similarity between consecutive states
    """
    history = getattr(trajectory, "belief_state_history", [])
    if not history:
        return {
            "n_updates": 0,
            "avg_length_words": 0,
            "avg_length_chars": 0,
            "length_trend": "none",
            "self_correction_count": 0,
            "self_correction_phrases": [],
            "consecutive_similarity": [],
        }

    n = len(history)
    lengths_words = [len(h.split()) for h in history]
    lengths_chars = [len(h) for h in history]

    # Length trend
    if n >= 3:
        first_half = sum(lengths_words[:n//2]) / max(n//2, 1)
        second_half = sum(lengths_words[n//2:]) / max(n - n//2, 1)
        ratio = second_half / first_half if first_half > 0 else 1.0
        if ratio > 1.2:
            trend = "growing"
        elif ratio < 0.8:
            trend = "shrinking"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    # Self-correction detection
    correction_markers = [
        "i was wrong", "actually", "correction:", "contrary to",
        "previously thought", "mistaken", "revising", "re-evaluating",
        "not actually", "turns out", "in fact,", "reconsidering",
    ]
    corrections = []
    for i, h in enumerate(history):
        h_lower = h.lower()
        for marker in correction_markers:
            if marker in h_lower:
                # Extract the sentence containing the marker
                for sent in h.split("."):
                    if marker in sent.lower():
                        corrections.append(f"update_{i+1}: {sent.strip()[:150]}")
                        break
                break  # one correction per update is enough

    # Bag-of-words similarity between consecutive states
    similarities = []
    for i in range(1, len(history)):
        words_prev = set(history[i-1].lower().split())
        words_curr = set(history[i].lower().split())
        if words_prev or words_curr:
            intersection = len(words_prev & words_curr)
            union = len(words_prev | words_curr)
            sim = intersection / union if union > 0 else 0.0
            similarities.append(round(sim, 4))

    return {
        "n_updates": n,
        "avg_length_words": round(sum(lengths_words) / n, 1),
        "avg_length_chars": round(sum(lengths_chars) / n, 1),
        "length_trend": trend,
        "self_correction_count": len(corrections),
        "self_correction_phrases": corrections,
        "consecutive_similarity": similarities,
    }
