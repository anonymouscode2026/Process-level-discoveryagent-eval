"""
compute_metrics.py — Recompute the FULL behavioral metrics CSV using
postproc'd labels as the SINGLE source of truth for primary_label and
was_reverted.

NEVER reads strategy_label / strategy_name / edit_type from trajectory JSON.

Outputs ~50-60 cols:
  - 4 id  (model, task, level, run_id)
  - A: score-based (label-independent)
  - B: edit-type distribution (18 closed labels)
  - C: diversity entropy quartiles
  - D: strategic decisions
  - E: was_reverted-derived

Run:
    python3 -W ignore scripts/compute_metrics.py 2>&1 | tee data/strategy_labels/_compute.log
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parents[1]

ROOT = _PKG_ROOT
TRAJ_ROOT = ROOT / "data" / "trajectories"
LABELS_ROOT = ROOT / "data" / "strategy_labels" / "postproc"
OUT_CSV = ROOT / "data" / "strategy_labels" / "behavioral_metrics.csv"

MODELS = [
    "claude-opus-4-6",
    "deepseek-v3.2",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    "gpt-oss-120b",
    "qwen3-next-80b-instruct",
]

CLOSED_VOCAB = [
    "bug_fix",
    "harness_or_scope_change",
    "no_change",
    "algorithm_switch",
    "representation_change",
    "data_structure_change",
    "search_space_change",
    "algorithm_refinement",
    "initialization_change",
    "parameter_tuning",
    "autotune_configuration",
    "loop_optimization",
    "memory_optimization",
    "kernel_restructuring",
    "precision_handling",
    "feature_engineering",
    "code_cleanup",
    "other",
]

SENTINEL_RE = re.compile(r"^__.*__$")
DELIBERATE_SHIFT = {"algorithm_switch", "representation_change"}


# ----------------------------------------------------------------------------
# Helpers ported from src/metrics/behavioral.py — IDENTICAL formulas
# ----------------------------------------------------------------------------
def _shannon_entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    e = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            e -= p * math.log2(p)
    return e


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _adaptive_plateau_epsilon(scores: list[float]) -> float:
    if not scores:
        return 1e-6
    scale = max(abs(s) for s in scores if s is not None) if any(s is not None for s in scores) else 0.0
    return 1e-6 if scale <= 1.0 else scale * 1e-6


def _quarter_split(items: list, n: int) -> tuple[list, list, list, list]:
    if n == 0:
        return [], [], [], []
    q_size = n / 4.0
    qs: list[list] = [[], [], [], []]
    for i, x in enumerate(items):
        idx = min(int(i / q_size), 3)
        qs[idx].append(x)
    return tuple(qs)  # type: ignore


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def _is_sentinel(label: str | None) -> bool:
    if label is None:
        return True
    return bool(SENTINEL_RE.match(label))


def _is_eval_success(eval_result) -> bool:
    if not eval_result or not isinstance(eval_result, dict):
        return False
    if eval_result.get("execution_success") is True:
        return True
    if eval_result.get("returncode") == 0:
        return True
    if eval_result.get("success") is True:
        return True
    return False


def _step_action(step: dict) -> str:
    """Return REWRITE / REFINE / '' for the step (sources: eval_result.mode_choice or decision.action_type)."""
    er = step.get("eval_result")
    if isinstance(er, dict):
        mc = er.get("mode_choice")
        if mc:
            mc = mc.upper()
            if "REWRITE" in mc:
                return "REWRITE"
            if "REFINE" in mc:
                return "REFINE"
    dec = step.get("decision")
    if isinstance(dec, dict):
        act = (dec.get("action_type") or "").upper()
        if "REWRITE" in act:
            return "REWRITE"
        if "REFINE" in act:
            return "REFINE"
    return ""


def _load_traj(model: str, task: str, level: str, run: int) -> dict | None:
    p = TRAJ_ROOT / model / task / f"{level}__run{run}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_labels(model: str, task: str, level: str, run: int) -> dict | None:
    p = LABELS_ROOT / model / task / f"{level}__run{run}.labels.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Per-trajectory metrics
# ----------------------------------------------------------------------------
def compute_one(traj: dict, labels_doc: dict, model: str, task: str, level: str, run: int) -> dict:
    steps = traj.get("steps") or []
    n = len(steps)
    labels = labels_doc.get("labels") or []

    # Verify alignment: labels[i].step_idx should be i, and there must be one per step.
    if len(labels) != n:
        raise RuntimeError(
            f"label_count_mismatch: {model}/{task}/{level}/run{run}: "
            f"steps={n} labels={len(labels)}"
        )
    for i, L in enumerate(labels):
        if L.get("step_idx") != i:
            raise RuntimeError(
                f"step_idx_mismatch: {model}/{task}/{level}/run{run} at i={i}: "
                f"label.step_idx={L.get('step_idx')}"
            )

    # ------------------------------------------------------------------
    # Per-step extracts
    # ------------------------------------------------------------------
    primary_labels: list[str] = [L.get("primary_label") or "other" for L in labels]
    was_reverted_arr: list[bool] = [bool(L.get("was_reverted")) for L in labels]
    scores: list[float] = []
    score_deltas: list[float] = []
    eval_success: list[bool] = []
    actions: list[str] = []
    for s in steps:
        sc = s.get("score")
        scores.append(float(sc) if sc is not None else 0.0)
        sd = s.get("score_delta")
        score_deltas.append(float(sd) if sd is not None else 0.0)
        eval_success.append(_is_eval_success(s.get("eval_result")))
        actions.append(_step_action(s))

    is_sentinel = [_is_sentinel(lb) for lb in primary_labels]
    nonsent_idx = [i for i in range(n) if not is_sentinel[i]]

    # Running best
    running_best: list[float] = []
    cur = float("-inf")
    for sc in scores:
        if sc > cur:
            cur = sc
        running_best.append(cur)

    final_best_score = running_best[-1] if running_best else None

    def score_at_frac(f):
        if n == 0:
            return None
        idx = max(0, min(int(math.floor(f * n)), n - 1))
        return running_best[idx]

    score_at_25pct = score_at_frac(0.25)
    score_at_50pct = score_at_frac(0.50)
    score_at_75pct = score_at_frac(0.75)

    # ------------------------------------------------------------------
    # A. Plateau detection (adaptive epsilon, ports behavioral.py)
    # ------------------------------------------------------------------
    num_plateaus = 0
    plateau_lengths: list[int] = []
    plateau_escapes = 0
    plateau_end_indices: set[int] = set()

    if n >= 3:
        epsilon = _adaptive_plateau_epsilon(scores)
        improvements = [0.0] + [
            running_best[i] - running_best[i - 1] for i in range(1, n)
        ]
        i = 0
        while i < n:
            if improvements[i] < epsilon:
                run_start = i
                while i < n and improvements[i] < epsilon:
                    i += 1
                run_length = i - run_start
                if run_length >= 3:
                    num_plateaus += 1
                    plateau_lengths.append(run_length)
                    if i < n:
                        plateau_end_indices.add(i)
                        if improvements[i] >= epsilon:
                            plateau_escapes += 1
            else:
                i += 1

    avg_plateau_length = (
        sum(plateau_lengths) / len(plateau_lengths) if plateau_lengths else 0.0
    )
    plateau_escape_rate = plateau_escapes / num_plateaus if num_plateaus else 0.0

    # ------------------------------------------------------------------
    # A. Negative optimization
    # ------------------------------------------------------------------
    total_successful = 0
    neg_opt_count = 0
    worst_regression = 0.0
    for i in range(1, n):
        if not eval_success[i]:
            continue
        total_successful += 1
        prev_sc = scores[i - 1]
        curr_sc = scores[i]
        if curr_sc < prev_sc:
            neg_opt_count += 1
            drop = prev_sc - curr_sc
            if drop > worst_regression:
                worst_regression = drop
    neg_opt_rate = neg_opt_count / total_successful if total_successful else 0.0
    worst_regression_magnitude = worst_regression if neg_opt_count > 0 else 0.0

    # ------------------------------------------------------------------
    # A. Serendipity index — port logic; use primary_label-derived "deliberate"
    # ------------------------------------------------------------------
    count_positive = 0
    count_unexpected = 0
    recent_pos: list[float] = []
    for i in range(n):
        d = score_deltas[i]
        if d <= 0:
            continue
        count_positive += 1
        cond_a = False
        if len(recent_pos) >= 3:
            mean_r = sum(recent_pos) / len(recent_pos)
            var_r = sum((x - mean_r) ** 2 for x in recent_pos) / len(recent_pos)
            std_r = math.sqrt(var_r)
            cond_a = d > mean_r + 2 * std_r
        # Past 2 steps did NOT use a deliberate strategy shift
        past_2_deliberate = False
        for j in (i - 1, i - 2):
            if 0 <= j < n and primary_labels[j] in DELIBERATE_SHIFT:
                past_2_deliberate = True
                break
        cond_b = not past_2_deliberate
        if cond_a and cond_b:
            count_unexpected += 1
        recent_pos.append(d)
        if len(recent_pos) > 10:
            recent_pos.pop(0)
    serendipity_idx = count_unexpected / count_positive if count_positive else 0.0

    # ------------------------------------------------------------------
    # A. Code success rate
    # ------------------------------------------------------------------
    code_success_rate = sum(eval_success) / n if n else 0.0

    # ------------------------------------------------------------------
    # A. early_exit_flag
    # ------------------------------------------------------------------
    step_budget = traj.get("step_budget") or (
        traj.get("metadata") or {}).get("step_budget") or n
    early_exit_flag = 1 if n < step_budget else 0

    # ------------------------------------------------------------------
    # A. error_recovery_rate
    # Of all failed steps, fraction where running_best improves within next 3 steps.
    # ------------------------------------------------------------------
    fail_total = 0
    fail_recovered = 0
    for i in range(n):
        if eval_success[i]:
            continue
        fail_total += 1
        rb_at_i = running_best[i]
        for k in range(1, 4):
            j = i + k
            if 0 <= j < n and running_best[j] > rb_at_i:
                fail_recovered += 1
                break
    error_recovery_rate = fail_recovered / fail_total if fail_total else 0.0

    # ------------------------------------------------------------------
    # B. Edit-type fractions over non-sentinel steps
    # ------------------------------------------------------------------
    nonsent_labels = [primary_labels[i] for i in nonsent_idx]
    nonsent_total = len(nonsent_labels)
    label_counter = Counter(nonsent_labels)
    edit_fracs: dict[str, float] = {}
    for lb in CLOSED_VOCAB:
        edit_fracs[f"edit_type_{lb}_frac"] = (
            label_counter.get(lb, 0) / nonsent_total if nonsent_total else 0.0
        )
    num_unique_strategies = len(set(nonsent_labels))
    rewrite_ratio = (
        sum(1 for a in actions if a == "REWRITE") / n if n else 0.0
    )

    # ------------------------------------------------------------------
    # C. Diversity entropy quartiles (sentinels excluded — drop in place)
    # ------------------------------------------------------------------
    q1, q2, q3, q4 = _quarter_split(
        [(primary_labels[i], is_sentinel[i]) for i in range(n)], n
    )

    def _ent_q(qlist):
        return _shannon_entropy([lb for lb, sn in qlist if not sn])

    diversity_entropy_q1 = _ent_q(q1)
    diversity_entropy_q2 = _ent_q(q2)
    diversity_entropy_q3 = _ent_q(q3)
    diversity_entropy_q4 = _ent_q(q4)
    diversity_entropy_trend = _linear_slope(
        [diversity_entropy_q1, diversity_entropy_q2, diversity_entropy_q3, diversity_entropy_q4]
    )

    # ------------------------------------------------------------------
    # D. Strategic decisions
    # ------------------------------------------------------------------
    # mode_choice_ratio = REFINE_count / total
    refine_count = sum(1 for a in actions if a == "REFINE")
    mode_choice_ratio = refine_count / n if n else 0.0

    # rewrite_after_plateau_rate
    rewrite_after_plateau_rate = 0.0
    if num_plateaus > 0:
        rewrites_after_plateau = sum(
            1 for idx in plateau_end_indices if 0 <= idx < n and actions[idx] == "REWRITE"
        )
        rewrite_after_plateau_rate = rewrites_after_plateau / num_plateaus

    # punctuation_index_top3 = top-3 label freq concentration over non-sentinel
    if nonsent_total:
        top3_sum = sum(c for _, c in label_counter.most_common(3))
        punctuation_index_top3 = top3_sum / nonsent_total
    else:
        punctuation_index_top3 = 0.0

    # ------------------------------------------------------------------
    # E. Was-reverted-derived
    # ------------------------------------------------------------------
    nonsent_reverted = [was_reverted_arr[i] for i in nonsent_idx]
    nonsent_reverted_count = sum(nonsent_reverted)
    overall_revert_rate = (
        nonsent_reverted_count / nonsent_total if nonsent_total else 0.0
    )

    revert_rates: dict[str, float] = {}
    for lb in CLOSED_VOCAB:
        denom = label_counter.get(lb, 0)
        if denom == 0:
            revert_rates[f"{lb}_revert_rate"] = float("nan")
        else:
            num = sum(
                1 for i in nonsent_idx
                if primary_labels[i] == lb and was_reverted_arr[i]
            )
            revert_rates[f"{lb}_revert_rate"] = num / denom

    # thrashing_index: # consecutive (i, i+1) both reverted (within non-sentinel) / nonsent_total
    thrashing_count = 0
    for k in range(len(nonsent_idx) - 1):
        i1 = nonsent_idx[k]
        i2 = nonsent_idx[k + 1]
        # Strict consecutive in original step ordering OR consecutive in non-sentinel sequence?
        # Spec: "consecutive steps i and i+1 both was_reverted=True". Use original sequence.
        if i2 == i1 + 1 and was_reverted_arr[i1] and was_reverted_arr[i2]:
            thrashing_count += 1
    thrashing_index = thrashing_count / nonsent_total if nonsent_total else 0.0

    # recovery_after_revert_rate: of reverted steps in this trajectory, fraction
    # where running_best improves within next 5 steps. NaN if no reverted steps.
    revert_indices = [i for i in nonsent_idx if was_reverted_arr[i]]
    if not revert_indices:
        recovery_after_revert_rate = float("nan")
    else:
        recovered = 0
        for i in revert_indices:
            rb_at_i = running_best[i]
            for k in range(1, 6):
                j = i + k
                if 0 <= j < n and running_best[j] > rb_at_i:
                    recovered += 1
                    break
        recovery_after_revert_rate = recovered / len(revert_indices)

    # initial_generation_reverted_rate: 1 if step 0 sentinel was_reverted else 0
    if n >= 1 and is_sentinel[0]:
        initial_generation_reverted_rate = 1.0 if was_reverted_arr[0] else 0.0
    else:
        initial_generation_reverted_rate = float("nan")

    # ------------------------------------------------------------------
    # Assemble row
    # ------------------------------------------------------------------
    row = {
        "model": model,
        "task": task,
        "level": level,
        "run_id": f"{task}__{level}__run{run}",
        # A
        "final_best_score": final_best_score,
        "score_at_25pct": score_at_25pct,
        "score_at_50pct": score_at_50pct,
        "score_at_75pct": score_at_75pct,
        "num_plateaus": num_plateaus,
        "avg_plateau_length": avg_plateau_length,
        "plateau_escape_rate": plateau_escape_rate,
        "neg_opt_rate": neg_opt_rate,
        "neg_opt_count": neg_opt_count,
        "serendipity_index": serendipity_idx,
        "worst_regression_magnitude": worst_regression_magnitude,
        "early_exit_flag": early_exit_flag,
        "error_recovery_rate": error_recovery_rate,
        "code_success_rate": code_success_rate,
        # B
        "num_unique_strategies": num_unique_strategies,
        **edit_fracs,
        "rewrite_ratio": rewrite_ratio,
        # C
        "diversity_entropy_q1": diversity_entropy_q1,
        "diversity_entropy_q2": diversity_entropy_q2,
        "diversity_entropy_q3": diversity_entropy_q3,
        "diversity_entropy_q4": diversity_entropy_q4,
        "diversity_entropy_trend": diversity_entropy_trend,
        # D
        "mode_choice_ratio": mode_choice_ratio,
        "rewrite_after_plateau_rate": rewrite_after_plateau_rate,
        "punctuation_index_top3": punctuation_index_top3,
        # E
        **revert_rates,
        "overall_revert_rate": overall_revert_rate,
        "thrashing_index": thrashing_index,
        "recovery_after_revert_rate": recovery_after_revert_rate,
        "initial_generation_reverted_rate": initial_generation_reverted_rate,
    }
    return row


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
LEVELS = ["L0", "L0f", "L1", "L2b"]
RUNS = [1, 2, 3]


def list_tasks(model: str) -> list[str]:
    d = LABELS_ROOT / model
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def main():
    rows: list[dict] = []
    issues: list[str] = []
    for model in MODELS:
        tasks = list_tasks(model)
        print(f"[{model}] {len(tasks)} tasks", flush=True)
        for task in tasks:
            for level in LEVELS:
                for run in RUNS:
                    traj = _load_traj(model, task, level, run)
                    labels_doc = _load_labels(model, task, level, run)
                    if traj is None or labels_doc is None:
                        issues.append(
                            f"missing: model={model} task={task} {level} run{run} "
                            f"traj={traj is not None} labels={labels_doc is not None}"
                        )
                        continue
                    try:
                        row = compute_one(traj, labels_doc, model, task, level, run)
                    except Exception as e:
                        issues.append(
                            f"error: {model}/{task}/{level}/run{run}: {type(e).__name__}: {e}"
                        )
                        continue
                    rows.append(row)
        print(f"[{model}] processed; rows so far: {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: rows={len(df)} cols={len(df.columns)}", flush=True)
    print("Columns:")
    for c in df.columns:
        print(f"  {c}")
    if issues:
        print(f"\n{len(issues)} issue(s):", file=sys.stderr)
        for i in issues[:50]:
            print("  -", i, file=sys.stderr)
    else:
        print("\nNo issues.", flush=True)


if __name__ == "__main__":
    main()
