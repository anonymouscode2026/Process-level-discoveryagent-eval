# Analysis D — Binary fingerprint classifier (frontier vs OSS)

Setup mirrors the 6-class Q4 pipeline exactly: per-(task, level, run) row, RandomForest(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1), within-task rank-normalised features ((rank-1)/(n-1) in [0,1]) with NaN replaced by 0.5, 24-fold leave-one-task-out CV. Label `is_frontier`: 1 for {claude-opus-4-6, gpt-5.4, gemini-3.1-pro-preview}, 0 for {deepseek-v3.2, gpt-oss-120b, qwen3-next-80b-instruct}.

- rows used: 1728 (frontier=864, oss=864)
- features used: 60 (final_best_score, score_at_*pct, abs_score, area_under_best_curve excluded)
- folds: 24
- chance accuracy = 0.5000

## Headline

- **mean LOTO accuracy = 0.8079**
- **macro-F1 = 0.8077**
- **chance-adjusted ratio = 1.616×** (vs 6-class 3.49× over chance=0.167)
- **Verdict**: frontier-vs-OSS fingerprint signature is recoverable from behavioural profile alone, far above chance and above the 6-class accuracy of 0.582

## Per-class F1

| class | F1 |
|---|---:|
| frontier | 0.8131 |
| oss | 0.8024 |

## 2×2 confusion matrix (rows = true, cols = predicted)

Raw counts:

| true \\ pred | OSS | frontier |
|---|---:|---:|
| **OSS** | 674 | 190 |
| **frontier** | 142 | 722 |

Row-normalised rates:

| true \\ pred | OSS | frontier |
|---|---:|---:|
| **OSS** | 0.780 | 0.220 |
| **frontier** | 0.164 | 0.836 |

## Top-15 features by mean RF importance

| rank | feature | importance | frontier−OSS (z) | description |
|---:|---|---:|---:|---|
| 1 | `edit_type_bug_fix_frac` | 0.0660 | -0.214 | fraction of edits classified as bug fixes |
| 2 | `edit_type_algorithm_switch_frac`† | 0.0542 | -0.169 | fraction of edits that switch algorithm |
| 3 | `worst_regression_magnitude`† | 0.0536 | +0.162 | max single-step score regression |
| 4 | `edit_type_no_change_frac`† | 0.0383 | -0.124 | fraction of edits classified as no-change |
| 5 | `edit_type_code_cleanup_frac` | 0.0336 | -0.155 | fraction of edits classified as code cleanup |
| 6 | `code_success_rate` | 0.0331 | +0.100 | fraction of edits that produced runnable code |
| 7 | `edit_type_harness_or_scope_change_frac` | 0.0328 | -0.122 | fraction of harness/scope-change edits |
| 8 | `mode_choice_ratio`† | 0.0323 | +0.160 | agent's REFINE/REWRITE mode mix |
| 9 | `rewrite_ratio`† | 0.0321 | -0.160 | fraction of edits that are full rewrites vs refinements |
| 10 | `algorithm_refinement_revert_rate` | 0.0271 | -0.142 | fraction of algorithm refinement edits that are later reverted |
| 11 | `neg_opt_count` | 0.0266 | +0.161 | number of edits that worsen the current best score |
| 12 | `edit_type_memory_optimization_frac` | 0.0256 | +0.081 | fraction of memory-optimization edits |
| 13 | `edit_type_parameter_tuning_frac` | 0.0231 | +0.090 | fraction of edits that tune parameters |
| 14 | `diversity_entropy_q2` | 0.0230 | -0.161 | entropy of strategy distribution in trajectory quartile 2 |
| 15 | `neg_opt_rate` | 0.0214 | +0.148 | fraction of edits that worsen the current best score |

† marks features that also appear in the 6-class classifier's top-5 (`rewrite_ratio`, `mode_choice_ratio`, `worst_regression_magnitude`, `edit_type_no_change_frac`, `edit_type_algorithm_switch_frac`).

## Feature direction (top-10)

Sign of `frontier_minus_oss` indicates which class scores higher on the rank-normalised feature ([0,1] within task). Positive ⇒ frontier higher; negative ⇒ OSS higher.

| rank | feature | frontier−OSS (z) | reads as |
|---:|---|---:|---|
| 1 | `edit_type_bug_fix_frac` | -0.214 | frontier lower on `edit_type_bug_fix_frac` |
| 2 | `edit_type_algorithm_switch_frac` | -0.169 | frontier lower on `edit_type_algorithm_switch_frac` |
| 3 | `worst_regression_magnitude` | +0.162 | frontier higher on `worst_regression_magnitude` |
| 4 | `edit_type_no_change_frac` | -0.124 | frontier lower on `edit_type_no_change_frac` |
| 5 | `edit_type_code_cleanup_frac` | -0.155 | frontier lower on `edit_type_code_cleanup_frac` |
| 6 | `code_success_rate` | +0.100 | frontier higher on `code_success_rate` |
| 7 | `edit_type_harness_or_scope_change_frac` | -0.122 | frontier lower on `edit_type_harness_or_scope_change_frac` |
| 8 | `mode_choice_ratio` | +0.160 | frontier higher on `mode_choice_ratio` |
| 9 | `rewrite_ratio` | -0.160 | frontier lower on `rewrite_ratio` |
| 10 | `algorithm_refinement_revert_rate` | -0.142 | frontier lower on `algorithm_refinement_revert_rate` |

## Per-fold breakdown

Mean per-fold accuracy: 0.8079 (σ = 0.0894, range [0.6111, 0.9583])

Mean per-fold macro-F1: 0.8071 (σ = 0.0902)

## Sanity-check status

- Class balance: frontier=864, OSS=864 — PASS
- Accuracy > 6-class 0.582: 0.8079 > 0.582 — PASS
- Held-out task absent from training (spot-check on `cross_entropy`) — PASS
