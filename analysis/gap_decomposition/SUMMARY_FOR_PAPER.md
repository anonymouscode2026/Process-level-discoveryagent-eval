# Section 6.4 — Summary for paper

Two analyses decompose the frontier-vs-OSS gap into a process signature and a score-controlled per-feature gap.

## Analysis D headline (binary fingerprint classifier)

A binary classifier trained to recover `is_frontier` from the 60-feature behavioural profile, under 24-fold leave-one-task-out CV, reaches a mean accuracy of **0.8079** (macro-F1 **0.8077**), a factor of **1.616× chance**. The three highest-importance features are `edit_type_bug_fix_frac`, `edit_type_algorithm_switch_frac`, `worst_regression_magnitude`; their directional signs (frontier − OSS) are -0.214 on `edit_type_bug_fix_frac`, -0.169 on `edit_type_algorithm_switch_frac`, +0.162 on `worst_regression_magnitude`.

## Analysis C headline (score-controlled gap on striking cases)

Of the 81 striking same-score, different-process cases, **40** are cross-group (one frontier model, one OSS model), spanning **16/24** tasks. The strongest within-case gaps are `diversity_entropy_q2` (-1.483z, consistency 0.85, p=7.7e-10); `diversity_entropy_q3` (-1.255z, consistency 0.82, p=1.9e-07); `diversity_entropy_q1` (-1.249z, consistency 0.82, p=1.8e-07). Top-10 of C agree in sign with D on **10/10** features.

## Combined claim sentence (paper draft)

> Frontier-vs-OSS is not a single-axis score difference: it has a process signature recoverable from the behavioural profile alone (LOTO accuracy 0.808, 1.62× chance), with the strongest fingerprint axes being `edit_type_bug_fix_frac`, `edit_type_algorithm_switch_frac`, `worst_regression_magnitude` (Analysis D); restricted to the 40 score-matched frontier × OSS pairs, this signature persists with the largest within-case gaps on `diversity_entropy_q2`, `diversity_entropy_q3`, `diversity_entropy_q1` (Analysis C; agreement with D on 10/10 features).

## Saved artefacts

- `d_binary_fingerprint.md` — full Analysis D report
- `d_binary_fingerprint_per_fold.csv` — per-fold accuracy + F1
- `d_feature_ranking.csv` — full 60-feature ranking with importance and frontier−OSS direction
- `c_score_controlled_gap.md` — full Analysis C report
- `c_per_feature_gap.csv` — per-feature signed_diff statistics for the 24 striking-case features
- `c_cross_group_cases.csv` — per-case signed_diff vector for each cross-group striking case
