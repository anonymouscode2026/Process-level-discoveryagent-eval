# Full-panel FRONTIER vs OSS — per-feature Δz

Per-feature mean signed difference Δz = z_frontier − z_oss, computed on the full 1,728-trajectory panel. Within-task z-scoring across the 72 same-task trajectories, per-task FRONTIER and OSS group means in z-units, paired-t on the 24 task-level Δz values against H0: mean = 0. Bonferroni correction across the 24 features tested (α = 0.05 / 24 ≈ 0.0021). FRONTIER = {claude-opus-4-6, gpt-5.4, gemini-3.1-pro-preview}; OSS = {deepseek-v3.2, gpt-oss-120b, qwen3-next-80b-instruct}.

## Top 10 features by |mean Δz|

| Rank | Feature | mean Δz | consistency | Bonferroni p | RF rank | Direction |
|---:|---|---:|---:|---:|---:|---|
| 1 | `mode_choice_ratio` | +0.590 | 1.00 | 1.91e-10 | 8 | frontier favors refinement |
| 2 | `rewrite_ratio` | -0.590 | 1.00 | 1.91e-10 | 9 | OSS rewrites more |
| 3 | `diversity_entropy_q2` | -0.552 | 0.92 | 5.40e-05 | 14 | OSS more diverse |
| 4 | `diversity_entropy_q1` | -0.511 | 0.88 | 1.69e-03 | 16 | OSS more diverse |
| 5 | `diversity_entropy_q3` | -0.487 | 0.83 | 2.80e-04 | 21 | OSS more diverse |
| 6 | `neg_opt_count` | +0.483 | 0.79 | 4.43e-03 | 11 | (frontier higher) |
| 7 | `diversity_entropy_q4` | -0.483 | 0.88 | 3.29e-04 | 18 | OSS more diverse |
| 8 | `neg_opt_rate` | +0.481 | 0.83 | 4.66e-03 | 15 | (frontier higher) |
| 9 | `overall_revert_rate` | -0.458 | 0.88 | 4.47e-04 | 17 | OSS reverts more |
| 10 | `punctuation_index_top3` | +0.455 | 0.67 | 2.80e-02 | 23 | frontier concentrates on top-3 strategies |

## Features that fail Bonferroni-corrected significance (p ≥ 0.05)

| Feature | mean Δz | Bonferroni p |
|---|---:|---:|
| `worst_regression_magnitude` | +0.381 | 2.511e-01 |
| `num_unique_strategies` | -0.350 | 4.711e-01 |
| `num_plateaus` | +0.223 | 3.558e-01 |
| `code_success_rate` | +0.213 | 1.000e+00 |
| `rewrite_after_plateau_rate` | -0.207 | 5.019e-01 |
| `avg_plateau_length` | -0.179 | 3.828e-01 |
| `plateau_escape_rate` | +0.176 | 5.749e-01 |
| `recovery_after_revert_rate` | +0.159 | 1.000e+00 |
| `error_recovery_rate` | +0.073 | 1.000e+00 |
| `serendipity_index` | +0.054 | 1.000e+00 |
| `diversity_entropy_trend` | -0.029 | 1.000e+00 |
| `initial_generation_reverted_rate` | -0.020 | 1.000e+00 |
| `early_exit_flag` | -0.017 | 1.000e+00 |

## Directional cross-check vs Slice 2 (score-matched)

Sign agreement between full-panel mean Δz and the existing score-matched mean signed difference, on the 24-feature testing set:

- **Sign agreement: 16/24** features.
- Disagreements:
  - `worst_regression_magnitude`: full-panel +0.381, score-matched -0.225
  - `num_plateaus`: full-panel +0.223, score-matched -0.188
  - `code_success_rate`: full-panel +0.213, score-matched -0.155
  - `avg_plateau_length`: full-panel -0.179, score-matched +0.083
  - `plateau_escape_rate`: full-panel +0.176, score-matched -0.268
  - `error_recovery_rate`: full-panel +0.073, score-matched -0.341
  - `initial_generation_reverted_rate`: full-panel -0.020, score-matched +0.000
  - `early_exit_flag`: full-panel -0.017, score-matched +0.093

## Top-10 overlap with Slice 2 score-matched top-10

- Common to both top-10: **8** features: `diversity_entropy_q1`, `diversity_entropy_q2`, `diversity_entropy_q3`, `diversity_entropy_q4`, `mode_choice_ratio`, `overall_revert_rate`, `punctuation_index_top3`, `rewrite_ratio`
- Only in full-panel top-10: `neg_opt_count`, `neg_opt_rate`
- Only in score-matched top-10: `num_unique_strategies`, `thrashing_index`
