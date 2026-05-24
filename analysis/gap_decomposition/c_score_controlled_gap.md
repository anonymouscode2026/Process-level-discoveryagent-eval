# Analysis C — Score-controlled gap on cross-group striking cases

Setup: filter the 81 striking cases (Q1) to the subset where one model is in FRONTIER and the other is in OSS. For each cross-group case, compute per-feature `signed_diff = z_frontier - z_oss` on the 24 striking-case profile features (label-independent + label-aggregated, z-scored within task across the 6 models on the (model, task) cell mean). Then aggregate signed_diff statistics across cases.

- striking cases (total): 81
- cross-group cases (frontier × OSS): **40**
- distinct tasks covered: 16 / 24

## Headline

- n_cross_group_cases = **40**
- n_distinct_tasks = **16**

## Top-10 features by |mean signed_diff|

| rank | feature | mean signed_diff (z) | consistency | paired t p-value | interpretation |
|---:|---|---:|---:|---:|---|
| 1 | `diversity_entropy_q2` | -1.483 | 0.85 | 7.74e-10 | frontier lower on `diversity_entropy_q2` |
| 2 | `diversity_entropy_q3` | -1.255 | 0.82 | 1.95e-07 | frontier lower on `diversity_entropy_q3` |
| 3 | `diversity_entropy_q1` | -1.249 | 0.82 | 1.84e-07 | frontier lower on `diversity_entropy_q1` |
| 4 | `diversity_entropy_q4` | -1.223 | 0.78 | 5.85e-06 | frontier lower on `diversity_entropy_q4` |
| 5 | `num_unique_strategies` | -1.100 | 0.78 | 7.67e-05 | frontier lower on `num_unique_strategies` |
| 6 | `punctuation_index_top3` | +1.089 | 0.70 | 1.06e-05 | frontier higher on `punctuation_index_top3` |
| 7 | `overall_revert_rate` | -0.995 | 0.68 | 1.49e-04 | frontier reverts less (overall_revert_rate) |
| 8 | `thrashing_index` | -0.982 | 0.78 | 6.14e-05 | frontier lower on `thrashing_index` |
| 9 | `rewrite_ratio` | -0.818 | 0.75 | 5.08e-05 | frontier refines more (rewrites less) |
| 10 | `mode_choice_ratio` | +0.818 | 0.75 | 5.08e-05 | frontier higher on `mode_choice_ratio` |

## Direction agreement with Analysis D

- top-10 of C: 10 features
- of those, **10/10** carry the same sign in D's frontier_minus_oss.
- Apples-to-apples top-10 overlap (D ranking restricted to the same $24$ striking-case profile features that C uses): **6/10** -- `diversity_entropy_q1`, `diversity_entropy_q2`, `diversity_entropy_q4`, `mode_choice_ratio`, `overall_revert_rate`, `rewrite_ratio`.
- For reference, D's top-10 over the full $60$-feature set shares 2 feature(s) with C's top-10 (`mode_choice_ratio`, `rewrite_ratio`); D's full-feature top is dominated by per-label `edit_type_*_frac` and `*_revert_rate` features that C's 24-feature profile does not include, so the apples-to-apples comparison is the meaningful one.

## Selected case studies (largest cos_dist cross-group cases, excluding signal_processing)

### `llm_sql` — `gpt-5.4` (frontier) vs `qwen3-next-80b-instruct` (OSS)

- score_frontier = 0.405; score_oss = 0.447; |Δs| = 0.042
- cos_dist = 1.769
- top-5 features by |signed_diff| (frontier − OSS, z units):
    - `avg_plateau_length`: -3.254
    - `punctuation_index_top3`: +3.195
    - `diversity_entropy_q1`: -3.141
    - `plateau_escape_rate`: +3.141
    - `error_recovery_rate`: -3.085

### `eplb` — `gpt-5.4` (frontier) vs `qwen3-next-80b-instruct` (OSS)

- score_frontier = 0.313; score_oss = 0.332; |Δs| = 0.019
- cos_dist = 1.717
- top-5 features by |signed_diff| (frontier − OSS, z units):
    - `diversity_entropy_q2`: -3.138
    - `diversity_entropy_q3`: -3.053
    - `mode_choice_ratio`: +2.990
    - `rewrite_ratio`: -2.990
    - `diversity_entropy_q4`: -2.841

## Sanity-check status

- n_cross_group in [30, 81]: 40 — PASS
- Apples-to-apples top-10 overlap (D restricted to 24 striking features) ≥ 3: 6 — PASS
- Direction agreement with D on every feature in C's top-10: 10/10 — PASS
- top-10 consistency rate ≥ 0.6 on every feature — PASS
