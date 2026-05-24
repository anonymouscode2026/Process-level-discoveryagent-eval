#!/usr/bin/env python3
"""
relabel_strategies.py — closed-vocabulary strategy re-annotation.

Two-pass pipeline:
  Pass 1: assign primary_label (1 of 18) + confidence + label_notes per step
          (skips step_idx=0; uses sentinel __INITIAL_GENERATION__).
  Pass 2: assign was_reverted boolean + reason per step (runs on all steps
          including step_idx=0).

Output: <traj>.raw.json sibling files. Trajectories are NEVER modified.

Usage:
    python scripts/relabel_strategies.py \
        --pilot \
        --concurrency 30 \
        --output-dir data/strategy_labels/raw

Or full run:
    python scripts/relabel_strategies.py \
        --concurrency 60 \
        --output-dir data/strategy_labels/raw
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import difflib
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import anthropic
from anthropic import AsyncAnthropic

_PKG_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELER_MODEL = "claude-opus-4-6"
LABELER_TEMPERATURE = 0
LABELER_MAX_TOKENS = 1000

PRIMARY_LABELS = [
    "bug_fix",
    "algorithm_switch",
    "algorithm_refinement",
    "representation_change",
    "data_structure_change",
    "search_space_change",
    "initialization_change",
    "parameter_tuning",
    "autotune_configuration",
    "loop_optimization",
    "memory_optimization",
    "kernel_restructuring",
    "precision_handling",
    "feature_engineering",
    "harness_or_scope_change",
    "code_cleanup",
    "no_change",
    "other",
]
LABEL_REGEX = re.compile(r"^(" + "|".join(re.escape(l) for l in PRIMARY_LABELS) + r")$")
SENTINEL_INITIAL = "__INITIAL_GENERATION__"
SENTINEL_FAILED = "__ANNOTATION_FAILED__"

# Pilot scope: 4 tasks × gpt-oss-120b × 4 levels × 3 runs = 48 trajectories
PILOT_MODELS = ["gpt-oss-120b"]
PILOT_TASKS = ["fcs_algo_46", "signal_processing", "gemm_optimization", "grammar_fuzzing"]
PILOT_LEVELS = ["L0", "L0f", "L1", "L2"]  # L2 is L2b in raw data
PILOT_RUNS = ["run1", "run2", "run3"]

FULL_MODELS = ["claude-opus-4-6", "deepseek-v3.2", "gemini-3.1-pro-preview", "gpt-5.4", "gpt-oss-120b", "qwen3-next-80b-instruct"]

CODE_TRUNC_PASS1 = 4000  # chars
REASONING_TRUNC = 500
DIFF_TRUNC_PASS2 = 2000

# ---------------------------------------------------------------------------
# Per-task family lists (for algorithm_switch vs algorithm_refinement)
# ---------------------------------------------------------------------------

TASK_FAMILY_LISTS: dict[str, str] = {
    "fcs_algo_46": """\
families:
  exact_solver: [constraint_programming, integer_programming, branch_and_bound]
  stochastic_metaheuristic: [simulated_annealing, genetic_algorithm, tabu_search, random_restart]
  constructive_heuristic: [priority_dispatch, greedy_construction, dispatch_rule]
  hand_designed_special_case: [special_case_for_small_N]
notes: |
  expected_approaches: constraint_programming, genetic_algorithm, simulated_annealing,
  priority_dispatch, tabu_search.
  SA→tabu_search is INTRA-family (algorithm_refinement).
  SA→constraint_programming is INTER-family (algorithm_switch).
""",
    "signal_processing": """\
families:
  frequency_domain_transform: [fourier_methods, wavelet_analysis, spectral_decomposition]
  parametric_model_fit: [polynomial_fitting, exponential_fitting, kalman_filter]
  learned_predictor: [neural_approximation, regression_model]
  hand_designed_filter: [moving_average, median_filter, savitzky_golay]
notes: |
  expected_approaches: fourier_methods, wavelet_analysis, parameter_tuning,
  neural_approximation, polynomial_fitting.
  fourier→wavelet is INTRA-family (frequency_domain_transform).
  fourier→neural_approximation is INTER-family.
  parameter_tuning is NOT a family — it's its own primary label.
""",
    "gemm_optimization": """\
families:
  naive_implementation: [direct_loop, reference_unfused]
  tiled_blocked: [block_tiling, register_tiling]
  warp_level_primitives: [warp_shuffle, warp_reduce, ballot]
  shared_memory_staged: [shared_memory_buffering, cache_optimization]
  tensor_core_mma: [tensor_core, wmma, mma_sync]
  library_call_wrapper: [cublas_call, cudnn_call]
notes: |
  expected_approaches: tiling_strategy, vectorization, loop_reordering,
  cache_optimization, tensor_core.
  tiling_strategy→tensor_core is INTER-family (algorithm_switch).
  vectorization, loop_reordering belong to loop_optimization (a primary label,
  not a family). Do NOT call those algorithm_switch.
""",
    "grammar_fuzzing": """\
families:
  grammar_based: [grammar_mutation, rule_based, cfg_expansion]
  coverage_feedback: [coverage_guided, hybrid_coverage]
  generative_ml: [generative_model, llm_based_generation]
  evolutionary_search: [evolutionary_fuzzing, genetic_mutation]
notes: |
  expected_approaches: grammar_mutation, coverage_guided, generative_model,
  rule_based, evolutionary_fuzzing. Each maps to a distinct family.
  grammar_mutation→rule_based is INTRA-family.
  grammar_mutation→evolutionary_fuzzing is INTER-family.
""",
    # -- non-pilot tasks (still defined so full-run works without changes) --
    "hexagon_packing": """\
families:
  geometric_construction: [geometric_construction, lattice_packing, symmetry_based]
  physical_simulation: [gradient_descent, force_directed, repulsion_simulation]
  stochastic_search: [simulated_annealing, evolutionary_algorithm, random_restart]
  exact_solver: [linear_programming, branch_and_bound]
  hand_designed_special_case: [special_case_for_12_hexagons]
notes: |
  expected_approaches: gradient_descent, simulated_annealing, geometric_construction,
  force_directed, linear_programming.
""",
    "heilbronn_convex": """\
families:
  geometric_construction: [geometric_construction, symmetric_placement]
  physical_simulation: [gradient_descent, force_directed]
  stochastic_search: [simulated_annealing, random_search, evolutionary_algorithm]
  exact_solver: [convex_optimization, linear_programming]
  hand_designed_special_case: [special_case_for_14_points]
notes: |
  expected_approaches: simulated_annealing, gradient_descent, random_search,
  convex_optimization, evolutionary_algorithm.
""",
    "llm_sql": """\
families:
  greedy_heuristic: [greedy_heuristic, column_reordering, statistical_analysis]
  priority_based: [prefix_optimization, priority_scoring]
  graph_based: [dependency_graph, topological_sort]
  ml_predictor: [classifier_based, learned_router]
  exact_optimal: [dynamic_programming, ilp]
notes: |
  expected_approaches: column_reordering, prefix_optimization, greedy_heuristic,
  dynamic_programming, statistical_analysis.
""",
    "txn_scheduling": """\
families:
  greedy_heuristic: [greedy_heuristic, dispatch_rule, priority_scheduling]
  priority_based: [priority_dispatch, edd, srpt]
  graph_based: [dependency_graph, flow_based, topological_sort]
  ml_predictor: [classifier_based, learned_dispatcher]
  exact_optimal: [constraint_optimization, ilp, branch_and_bound]
notes: |
  expected_approaches: priority_scheduling, dependency_graph, greedy_heuristic,
  constraint_optimization, batch_processing.
""",
    "vecadd": """\
families:
  naive_implementation: [direct_loop, reference_unfused]
  tiled_blocked: [block_tiling, vectorized_tiles]
  warp_level_primitives: [warp_shuffle, ballot, warp_reduce]
  shared_memory_staged: [shared_memory_buffering]
  tensor_core_mma: [tensor_core]
  library_call_wrapper: [cublas, cuda_runtime]
notes: |
  expected_approaches: memory_coalescing, thread_optimization, vectorization,
  tiling_strategy, shared_memory.
""",
    "grayscale": """\
families:
  naive_implementation: [direct_loop]
  tiled_blocked: [block_tiling, pixel_batching]
  warp_level_primitives: [warp_optimization, warp_shuffle]
  shared_memory_staged: [shared_memory]
  tensor_core_mma: [tensor_core]
  library_call_wrapper: [cudnn]
notes: |
  expected_approaches: memory_coalescing, thread_optimization, pixel_batching,
  shared_memory, warp_optimization.
""",
    "trimul": """\
families:
  naive_implementation: [direct_loop, reference_unfused]
  tiled_blocked: [tiling_strategy, register_tiling]
  warp_level_primitives: [warp_shuffle, warp_reduce]
  shared_memory_staged: [shared_memory]
  tensor_core_mma: [tensor_core, mma_sync]
  matrix_decomposition_based: [matrix_decomposition]
  library_call_wrapper: [cublas]
notes: |
  expected_approaches: tiling_strategy, shared_memory, loop_unrolling, tensor_core,
  matrix_decomposition.
""",
    "cross_entropy": """\
families:
  naive_implementation: [direct_loop, log_softmax_then_nll]
  tiled_blocked: [block_tiling]
  warp_level_primitives: [warp_reduce, warp_shuffle]
  shared_memory_staged: [shared_memory_buffering]
  fused_kernel: [kernel_fusion, fused_softmax_ce]
  tensor_core_mma: [tensor_core]
  library_call_wrapper: [pytorch_native, cudnn]
notes: |
  expected_approaches: block_tiling, memory_coalescing, numerical_stability,
  kernel_fusion, vectorization.
""",
    "group_gemm": """\
families:
  naive_implementation: [direct_loop_over_groups]
  tiled_blocked: [block_tiling, batched_tiles]
  warp_level_primitives: [warp_optimization, warp_shuffle]
  shared_memory_staged: [shared_memory]
  tensor_core_mma: [tensor_core]
  library_call_wrapper: [cublas_grouped]
notes: |
  expected_approaches: block_tiling, memory_coalescing, loop_unrolling,
  shared_memory, warp_optimization.
""",
    "llm_router": """\
families:
  threshold_policy: [threshold_policy, fixed_threshold]
  feature_engineering: [feature_engineering, learned_features]
  classifier_based: [classifier_based, learned_router, ml_predictor]
  cost_aware_routing: [cost_aware_routing, budget_aware]
  heuristic_rules: [heuristic_rules, hand_crafted_rules]
notes: |
  expected_approaches: threshold_policy, feature_engineering, classifier_based,
  cost_aware_routing, heuristic_rules.
""",
    "eplb": """\
families:
  greedy_heuristic: [greedy_balancing]
  exact_optimal: [integer_programming, constraint_optimization]
  graph_based: [graph_partitioning]
  evolutionary_search: [evolutionary_algorithm]
  ml_predictor: [learned_balancer]
notes: |
  expected_approaches: greedy_balancing, integer_programming, graph_partitioning,
  evolutionary_algorithm, constraint_optimization.
""",
    # FCS algo defaults: domain-default
    "_default_fcs_algo": """\
families:
  brute_force: [brute_force, exhaustive_search]
  greedy: [greedy_algorithm, greedy_construction, greedy_approximation]
  dynamic_programming: [dynamic_programming]
  divide_and_conquer: [divide_and_conquer]
  graph_algorithm: [graph_search, graph_coloring, hamiltonian_path]
  randomized: [randomized, simulated_annealing, monte_carlo]
  approximation_or_heuristic: [heuristic_search, approximation, local_search]
  number_theoretic: [number_theory, constructive_algorithm, mathematical_optimization]
notes: |
  Default fcs_algo families. The agent operates on a competitive-programming task.
""",
}

def get_family_block(task_name: str) -> str:
    """Return per-task family block; falls back to fcs_algo default."""
    if task_name in TASK_FAMILY_LISTS:
        return TASK_FAMILY_LISTS[task_name]
    if task_name.startswith("fcs_algo_"):
        return TASK_FAMILY_LISTS["_default_fcs_algo"]
    return "families: (none defined for this task)\n"

# ---------------------------------------------------------------------------
# System prompt (cacheable — same for all calls within a task)
# ---------------------------------------------------------------------------

PASS1_SYSTEM_TEMPLATE = """\
You are a code-change classifier for an LLM agent's optimization trajectory. Given a single step (previous code → new code), assign EXACTLY ONE primary label from the closed vocabulary, plus a confidence and optional notes.

OUTPUT FORMAT — strict JSON, no prose, no markdown, no code fences:
{{"primary_label": "<one_of_18_labels>", "confidence": <0.0-1.0>, "label_notes": "<string or null>"}}

EXTENDED LABEL DEFINITIONS (boundary cases — read carefully):

bug_fix vs algorithm_refinement:
- A change that "fixes a defect that was producing wrong outputs" is bug_fix even if it touches algorithm internals.
- A change that "improves correctness on edge cases without a prior crash/wrong-output" leans algorithm_refinement (adding defensive handling for an edge case).
- Numerical-stability tweaks that prevent NaN/inf are bug_fix; those that improve precision without preventing failure are precision_handling.

algorithm_switch vs algorithm_refinement (the most important boundary):
- Use the per-task family list. INTER-family change → algorithm_switch. INTRA-family change → algorithm_refinement.
- "Heuristic refinement", "scoring tweak", "reordering iteration", "different pruning predicate", "modify weights" — all INTRA-family unless they cross to a different family.
- "Complete rewrite" or "from scratch" with different overall structure → algorithm_switch.
- A change that swaps a SUB-COMPONENT (e.g., swapping the inner loop's distance metric) within an otherwise unchanged outer algorithm → algorithm_refinement.
- Replacing PSO with SA (both stochastic_metaheuristic) → algorithm_refinement (intra-family).
- Replacing SA with constraint_programming (different families) → algorithm_switch.

representation_change vs data_structure_change (often confused):
- representation_change is about WHAT the data MEANS — different encoding of the same problem. Example: switching from Cartesian (x,y) coordinates to polar (r,θ) representation of the same 2D points.
- data_structure_change is about HOW the data is STORED with the same meaning. Example: changing a list of points to a hash grid, or adding a memoization cache.
- Mnemonic: representation = different basis; data_structure = different container.

search_space_change vs algorithm_refinement:
- search_space_change modifies the SET of candidates considered. Example: increasing beam width from 4 to 16, or restricting candidates to even integers.
- algorithm_refinement modifies HOW the algorithm navigates a fixed candidate set. Example: changing the priority order in which candidates are explored.
- If a change adds an early-termination predicate that prunes the search → search_space_change (it changes which candidates get evaluated).
- If a change reorders evaluation but eventually visits the same candidates → algorithm_refinement.

parameter_tuning vs autotune_configuration:
- parameter_tuning = changing scalar constants in the runtime path (lr, max_iter, threshold, batch_size, regularization weight).
- autotune_configuration = changing the SET of candidate configurations a tuner searches (Triton @autotune list, autotuner key columns, candidate set add/remove).
- Mnemonic: parameter_tuning = "set the value to X"; autotune_configuration = "consider these candidate values".

loop_optimization vs memory_optimization vs kernel_restructuring (GPU triplet):
- loop_optimization = transformations to loops: unroll, fuse, split, vectorize, software-pipeline.
- memory_optimization = how data flows through memory: coalescing, shared memory tiles, prefetching, alignment, cache hints.
- kernel_restructuring = high-level kernel structure: kernel fusion/split, grid/block dim changes, kernel specialization.
- A change that adds shared memory tiling for coalesced loads is BOTH memory_optimization AND loop_optimization. Pick by the dominant intent. If the diff says "for coalescing" → memory_optimization; if "to enable unrolling" → loop_optimization.

precision_handling — when in doubt:
- Adding `+ eps` to prevent log(0) → bug_fix (prevents NaN).
- Adding log-sum-exp stabilization → precision_handling (numerical accuracy).
- Casting fp32 → fp16 for speed → precision_handling.
- Changing accumulator dtype to avoid overflow → bug_fix if it was crashing, else precision_handling.

harness_or_scope_change is a strong attractor — apply when:
- Adding `print()` debug calls
- Adding logging / tracing instrumentation
- Adding input validation at the entry point
- Restricting task scope (e.g., "only handle N <= 100", early-return for large inputs)
- Modifying the evaluation harness itself
- Changing output format/schema for downstream consumers

code_cleanup vs no_change:
- code_cleanup = semantic-preserving but non-trivial: rename variables, remove dead code, reorganize, add/remove comments with substance.
- no_change = trivial: only whitespace, only comment-only changes that add no information, or literally identical code.

other — used very rarely:
- Use only when the diff genuinely doesn't fit any of the 17 substantive labels. label_notes is REQUIRED.
- "I'm not sure" alone is NOT enough — pick the closest label and lower confidence instead.

CLOSED VOCABULARY (18 labels, in priority order — when multiple apply, smallest-numbered wins):

1.  bug_fix — fixes a correctness defect (out-of-bounds, wrong indexing, type error, infinite loop, NaN/inf, crash, divide-by-zero). Even if the fix changes the algorithm, label as bug_fix.
2.  harness_or_scope_change — modifies evaluation harness, input parsing, output format, logging, debug prints, or restricts task scope (e.g., "only handle N <= 100"). Even if some algorithm code is touched, if the *intent* is harness/scope, label here.
3.  no_change — trivial whitespace, comment-only, or literally identical code.
4.  algorithm_switch — replaces one algorithm FAMILY with another fundamentally different one (see per-task family list below). NOT same family with refinements.
5.  representation_change — rephrases what the problem IS: LP→SAT, geometric→algebraic, primal→dual, (x,y)→polar.
6.  data_structure_change — modifies HOW solutions are stored: list→hash map, dense→sparse, raw points→hash grid, add LRU/cache/memoization.
7.  search_space_change — modifies WHICH SET we search over: R^2→Z^2, bounded→unbounded, larger neighborhood, expanded beam, added termination predicate, search_strategy_change.
8.  algorithm_refinement — same family, structural changes within it (heuristic refinement, scoring tweak, reorder, pruning predicate).
9.  initialization_change — modifies how the algorithm is initialized: seed, starting solution, prior, multi-start config, weight init scheme, initial temperature.
10. parameter_tuning — changes ONLY numeric constants (lr, threshold, max_iter, batch_size, momentum). NO structural code change.
11. autotune_configuration — changes the autotuner's config space (Triton @autotune list, candidate add/remove/prune, tuning keys).
12. loop_optimization — loop-level: unrolling, fusion, splitting, tiling, vectorization, software pipelining, compiler hints.
13. memory_optimization — memory access patterns, coalescing, shared memory, prefetching, alignment, cache hints.
14. kernel_restructuring — GPU kernel structural: kernel fusion/split, grid/block dim, kernel specialization, parallelization_strategy, operation_fusion.
15. precision_handling — numerical precision (fp32↔fp16↔bf16), mixed precision, casting, log-sum-exp / numerical stability tweaks.
16. feature_engineering — adds/removes/transforms input features (ML / ADRS / LLM-router tasks).
17. code_cleanup — refactoring with NO semantic change: rename, dead-code removal, reorganization, comments, whitespace, variable_renaming.
18. other — doesn't fit any above. Use SPARINGLY (<3% target). Filling label_notes is REQUIRED when picking other.

PRIORITY ORDER (smallest wins when multiple apply):
1. bug_fix > 2. harness_or_scope_change > 3. no_change > 4. algorithm_switch > 5. representation_change > 6. data_structure_change > 7. search_space_change > 8. algorithm_refinement > 9. initialization_change > 10. parameter_tuning > 11. autotune_configuration > 12. loop_optimization > 13. memory_optimization > 14. kernel_restructuring > 15. precision_handling > 16. feature_engineering > 17. code_cleanup > 18. other

REPRESENTATION/DATA-STRUCTURE/SEARCH-SPACE DECISION TREE (apply in order):
- if change rephrases what the problem IS → representation_change
- elif change modifies how solutions are STORED/ENCODED → data_structure_change
- elif change modifies WHICH SET is searched → search_space_change
- else apply other rules

DISAMBIGUATION:
- "numerical_stability_improvement" / "epsilon_addition" → bug_fix (if prevents NaN/inf), else precision_handling.
- "early_termination" / "search_strategy_change" → search_space_change.
- "parallelization_strategy" / "operation_fusion" → kernel_restructuring (GPU) else loop_optimization.
- "caching_reuse" / "memoization_addition" / "lookup_table_addition" → data_structure_change.
- "framework_migration" → algorithm_switch (if substantive) else code_cleanup.
- "hardcoded_solution" / "puzzle_redesign" → algorithm_refinement.
- "architecture_redesign" / "complete_rewrite" → algorithm_switch.
- "robustness_improvement" → bug_fix if patches a defect; else algorithm_refinement.
- "variable_renaming" / "comment_cleanup" / "dead_code_removal" → code_cleanup.

PER-TASK ALGORITHM FAMILIES (REFERENCE ONLY — these family names are NOT primary labels; they are categories used internally to decide algorithm_switch vs algorithm_refinement). The primary_label MUST be one of the 18 labels listed in the CLOSED VOCABULARY above (bug_fix, algorithm_switch, algorithm_refinement, ..., other). Names like `classifier_based`, `stochastic_metaheuristic`, `tiled_blocked`, `frequency_domain_transform`, etc. that appear in this family block are NEVER valid primary labels.

{family_block}

CONFIDENCE GUIDANCE:
- 0.9-1.0: clear-cut, fits one label decisively.
- 0.7-0.9: confident but one alternative could plausibly apply.
- 0.5-0.7: uncertain — two labels both fit; picked by priority. label_notes REQUIRED.
- <0.5: very uncertain, likely "other" candidate. label_notes REQUIRED.

label_notes: REQUIRED when primary_label="other" OR confidence<0.6. Otherwise null or short note.

OUTPUT FORMAT (CRITICAL): Output ONLY the JSON object on a single line. Do NOT include explanatory text before or after the JSON. Do NOT use markdown code fences. Do NOT include multiple JSON objects. The very first character of your response must be `{{` and the very last must be `}}`.

EXAMPLES (a representative sample of clear cases):

Example 1 — bug_fix:
  Diff: `loss = log(p)` → `loss = log(p + 1e-12)`
  Output: {{"primary_label": "bug_fix", "confidence": 0.95, "label_notes": "Adds epsilon to prevent log(0) producing NaN"}}

Example 2 — algorithm_switch (cross-family):
  Diff: replaces gradient_descent loop with simulated_annealing scaffold (random restart, temperature schedule)
  Output: {{"primary_label": "algorithm_switch", "confidence": 0.92, "label_notes": null}}

Example 3 — algorithm_refinement (within-family):
  Diff: tweaks the SA cooling schedule from geometric (T*=0.99) to logarithmic (T = T0/log(t+1))
  Output: {{"primary_label": "algorithm_refinement", "confidence": 0.88, "label_notes": null}}

Example 4 — parameter_tuning:
  Diff: only changes `lr=0.01` to `lr=0.005`, no structural change
  Output: {{"primary_label": "parameter_tuning", "confidence": 0.99, "label_notes": null}}

Example 5 — autotune_configuration:
  Diff: adds `triton.Config({{"BLOCK_SIZE": 4096}}, num_warps=8)` to the @triton.autotune list
  Output: {{"primary_label": "autotune_configuration", "confidence": 0.95, "label_notes": null}}

Example 6 — data_structure_change:
  Diff: replaces a Python list with a deque + adds an LRU cache decorator
  Output: {{"primary_label": "data_structure_change", "confidence": 0.90, "label_notes": null}}

Example 7 — representation_change:
  Diff: switches solution storage from raw (x,y) coords to polar (r,θ) form; downstream code rewrites
  Output: {{"primary_label": "representation_change", "confidence": 0.87, "label_notes": null}}

Example 8 — search_space_change:
  Diff: increases neighborhood radius from 1 to 5 in local search; expands candidate set
  Output: {{"primary_label": "search_space_change", "confidence": 0.85, "label_notes": null}}

Example 9 — initialization_change:
  Diff: replaces zero-init with random multi-start (n_starts=10) seeded by Sobol sequence
  Output: {{"primary_label": "initialization_change", "confidence": 0.92, "label_notes": null}}

Example 10 — loop_optimization:
  Diff: adds `#pragma unroll` and SIMD vectorization to the inner loop
  Output: {{"primary_label": "loop_optimization", "confidence": 0.95, "label_notes": null}}

Example 11 — memory_optimization:
  Diff: rearranges array indexing for coalesced GPU loads, adds shared-memory tile
  Output: {{"primary_label": "memory_optimization", "confidence": 0.93, "label_notes": null}}

Example 12 — kernel_restructuring:
  Diff: splits one large Triton kernel into two separate kernels with different grid dims
  Output: {{"primary_label": "kernel_restructuring", "confidence": 0.92, "label_notes": null}}

Example 13 — precision_handling:
  Diff: casts intermediate accumulator from fp32 to fp16; adds a final fp32 cast for the output
  Output: {{"primary_label": "precision_handling", "confidence": 0.90, "label_notes": null}}

Example 14 — feature_engineering:
  Diff: adds two new derived columns (interaction term, log-transformed price) to the feature matrix
  Output: {{"primary_label": "feature_engineering", "confidence": 0.92, "label_notes": null}}

Example 15 — harness_or_scope_change:
  Diff: adds `print(score)` and a `if N > 100: return 0` early-out check to the entry function
  Output: {{"primary_label": "harness_or_scope_change", "confidence": 0.85, "label_notes": "Adds debug print and a scope restriction"}}

Example 16 — code_cleanup:
  Diff: renames `tmp` to `intermediate_buffer`, removes 5 unreachable lines, no semantic change
  Output: {{"primary_label": "code_cleanup", "confidence": 0.95, "label_notes": null}}

Example 17 — no_change:
  Diff: only whitespace differs from previous step
  Output: {{"primary_label": "no_change", "confidence": 1.0, "label_notes": "Empty diff"}}

Example 18 — other (used sparingly):
  Diff: agent dumps an extensive comment block summarizing prior attempts but no code change
  Output: {{"primary_label": "code_cleanup", "confidence": 0.65, "label_notes": "Comment-only change; ambiguous between code_cleanup and harness/logging"}}

Example 19 — priority disambiguation (bug_fix wins over algorithm_switch):
  Diff: switches algorithm to fix an off-by-one indexing bug that was crashing
  Output: {{"primary_label": "bug_fix", "confidence": 0.85, "label_notes": "Algorithm change is incidental; primary intent is fixing the off-by-one"}}

Example 20 — priority disambiguation (harness_or_scope wins):
  Diff: adds task-input-validation and an early return for N > 1e6, while also tweaking heuristic
  Output: {{"primary_label": "harness_or_scope_change", "confidence": 0.78, "label_notes": "Scope restriction is the dominant intent"}}

Remember:
- Output JSON only, single line, no markdown, no prose.
- First char `{{`, last char `}}`.
- Use the priority order to break ties.
- Use the per-task family list for algorithm_switch vs algorithm_refinement.
"""

PASS2_SYSTEM_TEMPLATE = """\
You are a code-revert detector. Given two consecutive steps in an LLM agent's trajectory, decide if step (i+1) reverts step (i).

OUTPUT — strict JSON only, no prose:
{"was_reverted": <true|false>, "reason": "<one sentence>"}

REVERT CRITERIA — was_reverted=true requires step (i+1) to do AT LEAST ONE of:
1. Remove most of the code added in step (i), OR
2. Restore code that step (i) had removed, OR
3. Agent's reasoning explicitly says "revert", "undo", "going back to", "that didn't work", AND the diff is consistent.

NOT a revert: replacing step (i)'s approach with a DIFFERENT approach that solves the same problem differently. That's a normal algorithm_switch, not a revert.

Be strict — when uncertain, was_reverted=false.

Output the JSON object on a SINGLE LINE. Nothing else.
"""

# ---------------------------------------------------------------------------
# Trajectory loading & step-pair extraction
# ---------------------------------------------------------------------------

def collect_trajectory_paths(args: argparse.Namespace) -> list[Path]:
    base = Path(args.trajectories_dir)
    if args.pilot:
        models = PILOT_MODELS
    elif args.models:
        models = args.models.split(",")
    else:
        models = FULL_MODELS

    levels_filter = set(args.levels.split(",")) if args.levels else None
    runs_filter = set(args.runs.split(",")) if args.runs else None

    paths: list[Path] = []
    for m in models:
        for task_dir in (base / m).iterdir() if (base / m).exists() else []:
            if not task_dir.is_dir():
                continue
            if task_dir.name == ".omc":
                continue  # filter stray
            task_name = task_dir.name
            if args.pilot and task_name not in PILOT_TASKS:
                continue
            if args.tasks:
                tasks_filter = set(args.tasks.split(","))
                if task_name not in tasks_filter:
                    continue
            for fp in sorted(task_dir.glob("L*__run*.json")):
                # Determine level from filename
                stem = fp.stem  # e.g. "L0__run1", "L0f__run1", "L2b__run1"
                level_part = stem.split("__")[0]
                run_part = stem.split("__")[1] if "__" in stem else ""
                # Map L2b → L2 (per user spec)
                eff_level = "L2" if level_part == "L2b" else level_part
                if eff_level not in PILOT_LEVELS:
                    continue  # drop  entirely
                if args.pilot and eff_level not in PILOT_LEVELS:
                    continue
                if levels_filter and eff_level not in levels_filter:
                    continue
                if runs_filter and run_part not in runs_filter:
                    continue
                paths.append(fp)
    return paths


def code_diff(prev: str, new: str, max_chars: int = 4000) -> str:
    """Compact unified-diff between prev and new, truncated."""
    diff = "".join(difflib.unified_diff(
        prev.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="prev",
        tofile="new",
        n=2,
    ))
    if len(diff) > max_chars:
        head = diff[: max_chars // 2]
        tail = diff[-max_chars // 2:]
        return f"{head}\n... [diff truncated; total {len(diff)} chars] ...\n{tail}"
    return diff


def truncate(s: str, max_chars: int) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2:]
    return f"{head}\n... [truncated; total {len(s)} chars] ...\n{tail}"


# ---------------------------------------------------------------------------
# Annotation calls (Pass 1 and Pass 2)
# ---------------------------------------------------------------------------

_JSON_DECODER = json.JSONDecoder()


def _extract_first_json(text: str) -> dict:
    """Extract the first JSON object from text, ignoring trailing content.
    Handles markdown fences and labeler responses with explanatory suffixes."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```\s*$", "", s).strip()
    # Find the first '{' and try raw_decode from there
    start = s.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' found", s, 0)
    obj, _end = _JSON_DECODER.raw_decode(s[start:])
    return obj


async def call_with_retries(
    client: AsyncAnthropic,
    system: list[dict],
    messages: list[dict],
    max_retries: int = 3,
) -> Optional[dict]:
    """Call API and parse JSON; retry up to max_retries on parse failure."""
    last_text = None
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.messages.create(
                model=LABELER_MODEL,
                max_tokens=LABELER_MAX_TOKENS,
                temperature=LABELER_TEMPERATURE,
                system=system,
                messages=messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            text = resp.content[0].text if resp.content else ""
            last_text = text
            obj = _extract_first_json(text)
            return {
                "obj": obj,
                "raw": text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            }
        except (json.JSONDecodeError, anthropic.APIError, anthropic.APITimeoutError) as e:
            last_err = str(e)
            if attempt == max_retries - 1:
                return {"obj": None, "raw": last_text, "error": last_err,
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read_tokens": 0, "cache_creation_tokens": 0}
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def annotate_pass1(
    client: AsyncAnthropic,
    task_name: str,
    task_description: str,
    step_idx: int,
    step: dict,
    prev_code: str,
    prev_labels: list[str],
) -> dict:
    """Pass 1: assign primary_label."""
    new_code = (step.get("generation") or {}).get("code") or ""
    diff_text = code_diff(prev_code, new_code, CODE_TRUNC_PASS1)
    reasoning = (step.get("decision") or {}).get("reasoning") or ""
    action_type = (step.get("decision") or {}).get("action_type") or ""

    family_block = get_family_block(task_name)
    system_text = PASS1_SYSTEM_TEMPLATE.format(family_block=family_block)
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    prev_labels_str = "\n".join([f"- step {step_idx-len(prev_labels)+i}: {lbl}"
                                  for i, lbl in enumerate(prev_labels)]) or "(none, this is the first labeled step)"
    user_prompt = f"""\
Task: {task_name}
Task description: {truncate(task_description, 200)}

Previous step labels (most recent {len(prev_labels)}, oldest first):
{prev_labels_str}

Current step (step_idx={step_idx}):
Mode chosen by agent: {action_type}
Code change (unified diff vs previous step):
```
{diff_text}
```

Brief reasoning the agent gave (if available):
{truncate(reasoning, REASONING_TRUNC)}

Output JSON only, no prose, single line:"""

    return await call_with_retries(client, system, [{"role": "user", "content": user_prompt}])


async def annotate_pass2(
    client: AsyncAnthropic,
    step_i_label: str,
    step_i_notes: Optional[str],
    step_i_code: str,
    step_i1_code: str,
    step_i1_label: str,
    step_i1_reasoning: str,
) -> dict:
    """Pass 2: was_reverted boolean."""
    diff_i = code_diff("", step_i_code, DIFF_TRUNC_PASS2)
    diff_i1 = code_diff(step_i_code, step_i1_code, DIFF_TRUNC_PASS2)

    system_text = PASS2_SYSTEM_TEMPLATE
    # Pass 2 system is short — caching may not trigger. That's OK; same prompt across calls.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    user_prompt = f"""\
Step (i) summary: primary_label={step_i_label}, notes={step_i_notes or ''}
Step (i) code (full code at end of step, truncated):
```
{truncate(step_i_code, DIFF_TRUNC_PASS2)}
```

Step (i+1) summary: primary_label={step_i1_label}
Step (i+1) diff vs step (i):
```
{diff_i1}
```

Step (i+1) agent reasoning:
{truncate(step_i1_reasoning, REASONING_TRUNC)}

Did step (i+1) revert step (i)? Output JSON only, single line:"""

    return await call_with_retries(client, system, [{"role": "user", "content": user_prompt}])


# ---------------------------------------------------------------------------
# Per-trajectory orchestration
# ---------------------------------------------------------------------------

async def annotate_trajectory(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    traj_path: Path,
    output_dir: Path,
    task_descriptions: dict[str, str],
    stats: dict,
    skip_existing: bool = True,
) -> tuple[Path, Optional[Path], str]:
    """Annotate one trajectory file. Returns (input_path, output_path or None, status)."""
    rel = traj_path.relative_to(traj_path.parents[2])  # <model>/<task>/<file>
    out_path = output_dir / rel.parent / (rel.name + ".raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_path.exists():
        return traj_path, out_path, "skipped_exists"

    try:
        traj = json.loads(traj_path.read_text())
    except Exception as e:
        return traj_path, None, f"load_fail: {e}"

    steps = traj.get("steps") or []
    if not steps:
        return traj_path, None, "empty_trajectory"

    task_id = traj.get("task_id", "")
    # Strip path prefix from task_id (math/hexagon_packing → hexagon_packing)
    task_name = task_id.split("/")[-1]
    # Special-case: gemm_optimization/annoying → gemm_optimization
    if "annoying" in task_id and "gemm" in task_id:
        task_name = "gemm_optimization"
    if task_id.startswith("grammar_fuzzing"):
        task_name = "grammar_fuzzing"
    if re.match(r"^\d+$", task_id):
        task_name = f"fcs_algo_{task_id}"

    task_desc = task_descriptions.get(task_name, "(no description available)")

    n_steps = len(steps)
    labels = [None] * n_steps  # filled in
    pass1_results = [None] * n_steps

    # ---------- Pass 1 (skip step_idx=0) ----------
    pass1_tasks = []
    for i in range(1, n_steps):
        prev_code = (steps[i-1].get("generation") or {}).get("code") or ""
        prev_labels = []  # we feed this AFTER pass 1 starts; for now use empty
        pass1_tasks.append((i, prev_code))

    # We do pass1 sequentially to use prev_labels chain; OR we do it in parallel
    # without prev_labels (only most recent will be empty initially). Do PARALLEL.
    async def _do_pass1(step_idx: int, prev_code: str) -> tuple[int, dict]:
        async with sem:
            stats["pass1_calls"] += 1
            res = await annotate_pass1(client, task_name, task_desc, step_idx,
                                        steps[step_idx], prev_code, prev_labels=[])
            return step_idx, res

    p1_results = await asyncio.gather(*[_do_pass1(i, pc) for i, pc in pass1_tasks])
    for step_idx, res in p1_results:
        pass1_results[step_idx] = res

    # Step 0 sentinel
    pass1_results[0] = {"obj": {"primary_label": SENTINEL_INITIAL, "confidence": 1.0,
                                  "label_notes": "Sentinel: initial generation, not an edit"},
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read_tokens": 0, "cache_creation_tokens": 0}

    # ---------- Validation: ensure label is in vocab or sentinel ----------
    for i in range(n_steps):
        r = pass1_results[i]
        if r is None or r.get("obj") is None:
            stats["pass1_failed"] += 1
            r = pass1_results[i] = {"obj": {"primary_label": SENTINEL_FAILED,
                                              "confidence": 0.0, "label_notes": (r or {}).get("error", "parse_failed")},
                                     "input_tokens": (r or {}).get("input_tokens", 0),
                                     "output_tokens": (r or {}).get("output_tokens", 0),
                                     "cache_read_tokens": (r or {}).get("cache_read_tokens", 0),
                                     "cache_creation_tokens": (r or {}).get("cache_creation_tokens", 0)}
        else:
            lbl = r["obj"].get("primary_label", "")
            if lbl == SENTINEL_INITIAL:
                pass
            elif not LABEL_REGEX.match(lbl):
                stats["pass1_invalid_label"] += 1
                r["obj"]["primary_label"] = SENTINEL_FAILED
                r["obj"]["label_notes"] = f"invalid_label: {lbl!r}"
            else:
                stats["label_dist"][lbl] += 1

        stats["input_tokens"] += r.get("input_tokens", 0)
        stats["output_tokens"] += r.get("output_tokens", 0)
        stats["cache_read_tokens"] += r.get("cache_read_tokens", 0)
        stats["cache_creation_tokens"] += r.get("cache_creation_tokens", 0)

    # ---------- Pass 2 (was_reverted on each step except last) ----------
    pass2_results = [None] * n_steps  # last one stays None

    async def _do_pass2(i: int) -> tuple[int, dict]:
        async with sem:
            stats["pass2_calls"] += 1
            si_code = (steps[i].get("generation") or {}).get("code") or ""
            si1_code = (steps[i+1].get("generation") or {}).get("code") or ""
            si1_reasoning = (steps[i+1].get("decision") or {}).get("reasoning") or ""
            si_label = pass1_results[i]["obj"].get("primary_label", "?")
            si_notes = pass1_results[i]["obj"].get("label_notes")
            si1_label = pass1_results[i+1]["obj"].get("primary_label", "?")
            res = await annotate_pass2(client, si_label, si_notes, si_code, si1_code, si1_label, si1_reasoning)
            return i, res

    p2_tasks = [_do_pass2(i) for i in range(n_steps - 1)]
    p2_results = await asyncio.gather(*p2_tasks)
    for i, res in p2_results:
        pass2_results[i] = res

    # Validate
    for i in range(n_steps - 1):
        r = pass2_results[i]
        if r is None or r.get("obj") is None:
            stats["pass2_failed"] += 1
            pass2_results[i] = {"obj": {"was_reverted": False, "reason": (r or {}).get("error", "parse_failed")},
                                  "input_tokens": (r or {}).get("input_tokens", 0),
                                  "output_tokens": (r or {}).get("output_tokens", 0),
                                  "cache_read_tokens": 0, "cache_creation_tokens": 0}
        else:
            wr = r["obj"].get("was_reverted")
            if wr is True:
                stats["was_reverted_true"] += 1

        rr = pass2_results[i]
        stats["input_tokens"] += rr.get("input_tokens", 0)
        stats["output_tokens"] += rr.get("output_tokens", 0)
        stats["cache_read_tokens"] += rr.get("cache_read_tokens", 0)
        stats["cache_creation_tokens"] += rr.get("cache_creation_tokens", 0)

    # ---------- Assemble output ----------
    label_records = []
    for i in range(n_steps):
        p1 = pass1_results[i]["obj"]
        p2 = pass2_results[i]["obj"] if pass2_results[i] is not None else {"was_reverted": False, "reason": "(last step, no next-step)"}
        label_records.append({
            "step_idx": i,
            "step_number": steps[i].get("step_number"),
            "primary_label": p1.get("primary_label"),
            "confidence": p1.get("confidence", 0.0),
            "label_notes": p1.get("label_notes"),
            "was_reverted": p2.get("was_reverted", False),
            "revert_reason": p2.get("reason", ""),
        })

    output = {
        "vocabulary_version": "v2",
        "annotation_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "labeler_model": LABELER_MODEL,
        "labeler_settings": {"temperature": LABELER_TEMPERATURE, "top_p": 1, "max_tokens": LABELER_MAX_TOKENS},
        "pass1_skipped_step_indices": [0],
        "task_id": task_id,
        "task_name": task_name,
        "model": traj.get("model"),
        "level": traj.get("level"),
        "run_id": traj.get("run_id"),
        "n_steps": n_steps,
        "labels": label_records,
    }

    out_path.write_text(json.dumps(output, indent=2))
    stats["trajectories_done"] += 1
    return traj_path, out_path, "ok"


# ---------------------------------------------------------------------------
# Task descriptions (loaded from configs/tasks.yaml)
# ---------------------------------------------------------------------------

def load_task_descriptions(repo_root: Path) -> dict[str, str]:
    import yaml
    tasks_yaml = repo_root / "configs" / "tasks.yaml"
    if not tasks_yaml.exists():
        return {}
    cfg = yaml.safe_load(tasks_yaml.read_text())
    out = {}
    for t in cfg.get("tasks", []):
        out[t["name"]] = t.get("description", "")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> int:
    api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.stderr.write("ERROR: CLAUDE_API_KEY (or ANTHROPIC_API_KEY) not set in env. "
                          "Set CLAUDE_API_KEY or ANTHROPIC_API_KEY in your shell env first.\n")
        return 1

    client = AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    repo_root = _PKG_ROOT
    task_descriptions = load_task_descriptions(repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = collect_trajectory_paths(args)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.stderr.write("ERROR: no trajectories matched scope.\n")
        return 1

    sys.stderr.write(f"Trajectories to annotate: {len(paths)}\n")
    sys.stderr.write(f"Output dir: {output_dir}\n")
    sys.stderr.write(f"Concurrency: {args.concurrency}\n")

    stats = defaultdict(int)
    stats["label_dist"] = Counter()
    t_start = time.time()

    # Process trajectories with progress reporting
    async def _drive(traj_path):
        return await annotate_trajectory(client, sem, traj_path, output_dir,
                                          task_descriptions, stats, skip_existing=args.skip_existing)

    # ---- (model, task) bucketing ----
    # Cache locality: all trajectories within the same (model, task) share the SAME
    # system prompt (same family block). Process one bucket at a time so the cache
    # for that bucket's prompt stays warm across all ~12 trajectories.
    from itertools import groupby

    def _bucket_key(p: Path) -> tuple[str, str]:
        # Layout: <trajectories_dir>/<model>/<task>/<file>
        return p.parent.parent.name, p.parent.name
    paths_sorted = sorted(paths, key=_bucket_key)
    buckets = [(k, list(g)) for k, g in groupby(paths_sorted, key=_bucket_key)]
    sys.stderr.write(f"Bucketing into {len(buckets)} (model, task) groups\n")
    for (m, t), bps in buckets[:5]:
        sys.stderr.write(f"  example bucket: {m}/{t} ({len(bps)} trajectories)\n")

    results = []
    n_done = 0
    for bi, ((m, t), bucket_paths) in enumerate(buckets):
        bt0 = time.time()
        sys.stderr.write(f"\n>>> Bucket {bi+1}/{len(buckets)}: {m}/{t} ({len(bucket_paths)} trajectories)\n")
        bucket_in_flight = [asyncio.create_task(_drive(p)) for p in bucket_paths]
        for fut in asyncio.as_completed(bucket_in_flight):
            traj_path, out_path, status = await fut
            n_done += 1
            results.append((traj_path, out_path, status))
        b_elapsed = time.time() - bt0
        total_elapsed = time.time() - t_start
        sys.stderr.write(f"<<< Bucket {bi+1}/{len(buckets)} done in {b_elapsed:.1f}s "
                          f"({n_done}/{len(paths)} total, elapsed {total_elapsed:.1f}s, "
                          f"cache_read so far: {stats['cache_read_tokens']:,})\n")

    # Final report
    elapsed = time.time() - t_start
    sys.stderr.write("\n=== SUMMARY ===\n")
    sys.stderr.write(f"  trajectories_done: {stats['trajectories_done']}\n")
    sys.stderr.write(f"  pass1_calls: {stats['pass1_calls']}\n")
    sys.stderr.write(f"  pass1_failed: {stats['pass1_failed']}\n")
    sys.stderr.write(f"  pass1_invalid_label: {stats['pass1_invalid_label']}\n")
    sys.stderr.write(f"  pass2_calls: {stats['pass2_calls']}\n")
    sys.stderr.write(f"  pass2_failed: {stats['pass2_failed']}\n")
    sys.stderr.write(f"  was_reverted_true: {stats['was_reverted_true']}\n")
    sys.stderr.write(f"  input_tokens: {stats['input_tokens']:,}\n")
    sys.stderr.write(f"  output_tokens: {stats['output_tokens']:,}\n")
    sys.stderr.write(f"  cache_read_tokens: {stats['cache_read_tokens']:,}\n")
    sys.stderr.write(f"  cache_creation_tokens: {stats['cache_creation_tokens']:,}\n")
    sys.stderr.write(f"  elapsed_sec: {elapsed:.1f}\n")
    sys.stderr.write(f"  label distribution (top 20):\n")
    for lbl, c in stats["label_dist"].most_common(20):
        sys.stderr.write(f"    {c:6d}  {lbl}\n")

    # Write summary JSON
    summary = {
        "n_trajectories": len(paths),
        "trajectories_done": stats["trajectories_done"],
        "pass1_calls": stats["pass1_calls"],
        "pass1_failed": stats["pass1_failed"],
        "pass1_invalid_label": stats["pass1_invalid_label"],
        "pass2_calls": stats["pass2_calls"],
        "pass2_failed": stats["pass2_failed"],
        "was_reverted_true": stats["was_reverted_true"],
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "cache_read_tokens": stats["cache_read_tokens"],
        "cache_creation_tokens": stats["cache_creation_tokens"],
        "elapsed_sec": elapsed,
        "label_distribution": dict(stats["label_dist"]),
    }
    summary_path = output_dir / "_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    sys.stderr.write(f"\nSummary written to: {summary_path}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories-dir",
                        default=str(_PKG_ROOT / "data/trajectories"))
    parser.add_argument("--output-dir",
                        default=str(_PKG_ROOT / "data/strategy_labels/raw"))
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot scope: gpt-oss-120b × 4 tasks × 4 levels × 3 runs = 48 trajectories")
    parser.add_argument("--models", default=None, help="Comma-separated; overrides --pilot's model filter")
    parser.add_argument("--tasks", default=None, help="Comma-separated; overrides --pilot's task filter")
    parser.add_argument("--levels", default=None, help="Comma-separated levels filter (e.g. L0,L1)")
    parser.add_argument("--runs", default=None, help="Comma-separated runs filter (e.g. run1)")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of trajectories to process")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip trajectories whose output already exists")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
