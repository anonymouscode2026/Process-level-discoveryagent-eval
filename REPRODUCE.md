# Reproduction guide — three tiers

| Tier | What it reproduces | Inputs needed | Wall time | Cost |
|------|-------------------|---------------|-----------|------|
| 1    | All paper tables & figures | shipped `data/metrics/` + `analysis/*.csv` | minutes | $0 |
| 2    | Behavioral metrics from raw trajectories | full corpus from `data_external/DOWNLOAD.md` (649 MB download) | hours / 1 node | $0 |
| 3    | Trajectories from a fresh pilot run | provider API keys + GPU | days / 1 node | $ |

All tiers assume you have run the Quickstart in `README.md` to install
`requirements.txt` into a Python 3.12 virtual environment.

---

## Tier 1 — paper tables/figures from precomputed CSVs (~minutes)

This is what most reviewers will run. Outputs are byte-identical to the
versions in the submitted PDF (verified during package preparation).

### Headline figures

```bash
# Figure 2: harness×family heatmap + per-model harness profile
PYTHONPATH=. python analysis/harness_by_task/render_harness_main_figure.py
PYTHONPATH=. python analysis/harness_by_task/make_plots.py
# → analysis/harness_by_task/figures/harness_main.{pdf,png}
# → analysis/harness_by_task/{family_x_level,quadrant_x_level}_heatmap.png

# Figure 3: frontier-vs-open-weight per-feature gap (two slices)
PYTHONPATH=. python analysis/gap_decomposition/make_gap_figure.py
# → analysis/gap_decomposition/figures/gap_two_slices.pdf

```

### Headline statistics

```bash
# Appendix A.2: Fleiss κ + ICC(3,5) on per-rater capability annotations
PYTHONPATH=. python data/capability_per_rater/compute_kappa_icc.py
# → data/capability_per_rater/kappa_icc.md
# Expected: κ_alg = 0.8994, κ_repr = 0.9745, ICC = 0.9903 / 0.9975
```

### Browsing precomputed artifacts directly

If you only want to inspect numbers without running code, here's the map from
shipped CSVs to paper claims:

| CSV | Contains | Paper reference |
|-----|----------|-----------------|
| `data/metrics/all_metrics.csv` | full 1,728-row × 60-feature panel | Tables 1, 2; Fig 2, 3 source data |
| `analysis/harness_by_task/per_model_family_level.csv` | per-(model, family, level) means | Table A.7 |
| `analysis/harness_by_task/family_x_level_means.csv` | aggregate family×level means | Fig 2 left |
| `analysis/gap_decomposition/c_per_feature_gap.csv` | Slice-1 per-feature Δz, paired-t, consistency | Table 3 |
| `analysis/gap_decomposition/full_panel_delta_z.csv` | Slice-2 (score-matched) per-feature Δz | Table A.10 |

---

## Tier 2 — behavioral metrics from raw trajectories (~hours)

Use this if you want to verify the full pipeline from trajectory JSONs.

```bash
# Fetch the full corpus (649 MB download)
cat data_external/DOWNLOAD.md   # follow the gdown + sha256sum -c instructions
tar -xzf raw_data.tar.gz        # extracts raw_data/{trajectories,step_labels}/

# Relabel with the strategy labeler (requires anthropic key for fresh runs; the
# shipped labels in data/metrics/ were produced with claude-opus-4-6)
PYTHONPATH=. python scripts/relabel_strategies.py \
  --concurrency 60 \
  --output-dir data/strategy_labels/raw

# Post-process per the 7-rule scheme of Appendix A.1
PYTHONPATH=. python scripts/postproc_labels.py

# Recompute the 60-feature behavioral profile per cell
PYTHONPATH=. python scripts/compute_metrics.py
# → data/metrics/all_metrics.csv (should match the shipped CSV up to rounding)
```

Expected wall time on a 60-core box: ~6 h for relabeling (LLM-bound), <1 h for
post-proc + metrics.

---

## Tier 3 — re-run the full pilot (~days, $$$)

Generates a fresh 6×24×4×3 trajectory set. Requires provider credentials for
each model and a GPU node for the kernel tasks.

```bash
# Set provider credentials (one per model — see configs/models.yaml for the
# expected env-var names: e.g. GPT54_BASE_URL, GPT54_API_KEY, ANTHROPIC_API_KEY,
# DEEPSEEK_BASE_URL, GEMINI_VERTEX_CREDENTIALS, …)
export ANTHROPIC_API_KEY=sk-...
export GPT54_BASE_URL=https://your-deployment/openai/v1/
export GPT54_API_KEY=...
# ... etc

# Optional: tell the GPU-pinning code which user owns our own evaluator
# processes (used to skip self-contention on shared hosts)
export EVAL_USER=$(whoami)

# Optional: tell the codebase where your project root lives. Defaults to the
# script's parent directories — only set this if you've installed the package
# in a non-standard layout.
export PROJECT_ROOT=$(pwd)

# External-repo paths (only needed for Tier 3 fresh pilot runs — these repos
# aren't shipped with the package).
export SKYDISCOVER_ROOT=/path/to/SkyDiscover/benchmarks
export FRONTIER_CS_ROOT=/path/to/Frontier-CS

# Run the 6×24×4×3 sweep (≈1,728 trajectories)
PYTHONPATH=. python scripts/run.py \
  --models claude-opus-4-6 deepseek-v3.2 gemini-3.1-pro-preview \
           gpt-5.4 gpt-oss-120b qwen3-next-80b-instruct \
  --tasks-config configs/tasks.yaml \
  --levels L0 L0f L1 L2b \
  --runs 3 \
  --step-budget 150 \
  --concurrency 60
# → data/trajectories/<model>/<task>/<level>__run<k>.json
```

### Hardware notes

- GPU kernel tasks (`vecadd`, `grayscale`, `trimul`, `gemm_optimization`,
  `group_gemm`, `cross_entropy`) need a CUDA-capable GPU. We used a single
  A100. The evaluator serializes timing-sensitive runs via a shared file lock
  (`OMC_GPU_EVAL_LOCK_DIR`, default `/tmp/omc_gpu_eval_locks`) to avoid SM
  contention.
- Frontier-CS research/algorithmic tasks need the upstream Frontier-CS
  evaluator. Pin the FCS commit hash referenced in `configs/tasks.yaml`. The
  evaluator runs in a Docker container; ensure `nvidia-container-toolkit` is
  installed if you want GPU-aware evaluation.

---