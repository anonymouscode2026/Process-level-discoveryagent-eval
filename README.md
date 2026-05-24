# Process-Level Evaluation of LLM Discovery Agents — Reviewer Package

Companion code and data for the NeurIPS 2026 Evaluations & Datasets Track
submission *A Process-Level Evaluation of LLM Discovery Agents*.

## What's in the box

The submission introduces a four-rung harness ladder and a process-level
evaluation framework for discovery-agent trajectories. The framework is
exercised by a fully-crossed sweep of **6 contemporary LLMs × 24 discovery
tasks × 4 harness rungs × 3 runs = 1,728 trajectories**. This package ships
every released artifact in that contribution:

| Artifact | Path | Paper section |
|---|---|---|
| 4-level harness implementations (L_0, L_0<sub>f</sub>, L_1, L_2) | `src/agent/` | §3 |
| Trajectory schema + recorder | `src/trajectory/` | §3 |
| 60-feature behavioral profile pipeline | `src/metrics/behavioral.py` | §3, App. A.1 |
| 18-label strategy labeler + prompt + audit ground truth (n=60) | `src/metrics/edit_classifier.py`, `data/strategy_labeler/` | App. A.1 |
| Per-step strategy label annotations (separate file per trajectory) | `data/strategy_labels/postproc/` | App. A.1 |
| Capability-demand rubric + 5-rater per-task annotations | `data/capability_rubric.md`, `data/capability_per_rater/` | §3, App. A.2 |
| 24-task definitions (math, GPU kernel, ADRS, FCS algorithmic) | `configs/tasks.yaml` | §4 |
| Behavioral metrics for all 1,728 trajectories | `data/metrics/all_metrics.csv` (1.1 MB) | §5 |
| Paper tables + figures (regenerated from shipped CSVs) | `analysis/` (3 subdirs) | §5 |
| **25-trajectory representative sample** | `data/trajectories_sample/` (58 MB) | — |
| **Full 1,728-trajectory + label corpus** | external Google Drive deposit (649 MB compressed) | §4 |

See `data_external/DOWNLOAD.md` for the full corpus; everything else is local.

## Quickstart

```bash
# 1. Set up a fresh environment (we tested on cpython 3.12 with the pins below)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Reproduce the two headline paper figures from shipped CSVs (~10 seconds)
PYTHONPATH=. python analysis/harness_by_task/render_harness_main_figure.py
PYTHONPATH=. python analysis/gap_decomposition/make_gap_figure.py

# 3. Recompute the inter-rater reliability headlines from raw rater YAMLs
PYTHONPATH=. python data/capability_per_rater/compute_kappa_icc.py

# 4. Run the test suite
PYTHONPATH=. pytest tests/
```

Outputs land next to the scripts:
- `analysis/harness_by_task/figures/harness_main.pdf`
- `analysis/gap_decomposition/figures/gap_two_slices.pdf`

See `REPRODUCE.md` for the three-tier reproduction guide (cheap → full).

## Directory layout

```
.
├── README.md                       this file
├── REPRODUCE.md                    three reproduction tiers, with timings
├── TRAJECTORY_SCHEMA.md            JSON schema + annotated 50-line excerpt
├── LICENSE                         code: MIT; data: CC-BY-4.0
├── requirements.txt
├── pyproject.toml
│
├── src/                            agent harnesses + behavioral metrics + trajectory schema
├── scripts/                        pipeline entry points (run, compute_metrics, …)
├── configs/                        tasks.yaml + models.yaml + experiments.yaml
├── analysis/                       3 paper-figure subdirs + cross-model aggregates
├── tests/                          23 unit tests on schema + behavioral pipeline
│
├── data/
│   ├── metrics/                    all_metrics.csv (1,728 × 68 behavioral profile panel)
│   ├── capability_rubric.md        scoring protocol
│   ├── capability_per_rater/       5 worker YAMLs (per-rater bin/value scores) +
│   │                               final_labels
│   ├── strategy_labeler/           labeler prompt + 60-step audit ground truth
│   └── trajectories_sample/        25-file representative slice + SHA256SUMS
└── data_external/
    └── DOWNLOAD.md                 anonymous deposit link for the full 1,728-trajectory corpus
```

## What this package does *not* include

- The full 1,728-trajectory + label corpus (649 MB compressed / 7.8 GB uncompressed) —
  fetch from the anonymous Google Drive deposit; see `data_external/DOWNLOAD.md`.
- Provider API keys and SDK credentials — set them via env vars if you want to
  re-run the agent loop (Tier 3 in `REPRODUCE.md`). Tier 1 + Tier 2 reproduction
  needs no keys.
- The SkyDiscover and Frontier-CS benchmark repos (only needed for Tier 3 fresh
  pilot runs — not required for Tier 1/2 reproduction). See `REPRODUCE.md`.

## License

- **Code** (`src/`, `scripts/`, `analysis/`, `tests/`): MIT
- **Data** (`data/`): CC-BY-4.0

