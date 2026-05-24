# Full trajectory and strategy-label corpus — external download

The package ships a **25-file representative sample** in
`data/trajectories_sample/` (covers `hexagon_packing` across all 6 models × 4
levels).

The complete **1,728-trajectory + 1,728-label** corpus is hosted externally:

**Google Drive**: <https://drive.google.com/file/d/1hpfXOIXyvUrQOeBjIp8KCoU1RH6XLZer/view?usp=sharing>


## Fetch

```bash
# Option 1: download via browser from the Google Drive link above

# Option 2: command-line with gdown (pip install gdown)
gdown 1hpfXOIXyvUrQOeBjIp8KCoU1RH6XLZer -O raw_data.tar.gz

# Extract at the package root — files land directly into data/
cd <package_root>
tar -xzf raw_data.tar.gz
```

## Layout after extraction

Extracting at the package root populates exactly the paths that scripts expect:

```
data/
├── trajectories/
│   ├── claude-opus-4-6/<task>/L{0,0f,1,2b}__run{1,2,3}.json
│   ├── deepseek-v3.2/...
│   ├── gemini-3.1-pro-preview/...
│   ├── gpt-5.4/...
│   ├── gpt-oss-120b/...
│   └── qwen3-next-80b-instruct/...
│   (6 models × 24 tasks × 4 levels × 3 runs = 1,728 trajectory JSONs)
│
└── strategy_labels/postproc/
    └── <model>/<task>/L{0,0f,1,2b}__run{1,2,3}.labels.json
    (1,728 post-processed strategy-label JSONs, one per trajectory)
```

Each trajectory file is self-contained — see `TRAJECTORY_SCHEMA.md` for the
field definitions, per-step structure, and how to join trajectories with their
strategy labels.

## Using the corpus

```bash
# Tier-1 reproduction (paper tables/figures) does NOT need this corpus —
# the shipped data/metrics/all_metrics.csv (1.1 MB) already encodes every
# paper claim. See REPRODUCE.md Tier 1.

# Tier-2: recompute behavioral metrics from raw trajectories + labels
export PROJECT_ROOT=$(pwd)
PYTHONPATH=. python scripts/compute_metrics.py
```
