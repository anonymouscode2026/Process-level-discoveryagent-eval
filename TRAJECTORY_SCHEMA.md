# Data schema — trajectories and strategy labels

For every (model, task, level, run) cell we produce **two independent JSON
files** that live in separate directories:

```
data/trajectories/<model>/<task>/<level>__run<k>.json                          # the trajectory
data/strategy_labels/postproc/<model>/<task>/<level>__run<k>.labels.json       # post-hoc strategy labels
```

The trajectory file is what the harness writes during a fresh agent run
(`scripts/run.py`). The label file is a **separate** post-hoc annotation
produced offline by `scripts/relabel_strategies.py` (which calls an LLM
labeler over each step's code) followed by `scripts/postproc_labels.py` (which
applies the 7-rule cleanup from Appendix A.1). Trajectories are **never**
modified during labeling — the two artifacts are write-once, read-many, and
joinable on `(model, task, level, run)` plus per-step `step_idx`.

This document covers the trajectory schema first, then the label schema, then
how to join them.

---

## Trajectory JSON

The 25-file sample in `data/trajectories_sample/` is structurally identical to
the external deposit; the deposit just contains the remaining
1,703 cells.

### Top-level structure (14 keys)

| Key | Type | Description |
|---|---|---|
| `run_id` | str | Cell identifier, e.g. `claude-opus-4-6__hexagon_packing__L0__run1` |
| `task_id` | str | Canonical task name from `configs/tasks.yaml` |
| `model` | str | Model name from `configs/models.yaml` |
| `level` | str | Harness rung label: `L0`, `L0f`, `L1`, `L2b` |
| `started_at` | null | **Scrubbed in this release** (would be ISO 8601 UTC if regenerated) |
| `finished_at` | null | **Scrubbed in this release** (would be ISO 8601 UTC if regenerated) |
| `step_budget` | int | Maximum steps for this cell (always 150 in the paper) |
| `steps` | list[Step] | Per-step records — see "Per-step structure" below |
| `best_score` | float | Highest score achieved across all steps |
| `best_code` | str | Source code corresponding to `best_score` |
| `tokens_used` | int | Total tokens consumed across all LLM calls |
| `total_prompt_tokens` | int | Input-token component |
| `total_completion_tokens` | int | Output-token component |
| `metadata` | dict | Run-level config (see below) |
| `belief_state_history` | list[dict] | L2 reflection memory snapshots (empty for L0/L0f/L1) |

### Per-step structure (10 keys)

Each element of `steps[]` represents one revision iteration:

| Key | Type | Description |
|---|---|---|
| `step_number` | int | 0-indexed step counter |
| `timestamp` | null | **Scrubbed in this release** (would be ISO 8601 UTC if regenerated) |
| `decision` | dict | Which **harness primitive** fired this step. Only 4 distinct values across the corpus (`REFINE`, `REFINE_FORCED`, `REWRITE`, `REWRITE_FORCED`); **this is not the paper's strategy label** — see the warning below |
| `generation` | dict | LLM call metadata: `code` (the emitted source), `prompt_tokens`, `completion_tokens`, optional `reasoning_tokens` |
| `score` | float \| None | Evaluator score; `None` when the evaluator rejected the program (parse error, timeout, sandbox failure) |
| `score_delta` | float | `score - prev_score`, `None` for step 0 |
| `is_new_best` | bool | Whether this step set a new best across the run |
| `eval_result` | dict | Sandbox + evaluator output: stdout, stderr, traceback, runtime, exit code |
| `prompt_summary` | str | Short tag for what entered the prompt this step (used by behavioral profiling) |
| `belief_state` | dict \| None | L₂ memory contents at the time of this step; `None` for L₀/L₀<sub>f</sub>/L₁ |

### `metadata` block

Three fields, all safe to share:

| Field | Notes |
|---|---|
| `step_budget` | mirrors top-level field |
| `seeded_with_initial_code` | bool — whether the harness started from a non-empty seed (`initial_program.py` for SkyDiscover, baseline for FCS) |
| `l0_rewrite_every` | int — L₀ rewrite-cadence parameter (only present for level L0); how many steps the agent runs before doing a full rewrite |

We **do not** record hostname, username, or any GPU UUID in the metadata block.
Verified during package preparation across a stratified sample of trajectories.

### Annotated 50-line excerpt

The first ~50 lines of `data/trajectories_sample/claude-opus-4-6/hexagon_packing/L0__run1.json`
look approximately like:

```json
{
  "run_id": "claude-opus-4-6__hexagon_packing__L0__run1",
  "task_id": "hexagon_packing",
  "model": "claude-opus-4-6",
  "level": "L0",
  "started_at": null,
  "step_budget": 150,
  "finished_at": null,
  "steps": [
    {
      "step_number": 0,
      "timestamp": null,
      "decision": {"primitive": "initial_generation", "from_seed": true},
      "generation": {
        "code": "import numpy as np\nfrom scipy.optimize import minimize\n\ndef pack_hexagons(...): ...",
        "prompt_tokens": 4231,
        "completion_tokens": 1842
      },
      "score": 0.6324,
      "score_delta": null,
      "is_new_best": true,
      "eval_result": {"runtime_s": 12.4, "stdout": "...", "stderr": "", "exit_code": 0},
      "prompt_summary": "L0 initial; seed initial_program.py present",
      "belief_state": null
    },
    {
      "step_number": 1,
      "timestamp": null,
      "decision": {"primitive": "refine"},
      "generation": {
        "code": "import numpy as np\nfrom scipy.optimize import minimize\n\ndef pack_hexagons(...): ...",
        "prompt_tokens": 4892,
        "completion_tokens": 2103
      },
      "score": 0.6418,
      "score_delta": 0.0094,
      "is_new_best": true,
      "eval_result": {...},
      "prompt_summary": "L0 refine; prev score=0.6324",
      "belief_state": null
    }
    /* ... 148 more steps ... */
  ],
  "best_score": 0.8761,
  "best_code": "...",
  "tokens_used": 528341,
  "total_prompt_tokens": 312194,
  "total_completion_tokens": 216147,
  "metadata": {"step_budget": 150, "seeded_with_initial_code": true, "l0_rewrite_every": 5},
  "belief_state_history": []
}
```

### Parsing

The schema is plain JSON; any language works. From Python, the recommended way
is to use the typed wrappers in `src/trajectory/schemas.py`:

```python
import json
from src.trajectory.schemas import Trajectory

with open("data/trajectories_sample/claude-opus-4-6/hexagon_packing/L0__run1.json") as f:
    t = Trajectory.from_dict(json.load(f))

print(t.run_id, t.best_score, len(t.steps))
# claude-opus-4-6__hexagon_packing__L0__run1 0.8761 150
```

`src/trajectory/recorder.py` is the writer-side counterpart used by `runner.py`
during a fresh pilot run (Tier 3 in `REPRODUCE.md`).

---

### ⚠️ `steps[i].decision` is *not* the paper's strategy label

A trajectory step has a `decision` field that records which **harness
primitive** the agent invoked that step. It has only 4 possible values:

| `decision.action_type` | Meaning |
|---|---|
| `REFINE` | Model chose to refine the current code |
| `REFINE_FORCED` | Harness scheduled a refine (e.g. L₀ cadence) |
| `REWRITE` | Model chose to write from scratch |
| `REWRITE_FORCED` | Harness scheduled a full rewrite |

That is the *coarse* harness-level edit mode, not the *fine* 18-vocab
classification the paper analyses use. The paper's strategy label
(`algorithm_switch`, `parameter_tuning`, `data_structure_change`, ...) comes
from an offline LLM post-hoc annotation that compares each step's emitted
code to the previous step's. Those fine labels live in a **separate** file —
see "Strategy-label JSON" below.

**Concretely:** building behavioral metrics from the trajectory's `decision`
field alone will reproduce the `mode_choice_ratio` / `rewrite_ratio` features
but not the rest of the 60-feature panel. If a metric depends on
`primary_label`, `was_reverted`, or any of the 18-vocab fractions (e.g.
`diversity_entropy_q1`), you must read the label file.

---

## Strategy-label JSON

Lives at `data/strategy_labels/postproc/<model>/<task>/<level>__run<k>.labels.json`.
The 25-file sample mirrors `data/trajectories_sample/` one-for-one; running
the full Tier-2 pipeline regenerates the entire 1,728-file label corpus.

### Top-level structure (11 keys)

| Key | Type | Description |
|---|---|---|
| `vocabulary_version` | str | Identifier for the 18-label vocabulary (paper §A.1) |
| `annotation_date` | null | **Scrubbed in this release** (would be ISO 8601 UTC if regenerated) |
| `labeler_model` | str | LLM used for labeling (e.g. `claude-opus-4-6`) |
| `labeler_settings` | dict | Temperature, max-tokens, retry config used by the labeler |
| `pass1_skipped_step_indices` | list[int] | Steps the labeler chose not to score in Pass 1 (e.g. step 0) — filled by Pass 2 |
| `task_id`, `task_name`, `model`, `level`, `run_id` | str | Join keys; mirror the trajectory's identical fields |
| `n_steps` | int | Number of labeled records (one per trajectory step) |
| `labels` | list[LabelRecord] | Per-step annotations — see below |

### Per-step LabelRecord (8 keys)

| Key | Type | Description |
|---|---|---|
| `step_idx` | int | Index into the trajectory's `steps[]` list (0-based) |
| `step_number` | int | Mirrors trajectory's `steps[i].step_number` (1-based in some pipelines) |
| `primary_label` | str | One of the 18-label vocabulary (paper §A.1), or sentinel `__INITIAL_GENERATION__` / `__ANNOTATION_FAILED__` |
| `confidence` | float | Labeler-reported confidence in `[0, 1]` |
| `label_notes` | str | Free-text rationale from the labeler |
| `was_reverted` | bool | Whether this step was rolled back later in the run (Pass-2 annotation) |
| `revert_reason` | str \| None | If `was_reverted`, the Pass-2 rationale |
| `postproc_rule_applied` | str \| None | Which of the 7 cleanup rules (A–G) reclassified this label, or `None` if untouched |

The 18-label vocabulary is documented in `data/strategy_labeler/prompt.md`;
the 7 post-processing rules (with audit numbers) are in
`scripts/postproc_labels.py`. The original raw-label audit ground truth
(60 hand-corrected cases) lives at
`data/strategy_labeler/audit_{representation,reverse}.md`.

---

## Joining trajectory and labels

Both files share the **same** `(task_id, model, level, run_id)` tuple at the
top level, and the labels' `step_idx` joins to the trajectory's `steps[i]` by
position. The trajectory tells you *what* the agent emitted and *how* the
harness routed it; the label file tells you *what kind* of edit it was on
the 18-vocab axis (which is what the paper measures). Reading both for one cell:

```python
import json

cell = ("claude-opus-4-6", "hexagon_packing", "L0", 1)
m, t, lv, r = cell
traj_path = f"data/trajectories_sample/{m}/{t}/{lv}__run{r}.json"
lbl_path  = f"data/strategy_labels/postproc/{m}/{t}/{lv}__run{r}.labels.json"

traj = json.load(open(traj_path))
lbl  = json.load(open(lbl_path))

assert traj["model"] == lbl["model"]
assert traj["run_id"] == lbl["run_id"]
assert len(traj["steps"]) == lbl["n_steps"]

for step, label in zip(traj["steps"], lbl["labels"]):
    # step["generation"]["code"]  ← what the agent produced
    # label["primary_label"]      ← what the offline labeler said about that code
    ...
```

Behavioral metrics (`data/metrics/all_metrics.csv`) are precomputed from this
join — see `scripts/compute_metrics.py` for the exact formulas.
