# Capability Demand Rubric

**Purpose:** Score each task in `configs/tasks.yaml` on two dimensions
(`algorithm_demand`, `representation_demand`) so downstream analysis can
compare a priori expectations to empirical behavioral metrics.

Optimization effort is a **universal baseline** assumed for every iterative
discovery task. We do NOT score it as a separate dimension.

## 1. Dimensions

### 1.1 Algorithm demand — "Does the task reward knowing the right algorithm?"

This dimension measures whether **switching between algorithm families
unlocks score regimes that cannot be reached by tuning within a single
family.**

- **HIGH** — Multiple algorithm families exist AND switching between
  families produces qualitatively different score regimes. Within-family
  tuning plateaus at a score significantly below what a different family
  can achieve. Breakthroughs come from recognizing and switching, not from
  refinement.
- **MED** — Multiple families exist, but within-family tuning eventually
  closes most of the gap. Switching helps but a well-tuned single family
  reaches near-best scores.
- **LOW** — One family dominates, OR all "different" approaches are
  notational variants of the same underlying algorithm. Nearly all score
  comes from parameter and implementation tuning within that family.

### 1.2 Representation demand — "Does the task reward framing the problem differently?"

This dimension measures whether **different problem formulations expose
structurally different search landscapes**, such that a solver designed
for one formulation cannot be trivially adapted to another.

- **HIGH** — The task admits formulations where the same information is
  encoded in ways that require **fundamentally different solvers**. A
  gradient-based solver for formulation A may be useless for formulation
  B even though both describe the same problem.
- **MED** — Multiple formulations exist, and 2–3 of them change which
  aspects of the problem are easy or hard to manipulate. A solver
  designed for one formulation would need substantial modification for
  another.
- **LOW** — The natural formulation is essentially unique. "Alternatives"
  are notational rewrites that preserve the search landscape — a solver
  can be converted with rewrites of input/output, not re-design.

### 1.3 Operational test for representation demand

When unsure between bins, apply this test:

> *"If I had a working solver for formulation A achieving score X, would
> converting to formulation B require designing a **fundamentally
> different solver** — not just rewriting the input/output layer?"*
>
> - If yes for multiple formulations → **HIGH**
> - If yes for one alternative → **MED**
> - If no, it's only notational → **LOW**

**Example of notational vs structural:**
- A permutation can be represented as an array, as a bijection, as a
  cycle decomposition. These are **notational** — any good solver
  converts between them trivially. *Representation demand: LOW.*
- A set of points can be represented by coordinates, or as vertices of
  a graph with pairwise-distance edges. These are **structural** — a
  gradient solver works on the first, a graph algorithm on the second,
  and they explore different score neighborhoods. *Representation demand
  contribution: HIGH.*

### 1.4 Critical rule: demand is task-intrinsic

Do **not** rate demand based on whether any specific model (Opus, GPT-5,
etc.) can solve the task. Demand describes what *any capable solver*
would need to do well. A task with LOW algorithm demand can still be
hard (tight precision, implementation complexity, tricky edge cases) —
but that doesn't make it HIGH on this axis.

Concretely: do not let your impression of Opus's skills influence the
score. Rate as if scoring for an abstract competent solver.

---

## 2. Bins (discrete, 3-level)

Write exactly one of these numeric values. Do NOT use intermediate values
(0.3, 0.6, etc.) — the 3-level discretization is intentional.

| Bin | Value | Meaning |
| --- | --- | --- |
| **LOW** | `0.25` | See §1 definitions; roughly "one natural approach/formulation" |
| **MED** | `0.50` | Multiple approaches exist but gap is narrow / formulations are similar |
| **HIGH** | `0.75` | Switching meaningfully changes the achievable score regime or search space |

**Tie-breaking rule:** If you are genuinely unsure between two bins, pick
the LOWER one and set `confidence: medium` in the justification. Do NOT
split the difference with an intermediate value.

**Confidence levels:**
- `high`: you can cite specific algorithms/reformulations from your own
  knowledge or from the task's own README.
- `medium`: you can reason about the task from first principles but
  don't recognize specific named algorithms.
- `low`: you don't recognize the task domain and are scoring from the
  literal task description only.

---

## 3. Scoring protocol — files to read per task

For **every** task in `configs/tasks.yaml`, read the following files
before scoring. If a file is missing, note it in your reasoning and
proceed.

### SkyDiscover tasks (`source: skydiscover`)

Base path:
`<SKYDISCOVER_ROOT>/benchmarks/<task_id>/`

Read in order:
1. `README.md` — problem setting; look for a "known approaches" or
   "prior work" section.
2. `config.yaml` — `prompt.system_message` (the task author's intended
   direction).
3. `initial_program.py` first ~30 lines — the seed solution reveals the
   "obvious approach".
4. The evaluator's `evaluate()` return statement — tells you what is
   actually scored.

Some tasks (e.g. `hexagon_packing`, `minimizing_max_min_dist`) have a
numbered subdirectory like `12/` or `3/`; read inside that subdir.

### Frontier-CS research tasks (`source: frontier_cs_research`)

Base path:
`<FRONTIER_CS_ROOT>/research/problems/<task_id>/`

Read:
1. `readme` (or `README` / `README.md` / `problem.md`) — problem
   statement.
2. `config.yaml` if present — runtime/environment hints.
3. `resources/` directory listing — sometimes contains seed code.

### Frontier-CS algorithmic tasks (`source: frontier_cs_algorithmic`)

Base path:
`<FRONTIER_CS_ROOT>/algorithmic/problems/<task_id>/`

Read:
1. `statement.txt` — the competitive-programming problem statement.
2. `examples/` directory listing (sample inputs/outputs).

### Tools

Use the `Read` tool to open files by absolute path. If a file doesn't
exist at the expected path, use `Glob` to find the closest match. Do
NOT use `Bash` for this — Read is faster and more reliable.

---

## 4. Anchors — one task per quadrant

These four tasks span the 2×2 space and should be your primary
calibration references. When unsure about a task, ask: *"which of these
four is it closest to on each dimension independently?"*

Before reading the anchor reasoning, note that **each anchor
demonstrates independence** — e.g. `matmul` is HIGH on algorithm but
LOW on representation, showing that the two dimensions are genuinely
orthogonal.

### 4.1 The four quadrant anchors

**`matmul` — Algorithm HIGH (0.75), Representation LOW (0.25)**
- Algorithm: Tensor decomposition is a ~50-year active research area
  (Strassen, Laderman, Smirnov, AlphaTensor, approximate
  factorizations, field-specific decompositions). Each is a
  **fundamentally different algorithm** and switching unlocks rank
  improvements that tuning cannot reach. Clear HIGH.
- Representation: The tensor has fixed shape and indices. "Alternative
  formulations" (e.g. matrix view vs tensor view) are notational — the
  underlying search for low-rank factorizations is identical. LOW.

**`symbolic_regression/peaks` — Algorithm LOW (0.25), Representation HIGH (0.75)**
- Algorithm: One dominant paradigm (search over expression space).
  Within-paradigm choices (genetic programming vs. MCTS vs. gradient-
  guided) give similar final scores; they are different search
  strategies, not different algorithm families in the demand sense. LOW.
- Representation: Expressions as syntax trees, as token sequences, as
  compositional operators, as neural embeddings — each exposes a
  different solver class. A tree-based GP solver and a token-based
  transformer search see structurally different landscapes. HIGH.

**`grammar_fuzzing` — Algorithm HIGH (0.75), Representation HIGH (0.75)**
- Algorithm: Coverage-guided, generative, BNF-mutation, neural-guided,
  and symbolic-execution-based fuzzers are all accepted paradigms with
  very different score profiles per target. Switching matters. HIGH.
- Representation: Grammar-as-rules vs. grammar-as-examples vs. grammar-
  as-tree-skeleton encode the same object in structurally different
  ways — a grammar-rule-based fuzzer cannot trivially reuse the
  tree-skeleton representation's machinery. HIGH.

**`vecadd` — Algorithm LOW (0.25), Representation LOW (0.25)**
- Algorithm: Exactly one kernel structure (elementwise add). All
  improvements come from parameter tuning: block size, vectorization
  width, memory coalescing. No algorithmic choice. LOW.
- Representation: Fixed input tensor shape, no meaningful alternatives.
  LOW.

### 4.2 Secondary anchors (optional reference, not required)

These may help if your task's domain is underrepresented above:

- **Algorithm MED, Representation LOW:** `txn_scheduling` — 2–3 valid
  scheduling heuristics exist (greedy, cost-based, ILP-approximate),
  but the search space (permutations of transactions) is essentially
  fixed.
- **Algorithm MED, Representation MED:** `signal_processing` — 3-5
  filter families (moving average, Kalman, EMD, wavelets); time- vs.
  frequency- vs. state-space formulations are real alternatives but
  not structurally alien.
- **Algorithm LOW, Representation MED:** `hexagon_packing`,
  `circle_packing_rect`, `heilbronn_convex` — a few geometric
  parametrizations (coordinates, angles-radii) give MED representation
  demand; algorithmically, placement heuristics dominate and switching
  between them gives marginal gains.

### 4.3 Calibration reminders

- The bin scale is **relative to the full space of discovery tasks**,
  not relative to other tasks in this particular registry. If our
  registry happens to contain many HIGH-demand tasks, that's fine.
- **Independence matters.** A task can be HIGH on one dimension and LOW
  on the other. Do not anchor on a task's "overall difficulty"; score
  each axis independently.