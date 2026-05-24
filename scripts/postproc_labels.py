#!/usr/bin/env python3
"""
postproc_labels.py — Apply 7 post-processing rules to raw labels.

Inputs:
  - data/strategy_labels/raw/<model>/<task>/<file>.raw.json
  - data/trajectories/<model>/<task>/<file>.json (for Rule C diffs)

Outputs:
  - data/strategy_labels/postproc/<model>/<task>/<file>.labels.json
  - data/strategy_labels/postproc_summary.md
  - data/strategy_labels/behavioral_metrics.csv

Validity checks:
  - total reclassified > 30%
  - any single rule > 15%
  - Rule C > 5% of refinement labels
  - Rule G misses > 30%
"""
from __future__ import annotations

import csv
import difflib
import glob
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]

ROOT = _PKG_ROOT
RAW_LABELS_DIR = ROOT / "data/strategy_labels/raw"
LABELS_OUT_DIR = ROOT / "data/strategy_labels/postproc"
TRAJ_DIR = ROOT / "data/trajectories"
SUMMARY_PATH = ROOT / "data/strategy_labels/postproc_summary.md"
METRICS_PATH = ROOT / "data/strategy_labels/behavioral_metrics.csv"

VOCAB = {
    "bug_fix", "harness_or_scope_change", "no_change", "algorithm_switch",
    "representation_change", "data_structure_change", "search_space_change",
    "algorithm_refinement", "initialization_change", "parameter_tuning",
    "autotune_configuration", "loop_optimization", "memory_optimization",
    "kernel_restructuring", "precision_handling", "feature_engineering",
    "code_cleanup", "other",
}
SENTINEL_INITIAL = "__INITIAL_GENERATION__"
SENTINEL_FAILED = "__ANNOTATION_FAILED__"

# ---------------------------------------------------------------------------
# Rule keyword sets
# ---------------------------------------------------------------------------

RULE_A_KEYWORDS = [
    "exploit", "cheat", "trivially valid", "degenerate solution",
    "invalid configuration", "not enforced", "gaming the score",
    "scaling to a target", "hardcoded target value",
]
RULE_B_KEYWORDS = [
    "was causing errors", "broken code that was", "fixing the",
    "syntax error", "fixes a", "addresses the bug",
    "indentation error", "control-flow bug", "undefined typo",
]
# Rule C is regex-based on the diff
RULE_C_VAR_NUM_PATTERN = re.compile(
    r"^[+\-]\s*[a-z_][a-z_0-9.]*\s*=\s*[\-]?[0-9.eE+\-]+\s*(#.*)?$"
)
RULE_C_FORBIDDEN_KEYWORDS = ["def ", "class ", "import "]

RULE_D_INTRA_KEYWORDS = [
    "could also be algorithm_refinement", "within the same family",
    "stays within", "intra-family", "could be refinement",
    "still in the same family", "all within", "remains in",
]
RULE_D_FORBIDDEN_KEYWORDS = [
    "fundamentally different family", "cross-family",
    "different family per", "inter-family",
]
RULE_E_KEYWORDS = [
    "format discovery", "EMERGENCY", "explore interface",
    "lost validity", "test return formats", "evaluator interface",
    "format hypothesis", "discover the correct interface",
]
RULE_F_KEYWORDS = [
    "broken rewrite", "introduces a bug", "incomplete edit",
    "references undefined", "broken state", "failed rewrite",
    "new code is broken", "new code has bug", "broken/incomplete",
    "would cause a NameError", "introduces a typo",
]

# Rule G regex — TIGHTENED to only match valid 18-vocab labels (eliminates ~50% false positive rate from loose regex)
LABEL_RE = (
    r"(?:bug_fix|harness_or_scope_change|no_change|algorithm_switch|"
    r"representation_change|data_structure_change|search_space_change|"
    r"algorithm_refinement|initialization_change|parameter_tuning|"
    r"autotune_configuration|loop_optimization|memory_optimization|"
    r"kernel_restructuring|precision_handling|feature_engineering|"
    r"code_cleanup|other)"
)
RULE_G_PATTERN = re.compile(
    rf"by priority[,:\s]+({LABEL_RE})|"
    rf"should be ({LABEL_RE})|"
    rf"({LABEL_RE}) wins (?:over|by priority)|"
    rf"({LABEL_RE}) applies here|"
    rf"priority rule[,:\s]+({LABEL_RE})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def contains_any(text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match for any keyword."""
    if not text:
        return False
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in keywords)


def compute_step_diff_lines(prev_code: str, new_code: str) -> list[str]:
    """Return diff lines (only lines starting with + or -, no file headers)."""
    raw = list(difflib.unified_diff(
        prev_code.splitlines(keepends=False),
        new_code.splitlines(keepends=False),
        lineterm="",
    ))
    out = []
    for ln in raw:
        if ln.startswith("+++") or ln.startswith("---") or ln.startswith("@@"):
            continue
        if ln.startswith("+") or ln.startswith("-"):
            out.append(ln)
    return out


def rule_c_check(diff_lines: list[str]) -> bool:
    """Return True if Rule C should fire on this diff."""
    if not diff_lines:
        return False
    # Count non-comment, non-blank changed lines
    relevant = []
    for ln in diff_lines:
        body = ln[1:].strip()  # strip +/-
        if not body or body.startswith("#"):
            continue
        relevant.append(ln)
    if not relevant:
        return False
    # Forbid lines containing def/class/import (they're added)
    for ln in relevant:
        if ln.startswith("+"):
            for forbid in RULE_C_FORBIDDEN_KEYWORDS:
                if forbid in ln:
                    return False
    # Count matches
    matching = sum(1 for ln in relevant if RULE_C_VAR_NUM_PATTERN.match(ln))
    return matching / len(relevant) > 0.70


def rule_g_extract(notes: str) -> str | None:
    """Extract suggested label from priority-statement match. Returns None if no
    match or extracted token isn't a valid vocab label."""
    if not notes:
        return None
    m = RULE_G_PATTERN.search(notes)
    if not m:
        return None
    for grp in m.groups():
        if grp:
            tok = grp.lower().strip()
            if tok in VOCAB:
                return tok
            return "__INVALID__"  # had a match but wasn't a valid label
    return None


# ---------------------------------------------------------------------------
# Apply rules to one label record
# ---------------------------------------------------------------------------

def apply_rules(rec: dict, prev_code: str, new_code: str) -> tuple[str | None, str | None]:
    """Apply rules A→G to a single label record. Returns (rule_letter, new_label)
    or (None, None) if no rule fires."""
    primary = rec.get("primary_label")
    notes = rec.get("label_notes") or ""

    # Rule A — refinement → harness
    if primary == "algorithm_refinement" and contains_any(notes, RULE_A_KEYWORDS):
        return "A", "harness_or_scope_change"

    # Rule B — refinement → bug_fix
    if primary == "algorithm_refinement" and contains_any(notes, RULE_B_KEYWORDS):
        return "B", "bug_fix"

    # Rule C — refinement → parameter_tuning (regex on diff)
    if primary == "algorithm_refinement":
        diff_lines = compute_step_diff_lines(prev_code, new_code)
        if rule_c_check(diff_lines):
            return "C", "parameter_tuning"

    # Rule D — switch/parameter_tuning → refinement
    if primary in {"algorithm_switch", "parameter_tuning"}:
        has_intra = contains_any(notes, RULE_D_INTRA_KEYWORDS)
        has_inter = contains_any(notes, RULE_D_FORBIDDEN_KEYWORDS)
        if has_intra and not has_inter:
            return "D", "algorithm_refinement"

    # Rule E — switch → harness
    if primary == "algorithm_switch" and contains_any(notes, RULE_E_KEYWORDS):
        return "E", "harness_or_scope_change"

    # Rule F — bug_fix → refinement
    if primary == "bug_fix" and contains_any(notes, RULE_F_KEYWORDS):
        return "F", "algorithm_refinement"

    # Rule G — priority self-contradiction
    if primary in {"algorithm_switch", "parameter_tuning", "bug_fix"}:
        suggested = rule_g_extract(notes)
        if suggested == "__INVALID__":
            return "G_MISS", None
        if suggested is not None and suggested != primary:
            return "G", suggested

    return None, None


# ---------------------------------------------------------------------------
# Behavioral metric helpers
# ---------------------------------------------------------------------------

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def compute_metrics_for_traj(records: list[dict]) -> dict:
    """Compute behavioral metrics from a trajectory's label records."""
    real = [r for r in records
            if r["primary_label"] not in {SENTINEL_INITIAL, SENTINEL_FAILED}]
    n = len(real)
    if n == 0:
        return {}

    label_counts = Counter(r["primary_label"] for r in real)
    metrics = {
        "num_unique_strategies": len(label_counts),
        "algorithm_switch_frac": label_counts.get("algorithm_switch", 0) / n,
    }
    # Per-label fractions
    for lbl in VOCAB:
        metrics[f"label_frac_{lbl}"] = label_counts.get(lbl, 0) / n

    # Diversity entropy by quartile
    q_size = max(1, n // 4)
    for qi in range(1, 5):
        start = (qi - 1) * q_size
        end = qi * q_size if qi < 4 else n
        slice_labels = [r["primary_label"] for r in real[start:end]]
        ctr = Counter(slice_labels)
        metrics[f"diversity_entropy_q{qi}"] = shannon_entropy(ctr)

    # was_reverted-based metric (not affected by rules; recomputed for completeness)
    n_reverted = sum(1 for r in real if r.get("was_reverted"))
    metrics["reverted_frac"] = n_reverted / n

    return metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    sys.stderr.write(f"[postproc] Loading v2 labels from {RAW_LABELS_DIR}\n")
    label_files = sorted(RAW_LABELS_DIR.glob("*/*/*.raw.json"))
    sys.stderr.write(f"[postproc] {len(label_files)} label files\n")

    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate counters
    rule_counts = Counter()
    rule_g_misses = 0
    total_labels = 0
    total_real = 0
    rule_examples = defaultdict(list)  # rule_letter → list of (task, model, step_idx, old, new, snippet)

    # For tasks/models concentration analysis
    task_total = Counter()
    task_changed = Counter()
    model_total = Counter()
    model_changed = Counter()

    # Track ALL refinement labels for Rule C sanity check
    n_refinement = 0
    n_rule_c_fires = 0

    # Updated label distribution
    raw_dist = Counter()
    postproc_dist = Counter()

    # We also recompute metrics — collect rows
    metric_rows = []

    for fp in label_files:
        d = json.loads(fp.read_text())
        model = d["model"]
        task_name = d["task_name"]
        level = d["level"]
        run_id = d["run_id"]

        # Locate trajectory file (need code per step for Rule C)
        traj_path = TRAJ_DIR / fp.relative_to(RAW_LABELS_DIR).parent / fp.name.replace(".raw.json", "")
        traj_steps = None
        if traj_path.exists():
            try:
                traj = json.loads(traj_path.read_text())
                traj_steps = traj.get("steps") or []
            except Exception:
                traj_steps = None

        records = d["labels"]
        for i, rec in enumerate(records):
            total_labels += 1
            primary = rec.get("primary_label")
            raw_dist[primary] += 1
            task_total[task_name] += 1
            model_total[model] += 1

            if primary in {SENTINEL_INITIAL, SENTINEL_FAILED}:
                rec["postproc_rule_applied"] = None
                postproc_dist[primary] += 1
                continue
            total_real += 1
            if primary == "algorithm_refinement":
                n_refinement += 1

            # Compute prev_code/new_code for Rule C if needed
            prev_code, new_code = "", ""
            step_idx = rec.get("step_idx", i)
            if traj_steps and step_idx < len(traj_steps):
                if step_idx >= 1 and step_idx - 1 < len(traj_steps):
                    prev_code = (traj_steps[step_idx - 1].get("generation") or {}).get("code") or ""
                new_code = (traj_steps[step_idx].get("generation") or {}).get("code") or ""

            rule, new_label = apply_rules(rec, prev_code, new_code)

            if rule == "G_MISS":
                rule_g_misses += 1
                rule = None
                new_label = None

            if rule is not None and new_label is not None and new_label in VOCAB:
                rule_counts[rule] += 1
                if rule == "C":
                    n_rule_c_fires += 1
                # Capture sample for summary
                if len(rule_examples[rule]) < 200:  # keep first 200 to sample 5 from later
                    rule_examples[rule].append({
                        "task": task_name, "model": model, "level": level,
                        "step_idx": step_idx, "old": primary, "new": new_label,
                        "notes_snippet": (rec.get("label_notes") or "")[:200],
                    })
                # Apply reclassification
                old_notes = rec.get("label_notes") or ""
                rec["label_notes"] = f"[POSTPROC rule {rule}] reclassified from {primary}: {old_notes}"
                rec["primary_label"] = new_label
                rec["confidence"] = max(rec.get("confidence", 0.65), 0.65)
                rec["postproc_rule_applied"] = rule
                task_changed[task_name] += 1
                model_changed[model] += 1
                postproc_dist[new_label] += 1
            else:
                rec["postproc_rule_applied"] = None
                postproc_dist[primary] += 1

        # Save updated file
        out_fp = LABELS_OUT_DIR / fp.relative_to(RAW_LABELS_DIR).with_suffix("")
        out_fp = out_fp.with_suffix(".labels.json")  # …json.raw labels → strip and rename
        # The relative path includes ".raw.json" suffix; relative_to gives e.g. claude-opus-4-6/cross_entropy/L0__run1.json.raw.json
        # Easier: rebuild path
        rel = fp.relative_to(RAW_LABELS_DIR)
        out_fp = (LABELS_OUT_DIR / rel.parent /
                  (rel.name.replace(".raw.json", ".labels.json")))
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(d, indent=2, default=str))

        # Recompute metrics for this trajectory
        metrics = compute_metrics_for_traj(records)
        metric_rows.append({
            "model": model, "task": task_name, "level": level, "run_id": run_id,
            **metrics,
        })

    # Validity checks
    n_changed = sum(rule_counts.values())
    pct_changed = n_changed / max(1, total_real)
    sanity_alerts = []
    if pct_changed > 0.30:
        sanity_alerts.append(f"⚠ Total reclassified {pct_changed:.2%} > 30%")
    for r, c in rule_counts.items():
        if c / max(1, total_real) > 0.15:
            sanity_alerts.append(f"⚠ Rule {r} fired on {c/total_real:.2%} > 15%")
    if n_refinement > 0 and n_rule_c_fires / n_refinement > 0.05:
        sanity_alerts.append(f"⚠ Rule C fired on {n_rule_c_fires/n_refinement:.2%} of refinement labels > 5%")
    rule_g_total = rule_counts["G"] + rule_g_misses
    if rule_g_total > 0 and rule_g_misses / rule_g_total > 0.30:
        sanity_alerts.append(f"⚠ Rule G misses {rule_g_misses}/{rule_g_total} = {rule_g_misses/rule_g_total:.2%} > 30%")

    # ---------------------------------------------------------------------
    # Write postproc_summary.md
    # ---------------------------------------------------------------------
    md = []
    md.append("# Post-processing summary — raw → postproc'd labels")
    md.append("")
    md.append(f"**Total label files**: {len(label_files)}")
    md.append(f"**Total labels** (incl. sentinels): {total_labels:,}")
    md.append(f"**Real (non-sentinel) labels**: {total_real:,}")
    md.append(f"**Total reclassified**: {n_changed:,} ({pct_changed:.2%})")
    md.append(f"**Unchanged**: {total_real - n_changed:,} ({(total_real - n_changed)/total_real*100:.2f}%)")
    md.append(f"**Rule G misses (regex matched but token not in vocab)**: {rule_g_misses}")
    md.append("")
    if sanity_alerts:
        md.append("## ⚠️  Sanity-guard alerts")
        md.append("")
        for s in sanity_alerts:
            md.append(f"- {s}")
        md.append("")

    # Section 1
    md.append("## 1. Reclassification counts per rule")
    md.append("")
    md.append("| rule | description | count | % of real |")
    md.append("|------|-------------|------:|----------:|")
    descs = {
        "A": "refinement → harness", "B": "refinement → bug_fix",
        "C": "refinement → param_tuning", "D": "switch/param → refinement",
        "E": "switch → harness", "F": "bug_fix → refinement",
        "G": "priority self-contradiction",
    }
    for r in "ABCDEFG":
        c = rule_counts[r]
        md.append(f"| {r} | {descs[r]} | {c:,} | {c/max(1,total_real)*100:.3f}% |")
    md.append(f"| **Total** | | {n_changed:,} | {pct_changed*100:.3f}% |")
    md.append("")

    # Section 2: label distributions
    md.append("## 2. Updated label distribution")
    md.append("")
    md.append("| label | raw count | postproc count | delta |")
    md.append("|-------|---------:|------------------:|------:|")
    all_labels = sorted(set(raw_dist.keys()) | set(postproc_dist.keys()))
    for lbl in all_labels:
        raw_c = raw_dist.get(lbl, 0)
        pp_c = postproc_dist.get(lbl, 0)
        delta = pp_c - raw_c
        sign = "+" if delta > 0 else ""
        md.append(f"| `{lbl}` | {raw_c:,} | {pp_c:,} | {sign}{delta:,} |")
    md.append("")

    # Section 3: random samples per rule
    md.append("## 3. 5 random examples per rule that fired")
    md.append("")
    random.seed(42)
    for r in "ABCDEFG":
        examples = rule_examples.get(r, [])
        if not examples:
            md.append(f"### Rule {r} ({descs[r]}) — **0 fires** (no examples)")
            md.append("")
            continue
        sample = random.sample(examples, min(5, len(examples)))
        md.append(f"### Rule {r} ({descs[r]}) — {rule_counts[r]:,} fires")
        md.append("")
        md.append("| task | model | level | step_idx | old → new | label_notes snippet |")
        md.append("|------|-------|-------|---------:|-----------|---------------------|")
        for ex in sample:
            snippet = ex["notes_snippet"].replace("|", "\\|").replace("\n", " ")[:150]
            md.append(f"| `{ex['task']}` | `{ex['model']}` | {ex['level']} | "
                       f"{ex['step_idx']} | `{ex['old']}` → `{ex['new']}` | {snippet} |")
        md.append("")

    # Section 4: zero-fire report
    md.append("## 4. Rule-fire-zero report")
    md.append("")
    zero_rules = [r for r in "ABCDEFG" if rule_counts[r] == 0]
    if zero_rules:
        md.append(f"⚠ Rules with **zero fires**: {', '.join(zero_rules)}")
        md.append("This may indicate keyword lists are too narrow.")
    else:
        md.append("All 7 rules fired at least once. ✓")
    md.append("")

    # Section 5: suspicious patterns
    md.append("## 5. Suspicious concentration patterns")
    md.append("")
    md.append("### Tasks with > 20% reclassified")
    md.append("")
    md.append("| task | total | changed | pct |")
    md.append("|------|------:|--------:|----:|")
    found_task_concentration = False
    for t in sorted(task_total):
        tot = task_total[t]
        ch = task_changed[t]
        pct = ch / max(1, tot)
        if pct > 0.20:
            found_task_concentration = True
            md.append(f"| `{t}` | {tot:,} | {ch:,} | {pct*100:.2f}% |")
    if not found_task_concentration:
        md.append("(none)")
    md.append("")
    md.append("### Models with > 20% reclassified")
    md.append("")
    md.append("| model | total | changed | pct |")
    md.append("|-------|------:|--------:|----:|")
    found_model_concentration = False
    for m in sorted(model_total):
        tot = model_total[m]
        ch = model_changed[m]
        pct = ch / max(1, tot)
        if pct > 0.20:
            found_model_concentration = True
            md.append(f"| `{m}` | {tot:,} | {ch:,} | {pct*100:.2f}% |")
    if not found_model_concentration:
        md.append("(none)")
    md.append("")
    md.append(f"**Rule G regex misses (extracted token not in vocab)**: {rule_g_misses}")
    md.append("")

    SUMMARY_PATH.write_text("\n".join(md))
    sys.stderr.write(f"[postproc] Summary written: {SUMMARY_PATH}\n")

    # ---------------------------------------------------------------------
    # Write recomputed metrics CSV
    # ---------------------------------------------------------------------
    if metric_rows:
        all_keys = sorted(set().union(*[set(r.keys()) for r in metric_rows]))
        # Order: model, task, level, run_id first, then everything else
        front = ["model", "task", "level", "run_id"]
        ordered = front + sorted([k for k in all_keys if k not in front])
        with METRICS_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            for row in metric_rows:
                writer.writerow(row)
        sys.stderr.write(f"[postproc] Metrics CSV written: {METRICS_PATH}\n")

    # Console summary
    sys.stderr.write(f"\n=== POSTPROC DONE ===\n")
    sys.stderr.write(f"  total real labels: {total_real:,}\n")
    sys.stderr.write(f"  total reclassified: {n_changed:,} ({pct_changed*100:.2f}%)\n")
    sys.stderr.write(f"  per-rule counts:\n")
    for r in "ABCDEFG":
        sys.stderr.write(f"    {r}: {rule_counts[r]:,}\n")
    sys.stderr.write(f"  rule G misses: {rule_g_misses}\n")
    if sanity_alerts:
        sys.stderr.write("  ⚠ SANITY ALERTS:\n")
        for s in sanity_alerts:
            sys.stderr.write(f"    {s}\n")

    return 0 if not sanity_alerts else 1


if __name__ == "__main__":
    sys.exit(main())
