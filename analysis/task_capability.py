"""
Task × 2D capability space analysis.

Builds empirical algorithm_demand and representation_demand profiles from
behavioral metrics, compares them to a priori profiles from tasks.yaml,
and produces a scatter plot of the 2D capability space.

Usage:
    python analysis/task_capability.py [--metrics-csv data/metrics/all_metrics.csv]
                                       [--tasks-yaml configs/tasks.yaml]
                                       [--output data/analysis/capability_space.png]
                                       [--report data/analysis/profile_correlation.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Empirical profile computation
# ---------------------------------------------------------------------------

def compute_empirical_profile(metrics_df: pd.DataFrame, task_name: str) -> Optional[dict]:
    """
    Compute empirical algorithm_demand and representation_demand for a task.

    Algorithm demand is estimated from:
      - Frequency of algorithm_switch edits
      - Breakthrough magnitude on new algorithms (score jump after algorithm_switch)

    Representation demand is estimated from:
      - Frequency of representation_change edits
      - Number of distinct problem framings (proxied by representation_change count)

    Returns dict with keys algorithm_demand and representation_demand (floats 0-1),
    or None if the task has no data in metrics_df.
    """
    col_task = "task_id" if "task_id" in metrics_df.columns else "task_name" if "task_name" in metrics_df.columns else None
    if col_task is None:
        return None

    task_df = metrics_df[metrics_df[col_task] == task_name]
    if task_df.empty:
        return None

    algo_demand: Optional[float] = None
    repr_demand: Optional[float] = None

    # --- Algorithm demand ---
    if "edit_type_algorithm_switch_frac" in task_df.columns:
        algo_frac = task_df["edit_type_algorithm_switch_frac"].mean()
        # Normalize: algo_switch_frac of ~0.3+ is considered high demand
        algo_demand_raw = float(np.clip(algo_frac / 0.3, 0.0, 1.0))

        # Boost if breakthrough magnitude is available
        if "plateau_escape_rate" in task_df.columns:
            escape_rate = task_df["plateau_escape_rate"].mean()
            # High escape rate on algorithm switches = high algorithm demand
            algo_demand_raw = float(np.clip(algo_demand_raw * 0.7 + float(escape_rate) * 0.3, 0.0, 1.0))

        algo_demand = algo_demand_raw

    # --- Representation demand ---
    if "edit_type_representation_change_frac" in task_df.columns:
        repr_frac = task_df["edit_type_representation_change_frac"].mean()
        # Normalize: repr_change_frac of ~0.2+ is considered high demand
        repr_demand_raw = float(np.clip(repr_frac / 0.2, 0.0, 1.0))

        # Boost if structural changes are also common (proxy for reformulation)
        if "edit_type_structural_change_frac" in task_df.columns:
            struct_frac = task_df["edit_type_structural_change_frac"].mean()
            repr_demand_raw = float(np.clip(repr_demand_raw * 0.8 + float(struct_frac) * 0.2, 0.0, 1.0))

        repr_demand = repr_demand_raw

    if algo_demand is None and repr_demand is None:
        return None

    return {
        "algorithm_demand": algo_demand,
        "representation_demand": repr_demand,
    }


# ---------------------------------------------------------------------------
# Profile correlation
# ---------------------------------------------------------------------------

def compute_profile_correlation(tasks_yaml_path: str, metrics_df: pd.DataFrame) -> float:
    """
    Compute Pearson correlation between a priori and empirical profiles across tasks.

    For each task that has both an a priori profile (in tasks.yaml) and an empirical
    profile (computable from metrics_df), extract the (algorithm_demand,
    representation_demand) pair from each source. Concatenate into two flat vectors
    and compute Pearson r.

    Returns the Pearson r value, or nan if insufficient data.
    """
    import yaml

    with open(tasks_yaml_path) as f:
        config = yaml.safe_load(f)

    tasks = config.get("tasks", [])

    a_priori_vals: list[float] = []
    empirical_vals: list[float] = []

    for task in tasks:
        name = task.get("name")
        a_priori = task.get("a_priori_profile", {})
        if not a_priori:
            continue

        empirical = compute_empirical_profile(metrics_df, name)
        if empirical is None:
            continue

        for dim in ("algorithm_demand", "representation_demand"):
            ap_val = a_priori.get(dim)
            em_val = empirical.get(dim)
            if ap_val is not None and em_val is not None:
                a_priori_vals.append(float(ap_val))
                empirical_vals.append(float(em_val))

    if len(a_priori_vals) < 3:
        return float("nan")

    r = float(np.corrcoef(a_priori_vals, empirical_vals)[0, 1])
    return r


# ---------------------------------------------------------------------------
# 2D capability space scatter plot
# ---------------------------------------------------------------------------

def plot_2d_capability_space(
    tasks_yaml_path: str,
    metrics_df: Optional[pd.DataFrame],
    output_path: str,
) -> None:
    """
    Scatter plot of tasks in 2D capability space (algorithm_demand × representation_demand).

    Uses empirical profiles when available, falls back to a priori profiles otherwise.
    Tasks are colored by source (skydiscover vs frontier_cs_research vs frontier_cs_algorithmic).
    Task names are shown as labels.

    Saves PNG to output_path.
    """
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(tasks_yaml_path) as f:
        config = yaml.safe_load(f)

    tasks = config.get("tasks", [])

    source_colors = {
        "skydiscover": "#2196F3",           # blue
        "frontier_cs_research": "#E91E63",  # pink
        "frontier_cs_algorithmic": "#4CAF50",  # green
    }
    default_color = "#9E9E9E"

    fig, ax = plt.subplots(figsize=(11, 9))

    plotted: dict[str, list] = {}  # source -> list of (x, y, name)

    for task in tasks:
        name = task.get("name", "?")
        source = task.get("source", "unknown")

        # Try empirical profile first
        profile: Optional[dict] = None
        if metrics_df is not None:
            profile = compute_empirical_profile(metrics_df, name)

        # Fall back to a priori
        if profile is None:
            a_priori = task.get("a_priori_profile", {})
            if a_priori:
                profile = {
                    "algorithm_demand": a_priori.get("algorithm_demand"),
                    "representation_demand": a_priori.get("representation_demand"),
                }

        if profile is None:
            continue

        x = profile.get("algorithm_demand")
        y = profile.get("representation_demand")
        if x is None or y is None:
            continue

        if source not in plotted:
            plotted[source] = []
        plotted[source].append((float(x), float(y), name))

    for source, points in plotted.items():
        color = source_colors.get(source, default_color)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        label = source.replace("_", " ").title()
        ax.scatter(xs, ys, c=color, label=label, s=80, alpha=0.85, edgecolors="white", linewidths=0.5)
        for x, y, name in points:
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9,
            )

    ax.set_xlabel("Algorithm Demand\n(depends on finding right algorithm / building blocks)", fontsize=11)
    ax.set_ylabel("Representation Demand\n(depends on problem formulation / encoding)", fontsize=11)
    ax.set_title("2D Task Capability Space", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Add quadrant labels
    for qx, qy, qlabel in [
        (0.75, 0.75, "High algo\nHigh repr"),
        (0.25, 0.75, "Low algo\nHigh repr"),
        (0.75, 0.25, "High algo\nLow repr"),
        (0.25, 0.25, "Low algo\nLow repr"),
    ]:
        ax.text(qx, qy, qlabel, ha="center", va="center", fontsize=8,
                color="gray", alpha=0.4, style="italic")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Capability space plot saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="2D task capability space analysis.")
    parser.add_argument("--metrics-csv", default="data/metrics/all_metrics.csv",
                        help="Path to metrics CSV (optional — uses a priori profiles if missing)")
    parser.add_argument("--tasks-yaml", default="configs/tasks.yaml",
                        help="Path to tasks.yaml")
    parser.add_argument("--output", default="data/analysis/capability_space.png",
                        help="Output path for scatter plot PNG")
    parser.add_argument("--report", default="data/analysis/profile_correlation.txt",
                        help="Output path for profile correlation report")
    args = parser.parse_args()

    # Load metrics if available
    metrics_df: Optional[pd.DataFrame] = None
    csv_path = Path(args.metrics_csv)
    if csv_path.exists():
        metrics_df = pd.read_csv(csv_path)
        print(f"Loaded metrics from {csv_path} ({len(metrics_df)} rows)")
    else:
        print(f"Metrics CSV not found at {csv_path} — using a priori profiles only.")

    tasks_yaml = args.tasks_yaml
    if not Path(tasks_yaml).exists():
        print(f"ERROR: tasks.yaml not found: {tasks_yaml}")
        sys.exit(1)

    # Compute and report correlation if metrics are available
    if metrics_df is not None:
        r = compute_profile_correlation(tasks_yaml, metrics_df)
        report_lines = [
            "Profile Correlation Report",
            "=" * 40,
            f"Pearson r (a priori vs empirical): {r:.4f}" if not np.isnan(r) else "Pearson r: insufficient data",
            "",
            "Interpretation:",
            "  r > 0.7 : a priori profiles strongly validated by behavioral data",
            "  r 0.4-0.7: moderate alignment — some tasks behave unexpectedly",
            "  r < 0.4 : weak alignment — a priori taxonomy needs revision",
        ]
        report_text = "\n".join(report_lines)
        print(report_text)

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text + "\n")
        print(f"Correlation report saved to {report_path}")
    else:
        print("Skipping correlation computation (no metrics data).")

    # Generate scatter plot
    plot_2d_capability_space(tasks_yaml, metrics_df, args.output)


if __name__ == "__main__":
    main()
