"""
Render Figure 2 (fig:harness_main) — two views of task-conditioned harness response.

Panel (a): 8x4 heatmap
  - Rows: 8 non-empty (alg-demand, repr-demand) capability cells
    ordered low/low, low/med, med/low, med/med, med/high, high/low, high/med, high/high
  - Cols: L0, L0f, L1, L2
  - Cell: mean rank-normalized score, viridis colormap, range [0, 1]
  - Annotate each cell with numeric value

Panel (b): 6x4 grid
  - Rows: 6 models in panel order
  - Cols: 4 task families (math, GPU kernel, ADRS, FCS algorithmic)
  - Cell: best harness level per (model, family); colored by level
  - Cells with margin < 0.05 marked with †

Total figure: 7.0 in wide × 3.6 in tall.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

_PKG_ROOT = Path(__file__).resolve().parents[2]

ROOT = str(_PKG_ROOT)
FIG_DIR = f"{ROOT}/analysis/harness_by_task/figures"
QUAD_CSV = f"{ROOT}/analysis/harness_by_task/quadrant_x_level_means.csv"
PMFL_CSV = f"{ROOT}/analysis/harness_by_task/per_model_family_level.csv"
PDF_OUT = f"{FIG_DIR}/harness_main.pdf"
PNG_OUT = f"{FIG_DIR}/harness_main.png"

# Order of capability-demand cells per spec
CELL_ORDER = [
    ("low", "low"),
    ("low", "med"),
    ("med", "low"),
    ("med", "med"),
    ("med", "high"),
    ("high", "low"),
    ("high", "med"),
    ("high", "high"),
]
LEVELS = ["L0", "L0f", "L1", "L2"]
LEVEL_DISP = {"L0": r"$L_0$", "L0f": r"$L_{0f}$", "L1": r"$L_1$", "L2": r"$L_2$"}

MODELS_ORDER = [
    "claude-opus-4-6",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "deepseek-v3.2",
    "gpt-oss-120b",
    "qwen3-next-80b-instruct",
]
MODELS_DISPLAY = {
    "claude-opus-4-6": "claude",
    "gpt-5.4": "gpt-5.4",
    "gemini-3.1-pro-preview": "gemini",
    "deepseek-v3.2": "deepseek",
    "gpt-oss-120b": "gpt-oss",
    "qwen3-next-80b-instruct": "qwen3",
}
FAMILIES_ORDER = ["Math optimization", "GPU kernel", "ADRS", "FCS algorithmic"]
FAMILIES_DISP = {
    "Math optimization": "math",
    "GPU kernel": "GPU kernel",
    "ADRS": "ADRS",
    "FCS algorithmic": "FCS algo.",
}

# Discrete colors per level for panel (b)
LEVEL_COLORS = {
    "L0":  "#cccccc",  # light gray
    "L0f": "#7bccc4",  # teal
    "L1":  "#43a2ca",  # blue
    "L2":  "#0868ac",  # dark blue
}


def load_quadrant_means():
    """Return dict[(alg, repr, level)] -> mean."""
    out = {}
    with open(QUAD_CSV) as f:
        for row in csv.DictReader(f):
            out[(row["alg_demand"], row["repr_demand"], row["level"])] = float(row["mean"])
    return out


def load_pmfl():
    """Return dict[(model, family)] -> {level -> mean}."""
    out = defaultdict(dict)
    with open(PMFL_CSV) as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["family"])][row["level"]] = float(row["mean"])
    return out


def main():
    Path(f"{FIG_DIR}").mkdir(parents=True, exist_ok=True)
    quad = load_quadrant_means()
    pmfl = load_pmfl()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "text.usetex": False,
    })

    fig = plt.figure(figsize=(7.5, 3.8))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.0],
        wspace=0.50,
        left=0.10, right=0.96, top=0.86, bottom=0.20,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ============== Panel (a) ==============
    M_a = np.array([[quad[(alg, repr_, lv)] for lv in LEVELS]
                    for (alg, repr_) in CELL_ORDER])
    im = ax_a.imshow(M_a, cmap="viridis", aspect="auto",
                     vmin=0.0, vmax=1.0, origin="upper")
    # Cell annotations
    for ri in range(len(CELL_ORDER)):
        for ci in range(len(LEVELS)):
            v = M_a[ri, ci]
            txt_color = "white" if v < 0.55 else "black"
            ax_a.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                      fontsize=10, color=txt_color)
    # X ticks: harness levels
    ax_a.set_xticks(range(len(LEVELS)))
    ax_a.set_xticklabels([LEVEL_DISP[lv] for lv in LEVELS], fontsize=12)
    ax_a.set_xlabel("Harness level", fontsize=11, labelpad=4)
    # Y ticks: capability cells (alg/repr abbreviations)
    ax_a.set_yticks(range(len(CELL_ORDER)))
    ax_a.set_yticklabels([f"{alg[0].upper()}/{repr_[0].upper()}"
                          for (alg, repr_) in CELL_ORDER], fontsize=10)
    ax_a.set_ylabel("Capability cell (alg / repr)", fontsize=11, labelpad=4)
    # Subplot title
    ax_a.set_title("(a) Capability-demand cell × harness", fontsize=12,
                   fontweight="semibold", loc="left", pad=8)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_a, orientation="horizontal",
                        pad=0.20, fraction=0.06, aspect=30)
    cbar.set_label("Mean rank-normalized score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Remove imshow box outline
    for sp in ax_a.spines.values():
        sp.set_visible(False)

    # ============== Panel (b) ==============
    # Compute best level per (model, family) and margin
    best_levels = np.empty((len(MODELS_ORDER), len(FAMILIES_ORDER)), dtype=object)
    margins = np.zeros((len(MODELS_ORDER), len(FAMILIES_ORDER)))
    for ri, m in enumerate(MODELS_ORDER):
        for ci, fam in enumerate(FAMILIES_ORDER):
            cell = pmfl[(m, fam)]
            sorted_levels = sorted(LEVELS, key=lambda lv: -cell[lv])
            best = sorted_levels[0]
            second = sorted_levels[1]
            best_levels[ri, ci] = best
            margins[ri, ci] = cell[best] - cell[second]

    # Render grid
    cell_size = 1.0
    for ri in range(len(MODELS_ORDER)):
        for ci in range(len(FAMILIES_ORDER)):
            best = best_levels[ri, ci]
            margin = margins[ri, ci]
            color = LEVEL_COLORS[best]
            # Cell background (solid)
            ax_b.add_patch(Rectangle(
                (ci, len(MODELS_ORDER) - 1 - ri), cell_size, cell_size,
                facecolor=color, edgecolor="white", lw=1.0,
            ))
            # Best-level label inside cell
            text_color = "white" if best == "L2" else "black"
            ax_b.text(ci + cell_size / 2, len(MODELS_ORDER) - 1 - ri + cell_size / 2,
                      LEVEL_DISP[best],
                      ha="center", va="center",
                      fontsize=14, color=text_color, fontweight="bold")
    ax_b.set_xlim(0, len(FAMILIES_ORDER))
    ax_b.set_ylim(0, len(MODELS_ORDER))
    # X ticks: families
    ax_b.set_xticks([ci + 0.5 for ci in range(len(FAMILIES_ORDER))])
    ax_b.set_xticklabels([FAMILIES_DISP[f] for f in FAMILIES_ORDER],
                         fontsize=10)
    ax_b.set_xlabel("Task family", fontsize=11, labelpad=4)
    # Y ticks: models (top-to-bottom = MODELS_ORDER)
    ax_b.set_yticks([len(MODELS_ORDER) - 1 - ri + 0.5
                    for ri in range(len(MODELS_ORDER))])
    ax_b.set_yticklabels([MODELS_DISPLAY[m] for m in MODELS_ORDER], fontsize=10)
    ax_b.tick_params(axis="x", which="both", length=0)
    ax_b.tick_params(axis="y", which="both", length=0)
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    ax_b.set_title("(b) Per-model best level by family", fontsize=12,
                   fontweight="semibold", loc="left", pad=8)

    # Legend below panel (b)
    legend_elements = [
        Patch(facecolor=LEVEL_COLORS["L0"], edgecolor="white", label=r"$L_0$"),
        Patch(facecolor=LEVEL_COLORS["L0f"], edgecolor="white", label=r"$L_{0f}$"),
        Patch(facecolor=LEVEL_COLORS["L1"], edgecolor="white", label=r"$L_1$"),
        Patch(facecolor=LEVEL_COLORS["L2"], edgecolor="white", label=r"$L_2$"),
    ]
    ax_b.legend(
        handles=legend_elements,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=4, fontsize=10, framealpha=0.0, handlelength=1.4,
        columnspacing=1.4, borderpad=0.2,
    )

    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
