"""Render Figure 6.4.1 — Two slices of the frontier-vs-OSS gap.

Reads d_feature_ranking.csv and c_per_feature_gap.csv; writes
figures/gap_two_slices.pdf in the paper folder.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PKG_ROOT = Path(__file__).resolve().parents[2]

ROOT = _PKG_ROOT
GAP = ROOT / "analysis/gap_decomposition"
FIG_DIR = GAP / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

D_CSV = GAP / "d_feature_ranking.csv"
C_CSV = GAP / "c_per_feature_gap.csv"
OUT = FIG_DIR / "gap_two_slices.pdf"


def load_d():
    rows = []
    with D_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "feature": r["feature"],
                "rank": int(r["rank"]),
                "importance": float(r["mean_importance"]),
                "direction": float(r["frontier_minus_oss"]),
            })
    return rows


def load_c():
    def _f(s, default=float("nan")):
        try:
            return float(s)
        except (TypeError, ValueError):
            return default
    rows = []
    with C_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "feature": r["feature"],
                "signed_diff": _f(r["mean_signed_diff"]),
                "consistency": _f(r["consistency_rate"]),
                "p": _f(r["paired_t_pvalue"]),
                "abs_diff": _f(r["abs_mean_diff"]),
            })
    rows.sort(key=lambda r: -r["abs_diff"])
    return rows


COLOR_FRONTIER = "#1f78b4"  # blue
COLOR_OSS = "#ff7f00"       # orange


def fmt_label(name: str, dagger: bool) -> str:
    s = "\\texttt{" + name.replace("_", "\\_") + "}"
    if dagger:
        s = s + "$^{\\dagger}$"
    return s


def main():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    d_rows = load_d()
    c_rows = load_c()
    d_top = d_rows[:10]
    c_top = c_rows[:10]
    d_top_names = {r["feature"] for r in d_top}
    c_top_names = {r["feature"] for r in c_top}
    common = d_top_names & c_top_names

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 3.5),
        gridspec_kw={"wspace": 0.55},
    )

    # LEFT panel: D top-10 by importance.
    d_sorted = sorted(d_top, key=lambda r: r["importance"])
    y = list(range(len(d_sorted)))
    colors = [COLOR_FRONTIER if r["direction"] > 0 else COLOR_OSS
              for r in d_sorted]
    bars = axL.barh(y, [r["importance"] for r in d_sorted],
                    color=colors, edgecolor="black", linewidth=0.4)
    labels = []
    for r in d_sorted:
        nm = r["feature"]
        if nm in common:
            labels.append(nm + " †")
        else:
            labels.append(nm)
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontfamily="monospace")
    axL.set_xlabel("RF importance")
    axL.set_title("(a) full-profile classifier (60 features)")
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.tick_params(axis="y", length=0)
    axL.grid(axis="x", linestyle=":", alpha=0.4)

    # RIGHT panel: C top-10 by |signed_diff|.
    c_sorted = sorted(c_top, key=lambda r: r["abs_diff"])
    y2 = list(range(len(c_sorted)))
    colors2 = [COLOR_FRONTIER if r["signed_diff"] > 0 else COLOR_OSS
               for r in c_sorted]
    bars2 = axR.barh(y2, [r["abs_diff"] for r in c_sorted],
                     color=colors2, edgecolor="black", linewidth=0.4)
    labels2 = []
    for r in c_sorted:
        nm = r["feature"]
        if nm in common:
            labels2.append(nm + " †")
        else:
            labels2.append(nm)
    axR.set_yticks(y2)
    axR.set_yticklabels(labels2, fontfamily="monospace")
    axR.set_xlabel("|mean signed diff| (z)")
    axR.set_title("(b) score-matched per-case gap (24 features)")
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    axR.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend.
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=COLOR_FRONTIER, edgecolor="black",
              label="frontier higher"),
        Patch(facecolor=COLOR_OSS, edgecolor="black",
              label="OSS higher"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Two slices of the frontier-vs-OSS gap",
        fontsize=10, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print(f"[save] {OUT}")
    print(f"[info] features common to both panels (marked †): {sorted(common)}")


if __name__ == "__main__":
    main()
