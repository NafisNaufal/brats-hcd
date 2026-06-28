"""Generate HVR bar chart for the paper (Table IV → Figure).

Usage:
    python scripts/plot_hvr_figure.py --out paper/figures/hvr_bar.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HVR_DATA = [
    ("E0  Flat-Swin",       0.0099, 0.0022),
    ("E3  SE only",         0.0060, 0.0023),
    ("E4  Hier. only",      0.0099, 0.0018),
    ("E1  HCD-Swin",        0.0088, 0.0021),
    ("E2  HCD-ResNet50",    0.0278, 0.0047),
    ("E5  SwinUNETR",       0.0031, 0.0003),
]

COLORS_ET = "#e05c5c"
COLORS_TC = "#5c9de0"
HIGHLIGHT = "#c0392b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/hvr_bar.pdf")
    args = parser.parse_args()

    labels = [d[0] for d in HVR_DATA]
    hvr_et = [d[1] for d in HVR_DATA]
    hvr_tc = [d[2] for d in HVR_DATA]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_et = ax.bar(x - width / 2, hvr_et, width, label=r"HVR$_\mathrm{ET}$",
                     color=COLORS_ET, edgecolor="white", linewidth=0.5)
    bars_tc = ax.bar(x + width / 2, hvr_tc, width, label=r"HVR$_\mathrm{TC}$",
                     color=COLORS_TC, edgecolor="white", linewidth=0.5)

    # Annotate values on top of each bar
    for bar in list(bars_et) + list(bars_tc):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=6.5)

    # Shade E2 (ResNet50) in a different background to call it out
    e2_idx = 4
    ax.axvspan(e2_idx - 0.5, e2_idx + 0.5, color="#f9e4e4", zorder=0, label="ResNet50 (E2)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Hierarchy Violation Rate (↓)", fontsize=9)
    ax.set_ylim(0, max(hvr_et) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
