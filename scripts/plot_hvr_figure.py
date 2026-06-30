"""Generate HVR bar chart for the paper — publication quality.

Usage:
    python scripts/plot_hvr_figure.py --out paper/figures/hvr_bar.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# IEEE single-column width ≈ 3.5 in; double-column ≈ 7.16 in
COLUMN_WIDTH = 3.5  # inches

HVR_DATA = [
    ("E0\nFlat-Swin",    0.0099, 0.0022),
    ("E3\nSE only",      0.0060, 0.0023),
    ("E4\nHier. only",   0.0099, 0.0018),
    ("E1\nHCD-Swin",     0.0088, 0.0021),
    ("E2\nResNet50",     0.0278, 0.0047),
    ("E5\nSwinUNETR",    0.0031, 0.0003),
]

COLOR_ET = "#C0392B"
COLOR_TC = "#2980B9"


def pub_rc() -> dict:
    return {
        "font.family":       "serif",
        "font.size":         8,
        "axes.titlesize":    8,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.2,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/hvr_bar.pdf")
    args = parser.parse_args()

    labels  = [d[0] for d in HVR_DATA]
    hvr_et  = [d[1] for d in HVR_DATA]
    hvr_tc  = [d[2] for d in HVR_DATA]
    x       = np.arange(len(labels))
    width   = 0.32

    with plt.rc_context(pub_rc()):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 2, 2.4))

        bars_et = ax.bar(x - width / 2, hvr_et, width,
                         label=r"HVR$_\mathrm{ET}$",
                         color=COLOR_ET, edgecolor="white", linewidth=0.4)
        bars_tc = ax.bar(x + width / 2, hvr_tc, width,
                         label=r"HVR$_\mathrm{TC}$",
                         color=COLOR_TC, edgecolor="white", linewidth=0.4)

        for bar in list(bars_et) + list(bars_tc):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.00025,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=5.5,
                    color="#333333")

        # Subtle highlight for E2 (ResNet50)
        ax.axvspan(3.5, 4.5, color="#fdf0f0", zorder=0, linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="center", linespacing=1.3)
        ax.set_ylabel("Hierarchy Violation Rate ($\\downarrow$)")
        ax.set_ylim(0, max(hvr_et) * 1.28)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#bbbbbb")

        fig.tight_layout(pad=0.5)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
