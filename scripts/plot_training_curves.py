"""Pull training curves from W&B and save as a publication-quality PDF.

Usage:
    WANDB_API_KEY=<key> python scripts/plot_training_curves.py \
        --out paper/figures/training_curves.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ENTITY  = "kc_brats"
PROJECT = "brats-segmentation"
METRIC  = "val_dice"

COLUMN_WIDTH = 3.5  # IEEE single column

RUN_STYLE: dict[str, tuple[str, str, str, float]] = {
    "exp0_swin_flat":       ("E0  Flat-Swin",    "#888888", "-",  1.2),
    "exp1_swin_hcd":        ("E1  HCD-Swin",     "#2ECC71", "-",  1.8),
    "exp2_resnet50_hcd":    ("E2  HCD-ResNet50", "#E74C3C", "--", 1.2),
    "exp3_swin_hcd_nohier": ("E3  SE only",      "#E67E22", "-.", 1.2),
    "exp4_swin_hcd_nose":   ("E4  Hier. only",   "#9B59B6", ":",  1.2),
    "exp5_swinunetr":       ("E5  SwinUNETR",    "#2980B9", "-",  1.8),
}


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
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    }


def smooth(vals: list[float], w: int = 5) -> list[float]:
    if len(vals) < w:
        return vals
    kernel = np.ones(w) / w
    padded = np.pad(vals, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def fetch_runs() -> dict[str, tuple[list[int], list[float]]]:
    import wandb
    api  = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    data: dict[str, tuple[list[int], list[float]]] = {}

    for run in runs:
        name = run.name or run.id
        matched = next((p for p in RUN_STYLE if name.startswith(p)), None)
        if matched is None:
            continue
        history = run.history(keys=["epoch", METRIC], pandas=False)
        epochs = [r.get("epoch", i) for i, r in enumerate(history) if METRIC in r]
        vals   = [r[METRIC] for r in history if METRIC in r]
        if epochs and vals:
            data[matched] = (epochs, vals)
            print(f"  Fetched {matched}: {len(vals)} points")

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/training_curves.pdf")
    args = parser.parse_args()

    print("Fetching from W&B...")
    data = fetch_runs()
    if not data:
        print("No data found. Check WANDB_API_KEY and entity/project.")
        return

    with plt.rc_context(pub_rc()):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 2, 2.6))

        for prefix in RUN_STYLE:
            if prefix not in data:
                continue
            label, color, ls, lw = RUN_STYLE[prefix]
            epochs, vals = data[prefix]
            smoothed = smooth(vals, w=7)
            ax.plot(epochs, smoothed, label=label, color=color,
                    linestyle=ls, linewidth=lw, alpha=0.92)
            # faint raw trace behind smooth
            ax.plot(epochs, vals, color=color, linewidth=0.3, alpha=0.25)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Dice (mean, $\\uparrow$)")
        ax.set_ylim(0.50, 0.95)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.legend(loc="lower right", framealpha=0.9,
                  edgecolor="#cccccc", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle=":", linewidth=0.5, color="#bbbbbb")

        fig.tight_layout(pad=0.5)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=300, bbox_inches="tight", format="pdf")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
