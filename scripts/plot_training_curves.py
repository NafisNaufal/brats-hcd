"""Pull training curves from W&B and save as a PDF figure.

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
import numpy as np

ENTITY = "kc_brats"
PROJECT = "brats-segmentation"

# Map W&B run name prefix → display label and color
RUN_STYLE: dict[str, tuple[str, str, str]] = {
    "exp0_swin_flat":           ("E0  Flat-Swin",       "#888888", "-"),
    "exp1_swin_hcd":            ("E1  HCD-Swin",        "#2ca02c", "-"),
    "exp2_resnet50_hcd":        ("E2  HCD-ResNet50",    "#d62728", "--"),
    "exp3_swin_hcd_nohier":     ("E3  SE only",         "#ff7f0e", "-."),
    "exp4_swin_hcd_nose":       ("E4  Hier. only",      "#9467bd", ":"),
    "exp5_swinunetr":           ("E5  SwinUNETR",       "#1f77b4", "-"),
}

METRIC = "val_dice"


def fetch_runs() -> dict[str, tuple[list[int], list[float]]]:
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    data: dict[str, tuple[list[int], list[float]]] = {}
    for run in runs:
        name = run.name or run.id
        matched = None
        for prefix in RUN_STYLE:
            if name.startswith(prefix):
                matched = prefix
                break
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

    print("Fetching runs from W&B...")
    data = fetch_runs()

    if not data:
        print("No runs found. Check ENTITY/PROJECT and WANDB_API_KEY.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for prefix, (epochs, vals) in sorted(data.items()):
        label, color, ls = RUN_STYLE[prefix]
        ax.plot(epochs, vals, label=label, color=color, linestyle=ls,
                linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Dice (mean)", fontsize=10)
    ax.set_ylim(0.5, 0.95)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
