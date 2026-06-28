"""Generate hierarchy-violation figure for the paper.

Finds the validation case where E0 (flat) has the highest ET-outside-TC
violation, then renders a 3-panel figure:
  [T1ce | E0 prediction (violation highlighted) | E1 prediction (clean)]

The "violation" voxels (ET outside TC) are shown in bright magenta.

Usage (on GPU server):
    BRATS_ROOT=/path/to/BraTS2021 python scripts/plot_hvr_case.py \
        --ckpt_e0 outputs/exp0_swin_flat/best_dice_*.pt \
        --ckpt_e1 outputs/exp1_swin_hcd/best_dice_*.pt \
        --out paper/figures/hvr_case.pdf \
        --gpu 0
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.inferers import sliding_window_inference
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BraTSDataset, load_manifest
from data.transforms import build_val_transforms
from models.build_model import build_model


ALPHA = 0.55
COLORS = {
    "WT": np.array([0.2, 0.8, 0.2]),
    "TC": np.array([1.0, 0.6, 0.0]),
    "ET": np.array([1.0, 0.2, 0.2]),
    "VIOLATION": np.array([1.0, 0.0, 1.0]),  # magenta
}


def load_ckpt(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def predict(model, image: torch.Tensor, device, cfg) -> np.ndarray:
    patch_size = tuple(cfg["data"]["patch_size"])
    sw_batch = cfg["data"].get("sw_batch_size", 1)
    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast("cuda", enabled=cfg["training"].get("amp", True)):
            logits = sliding_window_inference(
                x, patch_size, sw_batch, model,
                overlap=0.5, mode="gaussian", progress=False,
            )
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    return (probs > 0.5).astype(bool)  # (3, H, W, D)  ET/TC/WT


def hvr_et(pred: np.ndarray) -> float:
    et, tc = pred[0], pred[1]
    return (et & ~tc).sum() / max(1, et.sum())


def overlay(base: np.ndarray, et, tc, wt, violations=None) -> np.ndarray:
    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
    rgb = np.stack([base_norm] * 3, axis=-1)

    def blend(mask, color):
        for c in range(3):
            rgb[:, :, c] = np.where(mask, rgb[:, :, c] * (1 - ALPHA) + color[c] * ALPHA, rgb[:, :, c])

    blend(wt, COLORS["WT"])
    blend(tc, COLORS["TC"])
    blend(et, COLORS["ET"])
    if violations is not None:
        blend(violations, COLORS["VIOLATION"])

    return np.clip(rgb, 0, 1)


def best_slice(mask3d: np.ndarray) -> int:
    sums = mask3d.sum(axis=(0, 1))
    return int(sums.argmax())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_e0", required=True)
    parser.add_argument("--ckpt_e1", required=True)
    parser.add_argument("--val_manifest", default="manifests/val.json")
    parser.add_argument("--out", default="paper/figures/hvr_case.pdf")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=30,
                        help="Scan this many val cases to find worst violation")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading E0...")
    model_e0, cfg_e0 = load_ckpt(args.ckpt_e0, device)
    print("Loading E1...")
    model_e1, cfg_e1 = load_ckpt(args.ckpt_e1, device)

    root = os.environ.get("BRATS_ROOT") or cfg_e0["data"].get("root")
    samples = load_manifest(args.val_manifest, root=root)[:args.max_cases]

    from torch.utils.data import DataLoader
    ds = BraTSDataset(samples, transform=build_val_transforms(cfg_e0))

    best_hvr = -1.0
    best_idx = 0

    print(f"Scanning {len(samples)} cases for worst E0 violation...")
    for i, batch in enumerate(DataLoader(ds, batch_size=1, num_workers=2)):
        img = batch["image"].to(device)
        with torch.no_grad():
            with autocast("cuda", enabled=cfg_e0["training"].get("amp", True)):
                logits = sliding_window_inference(
                    img, tuple(cfg_e0["data"]["patch_size"]),
                    cfg_e0["data"].get("sw_batch_size", 1),
                    model_e0, overlap=0.5, mode="gaussian", progress=False,
                )
        pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
        h = hvr_et(pred)
        print(f"  case {i:3d}: HVR_ET={h:.4f}")
        if h > best_hvr:
            best_hvr = h
            best_idx = i

    print(f"\nWorst case: index={best_idx}, HVR_ET={best_hvr:.4f}")

    # Re-run inference on best case with both models
    sample_batch = ds[best_idx]
    image_t = sample_batch["image"]

    pred_e0 = predict(model_e0, image_t, device, cfg_e0)
    pred_e1 = predict(model_e1, image_t, device, cfg_e1)

    img_np = image_t.numpy()  # (4, H, W, D)
    t1ce = img_np[1]           # T1ce is channel 1

    # violation mask for E0
    viol_e0 = pred_e0[0] & ~pred_e0[1]  # ET outside TC

    z = best_slice(pred_e0[0])  # axial slice with most ET

    sl_t1ce     = np.rot90(t1ce[:, :, z])
    sl_e0_et    = np.rot90(pred_e0[0, :, :, z])
    sl_e0_tc    = np.rot90(pred_e0[1, :, :, z])
    sl_e0_wt    = np.rot90(pred_e0[2, :, :, z])
    sl_e0_viol  = np.rot90(viol_e0[:, :, z])
    sl_e1_et    = np.rot90(pred_e1[0, :, :, z])
    sl_e1_tc    = np.rot90(pred_e1[1, :, :, z])
    sl_e1_wt    = np.rot90(pred_e1[2, :, :, z])

    e0_rgb = overlay(sl_t1ce, sl_e0_et, sl_e0_tc, sl_e0_wt, sl_e0_viol)
    e1_rgb = overlay(sl_t1ce, sl_e1_et, sl_e1_tc, sl_e1_wt)

    base_norm = (sl_t1ce - sl_t1ce.min()) / (sl_t1ce.max() - sl_t1ce.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.suptitle(
        f"Hierarchy violation example (axial slice {z})", fontsize=10
    )

    axes[0].imshow(base_norm, cmap="gray", interpolation="nearest")
    axes[0].set_title("T1ce input", fontsize=9)

    axes[1].imshow(e0_rgb, interpolation="nearest")
    axes[1].set_title("E0 Flat-Swin\n(magenta = ET outside TC)", fontsize=9)

    axes[2].imshow(e1_rgb, interpolation="nearest")
    axes[2].set_title("E1 HCD-Swin\n(no violations)", fontsize=9)

    for ax in axes:
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=COLORS["WT"],       label="WT"),
        mpatches.Patch(color=COLORS["TC"],       label="TC"),
        mpatches.Patch(color=COLORS["ET"],       label="ET"),
        mpatches.Patch(color=COLORS["VIOLATION"], label="ET outside TC"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
