"""Generate qualitative segmentation panels for the paper.

For each selected validation case, saves a PNG with 3 rows (axial/coronal/sagittal)
and 5 columns: T1ce | FLAIR | Ground Truth | HCD-Swin | HCD-ResNet50.

Usage:
    python scripts/visualize_predictions.py \
        --checkpoint_swin  outputs/exp1_swin_hcd/best_dice_*.pt \
        --checkpoint_res50 outputs/exp2_resnet50_hcd/best_dice_*.pt \
        --val_manifest     manifests/val.json \
        --out_dir          paper/figures/qual \
        --n_cases          4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from pathlib import Path

# Allow running from the repo root without installing as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from monai.transforms import Compose
from data.dataset import _load_image_from_sample, _load_array, _to_multilabel_brats
from data.transforms import NonZeroPercentileClipd, MinMaxNormalizeNonZerod, CropToBrainBBoxd
from models.build_model import build_model

COLORS = {
    "ET": np.array([1.0, 0.2, 0.2, 0.6]),   # red
    "TC": np.array([1.0, 0.6, 0.0, 0.4]),   # orange
    "WT": np.array([0.2, 0.8, 0.2, 0.3]),   # green
}
CLASS_NAMES = ["ET", "TC", "WT"]


def overlay_masks(base: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Composite segmentation masks over a grayscale base slice."""
    h, w = base.shape
    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
    rgb = np.stack([base_norm] * 3, axis=-1)
    alpha_acc = np.zeros((h, w), dtype=np.float32)

    for name, mask in masks.items():
        color = COLORS[name]
        fg = mask.astype(bool)
        a = color[3] * (1 - alpha_acc)
        for c in range(3):
            rgb[:, :, c] = np.where(fg, rgb[:, :, c] * (1 - a) + color[c] * a, rgb[:, :, c])
        alpha_acc = np.where(fg, np.minimum(alpha_acc + color[3], 1.0), alpha_acc)

    return np.clip(rgb, 0, 1)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    patch_size: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Return (3, H, W, D) binary prediction."""
    x = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
            progress=False,
        )
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    return (probs > 0.5).astype(np.uint8)


def best_slice(volume: np.ndarray, axis: int) -> int:
    """Find the slice along *axis* with the most foreground."""
    sums = np.any(volume > 0, axis=tuple(i for i in range(volume.ndim) if i != axis))
    idxs = np.where(sums)[0]
    return int(idxs[len(idxs) // 2]) if len(idxs) else volume.shape[axis] // 2


def get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    return np.take(arr, idx, axis=axis)


def save_case_figure(
    case_id: str,
    image: np.ndarray,          # (4, H, W, D)
    gt: np.ndarray,             # (3, H, W, D)
    pred_swin: np.ndarray,      # (3, H, W, D)
    pred_res50: np.ndarray,     # (3, H, W, D)
    out_path: Path,
) -> None:
    gt_any = gt.any(axis=0)
    slice_ax = best_slice(gt_any, axis=2)
    slice_cor = best_slice(gt_any, axis=1)
    slice_sag = best_slice(gt_any, axis=0)

    views = [
        ("Axial",    image[:, :, :, slice_ax],    gt[:, :, :, slice_ax],    pred_swin[:, :, :, slice_ax],    pred_res50[:, :, :, slice_ax]),
        ("Coronal",  image[:, :, slice_cor, :],   gt[:, :, slice_cor, :],   pred_swin[:, :, slice_cor, :],   pred_res50[:, :, slice_cor, :]),
        ("Sagittal", image[:, slice_sag, :, :],   gt[:, slice_sag, :, :],   pred_swin[:, slice_sag, :, :],   pred_res50[:, slice_sag, :, :]),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.suptitle(f"Case: {case_id}", fontsize=11)

    col_titles = ["T1ce", "FLAIR", "Ground Truth", "HCD-Swin (E1)", "HCD-ResNet50 (E2)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    for row, (view_name, img_sl, gt_sl, sw_sl, r50_sl) in enumerate(views):
        t1ce = img_sl[1]   # channel 1 = T1ce
        flair = img_sl[0]  # channel 0 = FLAIR

        def mask_dict(pred: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "WT": pred[2].astype(bool),
                "TC": pred[1].astype(bool),
                "ET": pred[0].astype(bool),
            }

        panels = [
            t1ce,
            flair,
            overlay_masks(t1ce, mask_dict(gt_sl)),
            overlay_masks(t1ce, mask_dict(sw_sl)),
            overlay_masks(t1ce, mask_dict(r50_sl)),
        ]

        axes[row, 0].set_ylabel(view_name, fontsize=9)
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if panel.ndim == 2:
                ax.imshow(np.rot90(panel), cmap="gray", interpolation="nearest")
            else:
                ax.imshow(np.rot90(panel, axes=(0, 1)), interpolation="nearest")
            ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=COLORS["WT"][:3], alpha=0.8, label="WT"),
        mpatches.Patch(color=COLORS["TC"][:3], alpha=0.8, label="TC"),
        mpatches.Patch(color=COLORS["ET"][:3], alpha=0.8, label="ET"),
    ]
    fig.legend(handles=legend_patches, loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_swin",  required=True)
    parser.add_argument("--checkpoint_res50", required=True)
    parser.add_argument("--val_manifest", default="manifests/val.json")
    parser.add_argument("--out_dir", default="paper/figures/qual")
    parser.add_argument("--n_cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    model_swin = load_model(args.checkpoint_swin, device)
    model_res50 = load_model(args.checkpoint_res50, device)

    with open(args.val_manifest) as f:
        val_samples = json.load(f)

    rng = random.Random(args.seed)
    selected = rng.sample(val_samples, min(args.n_cases, len(val_samples)))

    patch_size = tuple(args.patch_size)

    for sample in selected:
        label_path = Path(sample["label"])
        case_id = label_path.name.replace("_seg.nii.gz", "")
        print(f"Processing {case_id}...")

        preprocess = Compose([
            NonZeroPercentileClipd(keys=["image"], lower=1.0, upper=99.0),
            MinMaxNormalizeNonZerod(keys=["image"]),
            CropToBrainBBoxd(keys=["image", "label"], source_key="image"),
        ])

        image_np = _load_image_from_sample(sample)
        raw_label = _load_array(sample["label"])
        label_np = _to_multilabel_brats(raw_label)

        item = preprocess({"image": image_np, "label": label_np})
        image_pp = item["image"]
        label_pp = item["label"]

        image_t = torch.from_numpy(np.ascontiguousarray(image_pp, dtype=np.float32))

        pred_swin = predict_volume(model_swin, image_t, device, patch_size)
        pred_res50 = predict_volume(model_res50, image_t, device, patch_size)

        out_path = out_dir / f"{case_id}.png"
        save_case_figure(
            case_id=case_id,
            image=image_pp,
            gt=label_pp,
            pred_swin=pred_swin,
            pred_res50=pred_res50,
            out_path=out_path,
        )

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
