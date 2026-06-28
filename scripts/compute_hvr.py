"""Compute Hierarchy Violation Rate (HVR) from saved checkpoints.

HVR measures how often the trained model violates the anatomical constraint
ET ⊂ TC ⊂ WT in its binary predictions (threshold 0.5).

    HVR_ET = mean over cases of |ET_pred ∩ ¬TC_pred| / max(1, |ET_pred|)
    HVR_TC = mean over cases of |TC_pred ∩ ¬WT_pred| / max(1, |TC_pred|)

Lower is better. A perfect HCD model would score 0.000 on both.

Usage:
    python scripts/compute_hvr.py \\
        --configs configs/exp0_swin_flat.yaml configs/exp1_swin_hcd.yaml \\
        --output_dir outputs

Output: prints a markdown table ready to paste into the paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BraTSDataset, load_manifest
from data.transforms import build_val_transforms
from models.build_model import build_model


def deep_update(base: dict, update: dict) -> dict:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if cfg.get("inherit"):
        parent_path = (p.parent / cfg["inherit"]).resolve()
        with parent_path.open() as pf:
            parent = yaml.safe_load(pf)
        cfg = deep_update(parent, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg


def find_best_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("best_dice_*.pt"))
    if candidates:
        return candidates[-1]
    latest = output_dir / "latest.pt"
    return latest if latest.exists() else None


def compute_hvr_for_experiment(
    cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, float]:
    model = build_model(cfg).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    import os
    root = os.environ.get("BRATS_ROOT") or cfg["data"].get("root")
    val_samples = load_manifest(cfg["data"]["val_manifest"], root=root)
    val_ds = BraTSDataset(val_samples, transform=build_val_transforms(cfg))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    patch_size = tuple(cfg["data"]["patch_size"])
    sw_batch_size = cfg["data"].get("sw_batch_size", 2)
    overlap = cfg["data"].get("sw_overlap", 0.5)
    amp_enabled = cfg["training"].get("amp", True)

    hvr_et_list: list[float] = []
    hvr_tc_list: list[float] = []

    def _predict(x: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", enabled=amp_enabled):
            return model(x)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=_predict,
                overlap=overlap,
                mode="gaussian",
                progress=False,
            )

            probs = torch.sigmoid(logits)
            # (1, 3, H, W, D) → bool masks per channel
            et = (probs[:, 0] > 0.5)   # ET
            tc = (probs[:, 1] > 0.5)   # TC
            wt = (probs[:, 2] > 0.5)   # WT

            et_count = et.sum().item()
            tc_count = tc.sum().item()

            # ET voxels outside TC
            et_outside_tc = (et & ~tc).sum().item()
            hvr_et = et_outside_tc / max(1, et_count)

            # TC voxels outside WT
            tc_outside_wt = (tc & ~wt).sum().item()
            hvr_tc = tc_outside_wt / max(1, tc_count)

            hvr_et_list.append(hvr_et)
            hvr_tc_list.append(hvr_tc)

            del logits, images, probs, et, tc, wt

    return {
        "hvr_et": float(np.mean(hvr_et_list)),
        "hvr_tc": float(np.mean(hvr_tc_list)),
        "hvr_mean": float(np.mean(hvr_et_list + hvr_tc_list)),
        "n_cases": len(hvr_et_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment config paths")
    parser.add_argument("--output_dir", default="outputs", help="Root dir containing experiment outputs")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir)

    results: list[dict] = []

    for config_path in args.configs:
        cfg = load_config(config_path)
        exp_name = cfg["experiment_name"]
        exp_out = output_root / exp_name
        ckpt = find_best_checkpoint(exp_out)

        if ckpt is None:
            print(f"[SKIP] {exp_name}: no checkpoint found in {exp_out}")
            continue

        print(f"[{exp_name}] Loading checkpoint: {ckpt.name} ...")
        try:
            metrics = compute_hvr_for_experiment(cfg, ckpt, device)
            metrics["exp"] = exp_name
            results.append(metrics)
            print(f"  HVR_ET={metrics['hvr_et']:.4f}  HVR_TC={metrics['hvr_tc']:.4f}  "
                  f"HVR_mean={metrics['hvr_mean']:.4f}  (n={metrics['n_cases']})")
        except Exception as e:
            print(f"[ERROR] {exp_name}: {e}")

    if not results:
        print("No results to display.")
        return

    # Markdown table
    print("\n## Hierarchy Violation Rate (HVR) — lower is better\n")
    print(f"{'Experiment':<30} {'HVR_ET':>8} {'HVR_TC':>8} {'HVR_mean':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['exp']:<30} {r['hvr_et']:>8.4f} {r['hvr_tc']:>8.4f} {r['hvr_mean']:>10.4f}")


if __name__ == "__main__":
    main()
