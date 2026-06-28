"""Compute per-region DSC and HD95 from saved checkpoints.

Runs sliding-window inference on the 251 validation cases and reports
WT / TC / ET Dice and HD95 ready to paste into the paper table.

Usage (on GPU server):
    BRATS_ROOT=/path/to/BraTS2021 python scripts/evaluate.py \
        --configs configs/exp0_swin_flat.yaml configs/exp1_swin_hcd.yaml ... \
        --output_dir outputs --gpu 0 2>&1 | tee logs/eval_results.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
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


def load_config(path: str) -> dict:
    p = Path(path)
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if cfg.get("inherit"):
        parent = (p.parent / cfg["inherit"]).resolve()
        with parent.open() as pf:
            base = yaml.safe_load(pf)
        cfg = deep_update(base, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg


def find_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("best_dice_*.pt"))
    if candidates:
        return candidates[-1]
    latest = output_dir / "latest.pt"
    return latest if latest.exists() else None


def evaluate(cfg: dict, ckpt: Path, device: torch.device) -> dict:
    model = build_model(cfg).to(device)
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    root = os.environ.get("BRATS_ROOT") or cfg["data"].get("root")
    samples = load_manifest(cfg["data"]["val_manifest"], root=root)
    ds = BraTSDataset(samples, transform=build_val_transforms(cfg))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    patch_size = tuple(cfg["data"]["patch_size"])
    sw_batch   = cfg["data"].get("sw_batch_size", 2)
    amp        = cfg["training"].get("amp", True)

    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95,
                                          reduction="none", get_not_nans=False)

    def _predict(x: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", enabled=amp):
            return model(x)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = sliding_window_inference(
                images, patch_size, sw_batch, _predict,
                overlap=0.5, mode="gaussian", progress=False,
            )
            preds = (torch.sigmoid(logits) > 0.5).long()
            labels_b = labels.long()

            # MONAI metrics expect (B, C, ...) one-hot — already in that format
            dice_metric(preds, labels_b)
            hd95_metric(preds, labels_b)

            del logits, images, labels, preds, labels_b

    # shape: (N_cases, 3) — channels: ET, TC, WT  (model output order)
    dice_vals = dice_metric.aggregate().cpu().numpy()   # (N, 3)
    hd95_vals = hd95_metric.aggregate().cpu().numpy()   # (N, 3)

    # model channel order: 0=ET, 1=TC, 2=WT
    dice_et, dice_tc, dice_wt = dice_vals[:, 0].mean(), dice_vals[:, 1].mean(), dice_vals[:, 2].mean()
    hd95_et, hd95_tc, hd95_wt = hd95_vals[:, 0].mean(), hd95_vals[:, 1].mean(), hd95_vals[:, 2].mean()

    return {
        "dsc_wt": float(dice_wt), "dsc_tc": float(dice_tc), "dsc_et": float(dice_et),
        "dsc_mean": float(np.mean([dice_wt, dice_tc, dice_et])),
        "hd95_wt": float(hd95_wt), "hd95_tc": float(hd95_tc), "hd95_et": float(hd95_et),
        "n": dice_vals.shape[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = Path(args.output_dir)

    results = []
    for cfg_path in args.configs:
        cfg  = load_config(cfg_path)
        name = cfg["experiment_name"]
        ckpt = find_checkpoint(root / name)
        if ckpt is None:
            print(f"[SKIP] {name}: no checkpoint")
            continue
        print(f"[{name}] {ckpt.name} ...")
        try:
            m = evaluate(cfg, ckpt, device)
            m["exp"] = name
            results.append(m)
            print(f"  DSC  WT={m['dsc_wt']:.4f}  TC={m['dsc_tc']:.4f}  ET={m['dsc_et']:.4f}  mean={m['dsc_mean']:.4f}")
            print(f"  HD95 WT={m['hd95_wt']:.2f}  TC={m['hd95_tc']:.2f}  ET={m['hd95_et']:.2f}  (n={m['n']})")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not results:
        return

    print("\n## Results (paste into paper)\n")
    hdr = f"{'Experiment':<30} {'DSC_WT':>7} {'DSC_TC':>7} {'DSC_ET':>7} {'DSC_m':>7} {'HD_WT':>7} {'HD_TC':>7} {'HD_ET':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['exp']:<30} {r['dsc_wt']:>7.4f} {r['dsc_tc']:>7.4f} {r['dsc_et']:>7.4f} "
              f"{r['dsc_mean']:>7.4f} {r['hd95_wt']:>7.2f} {r['hd95_tc']:>7.2f} {r['hd95_et']:>7.2f}")


if __name__ == "__main__":
    main()
