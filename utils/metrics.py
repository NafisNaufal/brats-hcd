from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from monai.metrics import compute_hausdorff_distance
except Exception:
    compute_hausdorff_distance = None

EPS = 1e-5


def dice_per_sample(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute per-sample, per-class Dice. Returns shape (B, C).

    Classes where the ground-truth is empty for a sample get Dice=1.0
    (standard BraTS convention: a class absent in GT and prediction is perfect).
    """
    preds = (probs > 0.5).float()
    # (B, C) sums
    intersection = (preds * targets).sum(dim=(2, 3, 4))
    pred_sum = preds.sum(dim=(2, 3, 4))
    gt_sum = targets.sum(dim=(2, 3, 4))
    cardinality = pred_sum + gt_sum

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    # Where GT is empty AND prediction is also empty → Dice = 1.0 (correct)
    # Where GT is empty but prediction is non-empty → cardinality > 0, gets penalised
    return dice  # (B, C)


def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    compute_hd95: bool = False,
) -> dict[str, Any]:
    """Compute per-sample Dice (macro-averaged) and optional HD95.

    Channel order: [ET=0, TC=1, WT=2].
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # (B, C) per-sample per-class Dice, then mean over samples → (C,)
        dice_pc = dice_per_sample(probs, targets, eps=EPS).mean(dim=0)

        metrics: dict[str, Any] = {
            "dice_et": float(dice_pc[0].item()),
            "dice_tc": float(dice_pc[1].item()),
            "dice_wt": float(dice_pc[2].item()),
            "dice_mean": float(dice_pc.mean().item()),
        }

        if compute_hd95 and compute_hausdorff_distance is not None:
            hd95_vals: list[float] = []
            for c in range(preds.shape[1]):
                p_c = preds[:, c : c + 1]
                t_c = targets[:, c : c + 1]
                pred_empty = p_c.sum(dim=(2, 3, 4)) == 0
                gt_empty = t_c.sum(dim=(2, 3, 4)) == 0
                if (pred_empty | gt_empty).all():
                    continue
                valid = ~(pred_empty | gt_empty)
                valid_idx = valid.view(-1)   # always (B,) — squeeze() breaks at B=1
                hd = compute_hausdorff_distance(
                    y_pred=p_c[valid_idx],
                    y=t_c[valid_idx],
                    include_background=True,
                    percentile=95.0,
                )
                hd_np = hd.detach().cpu().numpy().astype(np.float32)
                hd95_vals.append(float(np.nanmean(hd_np)))
            metrics["hd95"] = float(np.mean(hd95_vals)) if hd95_vals else float("nan")
        else:
            metrics["hd95"] = float("nan")

    return metrics
