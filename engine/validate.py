from __future__ import annotations

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.amp import autocast
from torch.utils.data import DataLoader

from utils.metrics import batch_metrics


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
    compute_hd95: bool = False,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
) -> dict[str, float]:
    model.eval()

    val_loss = 0.0
    dice_et: list[float] = []
    dice_tc: list[float] = []
    dice_wt: list[float] = []
    dice_mean: list[float] = []
    hd95_vals: list[float] = []

    def _predict(x: torch.Tensor) -> torch.Tensor:
        with autocast("cuda", enabled=amp_enabled):
            return model(x)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=_predict,
                overlap=overlap,
                mode="gaussian",
                progress=False,
            )

            with autocast("cuda", enabled=amp_enabled):
                loss = loss_fn(logits, labels)

            metrics = batch_metrics(logits=logits, targets=labels, compute_hd95=compute_hd95)
            loss_val = float(loss.detach().item())
            if not (loss_val != loss_val):  # skip NaN
                val_loss += loss_val
            dice_et.append(metrics["dice_et"])
            dice_tc.append(metrics["dice_tc"])
            dice_wt.append(metrics["dice_wt"])
            dice_mean.append(metrics["dice_mean"])
            if not np.isnan(metrics["hd95"]):
                hd95_vals.append(metrics["hd95"])

            # Free large GPU tensors immediately so they don't linger until the
            # next loop iteration (251 full-volume logit tensors add up fast).
            del logits, images, labels, loss

    return {
        "val_loss": val_loss / max(1, len(loader)),
        "val_dice": float(np.mean(dice_mean)) if dice_mean else 0.0,
        "val_dice_et": float(np.mean(dice_et)) if dice_et else 0.0,
        "val_dice_tc": float(np.mean(dice_tc)) if dice_tc else 0.0,
        "val_dice_wt": float(np.mean(dice_wt)) if dice_wt else 0.0,
        "val_hd95": float(np.mean(hd95_vals)) if hd95_vals else float("nan"),
    }
