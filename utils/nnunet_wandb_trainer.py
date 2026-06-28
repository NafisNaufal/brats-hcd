from __future__ import annotations

import os

import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandB(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        wandb.init(
            project="brats-segmentation",
            entity="kc_brats",
            name="nnunet_baseline",
            config={
                "configuration": configuration,
                "fold": fold,
                "plans": plans,
            },
            mode=os.environ.get("WANDB_MODE", "online"),
            resume="allow",
        )

    def on_epoch_end(self):
        super().on_epoch_end()
        log = self.logger.my_fantastic_logging
        payload = {
            "epoch": self.current_epoch,
            "train_loss": log["train_losses"][-1],
            "val_loss": log["val_losses"][-1],
            "ema_fg_dice": log["ema_fg_dice"][-1],
            "lr": log["lrs"][-1],
        }
        if "mean_fg_dice" in log and log["mean_fg_dice"]:
            payload["mean_fg_dice"] = log["mean_fg_dice"][-1]
        wandb.log(payload)

    def on_train_end(self):
        super().on_train_end()
        wandb.finish()
