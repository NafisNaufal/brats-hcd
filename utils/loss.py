from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4)

        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        mask = cardinality > 0
        dice_per_class = dice_per_class[mask]
        if dice_per_class.numel() == 0:
            return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
        return 1.0 - dice_per_class.mean()


class HierarchicalConsistencyLoss(nn.Module):
    """Penalizes predictions that violate ET ⊂ TC ⊂ WT.

    Logit channel order: [ET=0, TC=1, WT=2].
    Loss = mean(relu(p_ET - p_TC)) + mean(relu(p_TC - p_WT))
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        et, tc, wt = p[:, 0], p[:, 1], p[:, 2]
        return F.relu(et - tc).mean() + F.relu(tc - wt).mean()


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.consistency = HierarchicalConsistencyLoss()

    def _single(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)

    def forward(
        self,
        logits: torch.Tensor | Sequence[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            main_logits = logits[-1]
            base_loss = torch.stack([self._single(pred, targets) for pred in logits]).mean()
        else:
            main_logits = logits
            base_loss = self._single(logits, targets)

        if self.consistency_weight > 0.0:
            base_loss = base_loss + self.consistency_weight * self.consistency(main_logits)

        return base_loss
