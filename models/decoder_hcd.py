from __future__ import annotations

"""Hierarchical Consistency Decoder (HCD) for BraTS segmentation."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """Squeeze-Excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * w


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
    )


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
    )


class HierarchicalConsistencyDecoder(nn.Module):
    """Hierarchical Consistency Decoder (HCD).

    Architecture:
      1. FPN fuses 4 encoder scales into a single stride-2 feature map p0.
      2. SE recalibrates p0 channel-wise (optional, ablation: use_se=False).
      3. Three prediction heads enforce ET ⊂ TC ⊂ WT by sequential soft
         conditioning — each head receives the previous head's sigmoid output
         as an extra input channel (optional, ablation: use_hierarchy=False).

    Output: (B, 3, H', W', D') logits ordered [ET, TC, WT].
    """

    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int = 3,
        hidden: int = 128,
        use_hierarchy: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 3, "HCD requires exactly 3 classes: ET, TC, WT"
        c0, c1, c2, c3 = encoder_channels
        self.use_hierarchy = use_hierarchy
        self.use_se = use_se

        # FPN lateral projections (all scales → hidden channels)
        self.lat0 = _proj(c0, hidden)
        self.lat1 = _proj(c1, hidden)
        self.lat2 = _proj(c2, hidden)
        self.lat3 = _proj(c3, hidden)

        # FPN top-down refinement
        self.fpn2 = _conv_block(hidden, hidden)
        self.fpn1 = _conv_block(hidden, hidden)
        self.fpn0 = _conv_block(hidden, hidden)

        # SE recalibration on the fused FPN features
        if use_se:
            self.se = SEBlock3D(hidden)

        # WT head — no conditioning, predicts the largest region first
        self.wt_conv = _conv_block(hidden, hidden)
        self.wt_out = nn.Conv3d(hidden, 1, kernel_size=1)

        # TC head — conditioned on WT soft mask if use_hierarchy
        tc_in = hidden + 1 if use_hierarchy else hidden
        self.tc_conv = _conv_block(tc_in, hidden)
        self.tc_out = nn.Conv3d(hidden, 1, kernel_size=1)

        # ET head — conditioned on TC soft mask if use_hierarchy
        et_in = hidden + 1 if use_hierarchy else hidden
        self.et_conv = _conv_block(et_in, hidden)
        self.et_out = nn.Conv3d(hidden, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3 = features  # strides: 2, 4, 8, 16

        # FPN top-down pathway
        p3 = self.lat3(f3)
        p2 = self.fpn2(self.lat2(f2) + F.interpolate(p3, size=f2.shape[2:], mode="trilinear", align_corners=False))
        p1 = self.fpn1(self.lat1(f1) + F.interpolate(p2, size=f1.shape[2:], mode="trilinear", align_corners=False))
        p0 = self.fpn0(self.lat0(f0) + F.interpolate(p1, size=f0.shape[2:], mode="trilinear", align_corners=False))

        # Optional SE recalibration
        feat = self.se(p0) if self.use_se else p0

        # WT prediction
        wt_logit = self.wt_out(self.wt_conv(feat))

        if self.use_hierarchy:
            wt_mask = torch.sigmoid(wt_logit)
            tc_logit = self.tc_out(self.tc_conv(torch.cat([feat, wt_mask], dim=1)))
            tc_mask = torch.sigmoid(tc_logit)
            et_logit = self.et_out(self.et_conv(torch.cat([feat, tc_mask], dim=1)))
        else:
            tc_logit = self.tc_out(self.tc_conv(feat))
            et_logit = self.et_out(self.et_conv(feat))

        # Channel order [ET, TC, WT] matches BraTS label convention
        return torch.cat([et_logit, tc_logit, wt_logit], dim=1)
