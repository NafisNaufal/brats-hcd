from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_hcd import HierarchicalConsistencyDecoder
from .encoder import ResNet50Encoder3D, SwinEncoder3D


class SwinUNETRWrapper(nn.Module):
    """Thin wrapper around MONAI SwinUNETR exposing the same interface as SegmentationModel."""

    def __init__(self, in_channels: int, out_channels: int, feature_size: int, use_checkpoint: bool) -> None:
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def encoder_parameters(self):
        return self.model.swinViT.parameters()

    def non_encoder_parameters(self):
        enc_ids = {id(p) for p in self.model.swinViT.parameters()}
        for p in self.model.parameters():
            if id(p) not in enc_ids:
                yield p


class SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        hidden: int = 128,
        use_hierarchy: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = HierarchicalConsistencyDecoder(
            encoder_channels=encoder.out_channels,
            num_classes=num_classes,
            hidden=hidden,
            use_hierarchy=use_hierarchy,
            use_se=use_se,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        feats = self.encoder(x)
        out = self.decoder(feats)
        return F.interpolate(out, size=input_size, mode="trilinear", align_corners=False)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def non_encoder_parameters(self):
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        for p in self.parameters():
            if id(p) not in encoder_ids:
                yield p


def build_model(cfg: dict[str, Any]) -> SegmentationModel:
    model_cfg = cfg["model"]
    encoder_type: str = model_cfg["encoder_type"]

    if encoder_type == "swinunetr":
        return SwinUNETRWrapper(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["num_classes"],
            feature_size=model_cfg.get("feature_size", 48),
            use_checkpoint=model_cfg.get("use_checkpoint", True),
        )

    if encoder_type == "swin":
        encoder = SwinEncoder3D(
            in_channels=model_cfg["in_channels"],
            feature_size=model_cfg.get("feature_size", 48),
            pretrained=model_cfg.get("pretrained", True),
            use_checkpoint=model_cfg.get("use_checkpoint", True),
        )
    elif encoder_type == "resnet50":
        encoder = ResNet50Encoder3D(
            in_channels=model_cfg["in_channels"],
            pretrained=model_cfg.get("pretrained", True),
            use_checkpoint=model_cfg.get("use_checkpoint", True),
        )
    else:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. Choose 'swin', 'resnet50', or 'swinunetr'."
        )

    return SegmentationModel(
        encoder=encoder,
        num_classes=model_cfg["num_classes"],
        hidden=model_cfg.get("hcd_hidden", 128),
        use_hierarchy=model_cfg.get("use_hierarchy", True),
        use_se=model_cfg.get("use_se", True),
    )
