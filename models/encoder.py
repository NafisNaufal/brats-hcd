from __future__ import annotations

"""3D encoders: SwinEncoder3D and ResNet50Encoder3D (Med3D pretrained)."""

import os
import urllib.request
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
# Swin Transformer encoder (MONAI SSL pretrained)
# ---------------------------------------------------------------------------

class SwinEncoder3D(nn.Module):
    """3D Swin Transformer encoder via MONAI's SwinTransformer.

    Output: 4 feature maps at strides [2, 4, 8, 16]
    Channels: [F, 2F, 4F, 8F] where F = feature_size (default 48)
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_size: int = 48,
        pretrained: bool = True,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer

        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )
        self.out_channels: List[int] = [
            feature_size,
            feature_size * 2,
            feature_size * 4,
            feature_size * 8,
        ]
        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self) -> None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "brats_weights")
        os.makedirs(cache_dir, exist_ok=True)
        weight_path = os.path.join(cache_dir, "model_swinvit.pt")

        if not os.path.isfile(weight_path):
            url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
            print("[SwinEncoder3D] Downloading pretrained weights...")
            urllib.request.urlretrieve(url, weight_path)

        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        raw = checkpoint.get("state_dict", checkpoint)
        cleaned = {k.split("swinViT.")[-1]: v for k, v in raw.items() if "swinViT." in k}
        missing, _ = self.swin.load_state_dict(cleaned, strict=False)
        print(f"[SwinEncoder3D] Pretrained loaded. Missing: {len(missing)} "
              f"(patch_embed expected for in_chans!=1)")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.swin(x)
        return [hidden[0], hidden[1], hidden[2], hidden[3]]


# ---------------------------------------------------------------------------
# ResNet50 3D encoder (Med3D pretrained)
# ---------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = _gn(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = _gn(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = _gn(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                _gn(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.act(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.act(out + residual)


def _make_layer(inplanes: int, planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
    blocks: list[nn.Module] = [Bottleneck3D(inplanes, planes, stride=stride)]
    inplanes = planes * Bottleneck3D.expansion
    for _ in range(1, num_blocks):
        blocks.append(Bottleneck3D(inplanes, planes))
    return nn.Sequential(*blocks)


class ResNet50Encoder3D(nn.Module):
    """3D ResNet50 encoder with Med3D-style architecture (no maxpool after stem).

    Pretrained weights from Med3D (https://github.com/Tencent/MedicalNet),
    trained on 8 medical segmentation datasets. Run scripts/download_weights.py
    to fetch them, or place resnet_50_med3d.pth in ~/.cache/brats_weights/.

    Output: 4 feature maps at strides [2, 4, 8, 16]
    Channels: [256, 512, 1024, 2048]
    """

    def __init__(
        self,
        in_channels: int = 4,
        pretrained: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = _gn(64)
        self.act = nn.ReLU(inplace=True)

        self.layer1 = _make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = _make_layer(256, 128, num_blocks=4, stride=2)
        self.layer3 = _make_layer(512, 256, num_blocks=6, stride=2)
        self.layer4 = _make_layer(1024, 512, num_blocks=3, stride=2)

        self.out_channels: List[int] = [256, 512, 1024, 2048]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if pretrained:
            self._load_pretrained(in_channels)

    def _load_pretrained(self, in_channels: int) -> None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "brats_weights")
        weight_path = os.path.join(cache_dir, "resnet_50_med3d.pth")

        if not os.path.isfile(weight_path):
            try:
                print("[ResNet50Encoder3D] Downloading Med3D pretrained weights...")
                urllib.request.urlretrieve(
                    "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50.pth",
                    weight_path,
                )
            except Exception as e:
                print(f"[ResNet50Encoder3D] Auto-download failed: {e}")
                print(f"  Place resnet_50.pth at: {weight_path}")
                print("[ResNet50Encoder3D] Continuing with random initialization.")
                return

        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        raw = checkpoint.get("state_dict", checkpoint)

        cleaned: dict = {}
        for k, v in raw.items():
            cleaned[k.replace("module.", "")] = v

        # Adapt first conv weights: pretrained (64,1,7,7,7) → ours (64,C,7,7,7)
        if "conv1.weight" in cleaned:
            w = cleaned["conv1.weight"]
            if w.shape[1] != in_channels:
                cleaned["conv1.weight"] = w.repeat(1, in_channels, 1, 1, 1) / in_channels

        missing, _ = self.load_state_dict(cleaned, strict=False)
        conv_missing = [k for k in missing if "conv" in k]
        print(f"[ResNet50Encoder3D] Med3D pretrained loaded OK. "
              f"Skipped norm keys (GN vs BN mismatch, expected). Conv keys missing: {len(conv_missing)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.act(self.gn1(self.conv1(x)))
        if self.use_checkpoint:
            f0 = grad_checkpoint(self.layer1, x, use_reentrant=False)
            f1 = grad_checkpoint(self.layer2, f0, use_reentrant=False)
            f2 = grad_checkpoint(self.layer3, f1, use_reentrant=False)
            f3 = grad_checkpoint(self.layer4, f2, use_reentrant=False)
        else:
            f0 = self.layer1(x)
            f1 = self.layer2(f0)
            f2 = self.layer3(f1)
            f3 = self.layer4(f2)
        return [f0, f1, f2, f3]
