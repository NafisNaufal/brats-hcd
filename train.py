from __future__ import annotations

"""Training entrypoint for BraTS-style 3D segmentation experiments."""

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset, load_manifest, scan_brats_root
from data.transforms import build_train_transforms, build_val_transforms
from engine.train_loop import train_one_epoch
from engine.validate import validate
from models.build_model import build_model
from utils.loss import DiceBCELoss
from utils.scheduler import build_warmup_cosine_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BraTS 3D Segmentation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str) -> dict[str, Any]:
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inherit_from = cfg.get("inherit")
    if inherit_from:
        parent_path = (cfg_path.parent / inherit_from).resolve()
        with parent_path.open("r", encoding="utf-8") as pf:
            parent_cfg = yaml.safe_load(pf)
        cfg = deep_update(parent_cfg, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg


def load_samples(cfg: dict[str, Any], seed: int) -> tuple[list[dict], list[dict]]:
    """Load train/val sample lists from manifests or by scanning the dataset root."""
    data_cfg = cfg["data"]
    train_manifest = data_cfg.get("train_manifest")
    val_manifest = data_cfg.get("val_manifest")
    root = os.environ.get("BRATS_ROOT") or data_cfg.get("root")

    if train_manifest and val_manifest:
        train_samples = load_manifest(train_manifest, root=root)
        val_samples = load_manifest(val_manifest, root=root)
        print(f"[data] Loaded {len(train_samples)} train / {len(val_samples)} val from manifests.")
        return train_samples, val_samples

    # Fall back to directory scan
    if not root:
        raise ValueError(
            "Config must specify either data.train_manifest + data.val_manifest "
            "or data.root (for automatic split)."
        )
    return scan_brats_root(root=root, val_ratio=float(data_cfg.get("val_ratio", 0.2)), seed=seed)


def create_optimizer(cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    enc_lr = cfg["optim"]["encoder_lr"]
    dec_lr = cfg["optim"]["decoder_lr"]
    weight_decay = cfg["optim"]["weight_decay"]
    return torch.optim.AdamW(
        [
            {"params": model.encoder_parameters(), "lr": enc_lr},
            {"params": model.non_encoder_parameters(), "lr": dec_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    repro = cfg.get("reproducibility", {})
    if repro.get("deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(repro.get("use_deterministic_algorithms", False), warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    num_workers = int(cfg["data"]["num_workers"])
    persistent_workers = bool(cfg["data"].get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg["data"].get("prefetch_factor", 2))

    train_samples, val_samples = load_samples(cfg, seed)
    train_ds = BraTSDataset(train_samples, transform=build_train_transforms(cfg))
    val_ds = BraTSDataset(val_samples, transform=build_val_transforms(cfg))

    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": data_loader_generator,
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    # Validation uses batch_size=1: full-volume inputs have variable spatial sizes
    # and sliding-window inference is applied per-sample inside validate().
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = build_model(cfg).to(device)

    loss_cfg = cfg.get("loss", {})
    loss_fn = DiceBCELoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        consistency_weight=float(loss_cfg.get("consistency_weight", 0.0)),
    )

    optimizer = create_optimizer(cfg, model)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=cfg["training"]["epochs"],
        warmup_epochs=cfg["scheduler"]["warmup_epochs"],
        eta_min=cfg["scheduler"]["eta_min"],
    )
    scaler = GradScaler("cuda", enabled=cfg["training"]["amp"])

    output_dir = Path(cfg["output"]["dir"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_freq = max(1, int(cfg["output"].get("save_freq", 5)))

    # HD95 is expensive; compute it only every N epochs and always at the last epoch.
    metrics_cfg = cfg.get("metrics", {})
    compute_hd95 = bool(metrics_cfg.get("compute_hd95", False))
    hd95_every = int(metrics_cfg.get("hd95_every", 10))

    patch_size = tuple(int(v) for v in cfg["data"]["patch_size"])
    sw_batch_size = int(cfg["data"].get("sw_batch_size", 2))
    sw_overlap = float(cfg["data"].get("sw_overlap", 0.5))

    use_wandb = cfg["logging"]["use_wandb"]
    if use_wandb:
        wandb.init(
            project=cfg["logging"]["project"],
            entity=cfg["logging"].get("entity") or None,
            name=cfg["experiment_name"],
            config=cfg,
            mode=cfg["logging"].get("mode", "online"),
        )

    best_dice = -1.0
    start_epoch = 0
    best_checkpoint_path: Path | None = None

    resume_path = cfg.get("resume")
    if resume_path:
        checkpoint = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_dice = float(checkpoint.get("best_dice", best_dice))
        best_checkpoint_path = next(output_dir.glob("best_dice_*.pt"), None)

    total_epochs = cfg["training"]["epochs"]

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=cfg["training"]["amp"],
            grad_clip=cfg["training"].get("grad_clip"),
        )

        is_last = epoch + 1 == total_epochs
        run_hd95 = compute_hd95 and ((epoch + 1) % hd95_every == 0 or is_last)

        model.eval()
        with torch.no_grad():
            val_stats = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                amp_enabled=cfg["training"]["amp"],
                compute_hd95=run_hd95,
                patch_size=patch_size,
                sw_batch_size=sw_batch_size,
                overlap=sw_overlap,
            )

        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        log_payload = {
            "epoch": epoch + 1,
            "lr_encoder": lrs[0],
            "lr_decoder": lrs[1],
            **train_stats,
            **val_stats,
        }

        print(
            f"Epoch [{epoch+1}/{total_epochs}] "
            f"train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"val_dice={val_stats['val_dice']:.4f} "
            f"ET={val_stats['val_dice_et']:.4f} "
            f"TC={val_stats['val_dice_tc']:.4f} "
            f"WT={val_stats['val_dice_wt']:.4f}"
            + (f" HD95={val_stats['val_hd95']:.2f}" if run_hd95 else "")
        )

        if use_wandb:
            wandb.log(log_payload)

        if (epoch + 1) % save_freq == 0 or is_last:
            latest_path = output_dir / "latest.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                latest_path,
            )

        if val_stats["val_dice"] > best_dice:
            best_dice = val_stats["val_dice"]
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                best_checkpoint_path.unlink()
            best_checkpoint_path = output_dir / f"best_dice_{best_dice:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                best_checkpoint_path,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # expandable_segments reduces fragmentation without artificially capping allocation size
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
