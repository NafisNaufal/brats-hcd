"""Wrap nnUNetv2_train and stream metrics to W&B.

nnUNetv2 has no native W&B hook. This wrapper:
  1. Launches nnUNetv2_train as a subprocess
  2. Parses its stdout line by line
  3. Logs train_loss, val_loss, and Pseudo Dice to W&B each epoch

Usage (called by run_nnunet.sh — not directly):
    python scripts/nnunet_wandb_wrapper.py \\
        --dataset_id 1 --config 3d_fullres --fold 0 \\
        --gpu 0 --extra_args "--npz"
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", default="1")
    p.add_argument("--config", default="3d_fullres")
    p.add_argument("--fold", default="0")
    p.add_argument("--gpu", default="0")
    p.add_argument("--extra_args", default="--npz")
    p.add_argument("--base_dir", default=None,
                   help="nnunet_workspace root; sets nnUNet_raw/preprocessed/results if provided")
    p.add_argument("--project", default="brats-segmentation")
    p.add_argument("--entity", default="kc_brats")
    p.add_argument("--run_name", default="nnunet_baseline")
    return p.parse_args()


# Patterns for nnUNetv2 stdout (covers v2.2–v2.5 output formats)
_RE_EPOCH     = re.compile(r"Epoch\s+(\d+)")
_RE_TRAIN     = re.compile(r"train[_ ]loss[:\s]+([-\d.]+)", re.IGNORECASE)
_RE_VAL       = re.compile(r"val[_ ]loss[:\s]+([-\d.]+)", re.IGNORECASE)
_RE_DICE_LIST = re.compile(r"Pseudo\s+[Dd]ice[:\s]+\[([^\]]+)\]", re.IGNORECASE)
_RE_DICE_MEAN = re.compile(r"mean[:\s]+([\d.]+)", re.IGNORECASE)
_RE_LR        = re.compile(r"lr[:\s]+([\d.eE+\-]+)", re.IGNORECASE)


def main() -> None:
    args = parse_args()

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "trainer": "nnUNetv2",
            "dataset_id": args.dataset_id,
            "config": args.config,
            "fold": args.fold,
        },
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.base_dir:
        base = args.base_dir
        env.setdefault("nnUNet_raw",          f"{base}/nnUNet_raw")
        env.setdefault("nnUNet_preprocessed", f"{base}/nnUNet_preprocessed")
        env.setdefault("nnUNet_results",      f"{base}/nnUNet_results")

    cmd = [
        "nnUNetv2_train",
        args.dataset_id,
        args.config,
        args.fold,
    ] + args.extra_args.split()

    print(f"[nnunet_wandb_wrapper] Running: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    epoch_buf: dict = {}

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        if m := _RE_EPOCH.search(line):
            if epoch_buf:
                wandb.log(epoch_buf)
            epoch_buf = {"epoch": int(m.group(1))}

        if m := _RE_TRAIN.search(line):
            epoch_buf["train_loss"] = float(m.group(1))

        if m := _RE_VAL.search(line):
            epoch_buf["val_loss"] = float(m.group(1))

        if m := _RE_DICE_LIST.search(line):
            vals = [float(x) for x in re.findall(r"\d+\.\d+", m.group(1))]
            if len(vals) >= 3:
                # nnUNet channel order: WT, TC, ET (BraTS challenge order)
                epoch_buf["dice_wt"] = vals[0]
                epoch_buf["dice_tc"] = vals[1]
                epoch_buf["dice_et"] = vals[2]
                epoch_buf["dice_mean"] = sum(vals) / len(vals)

        if m := _RE_LR.search(line):
            epoch_buf["lr"] = float(m.group(1))

    # flush last epoch
    if epoch_buf:
        wandb.log(epoch_buf)

    proc.wait()
    wandb.finish()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
