"""Convert BraTS 2021 to nnU-Net v2 Dataset format using our fixed train/val split.

Usage:
    python scripts/prepare_nnunet.py \
        --brats_root /path/to/BraTS2021_Training_Data \
        --nnunet_raw $nnUNet_raw \
        --train_manifest manifests/train.json \
        --val_manifest   manifests/val.json

The script creates:
    $nnUNet_raw/Dataset001_BraTS2021/
        dataset.json
        imagesTr/   (training images, 4 channels → _0000..._0003.nii.gz)
        labelsTr/   (training labels)
        imagesTs/   (validation images, used for inference only)

nnU-Net channel convention (BraTS):
    _0000 → T1
    _0001 → T1ce
    _0002 → T2
    _0003 → FLAIR

After running this script:
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
    nnUNetv2_train 1 3d_fullres 0 --npz  (use fold 0 = our val split)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import load_manifest


DATASET_ID = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_BraTS2021"

CHANNEL_MAP = {
    "t1":   "_0000",
    "t1ce": "_0001",
    "t2":   "_0002",
    "flair": "_0003",
}


def copy_image(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_label(src: str, dst: Path) -> None:
    """Copy a BraTS label file, remapping ET label 4 → 3 for nnU-Net compatibility."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(src)
    data = np.asarray(img.dataobj, dtype=np.uint8)
    data[data == 4] = 3
    nib.save(nib.Nifti1Image(data, img.affine, img.header), dst)


def convert_split(
    samples: list[dict],
    images_dir: Path,
    labels_dir: Path,
) -> list[str]:
    """Copy images and labels; return list of case identifiers."""
    case_ids: list[str] = []
    for sample in samples:
        label_path = Path(sample["label"])
        # Derive a clean case ID from the seg filename: BraTS2021_XXXXX
        case_id = label_path.name.replace("_seg.nii.gz", "")
        case_ids.append(case_id)

        for modality_key, suffix in CHANNEL_MAP.items():
            src = sample[modality_key]
            dst = images_dir / f"{case_id}{suffix}.nii.gz"
            if not dst.exists():
                copy_image(src, dst)

        dst_label = labels_dir / f"{case_id}.nii.gz"
        if not dst_label.exists():
            copy_label(sample["label"], dst_label)

    return case_ids


def build_dataset_json(
    out_dir: Path,
    train_case_ids: list[str],
    val_case_ids: list[str],
) -> None:
    """Write nnU-Net v2 dataset.json.

    We encode our fixed val split as fold 0 so nnUNetv2_train 1 3d_fullres 0
    trains on exactly our 1000 cases and validates on our 251.
    """
    dataset = {
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR",
        },
        "labels": {
            "background": 0,
            "NCR": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": len(train_case_ids) + len(val_case_ids),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "BraTS 2021 — fixed 1000/251 train/val split",
        "reference": "https://arxiv.org/abs/2107.02314",
        "licence": "CC-BY-NC 4.0",
        "release": "1.0",
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    # Write a splits_final.json that maps fold 0 → our exact val set.
    # nnU-Net respects this file and skips auto-splitting.
    splits = [
        {
            "train": train_case_ids,
            "val": val_case_ids,
        }
    ]
    preprocessed_dir = out_dir.parent.parent / "nnUNet_preprocessed" / DATASET_NAME
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    with open(preprocessed_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[prepare_nnunet] splits_final.json written to {preprocessed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats_root", required=True)
    parser.add_argument("--nnunet_raw", required=True, help="Path to $nnUNet_raw directory")
    parser.add_argument("--train_manifest", default="manifests/train.json")
    parser.add_argument("--val_manifest", default="manifests/val.json")
    args = parser.parse_args()

    train_samples = load_manifest(args.train_manifest, root=args.brats_root)
    val_samples = load_manifest(args.val_manifest, root=args.brats_root)

    out_dir = Path(args.nnunet_raw) / DATASET_NAME
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_ts = out_dir / "imagesTs"
    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_nnunet] Converting {len(train_samples)} training cases...")
    train_ids = convert_split(train_samples, images_tr, labels_tr)

    # Val images go to both imagesTr (so nnUNet preprocesses them for fold-0 val)
    # and imagesTs (for standalone inference). Labels always go to labelsTr.
    print(f"[prepare_nnunet] Converting {len(val_samples)} validation cases (→ imagesTr + imagesTs)...")
    val_ids = convert_split(val_samples, images_tr, labels_tr)
    convert_split(val_samples, images_ts, labels_tr)

    build_dataset_json(out_dir, train_ids, val_ids)
    print(f"[prepare_nnunet] Done. Dataset written to: {out_dir}")
    print()
    print("Next steps:")
    print(f"  export nnUNet_raw={args.nnunet_raw}")
    print(f"  export nnUNet_preprocessed=$(dirname {args.nnunet_raw})/nnUNet_preprocessed")
    print(f"  export nnUNet_results=$(dirname {args.nnunet_raw})/nnUNet_results")
    print(f"  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres")
    print(f"  nnUNetv2_train 1 3d_fullres 0 --npz")


if __name__ == "__main__":
    main()
