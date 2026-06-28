"""Download BraTS 2021 and generate reproducible train/val manifests.

The manifests store only case IDs (e.g. "BraTS2021_00001"), not absolute
paths, so they can be committed to git and used on any machine.

Usage:
    # Download automatically via kagglehub:
    python scripts/setup_data.py

    # Use data already on disk:
    python scripts/setup_data.py --root /path/to/BraTS2021_Training_Data

    # Override output directory for manifests:
    python scripts/setup_data.py --manifest_dir ./manifests

After this script:
  - manifests/train.json  → list of 1000 case IDs
  - manifests/val.json    → list of 251  case IDs
  - A line like "data_root: /path/to/data" is printed for your config.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


EXPECTED_TOTAL = 1251
MODALITIES = ["flair", "t1", "t1ce", "t2"]


def download_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError:
        print("[setup_data] ERROR: kagglehub is not installed.")
        print("  Run: uv pip install kagglehub")
        print("  Or provide --root /path/to/existing/data")
        sys.exit(1)

    print("[setup_data] Downloading BraTS 2021 via kagglehub...")
    path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
    return Path(path)


def find_brats_root(base: Path) -> Path:
    """Descend into the downloaded path to find the directory that has case subdirs."""
    # kagglehub sometimes nests: base/BraTS2021_Training_Data/BraTS2021_00001/...
    for candidate in [base, *base.iterdir()]:
        if candidate.is_dir() and any(candidate.glob("BraTS2021_*/BraTS2021_*_seg.nii.gz")):
            return candidate
    return base


def discover_cases(root: Path) -> list[str]:
    """Return sorted list of complete case IDs that are direct children of root."""
    case_ids: list[str] = []
    skipped = 0

    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("BraTS2021_"):
            continue
        stem = d.name
        seg = d / f"{stem}_seg.nii.gz"
        if seg.exists() and all((d / f"{stem}_{mod}.nii.gz").exists() for mod in MODALITIES):
            case_ids.append(stem)
        else:
            skipped += 1

    if skipped:
        print(f"[setup_data] Warning: skipped {skipped} incomplete cases (missing modalities).")

    return case_ids


def verify(case_ids: list[str], root: Path) -> None:
    n = len(case_ids)
    if n == 0:
        print(f"[setup_data] ERROR: No complete BraTS cases found under {root}")
        print("  Expected structure: <root>/BraTS2021_XXXXX/BraTS2021_XXXXX_{flair,t1,t1ce,t2,seg}.nii.gz")
        sys.exit(1)

    if n < EXPECTED_TOTAL * 0.95:
        print(f"[setup_data] Warning: found {n} cases, expected ~{EXPECTED_TOTAL}. Download may be incomplete.")
    else:
        print(f"[setup_data] Found {n} complete cases (expected {EXPECTED_TOTAL}). OK")


def generate_manifests(
    case_ids: list[str],
    manifest_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    n_val = 251
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "train.json", "w") as f:
        json.dump(train_ids, f, indent=2)
    with open(manifest_dir / "val.json", "w") as f:
        json.dump(val_ids, f, indent=2)

    return train_ids, val_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BraTS 2021 and generate manifests")
    parser.add_argument("--root", type=str, default=None,
                        help="Path to existing BraTS 2021 data (skip download)")
    parser.add_argument("--manifest_dir", type=str, default="manifests",
                        help="Output directory for manifest JSON files (default: manifests/)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.root:
        root = find_brats_root(Path(args.root))
        print(f"[setup_data] Using existing data at: {root}")
    else:
        downloaded = download_kagglehub()
        root = find_brats_root(downloaded)
        print(f"[setup_data] Data located at: {root}")

    case_ids = discover_cases(root)
    verify(case_ids, root)

    manifest_dir = Path(args.manifest_dir)
    train_ids, val_ids = generate_manifests(
        case_ids, manifest_dir, val_ratio=args.val_ratio, seed=args.seed
    )

    print(f"[setup_data] Manifests written to {manifest_dir}/")
    print(f"  train.json: {len(train_ids)} cases")
    print(f"  val.json:   {len(val_ids)} cases")
    print()
    print("Add this to your config (configs/base.yaml):")
    print(f"  data:")
    print(f"    root: {root}")
    print(f"    train_manifest: {manifest_dir / 'train.json'}")
    print(f"    val_manifest:   {manifest_dir / 'val.json'}")
    print()
    print("Or export as environment variable:")
    print(f"  export BRATS_ROOT={root}")


if __name__ == "__main__":
    main()
