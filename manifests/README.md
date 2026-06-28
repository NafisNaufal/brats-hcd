# Manifests

`train.json` and `val.json` define the fixed 1000/251 BraTS 2021 split used
in all experiments (seed=42, val_ratio=0.2).

They contain **case ID lists only** — no absolute paths — so they are
portable across machines and safe to commit to git.

## Generating manifests

Run once after downloading the data:

```bash
python scripts/setup_data.py --root /path/to/BraTS2021_Training_Data
```

The script discovers all complete cases, applies the fixed split, and
writes `manifests/train.json` and `manifests/val.json`.

After generating, commit them:

```bash
git add manifests/train.json manifests/val.json
git commit -m "data: add fixed train/val manifests (seed=42)"
```

## Format

```json
["BraTS2021_00001", "BraTS2021_00042", ...]
```

Absolute paths are resolved at runtime using the `data.root` key in
`configs/base.yaml`.
