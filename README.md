# Hierarchical Consistency Decoder for 3D Brain Tumor Segmentation

A single end-to-end decoder that enforces the anatomical nesting constraint
ET ⊂ TC ⊂ WT via sequential soft architectural conditioning, evaluated on
BraTS 2021.

## Overview

Standard segmentation models predict Enhancing Tumor (ET), Tumor Core (TC),
and Whole Tumor (WT) with independent sigmoid heads, allowing anatomically
impossible outputs where ET voxels fall outside TC or TC voxels fall outside
WT. The **Hierarchical Consistency Decoder (HCD)** encodes this constraint
into the decoder's forward computation:

1. A Feature Pyramid Network (FPN) fuses four encoder scales into a single feature map
2. A Squeeze-Excitation (SE) block recalibrates channel responses
3. WT is predicted first; TC is predicted conditioned on the WT soft mask; ET is predicted conditioned on the TC soft mask

The constraint is enforced architecturally (not just via loss), and the entire
pipeline remains end-to-end differentiable because soft (sigmoid) masks are
used rather than hard thresholded ones.

**Encoders:** Swin Transformer (SSL pretrained), ResNet50-3D (Med3D pretrained)  
**Dataset:** BraTS 2021 (1,251 glioma cases, 1,000 train / 251 val)  
**Metrics:** Dice Similarity Coefficient (DSC) and Hausdorff Distance 95% (HD95)
for Whole Tumor, Tumor Core, and Enhancing Tumor

### Experiments

| Run | Encoder | Hierarchy | SE | Role |
|---|---|---|---|---|
| nnU-Net | nnU-Net auto-config | — | — | SOTA baseline |
| SwinUNETR (E5) | Swin | — | — | Published Swin baseline |
| E0 | Swin | ✗ | ✗ | Internal flat decoder baseline |
| E1 | Swin | ✓ | ✓ | Main method |
| E2 | ResNet50-3D | ✓ | ✓ | Encoder generalization |
| E3 | Swin | ✗ | ✓ | Ablation: SE only |
| E4 | Swin | ✓ | ✗ | Ablation: hierarchy only |

---

## Setup on a new SSH server

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd BraTS
```

### 2. Create conda environment

```bash
conda create -n brats python=3.10 -y
conda activate brats
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle credentials (to download data)

Go to [kaggle.com](https://kaggle.com) → profile → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download data and generate manifests

```bash
python scripts/setup_data.py
```

This downloads BraTS 2021 (~12 GB in NIfTI format) via kagglehub, finds all 1,251 cases,
splits them 1,000/251 with seed=42, and writes `manifests/train.json` and
`manifests/val.json`. It prints the data path at the end — copy it.

If you already have the data on disk:

```bash
python scripts/setup_data.py --root /path/to/BraTS2021_Training_Data
```

### 6. Set the data root in config

Open `configs/base.yaml` and paste the path printed by the script:

```yaml
data:
  root: /path/printed/by/setup_data.py
```

### 7. Commit the manifests

```bash
git add manifests/train.json manifests/val.json
git commit -m "add train/val manifests"
```

### 8. Log in to Weights & Biases

```bash
wandb login
```

To disable W&B: set `logging.use_wandb: false` in `configs/base.yaml`.

---

## Running all experiments

```bash
BRATS_ROOT=/path/to/BraTS2021_Training_Data bash scripts/run_all.sh
```

This runs all 7 experiments across 3 GPUs:

```
GPU 0:  E0 (Swin flat) → E3 (no-hier ablation) → nnU-Net
GPU 1:  E1 (HCD-Swin)  → E4 (no-SE ablation)
GPU 2:  E2 (HCD-ResNet50) → E5 (SwinUNETR baseline)
```

Logs are written to `logs/<experiment_name>.gpu<id>.log`.

> **Recommended:** run inside tmux so training survives SSH disconnect.
> ```bash
> tmux new -s brats
> BRATS_ROOT=... bash scripts/run_all.sh
> # detach: Ctrl+B then D
> # re-attach: tmux attach -t brats
> ```

### Running a single experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py --config configs/exp1_swin_hcd.yaml
```

---

## Outputs

Checkpoints are saved to `outputs/<experiment_name>/`:

| File | Description |
|---|---|
| `latest.pt` | Most recent checkpoint (saved every N epochs) |
| `best_dice_<score>.pt` | Best validation Dice checkpoint |

To resume a run, add to your config:

```yaml
resume: outputs/exp1_swin_hcd/latest.pt
```

---

## Generating paper figures

```bash
# Qualitative segmentation panels (requires trained checkpoints)
python scripts/visualize_predictions.py \
    --checkpoint_swin  outputs/exp1_swin_hcd/best_dice_*.pt \
    --checkpoint_res50 outputs/exp2_resnet50_hcd/best_dice_*.pt \
    --out_dir paper/figures/qual

# Architecture diagram (requires LaTeX)
bash scripts/build_figures.sh
```

---

## Project structure

```
BraTS/
├── configs/
│   ├── base.yaml                   # shared training settings
│   ├── exp0_swin_flat.yaml         # flat FPN baseline (no HCD)
│   ├── exp1_swin_hcd.yaml          # HCD + Swin (main)
│   ├── exp2_resnet50_hcd.yaml      # HCD + ResNet50
│   ├── exp3_swin_hcd_nohier.yaml   # ablation: SE only
│   ├── exp4_swin_hcd_nose.yaml     # ablation: hierarchy only
│   └── exp5_swinunetr.yaml         # SwinUNETR baseline
├── data/
│   ├── dataset.py                  # BraTSDataset, manifest loader
│   └── transforms.py               # preprocessing and augmentation
├── engine/
│   ├── train_loop.py               # training step
│   └── validate.py                 # sliding-window validation
├── manifests/
│   ├── train.json                  # 1000 case IDs (generated by setup_data.py)
│   └── val.json                    # 251 case IDs
├── models/
│   ├── build_model.py              # model factory
│   ├── decoder_hcd.py              # HCD: FPN + SE + sequential heads
│   └── encoder.py                  # Swin and ResNet50-3D encoders
├── paper/
│   ├── main.tex                    # IEEE conference paper
│   └── figures/
│       └── architecture.tex        # TikZ architecture diagram
├── scripts/
│   ├── build_figures.sh            # compile TikZ figures to PDF
│   ├── prepare_nnunet.py           # convert data to nnU-Net format
│   ├── run_all.sh                  # launch all experiments
│   ├── run_nnunet.sh               # nnU-Net pipeline
│   ├── setup_data.py               # download data + generate manifests
│   └── visualize_predictions.py    # qualitative figure generator
├── utils/
│   ├── loss.py                     # Dice + BCE + consistency loss
│   ├── metrics.py                  # DSC and HD95
│   └── scheduler.py                # cosine annealing with warmup
├── train.py                        # training entrypoint
└── requirements.txt
```
