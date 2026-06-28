#!/usr/bin/env bash
# Run nnU-Net v2 baseline on BraTS 2021 using our fixed 1000/251 split.
#
# Prerequisites:
#   pip install nnunetv2
#   Set BRATS_ROOT, BASE_DIR, and GPU_ID below.
#
# This script:
#   1. Converts BraTS 2021 to nnU-Net Dataset001 format
#   2. Runs preprocessing and planning
#   3. Trains on fold 0 (= our train split) on a single GPU
#   4. Runs inference on the validation set
#   5. Computes Dice / HD95 with nnU-Net's evaluator
set -euo pipefail

# ---- Configure these ----
BRATS_ROOT="${BRATS_ROOT:-/path/to/BraTS2021_Training_Data}"
BASE_DIR="${BASE_DIR:-$(pwd)/nnunet_workspace}"
GPU_ID="${GPU_ID:-0}"
# -------------------------

export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "=== Step 1: Convert dataset ==="
python scripts/prepare_nnunet.py \
    --brats_root "${BRATS_ROOT}" \
    --nnunet_raw "${nnUNet_raw}" \
    --train_manifest manifests/train.json \
    --val_manifest   manifests/val.json

echo "=== Step 2: Plan and preprocess ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres -np 8

echo "=== Step 3: Train (fold 0 = our val split) ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python scripts/nnunet_wandb_wrapper.py \
        --dataset_id 1 --config 3d_fullres --fold 0 \
        --gpu "${GPU_ID}" --extra_args="--npz" \
        --base_dir "${BASE_DIR}" \
        --project brats-segmentation --entity kc_brats --run_name nnunet_baseline \
    2>&1 | tee logs/nnunet_train.log

echo "=== Step 4: Predict on validation set ==="
VAL_INPUT="${nnUNet_raw}/Dataset001_BraTS2021/imagesTs"
VAL_OUTPUT="${nnUNet_results}/nnunet_val_preds"
mkdir -p "${VAL_OUTPUT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    nnUNetv2_predict \
        -i "${VAL_INPUT}" \
        -o "${VAL_OUTPUT}" \
        -d 1 -c 3d_fullres -f 0 \
        --save_probabilities

echo "=== Step 5: Evaluate ==="
GT_DIR="${nnUNet_raw}/Dataset001_BraTS2021/labelsTr"
nnUNetv2_evaluate_folder \
    -ref "${GT_DIR}" \
    -pred "${VAL_OUTPUT}" \
    -djfile "${nnUNet_raw}/Dataset001_BraTS2021/dataset.json" \
    -pfile "${nnUNet_results}/Dataset001_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/predicted_next_stage.pkl" \
    2>&1 | tee logs/nnunet_eval.log

echo "nnU-Net baseline complete. Results in logs/nnunet_eval.log"
