#!/usr/bin/env bash
# Launch ALL experiments: HCD runs (E0–E5) + nnU-Net baseline.
#
# GPU assignment:
#   GPU 0: E0 (Swin flat baseline)  → E3 (no-hierarchy ablation)  → nnU-Net
#   GPU 1: E1 (HCD-Swin, main)      → E4 (no-SE ablation)
#   GPU 2: E2 (HCD-ResNet50)        → E5 (SwinUNETR baseline)
#
# Usage:
#   bash scripts/run_all.sh
set -euo pipefail

export WANDB_API_KEY="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"
BRATS_ROOT="${BRATS_ROOT:-/mnt/nas-hpg9/adhi/.cache/kagglehub/datasets/dschettler8845/brats-2021-task1/versions/1}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_sequential() {
    local gpu="$1"
    shift
    local cfgs=("$@")
    (
        for cfg in "${cfgs[@]}"; do
            exp="$(basename "${cfg}" .yaml)"
            log="${LOG_DIR}/${exp}.gpu${gpu}.log"
            echo "[gpu${gpu}] Starting ${exp} → ${log}" >&2
            if CUDA_VISIBLE_DEVICES="${gpu}" python -u train.py --config "${cfg}" \
                    > "${log}" 2>&1; then
                echo "[gpu${gpu}] Finished ${exp}" >&2
            else
                echo "[gpu${gpu}] FAILED ${exp} (exit $?) — see ${log}" >&2
            fi
        done
    ) &
}

# GPU 0: flat baseline → no-hier ablation  (nnU-Net runs after these finish)
run_sequential 0 \
    "configs/exp0_swin_flat.yaml" \
    "configs/exp3_swin_hcd_nohier.yaml"
pid0=$!

# GPU 1: main HCD-Swin → no-SE ablation
run_sequential 1 \
    "configs/exp1_swin_hcd.yaml" \
    "configs/exp4_swin_hcd_nose.yaml"
pid1=$!

# GPU 2: HCD-ResNet50 → SwinUNETR baseline
run_sequential 2 \
    "configs/exp2_resnet50_hcd.yaml" \
    "configs/exp5_swinunetr.yaml"
pid2=$!

echo "Waiting for HCD experiments (pids: ${pid0} ${pid1} ${pid2})..."
wait "${pid0}" "${pid1}" "${pid2}" || true  # failures are logged per-experiment; nnU-Net still runs
echo "All HCD experiments completed. Starting nnU-Net on GPU 0..."

BRATS_ROOT="${BRATS_ROOT}" GPU_ID=0 bash scripts/run_nnunet.sh \
    2>&1 | tee "${LOG_DIR}/nnunet.log"

echo "All experiments completed."
