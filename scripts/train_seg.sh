#!/usr/bin/env bash
set -euo pipefail
# Train the segmentation-only control model (single-control, frozen backbone).
# Dual-GPU example:  GPUS=0,1 NP=2 bash scripts/train_seg.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG="${CONFIG:-${PKG_ROOT}/t2i/configs_t2i/pixeldit_seg_control_v1_first200.yaml}"
GPUS="${GPUS:-0,1}"; NP="${NP:-2}"; MASTER_PORT="${MASTER_PORT:-29531}"
NAME="${NAME:-pixeldit_seg_control_v1_512_bs16x2_acc4_cycle002_first200}"

cd "${PKG_ROOT}/t2i"
echo "[train-seg] config=${CONFIG} gpus=${GPUS} np=${NP}"
CUDA_VISIBLE_DEVICES="${GPUS}" NP="${NP}" MASTER_PORT="${MASTER_PORT}" NAME="${NAME}" \
  bash train_control.sh "${CONFIG}"
