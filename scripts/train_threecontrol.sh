#!/usr/bin/env bash
set -euo pipefail
# Train the final mixed three-control model: independent depth/seg/edge
# branches + layer-wise gated fusion, all branches small LR, gate normal LR,
# 7-mode dropout sampling, all-three pyramid cycle losses (weight 0.005).
# Dual-GPU example:  GPUS=0,1 NP=2 bash scripts/train_threecontrol.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG="${CONFIG:-${PKG_ROOT}/t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml}"
GPUS="${GPUS:-0,1}"; NP="${NP:-2}"; MASTER_PORT="${MASTER_PORT:-29533}"
NAME="${NAME:-pixeldit_threecontrol_v1_mixed_cycle005_first200_from_mixed2k}"

cd "${PKG_ROOT}/t2i"
echo "[train-3ctrl] config=${CONFIG} gpus=${GPUS} np=${NP}"
CUDA_VISIBLE_DEVICES="${GPUS}" NP="${NP}" MASTER_PORT="${MASTER_PORT}" NAME="${NAME}" \
  bash train_control.sh "${CONFIG}"
