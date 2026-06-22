#!/usr/bin/env bash
set -euo pipefail
# Generate images from a trained checkpoint.
#
# Examples
#   # seg-only checkpoint, segmentation condition, 2000 images:
#   CKPT="$CKPT_SEG" CONFIG=.../pixeldit_seg_control_v1_first200.yaml \
#     MODES=seg MAX_SAMPLES=2000 OUT=outputs/infer_seg bash scripts/infer.sh
#
#   # final three-control checkpoint, all 7 modes, 50 images:
#   CKPT="$CKPT_THREE" CONFIG=.../pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml \
#     MODES="depth seg edge depth_seg depth_edge seg_edge depth_seg_edge" \
#     MAX_SAMPLES=50 OUT=outputs/infer_3ctrl bash scripts/infer.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CKPT="${CKPT:-${CKPT_THREE}}"
CONFIG="${CONFIG:-${PKG_ROOT}/t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml}"
OUT="${OUT:-${PKG_ROOT}/outputs/infer}"
MODES="${MODES:-depth seg edge depth_seg depth_edge seg_edge depth_seg_edge}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"; BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-50}"; CFG_SCALE="${CFG_SCALE:-2.75}"
SEED="${SEED:-2025}"; DEVICE="${DEVICE:-cuda:0}"; DTYPE="${DTYPE:-bf16}"

cd "${PKG_ROOT}/t2i"
echo "[infer] ckpt=${CKPT}"
echo "[infer] modes=${MODES} max_samples=${MAX_SAMPLES} -> ${OUT}"
python infer_threecontrol_val.py \
  --config_path "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --output_dir "${OUT}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --num_sampling_steps "${NUM_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --control_modes ${MODES}
