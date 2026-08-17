#!/usr/bin/env bash
set -euo pipefail
# Segmentation consistency (mIoU / pixel_acc / mAcc / boundary_F1 via SAM2.1).
# Re-runs SAM2 on the generated seg-only images, matches labels to GT SAM2.
#   GEN=outputs/infer_seg NAME=ours_seg bash scripts/eval_seg_sam2.sh
# NOTE: run this in the `deco` conda env (transformers + SAM2 weights).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

GEN="${GEN:?set GEN=<seg-only generated folder>}"
NAME="${NAME:-$(basename "${GEN}")}"
OUT="${OUT:-${PKG_ROOT}/outputs/seg_consistency_sam2}"
DEVICE="${DEVICE:-cuda:0}"; BATCH="${BATCH:-4}"; MIN_SAMPLES="${MIN_SAMPLES:-1}"

cd "${PKG_ROOT}"
python eval/eval_seg_consistency_sam2.py \
  --gen_dirs "${GEN}" --names "${NAME}" \
  --baseline_methods \
  --seg_root "${EVAL_SEG_ROOT}" \
  --sam2_model_dir "${SAM2_MODEL}" \
  --output_root "${OUT}" \
  --device "${DEVICE}" \
  --sam2_batch_size "${BATCH}" \
  --min_samples "${MIN_SAMPLES}"
