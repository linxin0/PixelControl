#!/usr/bin/env bash
set -euo pipefail
# Visual quality (FID / CLIP-text / CLIP-img / LPIPS) for one generated folder.
#   GEN=outputs/infer_seg NAME=ours_seg SUFFIX=seg bash scripts/eval_visual_quality.sh
# SUFFIX is the single-control suffix (seg/edge/depth); leave empty for <stem>.png.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

GEN="${GEN:?set GEN=<generated image folder>}"
NAME="${NAME:-$(basename "${GEN}")}"
SUFFIX="${SUFFIX:-}"
OUT="${OUT:-${PKG_ROOT}/outputs/visual_quality_${NAME}.json}"
DEVICE="${DEVICE:-cuda:0}"; BATCH="${BATCH:-16}"
METRICS="${METRICS:-fid clip_text clip_img lpips}"

cd "${PKG_ROOT}"
python eval/eval_visual_quality.py \
  --gen_dir "${GEN}" --name "${NAME}" \
  ${SUFFIX:+--control_suffix "${SUFFIX}"} \
  --image_root "${EVAL_IMAGE_ROOT}" \
  --clip_model "${CLIP_MODEL}" \
  --metrics ${METRICS} \
  --batch_size "${BATCH}" --device "${DEVICE}" \
  --output_json "${OUT}"
