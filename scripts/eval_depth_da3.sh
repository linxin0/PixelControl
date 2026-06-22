#!/usr/bin/env bash
set -euo pipefail
# Depth consistency for depth-only outputs using DepthAnything-3 (self-contained).
# Re-runs DA3 on generated depth-only images, compares to GT DA3 depth.
# Reports SI-RMSE / AbsRel / delta1-3 / pearson, and (optionally) saves DA3 maps.
#   GEN=outputs/infer_depth NAME=ours_depth SUFFIX=depth bash scripts/eval_depth_da3.sh
# NOTE: run in the `deco` conda env (DA3 + omegaconf + addict + imageio):
#   conda activate deco
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# DA3 needs user-site packages (omegaconf etc); do NOT isolate them here.
unset PYTHONNOUSERSITE || true

GEN="${GEN:?set GEN=<depth-only generated folder>}"
NAME="${NAME:-$(basename "${GEN}")}"
SUFFIX="${SUFFIX:-depth}"
OUT="${OUT:-${PKG_ROOT}/outputs/depth_consistency_da3_${NAME}.json}"
DEVICE="${DEVICE:-cuda:0}"; SAVE_MAPS="${SAVE_MAPS:-1}"

cd "${PKG_ROOT}"
python eval/eval_depth_consistency_da3.py \
  --gen_dir "${GEN}" --name "${NAME}" --control_suffix "${SUFFIX}" \
  --depth_root "${EVAL_DEPTH_ROOT}" \
  --da3_src "${DA3_SRC}" --da3_model_dir "${DA3_MODEL}" \
  --device "${DEVICE}" \
  ${SAVE_MAPS:+--save_maps} \
  --output_json "${OUT}"
