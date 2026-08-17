#!/usr/bin/env bash
set -euo pipefail
# Edge consistency (Edge-F1 / precision / recall / Chamfer / Soft-IoU).
# Both the generated RGB and the GT RGB are run through the SAME extractor:
#   RGB -> grayscale -> GaussianBlur(k=11) -> Canny(70,150)
# so the threshold-sensitive condition map is never used as direct GT.
#   GEN=outputs/infer_edge NAME=edge_iter12000 bash scripts/eval_edge_canny.sh
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

GEN="${GEN:?set GEN=<edge-only generated folder>}"
NAME="${NAME:-$(basename "${GEN}")}"
OUT="${OUT:-${PKG_ROOT}/outputs/edge_consistency_canny}"
MIN_SAMPLES="${MIN_SAMPLES:-1}"
BLUR="${BLUR:-11}"; CANNY_LOW="${CANNY_LOW:-70}"; CANNY_HIGH="${CANNY_HIGH:-150}"

cd "${PKG_ROOT}"
python eval/eval_edge_consistency_canny.py \
  --gen_dirs "${GEN}" --names "${NAME}" \
  --image_root "${EVAL_IMAGE_ROOT}" \
  --output_root "${OUT}" \
  --blur_kernel "${BLUR}" --canny_low "${CANNY_LOW}" --canny_high "${CANNY_HIGH}" \
  --min_samples "${MIN_SAMPLES}"
