#!/usr/bin/env bash
set -euo pipefail
# YOLOE open-vocabulary segmentation + object-size distribution analysis.
# Stage 1: run YOLOE on RGB images (needs ultralytics + a yoloe-*-seg.pt model).
# Stage 2: bucket every detected object into small/medium/large and report
#          object-size proportions for the supplied evaluation directory.
#   META_DIR=t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201 LIMIT=2000 bash scripts/eval_yolo_object_sizes.sh
# NOTE: run in an env with `ultralytics` installed.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

META_DIR="${META_DIR:-${EVAL_DEPTH_ROOT}}"
YOLO_OUT="${YOLO_OUT:-${PKG_ROOT}/outputs/yoloe_segmentation/sa_000201}"
MODEL="${MODEL:-yoloe-26x-seg.pt}"
LIMIT="${LIMIT:-2000}"; CONF="${CONF:-0.05}"; DEVICE="${DEVICE:-0}"
SIZE_MODE="${SIZE_MODE:-coco}"   # coco (32^2 / 96^2) or ratio (0.5% / 2% / 10%)

cd "${PKG_ROOT}"
echo "[yolo] stage1: detect -> ${YOLO_OUT}"
python eval/run_yoloe_seg.py \
  --meta_dir "${META_DIR}" --output_dir "${YOLO_OUT}" \
  --model "${MODEL}" --limit "${LIMIT}" --conf "${CONF}" --device "${DEVICE}"

echo "[yolo] stage2: object-size distribution (${SIZE_MODE})"
python eval/analyze_yolo_object_sizes.py \
  --yolo_dir "${YOLO_OUT}" --mode "${SIZE_MODE}" --min_score "${CONF}" \
  --output_json "${YOLO_OUT}/object_sizes_${SIZE_MODE}.json"
