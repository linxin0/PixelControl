#!/usr/bin/env bash
# Shared environment for all launchers in this package.
#
# PKG_ROOT is the PixelControl directory (parent of scripts/).
# Adjust the model paths below if you move the package to another server.

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PKG_ROOT

# --- runtime settings for the reference environment ----------------------
# Isolate user-site packages (avoids tokenizers/transformers version clash)
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# Pure-python protobuf parser (avoids onnx/protobuf descriptor TypeError)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
# Reduce CUDA fragmentation during long training
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- external model assets (place under t2i/pretrained, t2i/third_party,
#     or override with an absolute path on your machine) --------------------
export CLIP_MODEL="${CLIP_MODEL:-${PKG_ROOT}/t2i/pretrained/clip-vit-large-patch14}"
export SAM2_MODEL="${SAM2_MODEL:-${PKG_ROOT}/t2i/pretrained/sam2.1-hiera-large}"
export DA3_MODEL="${DA3_MODEL:-${PKG_ROOT}/t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1}"
export DA3_SRC="${DA3_SRC:-${PKG_ROOT}/t2i/third_party/depth-anything-3/src}"

# --- evaluation data roots (place under t2i/data, or override) ------------
export EVAL_IMAGE_ROOT="${EVAL_IMAGE_ROOT:-${PKG_ROOT}/t2i/data/blip/extracted_new/sa_000201}"
export EVAL_SEG_ROOT="${EVAL_SEG_ROOT:-${PKG_ROOT}/t2i/data/blip_sam2_large_extracted/sa_000201}"
export EVAL_EDGE_ROOT="${EVAL_EDGE_ROOT:-${PKG_ROOT}/t2i/data/blip_edge/sa_000201}"
export EVAL_DEPTH_ROOT="${EVAL_DEPTH_ROOT:-${PKG_ROOT}/t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201}"

# --- trained checkpoints (the 3 final models) -----------------------------
WORKDIRS="${PKG_ROOT}/t2i/universal_pix_t2i_workdirs"
export CKPT_SEG="${CKPT_SEG:-${WORKDIRS}/exp_pixeldit_seg_control_v1_512_bs16x2_acc4_cycle002_first200/checkpoints/epoch_1_step_6000.pth}"
export CKPT_EDGE="${CKPT_EDGE:-${WORKDIRS}/exp_pixeldit_edge_control_v1_512_bs16x2_acc4_noinj_softcanny001_first200/checkpoints/epoch_1_step_12000.pth}"
export CKPT_THREE="${CKPT_THREE:-${WORKDIRS}/exp_pixeldit_threecontrol_v1_mixed_cycle005_first200_from_mixed2k/checkpoints/epoch_1_step_10000.pth}"
