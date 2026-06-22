# Independent Multi-Control PixelDiT — Minimal Runnable Release

A self-contained copy of everything needed to **train**, **infer**, and
**evaluate** the three trained control models:

| Model | What it does | Checkpoint env var | Config |
|-------|--------------|--------------------|--------|
| **seg-only** | segmentation → image (single control) | `CKPT_SEG` | `pixeldit_seg_control_v1_first200.yaml` |
| **edge-only** | edge → image (single control, no injection, SoftCanny cycle) | `CKPT_EDGE` | `pixeldit_edge_control_v1_first200.yaml` |
| **three-control** | depth/seg/edge + any combination via gated fusion | `CKPT_THREE` | `pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml` |

The core innovation is **independent depth/seg/edge control branches with
layer-wise gated fusion**: single-condition inputs hard-select one branch
(gate ignored), multi-condition inputs use a masked softmax over only the
active branches. See `docs/01_OVERVIEW_AND_INNOVATIONS.md`.

## Layout

```text
release_my_network/
  pixdit_core/                 # backbone + control model (vendored, unchanged)
  t2i/
    diffusion/                 # training/inference framework (vendored)
    train_control.py           # training entry (unchanged)
    train_control.sh           # torchrun launcher (unchanged)
    infer_threecontrol_val.py  # inference/sampling entry (unchanged)
    configs_t2i/               # the 3 configs for the 3 checkpoints
    output/pretrained_models/  # null text embedding for CFG
  eval/                        # all metric scripts (see docs/04 & docs/05)
  scripts/                     # one-line launchers (train / infer / eval)
    _env.sh                    # EDIT paths here when moving servers
  reference_innovation_code/   # compact, framework-free extract of the core idea
  docs/                        # detailed documentation (read these)
  requirements.txt
```

> The `pixdit_core/` and `t2i/diffusion/` trees are **byte-identical copies**
> of the original repo, so the 10 GB checkpoints load with no surgery.

## Quickstart

```bash
# 0) edit absolute paths (models + data + checkpoints) once:
#    release_my_network/scripts/_env.sh

# 1) infer 50 images with the three-control model, all 7 modes
GPUS=0 MAX_SAMPLES=50 bash scripts/infer.sh

# 2) train seg-only on 2 GPUs
GPUS=0,1 NP=2 bash scripts/train_seg.sh

# 3) evaluate (examples)
GEN=outputs/infer NAME=ours SUFFIX=seg bash scripts/eval_visual_quality.sh
GEN=outputs/infer NAME=ours bash scripts/eval_edge_canny.sh        # edge F1
GEN=outputs/infer NAME=ours bash scripts/eval_seg_sam2.sh          # mIoU (deco env)
GEN=outputs/infer NAME=ours SUFFIX=depth bash scripts/eval_depth_da3.sh   # (deco env)
META_DIR=/...da3.../sa_000201 LIMIT=2000 bash scripts/eval_yolo_object_sizes.sh
```

## Documentation

1. `docs/01_OVERVIEW_AND_INNOVATIONS.md` — method, what's novel, architecture
2. `docs/02_USAGE.md` — install, train, infer, full command reference
3. `docs/03_PRETRAINED_MODELS.md` — every weight to download (Gemma, CLIP, SAM2, DA3, YOLOE, base PixelDiT)
4. `docs/04_METRICS.md` — depth/seg/edge accuracy + visual quality, **edge metric explained in detail**
5. `docs/05_YOLO.md` — YOLOE pipeline + small/medium/large object definitions + measured proportions
6. `docs/06_PARAMETERS.md` — control-injection params, pyramid-cycle-loss layer params, gate/LR scales

## Environment notes (this server)

The system Python needs two env vars to dodge packaging clashes (the launchers
set them for you):

```bash
export PYTHONNOUSERSITE=1                              # user-site tokenizers clash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # onnx/protobuf descriptor error
```

Depth (DA3) and segmentation (SAM2) metrics are best run in the `deco` conda env:

```bash
conda activate deco
```
