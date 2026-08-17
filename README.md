# PixelControl: Fine-Grained Condition Fidelity in Text-to-Image Diffusion

PixelControl is a pixel-space controllable text-to-image diffusion system for
preserving fine structures in **depth**, **segmentation**, and **edge**
conditions. It is built around a frozen PixelDiT-style text-to-image backbone
and adds independent modality-specific control branches, structure-aware
injection, multi-scale pyramid cycle supervision, and lightweight gated fusion
for multi-condition inputs.

**Project page:** [linxin0.github.io/pixelcontrol_homepage](https://linxin0.github.io/pixelcontrol_homepage/pixelcontrol-site/)  \
**Code repository:** [github.com/linxin0/PixelControl](https://github.com/linxin0/PixelControl)

> **Release status.** This repository contains the training, inference, and
> evaluation code and the configuration files. Large model weights, datasets,
> and third-party metric models are intentionally not committed. After those
> external assets are installed and paths are configured, the commands below
> reproduce the released workflow. It is not a zero-download turnkey bundle.

## What is included

| Capability | Entry point | Output |
|---|---|---|
| Segmentation-only training | `scripts/train_seg.sh` | a frozen-backbone seg-control checkpoint |
| Edge-only training | `scripts/train_edge.sh` | a frozen-backbone edge-control checkpoint |
| Three-control training | `scripts/train_threecontrol.sh` | depth/seg/edge gated checkpoint |
| Sampling | `scripts/infer.sh` → `t2i/infer_threecontrol_val.py` | PNGs named `sa_XXXXXX_<mode>.png` |
| Image quality | `scripts/eval_visual_quality.sh` | FID, CLIP-text, CLIP-image, LPIPS |
| Depth consistency | `scripts/eval_depth_da3.sh` | SI-RMSE, AbsRel, δ1–δ3, Pearson |
| Segmentation consistency | `scripts/eval_seg_sam2.sh` | mIoU, pixel accuracy, mAcc, boundary-F1, optional ARI/NMI |
| Edge consistency | `scripts/eval_edge_canny.sh` | F1, precision/recall, Chamfer, soft-IoU, MAE |
| Non-large regions | `eval/eval_nonlarge_conditioned_fidelity.py` | condition fidelity restricted by object size |
| Object-size analysis | `scripts/eval_yolo_object_sizes.sh` | small/medium/large object statistics |

The model supports seven control modes:

```text
depth, seg, edge, depth_seg, depth_edge, seg_edge, depth_seg_edge
```

The current implementation names its condition backends explicitly as
`depth_anything_v3`, `segment_anything_v2`, and `soft_canny`. Training uses
cached depth/segmentation/edge targets; the offline evaluators re-estimate
depth and segmentation from generated RGB images for closed-loop metrics.

## Repository map

```text
PixelControl/
├── pixdit_core/                 # PixelDiT backbone and control branches
├── t2i/
│   ├── train_control.py         # DDP training loop
│   ├── infer_threecontrol_val.py# deterministic validation sampler
│   ├── train_control.sh         # torchrun launcher
│   ├── configs_t2i/              # single-control and mixed-control YAMLs
│   ├── diffusion/                # data, model, losses, scheduler, checkpoint code
│   └── output/pretrained_models/ # small null text embedding shipped with the repo
├── eval/                         # metric implementations
├── scripts/                      # shell entry points and shared environment
├── docs/                         # method, assets, usage, metrics, reproducibility
├── reference_innovation_code/    # compact framework-free method extract
└── requirements.txt
```

Start with [docs/02_USAGE.md](docs/02_USAGE.md) for commands and
[docs/03_PRETRAINED_MODELS.md](docs/03_PRETRAINED_MODELS.md) for every external
asset. The exact experiment assumptions and data contract are in
[docs/07_REPRODUCIBILITY.md](docs/07_REPRODUCIBILITY.md).

## Quick start

### 1. Create the runtime

The reference environment is Python 3.10 with CUDA PyTorch (the development
environment used PyTorch 2.3). Install the repository dependencies inside a
fresh environment:

```bash
git clone https://github.com/linxin0/PixelControl.git
cd PixelControl
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Depth and SAM2 evaluators require the separate `deco` environment described in
[docs/03_PRETRAINED_MODELS.md](docs/03_PRETRAINED_MODELS.md). Run the cheap
setup check before using GPUs:

```bash
bash scripts/check_setup.sh
```

### 2. Configure paths

Copy the environment template if you need a machine-specific file, then edit
[`scripts/_env.sh`](scripts/_env.sh). It controls:

- `CLIP_MODEL`, `SAM2_MODEL`, `DA3_MODEL`, `DA3_SRC`
- `EVAL_IMAGE_ROOT`, `EVAL_DEPTH_ROOT`, `EVAL_SEG_ROOT`, `EVAL_EDGE_ROOT`
- `CKPT_SEG`, `CKPT_EDGE`, and `CKPT_THREE`

The YAML files use paths relative to `t2i/`. Keep the base PixelDiT checkpoint
at `t2i/pixeldit_t2i_v1.pth`, or edit `model.extra.pretrained_ckpt` in the
selected YAML. Do not commit private absolute paths or model weights.

### 3. Generate images

The default launcher makes a small smoke run. Set `MAX_SAMPLES=2000` for the
full released evaluation set:

```bash
# Three-control checkpoint: all seven modes, 2,000 images per mode.
CKPT="$CKPT_THREE" \
CONFIG=t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml \
MODES="depth seg edge depth_seg depth_edge seg_edge depth_seg_edge" \
MAX_SAMPLES=2000 BATCH_SIZE=8 NUM_STEPS=50 CFG_SCALE=2.75 \
OUT=outputs/infer_threecontrol_2000 GPUS=0 \
bash scripts/infer.sh

# Single-control checkpoints: one mode and 2,000 images.
CKPT="$CKPT_SEG" CONFIG=t2i/configs_t2i/pixeldit_seg_control_v1_first200.yaml \
MODES=seg MAX_SAMPLES=2000 OUT=outputs/infer_seg_2000 bash scripts/infer.sh

CKPT="$CKPT_EDGE" CONFIG=t2i/configs_t2i/pixeldit_edge_control_v1_first200.yaml \
MODES=edge MAX_SAMPLES=2000 OUT=outputs/infer_edge_2000 bash scripts/infer.sh
```

`GPUS` is translated to `CUDA_VISIBLE_DEVICES`; `DEVICE` can also be set
directly (for example `DEVICE=cuda:1`). Outputs are deterministic for a fixed
`SEED`, checkpoint, sampler settings, and input ordering.

### 4. Run metrics

Run metrics on the generated folder. The visual, edge, and YOLO evaluators run
in the main environment; depth and SAM2 segmentation evaluation is normally
run inside `deco`:

```bash
GEN=outputs/infer_seg_2000 NAME=seg_2000 SUFFIX=seg \
  METRICS="fid clip_text clip_img lpips" \
  bash scripts/eval_visual_quality.sh

GEN=outputs/infer_seg_2000 NAME=seg_2000 \
  bash scripts/eval_seg_sam2.sh       # conda activate deco first

GEN=outputs/infer_edge_2000 NAME=edge_2000 \
  bash scripts/eval_edge_canny.sh

GEN=outputs/infer_depth_2000 NAME=depth_2000 SUFFIX=depth \
  bash scripts/eval_depth_da3.sh      # conda activate deco first

META_DIR="$EVAL_DEPTH_ROOT" LIMIT=2000 \
  bash scripts/eval_yolo_object_sizes.sh
```

For the object-size breakdown described on the project page, first create the
YOLOE detections and then pass the resulting directory to
`eval/eval_nonlarge_conditioned_fidelity.py`; the full command is documented
in [docs/04_METRICS.md](docs/04_METRICS.md) and [docs/05_YOLO.md](docs/05_YOLO.md).

## Training

All three launchers use DDP through `torchrun`. The base PixelDiT backbone is
frozen; the control branches and (for the mixed model) the layer-wise gate are
optimized. The default effective batch is
`16 (per rank) × number of ranks × 4 gradient accumulation steps`.

```bash
# Two-GPU examples. Change GPUS/NP for your machine.
GPUS=0,1 NP=2 bash scripts/train_seg.sh
GPUS=0,1 NP=2 bash scripts/train_edge.sh
GPUS=0,1 NP=2 bash scripts/train_threecontrol.sh
```

Important configuration fields:

| Field | Current release value | Meaning |
|---|---:|---|
| `train.train_batch_size` | 16 | per-rank batch |
| `train.gradient_accumulation_steps` | 4 | effective batch multiplier |
| `train.optimizer.lr` | `2e-5` | base learning rate |
| `train.save_model_steps` | 2000 | checkpoint interval |
| `validation.every_n_steps` | 500 | lightweight validation interval |
| `control.control_probs` | 0.15/0.15/0.15/0.12/0.12/0.12/0.19 | seven-mode mixed sampling |
| `control.cycle_weight` | 0.02 / 0.01 / 0.005 | seg / edge / mixed |
| `cycle_scale_weights` | 0.75/0.5/0.5/0.25 | 512/256/128/64 pyramid |

Checkpoints are written below each config's `work_dir`. To resume, set
`model.resume_from.checkpoint` in the YAML to a saved `.pth`; the loader can
also restore optimizer, scheduler, and RNG state according to
`model.resume_from.resume_optimizer` and
`model.resume_from.resume_lr_scheduler`.

## Data contract

Each sample is keyed by a shared `sa_XXXXXX` stem:

```text
image_root/sa_000123/{stem}.jpg
image_root/sa_000123/{stem}.txt
depth_root/sa_000123/{stem}.depth.npy
seg_root/sa_000123/{stem}.sam2_label.npy
edge_root/sa_000123/{stem}.edge.png
```

Training configs use `sa_000000` through `sa_000199`. The full evaluation
split is `sa_000201`; the inference/evaluation scripts cap it at 2,000 samples
when `MAX_SAMPLES=2000` or `--max_samples 2000` is supplied. The repository
does not redistribute this dataset or the derived condition maps.

## Method summary

- **Structure-aware control injection:** condition Sobel structure maps
  modulate residuals at spatially sensitive locations; depth and segmentation
  use it, while edge injection is disabled because edge maps are already
  high-frequency.
- **Independent branches:** depth, segmentation, and edge have separate
  encoders/adapters.
- **Gated fusion:** single-control samples hard-select their active branch;
  multi-control samples use an active-only masked softmax gate per injection
  layer.
- **Pyramid cycle loss:** generated structure is checked at 512/256/128/64
  resolutions, balancing layout and boundary fidelity.

See [docs/01_OVERVIEW_AND_INNOVATIONS.md](docs/01_OVERVIEW_AND_INNOVATIONS.md)
for the architecture and [docs/06_PARAMETERS.md](docs/06_PARAMETERS.md) for
the complete parameter table.

## Reported project-page results

The project page reports the following representative comparisons. They are
included here as reference numbers from the project materials, not as a claim
that a fresh checkout has already regenerated them:

| Setting | Metric | Strongest baseline | PixelControl |
|---|---|---:|---:|
| Depth | AbsRel ↓ | 0.1949 | **0.1434** |
| Segmentation | Boundary-F1 ↑ | 0.5061 | **0.6973** |
| Edge | Chamfer ↓ | 11.53 | **5.456** |
| Non-large depth | AbsRel ↓ | 0.2727 | **0.1419** |
| Non-large edge | Chamfer ↓ | 4.381 | **2.603** |
| Non-large segmentation | Boundary-F1 ↑ | 0.4728 | **0.6409** |

Use the scripts in this repository to recompute metrics under a fixed
environment and record the resulting JSON files with the checkpoint and git
commit used.

## Reproducibility and limitations

- The release contains code and configs, not the base checkpoint, datasets,
  trained checkpoints, or third-party weights.
- SAM2 and DA3 are used by the offline evaluators to re-estimate generated
  structure. Training uses cached SAM2 labels and a differentiable surrogate;
  standard SAM2 mask generation is not differentiated through during the
  training backward pass.
- The repository exposes separate, composable commands rather than silently
  downloading private assets or running a long end-to-end job.
- Run `bash scripts/check_setup.sh` and record the exact YAML, checkpoint,
  sampler flags, and metric environment alongside any reported result.

## Citation

```bibtex
@article{lin2025pixelcontrol,
  title   = {PixelControl: Fine-Grained Condition Fidelity in Text-to-Image Diffusion},
  author  = {Lin, Xin and Li, Haodong and Zhang, Zhifei and Yang, Yutong and
             Zheng, Haitian and Tian, Juanxi and Lin, Zhe and Nguyen, Truong},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## License and third-party components

No license file is added by this release yet. Check the licenses of the
PixelDiT code, Gemma, DepthAnything-3, SAM2, CLIP, LPIPS, PyTorch-FID, and
YOLOE before redistribution or commercial use. The repository's
`requirements.txt` only declares Python dependencies; it does not grant rights
to their weights or datasets.
