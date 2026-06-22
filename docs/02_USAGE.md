# 02 — Usage

## 1. Install

```bash
cd release_my_network
pip install -r requirements.txt
# Optional metric backends (see docs/03):
#   pip install ultralytics            # YOLO object-size stats
#   pip install pytorch-fid lpips      # visual quality (if not already)
#   DepthAnything-3 + SAM2.1 weights   # depth / seg consistency
```

The package adds itself to `sys.path` automatically: the entry scripts insert
the package root (for `pixdit_core`) and `t2i/` (for `diffusion`). You run
everything from inside `release_my_network/`.

## 2. Configure paths once

Open `scripts/_env.sh` and set the absolute paths for your machine:

- model assets: `CLIP_MODEL`, `SAM2_MODEL`, `DA3_MODEL`, `DA3_SRC`
- eval data roots: `EVAL_IMAGE_ROOT`, `EVAL_SEG_ROOT`, `EVAL_EDGE_ROOT`, `EVAL_DEPTH_ROOT`
- checkpoints: `CKPT_SEG`, `CKPT_EDGE`, `CKPT_THREE`

The training **configs** (`t2i/configs_t2i/*.yaml`) also contain absolute paths
(`pretrained_ckpt`, dataset roots, `work_dir`). Edit those if you retrain on a
new server. The defaults match the original server.

## 3. Inference

`scripts/infer.sh` wraps `t2i/infer_threecontrol_val.py`.

```bash
# Three-control model, all 7 modes, 50 images
CKPT="$CKPT_THREE" \
CONFIG=t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml \
MODES="depth seg edge depth_seg depth_edge seg_edge depth_seg_edge" \
MAX_SAMPLES=50 OUT=outputs/infer_3ctrl GPUS=0 bash scripts/infer.sh

# Seg-only model, 2000 images
CKPT="$CKPT_SEG" \
CONFIG=t2i/configs_t2i/pixeldit_seg_control_v1_first200.yaml \
MODES=seg MAX_SAMPLES=2000 OUT=outputs/infer_seg GPUS=0 bash scripts/infer.sh

# Edge-only model, 2000 images
CKPT="$CKPT_EDGE" \
CONFIG=t2i/configs_t2i/pixeldit_edge_control_v1_first200.yaml \
MODES=edge MAX_SAMPLES=2000 OUT=outputs/infer_edge GPUS=0 bash scripts/infer.sh
```

Output filenames: `sa_XXXXXX_<mode>.png` (e.g. `sa_000201_seg.png`,
`sa_000201_depth_seg_edge.png`). For single-mode runs the suffix equals the
mode (`_seg`, `_edge`, `_depth`).

Key flags (pass via env to `infer.sh`, or directly to the python script):
`--max_samples`, `--batch_size`, `--num_sampling_steps` (default 50),
`--cfg_scale` (default 2.75), `--seed` (default 2025), `--dtype bf16`,
`--control_modes`, `--load_ema`.

## 4. Training

Backbone is frozen; only control branches + gate train. Two-GPU DDP examples:

```bash
GPUS=0,1 NP=2 bash scripts/train_seg.sh           # seg-only
GPUS=0,1 NP=2 bash scripts/train_edge.sh          # edge-only (no inj, SoftCanny 0.01)
GPUS=0,1 NP=2 bash scripts/train_threecontrol.sh  # final mixed gated model
```

Effective batch = `train_batch_size(16) × NP(2) × grad_accum(4) = 128`.
Checkpoints save every `save_model_steps` (2000); validation runs every
`every_n_steps` (500 single-control / per config). Outputs go to the config's
`work_dir` (`t2i/universal_pix_t2i_workdirs/<exp_name>/{checkpoints,val}`).

To resume / change LR see `train.override_lr_on_resume` and `optimizer.lr` in
the YAML.

## 5. Evaluation (one folder of generated images at a time)

| Metric | Script / launcher | Env | Output |
|--------|-------------------|-----|--------|
| Visual quality (FID, CLIP-text, CLIP-img, LPIPS) | `scripts/eval_visual_quality.sh` | system (PYTHONNOUSERSITE) | json + table |
| Edge accuracy (F1, P/R, Chamfer, Soft-IoU) | `scripts/eval_edge_canny.sh` | system | json/csv/md + edge maps |
| Seg accuracy (mIoU, pixel-acc, mAcc, boundary-F1) | `scripts/eval_seg_sam2.sh` | **deco** (SAM2) | json/csv/md + pred labels |
| Depth accuracy (SI-RMSE, AbsRel, δ1-3, Pearson) | `scripts/eval_depth_da3.sh` | **deco** (DA3) | json + DA3 maps |
| Object-size stats (small/med/large) | `scripts/eval_yolo_object_sizes.sh` | ultralytics | json |

Examples:

```bash
# visual quality of the seg outputs (sa_xxxxxx_seg.png) of any run
GEN=outputs/infer_seg NAME=ours_seg SUFFIX=seg METRICS="fid clip_img lpips" \
  bash scripts/eval_visual_quality.sh

# edge accuracy of edge-only outputs (sa_xxxxxx_edge.png)
GEN=outputs/infer_edge NAME=edge_iter12000 bash scripts/eval_edge_canny.sh

# seg accuracy (run inside deco)
conda activate deco
GEN=outputs/infer_seg NAME=ours_seg bash scripts/eval_seg_sam2.sh

# depth accuracy (run inside deco)
GEN=outputs/infer_depth NAME=ours_depth SUFFIX=depth bash scripts/eval_depth_da3.sh
```

See `docs/04_METRICS.md` for what each number means and how each metric is
computed, and `docs/05_YOLO.md` for the object-size analysis.

## 6. Smoke test the package (no checkpoint needed)

```bash
cd t2i
PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python - <<'PY'
import sys, types; from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent)); sys.path.insert(0, str(Path.cwd()))
sys.modules.setdefault("pyrallis", types.SimpleNamespace(wrap=lambda *a,**k:(lambda f:f)))
import diffusion.model.control_trainer, diffusion.data.datasets.control_datasets  # registrations
print("imports OK")
PY
```
