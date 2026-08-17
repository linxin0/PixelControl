# 07 — Reproducibility checklist

This document is the experiment contract for the public code release. Record
these values with every result so that two runs can be compared meaningfully.

## 1. Fixed experiment choices

| Item | Release setting |
|---|---|
| Image resolution | 512 × 512 |
| Backbone | PixelDiT pixel-space T2I, frozen during control training |
| Text encoder | Gemma-2-2b-it, 300 tokens, 2,304 channels |
| Training data | `sa_000000` … `sa_000199` |
| Evaluation data | `sa_000201`, up to 2,000 aligned samples |
| Flow sampler | `flow_dpm-solver` |
| Default sampling steps | 50 |
| Default CFG | 2.75 |
| Default inference seed | 2025 |
| Checkpoint interval | every 2,000 optimizer steps |
| Validation interval | every 500 optimizer steps |
| Mixed control modes | depth, seg, edge, depth_seg, depth_edge, seg_edge, depth_seg_edge |
| Mixed mode probabilities | 0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19 |

## 2. Data alignment

The four roots must contain the same image stems. A single sample is valid only
when its RGB image/caption and required condition files agree on the stem:

```text
<image_root>/<shard>/<stem>.jpg
<image_root>/<shard>/<stem>.txt
<depth_root>/<shard>/<stem>.depth.npy
<seg_root>/<shard>/<stem>.sam2_label.npy
<edge_root>/<shard>/<stem>.edge.png
```

Training uses the first 200 `sa_XXXXXX` shards. Evaluation uses the dedicated
`sa_000201` shard. Do not mix training shards into the reported evaluation
folder.

Before a long run, verify alignment with a small manifest and inspect at least
one sample from each modality. The dataset classes cache an index under
`outputs/`; delete the cache after moving or regenerating data.

## 3. Training protocol

```bash
GPUS=0,1 NP=2 MASTER_PORT=29533 \
  bash scripts/train_threecontrol.sh
```

The launcher passes the YAML to `torchrun`. The default mixed config uses a
per-rank batch of 16 and four accumulation steps. Save checkpoints and logs
under the YAML `work_dir`; do not overwrite a previous experiment directory.

To resume, edit the selected YAML:

```yaml
model:
  resume_from:
    checkpoint: ./universal_pix_t2i_workdirs/<run>/checkpoints/epoch_1_step_20000.pth
    resume_optimizer: true
    resume_lr_scheduler: true
```

If comparing a resumed run to a fresh run, record whether optimizer, scheduler,
EMA, and RNG state were restored. Changing `control_probs`, cycle weights,
resolution, or the base checkpoint creates a new experiment, not a continuation
of the same result.

## 4. Inference protocol

Use an explicit checkpoint, config, mode list, output directory, and sample
count. The following is the canonical 2,000-image command:

```bash
CKPT=/path/to/checkpoint.pth \
CONFIG=t2i/configs_t2i/pixeldit_threecontrol_v1_mixed_cycle005_from_mixed2k.yaml \
MODES="depth seg edge depth_seg depth_edge seg_edge depth_seg_edge" \
MAX_SAMPLES=2000 BATCH_SIZE=8 NUM_STEPS=50 CFG_SCALE=2.75 SEED=2025 \
OUT=outputs/ours_2000 GPUS=0 \
  bash scripts/infer.sh
```

Do not compare folders generated with different `CFG_SCALE`, sampler steps,
seed, prompt files, checkpoint EMA state, or sample ordering. The output name
is `sa_XXXXXX_<mode>.png`; evaluators use that suffix to match conditions.

## 5. Metric protocol

Run each metric on the same generated folder and save its JSON/CSV/MD output
under a run-specific directory. At minimum, report:

```text
FID, CLIP-text, CLIP-image, LPIPS,
depth SI-RMSE/AbsRel/δ1–δ3/Pearson,
seg mIoU/pixel-acc/mAcc/boundary-F1,
edge F1/precision/recall/Chamfer/soft-IoU,
non-large depth/seg/edge statistics when object-size analysis is enabled.
```

Depth and segmentation metrics require their evaluator environments and
weights. Record package versions, evaluator checkpoint identifiers, device,
sample count, and the exact command line. A metric computed on a 50-image
smoke run must not be reported as the 2,000-image result.

## 6. What is and is not paper-exact

- The code implements the released PixelControl control branches, gated fusion,
  structure-aware injection, and pyramid cycle losses.
- Training consumes cached condition maps. SAM2 and DA3 are re-run in the
  offline evaluation scripts; standard SAM2 mask generation is not part of a
  differentiable backward pass.
- The public checkout does not include private base/control checkpoints,
  datasets, or third-party weights. Reproducibility therefore requires the
  same external assets and environment.
