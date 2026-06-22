# 03 — Pretrained Models to Download

Every external weight, what it's for, where it goes, and how to get it.
Update the matching variable in `scripts/_env.sh` (or the YAML) after download.

## A. Required for TRAINING and INFERENCE

### 1. Base text-to-image PixelDiT backbone (`pixeldit_t2i_v1.pth`)
- **Role:** frozen backbone the control branches attach to. Referenced by every
  config as `model.extra.pretrained_ckpt`.
- **Size:** ~5.2 GB.
- **Local path (this server):** `t2i/pixeldit_t2i_v1.pth`
  (the configs point at the absolute repo path; copy it into the package's
  `t2i/` or edit `pretrained_ckpt` in the 3 YAMLs).
- This is the project's own checkpoint, not a public download.

### 2. Gemma-2-2B-it text encoder
- **Role:** caption → text embeddings (`txt_embed_dim=2304`, max 300 tokens).
- **Loaded as:** HuggingFace `Efficient-Large-Model/gemma-2-2b-it`
  (`AutoModelForCausalLM(...).get_decoder()`), bf16.
- **Download:**
  ```bash
  huggingface-cli download Efficient-Large-Model/gemma-2-2b-it
  ```
  or set `HF_HOME` to a cache that already has it. Resolution happens in
  `t2i/diffusion/model/builder.py::get_tokenizer_and_text_encoder`.

### 3. Null text embedding (provided)
- **Role:** classifier-free-guidance unconditional embedding.
- **Provided in package:**
  `t2i/output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth`
  (1.4 MB). Already wired via `train.null_embed_root: ./output/pretrained_models/`.

### 4. The 3 trained control checkpoints (your models)
| Model | Path (this server) | Size |
|-------|--------------------|------|
| seg-only | `.../exp_pixeldit_seg_control_v1_512_bs16x2_acc4_cycle002_first200/checkpoints/epoch_1_step_6000.pth` | ~11 GB |
| edge-only | `.../exp_pixeldit_edge_control_v1_512_bs16x2_acc4_noinj_softcanny001_first200/checkpoints/epoch_1_step_12000.pth` | ~11 GB |
| three-control | `.../exp_pixeldit_threecontrol_v1_mixed_cycle005_first200_from_mixed2k/checkpoints/epoch_1_step_10000.pth` | ~12 GB |

These are not bundled (size). Point `CKPT_SEG/CKPT_EDGE/CKPT_THREE` at them.

## B. Required for METRIC EVALUATION

### 5. CLIP ViT-L/14 — visual quality (CLIP-text, CLIP-img)
- **Download:** `openai/clip-vit-large-patch14` (or local mirror).
- **Default path (relative to package):** `t2i/pretrained/clip-vit-large-patch14`
- **Var:** `CLIP_MODEL`.

### 6. InceptionV3 — FID
- Auto-downloaded by `pytorch-fid` on first use (`pip install pytorch-fid`).
- LPIPS (`pip install lpips`) auto-downloads the AlexNet LPIPS weights.

### 7. SAM2.1-Hiera-Large — segmentation consistency
- **Role:** re-segment generated seg-only images, compare masks to GT SAM2.
- **Used via:** HuggingFace `transformers` mask-generation pipeline.
- **Default path (relative to package):** `t2i/pretrained/sam2.1-hiera-large`
- **Var:** `SAM2_MODEL`. Run inside the `deco` conda env.
- This is the SAME model that produced the GT seg labels, so predictions and
  targets are comparable.

### 8. DepthAnything-3 (DA3 NESTED-GIANT-LARGE-1.1) — depth consistency
- **Role:** re-estimate depth of generated depth-only images, compare to GT DA3
  depth (the model that produced the training depth targets).
- **Code (default):** `DA3_SRC=t2i/third_party/depth-anything-3/src`
- **Weights (default):** `DA3_MODEL=t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1`
- DA3 extra deps: `omegaconf, addict, imageio, e3nn, evo` (and stubs moviepy /
  export internally — our scripts stub the unused video-export path). Run
  inside the `deco` conda env. **Do not** set `PYTHONNOUSERSITE=1` for DA3
  (it hides `omegaconf`); the `eval_depth_da3.sh` launcher unsets it.

### 9. YOLOE-26x-seg — object-size statistics
- **Role:** open-vocabulary detection/segmentation to bucket objects into
  small/medium/large (the dataset object-size analysis).
- **Install:** `pip install ultralytics`; the `yoloe-26x-seg.pt` weight is
  fetched by ultralytics on first use (or pass a local `--model` path).
- See `docs/05_YOLO.md`.

## C. Data roots (the SA-1B / BLIP eval set)
| Var | Default | Contents |
|-----|---------|----------|
| `EVAL_IMAGE_ROOT` | `t2i/data/blip/extracted_new/sa_000201` | `{stem}.jpg` + `{stem}.txt` |
| `EVAL_DEPTH_ROOT` | `t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201` | `{stem}.depth.npy` + `.meta.json` |
| `EVAL_SEG_ROOT` | `t2i/data/blip_sam2_large_extracted/sa_000201` | `{stem}.sam2_label.npy` |
| `EVAL_EDGE_ROOT` | `t2i/data/blip_edge/sa_000201` | `{stem}.edge.png` |

Training uses the same modalities for subdirs `sa_000000`…`sa_000199`
(see config `data.data.subdir_range: [0, 199]`).
