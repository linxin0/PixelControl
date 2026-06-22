# 04 — Metrics

Four metric families. Each operates on a folder of generated images and reuses
the same learned/handcrafted extractor that produced the conditions, so the
"loop" is closed fairly. Direction arrows: ↓ = lower better, ↑ = higher better.

---

## 1. Visual quality — `eval/eval_visual_quality.py`

Self-contained FID / CLIP / LPIPS (no external repo needed). For each generated
`sa_XXXXXX[_<suffix>].png` we look up the GT image `<image_root>/sa_XXXXXX.{jpg,…}`
and caption `<image_root>/sa_XXXXXX.txt`.

| Metric | Backbone | Definition | Dir |
|--------|----------|------------|-----|
| **FID** | InceptionV3 (2048-d, pytorch-fid) | Fréchet distance between Inception feature distributions of generated vs real images (all resized to 512, center-crop). | ↓ |
| **CLIP-text** | CLIP ViT-L/14 | mean cos(CLIP_img(gen), CLIP_text(caption)). Caption alignment. | ↑ |
| **CLIP-img** | CLIP ViT-L/14 | mean cos(CLIP_img(gen), CLIP_img(real)). Similarity to GT image. | ↑ |
| **LPIPS** | AlexNet (lpips) | perceptual distance between paired gen/real in [-1,1]. | ↓ |

```bash
GEN=outputs/infer_seg NAME=ours_seg SUFFIX=seg \
  METRICS="fid clip_text clip_img lpips" bash scripts/eval_visual_quality.sh
```
`SUFFIX` lets it match single-control filenames (`_seg`/`_edge`/`_depth`); leave
empty for `sa_xxxxxx.png`.

---

## 2. Edge accuracy — `eval/eval_edge_consistency_canny.py`  (read this carefully)

**Problem:** the edge *condition* maps were created with
`GaussianBlur(k=5/11) + Canny(random threshold range)`. Because the threshold
is randomized per image, the condition map is **not** a stable ground truth —
you cannot directly compare a generated image's edges to the condition map.

**Solution — symmetric extraction.** We never use the condition map as GT.
Instead, **both** the generated RGB image **and** the original GT RGB image are
passed through the *same deterministic* extractor:

```
RGB ─► grayscale ─► GaussianBlur(k=11) ─► Canny(70,150) ─► soft edge map ∈ [0,1]
```

(`blurred_canny_soft`, `eval_edge_consistency_canny.py:219`). Optionally an
ensemble of threshold pairs is averaged into a soft map (`--ensemble_thresholds
"l1,h1;l2,h2"`); default is the single `(70,150)` pair. This mirrors the
edge-condition recipe (blur first, then Canny) so generated and GT are processed
identically and the comparison is well-defined.

Then both soft maps are binarized at 0.5 and compared:

| Metric | Definition | Dir |
|--------|------------|-----|
| **edge_f1** | tolerance-based F1 between predicted and GT edge pixels. A predicted edge pixel counts as a true positive if a GT edge pixel lies within `--boundary_tolerance` (default 2 px), via Euclidean distance transform. F1 = 2PR/(P+R). | ↑ |
| **edge_precision / recall** | the P and R behind that F1. | ↑ |
| **chamfer** | 0.5·(mean dist from each pred edge px to nearest GT edge px + mean dist from each GT edge px to nearest pred edge px). In pixels. | ↓ |
| **soft_iou** | Σ min(pred_soft, gt_soft) / Σ max(pred_soft, gt_soft) on the soft maps (before binarization). | ↑ |
| **edge_mae** | mean |pred_soft − gt_soft|. | ↓ |
| **density_ratio** | (#pred edge px) / (#GT edge px). ~1.0 means the model reproduces the right amount of edge content (≪1 = under-drawing, ≫1 = noisy over-edges). | →1 |

Why tolerance + chamfer: pixel-exact edge matching is brutally strict; a 2-px
tolerance and the distance-based chamfer reward edges that are *close* to GT
even if not pixel-aligned, which is the standard for boundary evaluation (BSDS).

```bash
GEN=outputs/infer_edge NAME=edge_iter12000 bash scripts/eval_edge_canny.sh
```
Outputs: per-sample JSONL, summary json/csv/md, and the extracted edge maps
(`extracted_edges_k11_canny70_150/`) for visualization. Example real numbers
(5-image smoke run): ours edge_f1≈0.69, baselines 0.22–0.44.

---

## 3. Segmentation accuracy — `eval/eval_seg_consistency_sam2.py`

**Re-segment the generated seg-only image with SAM2.1-Hiera-Large** (the same
model that produced the GT labels), then compare the predicted label map to the
GT SAM2 label map (`<seg_root>/{stem}.sam2_label.npy`).

Because SAM2 label IDs are arbitrary, predicted and GT segments are first
**matched by Hungarian assignment on IoU** before computing:

| Metric | Definition | Dir |
|--------|------------|-----|
| **mIoU** | mean IoU over matched segment pairs. | ↑ |
| **pixel_acc** | fraction of pixels with correct matched label. | ↑ |
| **mAcc** | mean per-class accuracy. | ↑ |
| **boundary_f1** | F1 of segment boundaries within `--boundary_tolerance` px. | ↑ |
| **ARI / NMI** | (optional) clustering agreement, label-permutation invariant. Disable with `--no_cluster_metrics`. | ↑ |

```bash
# run inside the deco conda env (needs transformers + SAM2 weights + GPU)
GEN=outputs/infer_seg NAME=ours_seg bash scripts/eval_seg_sam2.sh
```
Predicted labels are cached as `.npy` (re-runs are fast). `--sam2_only` just
caches predictions; `--sam2_batch_size N` batches images on one GPU.

---

## 4. Depth accuracy — `eval/eval_depth_consistency_da3.py`

**Re-estimate depth of the generated depth-only image with DepthAnything-3**
(the model that produced the training depth targets), compare to the GT DA3
depth (`<depth_root>/{stem}.depth.npy`). All scale-invariant via a per-image
least-squares `(a,b)` fit `a·pred + b ≈ gt`.

| Metric | Definition | Dir |
|--------|------------|-----|
| **si_rmse** | RMSE of `(a·pred+b) − gt` after the per-image affine fit. | ↓ |
| **absrel** | mean \|aligned−gt\|/gt over pixels with gt > `absrel_min_gt` (0.1). | ↓ |
| **delta1/2/3** | % pixels with max(p/g, g/p) < 1.25 / 1.25² / 1.25³. | ↑ |
| **pearson_r** | linear correlation of aligned pred vs gt. | ↑ |

```bash
# run inside the deco conda env (DA3); the launcher unsets PYTHONNOUSERSITE
GEN=outputs/infer_depth NAME=ours_depth SUFFIX=depth SAVE_MAPS=1 \
  bash scripts/eval_depth_da3.sh
```
`SAVE_MAPS=1` also dumps the DA3 depth `.npy` + visualization `.png` per image.

> The bundled `eval/save_depth_only_da3_maps.py` and
> `eval/eval_baseline_depth_only_da3.py` are the **baseline-comparison** variants
> (multi-method folder layout, shells out to the PixelGen DA3 script). The
> consolidated `eval_depth_consistency_da3.py` above is the self-contained path.

---

## 5. Object-size statistics — see `docs/05_YOLO.md`

`eval/run_yoloe_seg.py` + `eval/analyze_yolo_object_sizes.py` measure the
small/medium/large object composition of the dataset.
