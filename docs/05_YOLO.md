# 05 — YOLO Object Detection & Object-Size Analysis

This documents how we detect objects and how we define and measure
**small / medium / large** objects, including the proportions measured on our
2000-image evaluation set.

## 1. Detection — `eval/run_yoloe_seg.py`

We use **YOLOE-26x-seg** (Ultralytics open-vocabulary detector+segmenter). It
is *prompted with text labels*, so it finds objects relevant to each image
rather than a fixed COCO 80.

Pipeline per image:
1. Read the caption; derive candidate labels via `CAPTION_LABEL_ALIASES`
   (e.g. "people/pedestrians/tourists" → `person`), then **append a default
   label bank** of ~50 common categories (person, building, tree, car, sky,
   road, …). Mode is `--prompt_mode caption_plus_defaults` (default).
2. Set those labels as the open-vocab classes (`model.set_classes`).
3. Predict with `--conf` (default 0.05 for the object-size study, 0.25 default),
   `--iou 0.7`, `--imgsz 640`.
4. Save per image: annotated jpg, mask `.npz`, and `detections.json`
   (`box xyxy`, `score`, `label`, `polygon`), plus a dataset `summary.json`
   and contact sheets.

```bash
python eval/run_yoloe_seg.py \
  --meta_dir t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201 \
  --output_dir outputs/yoloe_segmentation/sa_000201 \
  --model yoloe-26x-seg.pt --limit 2000 --conf 0.05 --device 0
```
`--meta_dir` holds `*.meta.json` that point at the RGB + caption of each sample.

## 2. Object-size bucketing — `eval/analyze_yolo_object_sizes.py`

Consumes the `detections.json` files and classifies **every detected object**
into a size bucket. Object pixel area is taken from the **segmentation polygon**
when available (preferred, true shape area) and falls back to the **bounding-box
area** otherwise.

Two conventions are supported (both reported):

### (a) COCO absolute-area convention — `--mode coco` (default)
Object areas are first **rescaled to a canonical 512×512 frame** (so thresholds
are resolution-independent: `area_canon = area_px · 512² / (H·W)`), then bucketed
with the standard COCO rule:

| Bucket | Rule (canonical 512² frame) |
|--------|------------------------------|
| **small**  | area `< 32×32` (= 1024 px) |
| **medium** | `32×32 ≤ area < 96×96` (1024 – 9216 px) |
| **large**  | area `≥ 96×96` (= 9216 px) |

### (b) Relative area-ratio convention — `--mode ratio`
Bucket by object area as a fraction of the whole image (matches the depth
Structural Scale Benchmark in `eval/build_depth_structural_scale_benchmark.py`):

| Bucket | area / image |
|--------|--------------|
| **tiny**   | `< 0.5%` |
| **small**  | `0.5% – 2%` |
| **medium** | `2% – 10%` |
| **large**  | `≥ 10%` |

```bash
python eval/analyze_yolo_object_sizes.py \
  --yolo_dir outputs/yoloe_segmentation/sa_000201 --mode coco --min_score 0.05 \
  --output_json outputs/yoloe_segmentation/sa_000201/object_sizes_coco.json
```

## 3. Measured proportions (our eval set, 2000 images)

Source: `sa_000201_first2000_object_conf005`, conf ≥ 0.05, **19 973 objects**,
mean ≈ 10.0 objects/image.

**COCO absolute-area convention (`--mode coco`):**

| Bucket | Count | Proportion |
|--------|-------|------------|
| small  | 9 961 | **49.9%** |
| medium | 6 275 | **31.4%** |
| large  | 3 737 | **18.7%** |

**Relative area-ratio convention (`--mode ratio`):**

| Bucket | Count | Proportion |
|--------|-------|------------|
| tiny   | 10 974 | **54.9%** |
| small  | 4 111  | **20.6%** |
| medium | 2 712  | **13.6%** |
| large  | 2 176  | **10.9%** |

Takeaway: the benchmark is **small-object dominated** — roughly half the objects
are small under the COCO rule, and ~55% are "tiny" (<0.5% of the frame) under the
relative rule. This is why structural control (depth/seg/edge) and the
boundary-weighted cycle losses matter: a large fraction of the supervision lives
on small, high-frequency structures.

## 4. How "size" is determined — summary
- **Area source:** segmentation mask polygon area (true shape) → fallback bbox area.
- **Normalization:** rescaled to a 512² canonical frame so a 4K image and a
  512px image bucket the same object identically (COCO mode).
- **Thresholds:** COCO `32²/96²` (small/medium/large) or relative `0.5%/2%/10%`
  (tiny/small/medium/large). Pick with `--mode`.
- **Confidence gate:** `--min_score` drops low-confidence detections before
  bucketing (use the same value as detection `--conf`).
- **Per-label breakdown** is also written (`per_label_bucket_counts`) so you can
  see, e.g., that `person` skews small and `building/sky` skews large.
