#!/usr/bin/env python3
"""Analyze YOLOE detection object-size distribution (small / medium / large).

This consumes the per-image ``detections.json`` files produced by
``eval/run_yoloe_seg.py`` and classifies every detected object into a size
bucket, then reports the proportion of small / medium / large objects across
the dataset. This is the analysis behind the paper's object-size statistics.

How object size is determined
------------------------------
We support two conventions and report both:

1. COCO absolute-area convention (default). Object pixel area is computed from
   the segmentation mask if present, else from the bounding box. Areas are
   first rescaled to a canonical 512x512 frame so the thresholds are resolution
   independent, then bucketed with the standard COCO rule:

       small  : area  < 32 * 32          (< 1024 px)
       medium : 32*32 <= area < 96*96     (1024 .. 9216 px)
       large  : area >= 96 * 96           (>= 9216 px)

2. Relative area-ratio convention (``--mode ratio``). bucket by the object's
   area as a fraction of the whole image (matches the depth Structural Scale
   Benchmark in ``eval/build_depth_structural_scale_benchmark.py``):

       tiny   : ratio < 0.5%
       small  : 0.5% <= ratio < 2%
       medium : 2%   <= ratio < 10%
       large  : ratio >= 10%

Usage
-----
python eval/analyze_yolo_object_sizes.py \
  --yolo_dir outputs/yoloe_segmentation/sa_000201_first2000_object_conf005 \
  --output_json outputs/yolo_object_sizes.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


COCO_SMALL = 32 * 32
COCO_LARGE = 96 * 96
RATIO_TINY = 0.005
RATIO_SMALL = 0.02
RATIO_MEDIUM = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOE object-size distribution.")
    p.add_argument("--yolo_dir", required=True,
                   help="Output dir of run_yoloe_seg.py containing images/<id>/detections.json")
    p.add_argument("--mode", choices=["coco", "ratio"], default="coco")
    p.add_argument("--canonical_size", type=int, default=512,
                   help="COCO areas are rescaled to this square frame before bucketing.")
    p.add_argument("--min_score", type=float, default=0.0,
                   help="Ignore detections below this confidence.")
    p.add_argument("--output_json", default="outputs/yolo_object_sizes.json")
    return p.parse_args()


def image_hw(image_path: Optional[str]) -> Optional[Tuple[int, int]]:
    if not image_path or Image is None:
        return None
    try:
        with Image.open(image_path) as im:
            return im.height, im.width
    except Exception:
        return None


def polygon_area(poly: List[List[float]]) -> float:
    if not poly or len(poly) < 3:
        return 0.0
    pts = np.asarray(poly, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def object_area_px(det: Dict) -> float:
    """Object pixel area from polygon (preferred) else bbox."""
    poly = det.get("polygon") or []
    area = polygon_area(poly)
    if area > 0:
        return area
    box = det.get("box")
    if box and len(box) == 4:
        x1, y1, x2, y2 = box
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return 0.0


def coco_bucket(area_canon: float) -> str:
    if area_canon < COCO_SMALL:
        return "small"
    if area_canon < COCO_LARGE:
        return "medium"
    return "large"


def ratio_bucket(ratio: float) -> str:
    if ratio < RATIO_TINY:
        return "tiny"
    if ratio < RATIO_SMALL:
        return "small"
    if ratio < RATIO_MEDIUM:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    yolo_dir = Path(args.yolo_dir)
    det_files = sorted(yolo_dir.glob("images/*/detections.json"))
    if not det_files:
        # also accept a flat directory of detections.json
        det_files = sorted(yolo_dir.glob("**/detections.json"))
    if not det_files:
        raise FileNotFoundError(f"no detections.json under {yolo_dir}")

    buckets = Counter()
    per_label_bucket: Dict[str, Counter] = {}
    areas_canon: List[float] = []
    ratios: List[float] = []
    n_images = 0
    n_objects = 0
    images_with_size: List[Tuple[int, int]] = []

    for df in det_files:
        blob = json.loads(df.read_text(encoding="utf-8"))
        n_images += 1
        hw = image_hw(blob.get("image_path"))
        for det in blob.get("detections", []):
            if float(det.get("score", 0.0)) < args.min_score:
                continue
            area = object_area_px(det)
            if area <= 0:
                continue
            n_objects += 1
            label = det.get("label", "unknown")
            if args.mode == "ratio":
                if hw is None:
                    continue
                img_area = float(hw[0] * hw[1])
                ratio = area / max(1.0, img_area)
                ratios.append(ratio)
                bk = ratio_bucket(ratio)
            else:
                if hw is not None:
                    scale = (args.canonical_size ** 2) / float(hw[0] * hw[1])
                else:
                    scale = 1.0
                area_canon = area * scale
                areas_canon.append(area_canon)
                bk = coco_bucket(area_canon)
            buckets[bk] += 1
            per_label_bucket.setdefault(label, Counter())[bk] += 1

    total = sum(buckets.values())
    proportions = {k: (buckets[k] / total if total else 0.0) for k in buckets}

    summary = {
        "yolo_dir": str(yolo_dir),
        "mode": args.mode,
        "canonical_size": args.canonical_size if args.mode == "coco" else None,
        "thresholds": (
            {"small_px<": COCO_SMALL, "large_px>=": COCO_LARGE,
             "note": "areas rescaled to canonical_size^2 frame"}
            if args.mode == "coco" else
            {"tiny<": RATIO_TINY, "small<": RATIO_SMALL, "medium<": RATIO_MEDIUM,
             "note": "object area / image area"}
        ),
        "min_score": args.min_score,
        "num_images": n_images,
        "num_objects": total,
        "bucket_counts": dict(buckets),
        "bucket_proportions": {k: round(v, 4) for k, v in proportions.items()},
        "mean_objects_per_image": round(total / n_images, 3) if n_images else 0.0,
    }
    if args.mode == "coco" and areas_canon:
        arr = np.asarray(areas_canon)
        summary["area_canonical_px"] = {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
        }
    if args.mode == "ratio" and ratios:
        arr = np.asarray(ratios)
        summary["area_ratio"] = {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
        }
    summary["per_label_bucket_counts"] = {
        lbl: dict(c) for lbl, c in sorted(per_label_bucket.items(),
                                          key=lambda kv: -sum(kv[1].values()))
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[yolo-size] images={n_images} objects={total} mode={args.mode}")
    for k in ("tiny", "small", "medium", "large"):
        if k in buckets:
            print(f"  {k:>7s}: {buckets[k]:6d}  ({100*proportions[k]:5.1f}%)")
    print(f"[yolo-size] wrote {out}")


if __name__ == "__main__":
    main()
