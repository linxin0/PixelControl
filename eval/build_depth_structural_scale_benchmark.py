#!/usr/bin/env python3
"""Build a depth-derived Structural Scale Benchmark.

The benchmark intentionally avoids semantic or instance segmentation. It
extracts Depth-derived Structural Regions from ground-truth depth maps using
log-depth Sobel discontinuities, morphology, and connected components.

Example:
    conda activate deco
    python eval/build_depth_structural_scale_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT_DIR = Path("t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201")
DEFAULT_OUTPUT_DIR = Path("outputs/depth_structural_scale_benchmark/sa_000201_first50")
SCALE_NAMES = ("tiny", "small", "medium", "large")
SCALE_DISPLAY = {
    "tiny": "Tiny",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


@dataclass
class RegionRecord:
    image_id: str
    region_id: int
    source: str
    area_pixels: int
    area_ratio: float
    bbox_xyxy: Tuple[int, int, int, int]
    eval_bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    scale: str = ""


@dataclass
class ImageResult:
    image_id: str
    meta_path: Path
    depth_path: Path
    source_image: Optional[Path]
    depth_raw: np.ndarray
    depth_norm: np.ndarray
    gradient: np.ndarray
    boundary: np.ndarray
    labels: np.ndarray
    records: List[RegionRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Depth-derived Structural Regions for scale-wise ControlNet evaluation."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=50, help="Use the first N meta files after sorting.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many sorted meta files first.")
    parser.add_argument(
        "--meta_glob",
        default="*.meta.json",
        help="Metadata glob under input_dir. Metadata should contain depth_file and optionally source_image.",
    )
    parser.add_argument(
        "--threshold_method",
        choices=("percentile", "otsu"),
        default="percentile",
        help="How to threshold Sobel gradient magnitude into a binary depth-discontinuity map.",
    )
    parser.add_argument(
        "--gradient_percentile",
        type=float,
        default=90.0,
        help="Gradient percentile used when --threshold_method percentile.",
    )
    parser.add_argument("--closing_kernel", type=int, default=5)
    parser.add_argument("--opening_kernel", type=int, default=3)
    parser.add_argument(
        "--min_area_ratio",
        type=float,
        default=0.001,
        help="Discard regions smaller than this image-area ratio. Default is 0.1%%.",
    )
    parser.add_argument(
        "--include_discontinuity_regions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also extract connected components from depth-discontinuity blobs. "
            "This recovers small non-closed structures such as distant people."
        ),
    )
    parser.add_argument(
        "--min_discontinuity_area_ratio",
        type=float,
        default=0.00002,
        help="Minimum area ratio for discontinuity-blob regions. Default is 0.002%%.",
    )
    parser.add_argument(
        "--max_discontinuity_area_ratio",
        type=float,
        default=0.02,
        help="Keep discontinuity blobs only up to this area ratio to avoid duplicating large surfaces.",
    )
    parser.add_argument("--discontinuity_close_kernel", type=int, default=3)
    parser.add_argument("--discontinuity_dilate_kernel", type=int, default=3)
    parser.add_argument(
        "--scale_mode",
        choices=("fixed", "quantile"),
        default="fixed",
        help="fixed uses 0.5%%/2%%/10%% thresholds; quantile uses dataset quartiles.",
    )
    parser.add_argument(
        "--roi_margin",
        type=float,
        default=0.15,
        help="Expand each structural-region bbox by this fraction of bbox width/height for evaluation ROI.",
    )
    parser.add_argument(
        "--visualization_rows",
        type=int,
        default=50,
        help="Number of selected images to include in the RGB/depth/scale-ROI contact sheet.",
    )
    parser.add_argument(
        "--copy_source_depth",
        action="store_true",
        help="Copy the original depth file into each image output directory in addition to saving loaded .npy.",
    )
    parser.add_argument(
        "--save_region_pngs",
        action="store_true",
        help="Also save one binary PNG per structural region. labels.npz is always saved.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_meta_files(input_dir: Path, meta_glob: str, offset: int, limit: int) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {input_dir}")
    meta_files = sorted(input_dir.glob(meta_glob))
    if not meta_files:
        raise RuntimeError(f"found 0 metadata files under {input_dir} matching {meta_glob!r}")
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    if limit <= 0:
        selected = meta_files[offset:]
    else:
        selected = meta_files[offset : offset + limit]
    if not selected:
        raise RuntimeError(f"selected 0 metadata files with offset={offset}, limit={limit}")
    return selected


def load_depth(depth_path: Path) -> np.ndarray:
    if not depth_path.is_file():
        raise FileNotFoundError(f"depth file does not exist: {depth_path}")
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    elif suffix == ".npz":
        archive = np.load(depth_path)
        depth = archive[list(archive.keys())[0]]
    else:
        with Image.open(depth_path) as im:
            if im.mode in ("I", "I;16"):
                depth = np.asarray(im.convert("I"), dtype=np.float32)
            else:
                depth = np.asarray(im.convert("L"), dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth.mean(axis=-1)
    if depth.ndim != 2:
        raise ValueError(f"expected 2D depth, got shape={depth.shape} for {depth_path}")
    return depth


def robust_normalize(x: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [low_pct, high_pct])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y[~valid] = 0.0
    return y.astype(np.float32)


def log_depth_normalize(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros_like(depth, dtype=np.float32)

    safe_depth = depth.copy()
    median = float(np.median(safe_depth[valid]))
    safe_depth[~valid] = median
    min_val = float(np.min(safe_depth))
    if min_val < 0.0:
        safe_depth = safe_depth - min_val
    safe_depth = np.maximum(safe_depth, 0.0)
    return robust_normalize(np.log1p(safe_depth))


def odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def compute_gradient(depth_norm: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    return robust_normalize(np.sqrt(gx * gx + gy * gy))


def threshold_gradient(
    gradient: np.ndarray,
    method: str,
    percentile: float,
    closing_kernel: int,
    opening_kernel: int,
) -> np.ndarray:
    grad_u8 = np.clip(gradient * 255.0, 0, 255).astype(np.uint8)
    if method == "otsu":
        _, boundary = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        valid = gradient[np.isfinite(gradient)]
        threshold = float(np.percentile(valid, percentile)) if valid.size else 1.0
        boundary = (gradient >= threshold).astype(np.uint8) * 255

    if closing_kernel > 1:
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, odd_kernel(closing_kernel))
    if opening_kernel > 1:
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, odd_kernel(opening_kernel))
    return boundary > 0


def raw_discontinuity_mask(gradient: np.ndarray, method: str, percentile: float) -> np.ndarray:
    grad_u8 = np.clip(gradient * 255.0, 0, 255).astype(np.uint8)
    if method == "otsu":
        _, mask = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask > 0
    valid = gradient[np.isfinite(gradient)]
    threshold = float(np.percentile(valid, percentile)) if valid.size else 1.0
    return gradient >= threshold


def extract_structural_regions(
    boundary: np.ndarray,
    min_area_ratio: float,
) -> Tuple[np.ndarray, List[Tuple[int, int, Tuple[int, int, int, int], Tuple[float, float]]]]:
    h, w = boundary.shape
    min_area = max(1, int(math.ceil(h * w * min_area_ratio)))
    free_space = (~boundary).astype(np.uint8)
    num_labels, raw_labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=8)

    labels = np.zeros_like(raw_labels, dtype=np.int32)
    region_info: List[Tuple[int, str, int, Tuple[int, int, int, int], Tuple[float, float]]] = []
    next_id = 1
    for raw_id in range(1, num_labels):
        area = int(stats[raw_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[raw_id, cv2.CC_STAT_LEFT])
        y = int(stats[raw_id, cv2.CC_STAT_TOP])
        bw = int(stats[raw_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[raw_id, cv2.CC_STAT_HEIGHT])
        labels[raw_labels == raw_id] = next_id
        bbox = (x, y, x + bw, y + bh)
        centroid = (float(centroids[raw_id][0]), float(centroids[raw_id][1]))
        region_info.append((next_id, "surface_component", area, bbox, centroid))
        next_id += 1
    return labels, region_info


def extract_discontinuity_regions(
    gradient: np.ndarray,
    method: str,
    percentile: float,
    min_area_ratio: float,
    max_area_ratio: float,
    close_kernel: int,
    dilate_kernel: int,
    start_id: int,
) -> Tuple[np.ndarray, List[Tuple[int, str, int, Tuple[int, int, int, int], Tuple[float, float]]]]:
    h, w = gradient.shape
    min_area = max(1, int(math.ceil(h * w * min_area_ratio)))
    max_area = max(min_area, int(math.floor(h * w * max_area_ratio)))
    mask = raw_discontinuity_mask(gradient, method, percentile).astype(np.uint8) * 255
    if close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, odd_kernel(close_kernel))
    if dilate_kernel > 1:
        mask = cv2.dilate(mask, odd_kernel(dilate_kernel), iterations=1)

    num_labels, raw_labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    labels = np.zeros_like(raw_labels, dtype=np.int32)
    region_info: List[Tuple[int, str, int, Tuple[int, int, int, int], Tuple[float, float]]] = []
    next_id = start_id
    for raw_id in range(1, num_labels):
        area = int(stats[raw_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[raw_id, cv2.CC_STAT_LEFT])
        y = int(stats[raw_id, cv2.CC_STAT_TOP])
        bw = int(stats[raw_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[raw_id, cv2.CC_STAT_HEIGHT])
        labels[raw_labels == raw_id] = next_id
        bbox = (x, y, x + bw, y + bh)
        centroid = (float(centroids[raw_id][0]), float(centroids[raw_id][1]))
        region_info.append((next_id, "discontinuity_blob", area, bbox, centroid))
        next_id += 1
    return labels, region_info


def expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    height: int,
    width: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * max(0.0, margin)))
    pad_y = int(round(box_h * max(0.0, margin)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def process_one(meta_path: Path, args: argparse.Namespace) -> ImageResult:
    meta = load_json(meta_path)
    depth_path = Path(meta.get("depth_file", ""))
    if not depth_path.is_absolute():
        depth_path = meta_path.parent / depth_path
    source_image_value = meta.get("source_image")
    source_image = Path(source_image_value) if source_image_value else None
    if source_image is not None and not source_image.is_file():
        source_image = None

    depth_raw = load_depth(depth_path)
    depth_norm = log_depth_normalize(depth_raw)
    gradient = compute_gradient(depth_norm)
    boundary = threshold_gradient(
        gradient,
        method=args.threshold_method,
        percentile=args.gradient_percentile,
        closing_kernel=args.closing_kernel,
        opening_kernel=args.opening_kernel,
    )
    labels, region_info = extract_structural_regions(boundary, args.min_area_ratio)
    if args.include_discontinuity_regions:
        discontinuity_labels, discontinuity_info = extract_discontinuity_regions(
            gradient,
            method=args.threshold_method,
            percentile=args.gradient_percentile,
            min_area_ratio=args.min_discontinuity_area_ratio,
            max_area_ratio=args.max_discontinuity_area_ratio,
            close_kernel=args.discontinuity_close_kernel,
            dilate_kernel=args.discontinuity_dilate_kernel,
            start_id=(max([info[0] for info in region_info], default=0) + 1),
        )
        labels = np.where(discontinuity_labels > 0, discontinuity_labels, labels).astype(np.int32)
        region_info.extend(discontinuity_info)
    h, w = depth_norm.shape
    records = [
        RegionRecord(
            image_id=meta_path.name[: -len(".meta.json")] if meta_path.name.endswith(".meta.json") else meta_path.stem,
            region_id=region_id,
            source=source,
            area_pixels=area,
            area_ratio=float(area) / float(h * w),
            bbox_xyxy=bbox,
            eval_bbox_xyxy=expand_bbox(bbox, h, w, args.roi_margin),
            centroid_xy=centroid,
        )
        for region_id, source, area, bbox, centroid in region_info
    ]
    return ImageResult(
        image_id=records[0].image_id if records else meta_path.name.replace(".meta.json", ""),
        meta_path=meta_path,
        depth_path=depth_path,
        source_image=source_image,
        depth_raw=depth_raw,
        depth_norm=depth_norm,
        gradient=gradient,
        boundary=boundary,
        labels=labels,
        records=records,
    )


def fixed_scale(area_ratio: float) -> str:
    if area_ratio < 0.005:
        return "tiny"
    if area_ratio < 0.02:
        return "small"
    if area_ratio < 0.10:
        return "medium"
    return "large"


def assign_scales(results: Sequence[ImageResult], mode: str) -> Dict[str, Any]:
    ratios = np.asarray([r.area_ratio for item in results for r in item.records], dtype=np.float64)
    if ratios.size == 0:
        thresholds = [0.005, 0.02, 0.10]
        for item in results:
            for record in item.records:
                record.scale = fixed_scale(record.area_ratio)
        return {"mode": mode, "thresholds": thresholds}

    if mode == "quantile":
        thresholds = [float(x) for x in np.quantile(ratios, [0.25, 0.50, 0.75])]
    else:
        thresholds = [0.005, 0.02, 0.10]

    for item in results:
        for record in item.records:
            ratio = record.area_ratio
            if ratio < thresholds[0]:
                record.scale = "tiny"
            elif ratio < thresholds[1]:
                record.scale = "small"
            elif ratio < thresholds[2]:
                record.scale = "medium"
            else:
                record.scale = "large"
    return {"mode": mode, "thresholds": thresholds}


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def label_colors(max_label: int) -> np.ndarray:
    rng = np.random.default_rng(20260609)
    colors = rng.integers(32, 256, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    colors = label_colors(int(labels.max()))
    return colors[labels]


def overlay_labels(base_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = labels_to_rgb(labels)
    mask = labels > 0
    out = base_rgb.copy().astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color[mask].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def scale_exact_masks(item: ImageResult) -> Dict[str, np.ndarray]:
    masks = {}
    for scale in SCALE_NAMES:
        ids = [record.region_id for record in item.records if record.scale == scale]
        masks[scale] = np.isin(item.labels, ids)
    return masks


def scale_roi_masks(item: ImageResult) -> Dict[str, np.ndarray]:
    h, w = item.depth_norm.shape
    masks = {scale: np.zeros((h, w), dtype=bool) for scale in SCALE_NAMES}
    for record in item.records:
        x1, y1, x2, y2 = record.eval_bbox_xyxy
        masks[record.scale][y1:y2, x1:x2] = True
    return masks


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base_rgb.copy().astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_roi_boxes(
    image_rgb: np.ndarray,
    records: Sequence[RegionRecord],
    scale: Optional[str] = None,
    width: int = 2,
) -> np.ndarray:
    colors = {
        "tiny": (129, 114, 178),
        "small": (85, 168, 104),
        "medium": (204, 185, 116),
        "large": (196, 78, 82),
    }
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas)
    for record in records:
        if scale is not None and record.scale != scale:
            continue
        x1, y1, x2, y2 = record.eval_bbox_xyxy
        color = colors[record.scale]
        for inset in range(width):
            left = x1 + inset
            top = y1 + inset
            right = x2 - 1 - inset
            bottom = y2 - 1 - inset
            if right < left or bottom < top:
                break
            draw.rectangle([left, top, right, bottom], outline=color)
    return np.asarray(canvas, dtype=np.uint8)


def load_source_rgb(path: Optional[Path], target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (target_hw[1], target_hw[0]):
            im = im.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_image_result(item: ImageResult, output_dir: Path, args: argparse.Namespace) -> None:
    image_dir = output_dir / "images" / item.image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    h, w = item.depth_norm.shape

    np.save(image_dir / "original_depth.npy", item.depth_raw.astype(np.float32))
    if args.copy_source_depth:
        shutil.copy2(item.depth_path, image_dir / item.depth_path.name)
    np.savez_compressed(image_dir / "structural_region_labels.npz", labels=item.labels.astype(np.int32))

    exact_scale_masks = scale_exact_masks(item)
    roi_scale_masks = scale_roi_masks(item)
    np.savez_compressed(image_dir / "structural_scale_region_masks_exact.npz", **exact_scale_masks)
    np.savez_compressed(image_dir / "scale_region_masks.npz", **roi_scale_masks)
    np.savez_compressed(image_dir / "scale_roi_masks.npz", **roi_scale_masks)

    save_png(image_dir / "depth_log_normalized.png", to_uint8(item.depth_norm))
    save_png(image_dir / "depth_gradient_sobel.png", to_uint8(item.gradient))
    save_png(image_dir / "depth_discontinuities.png", item.boundary.astype(np.uint8) * 255)

    labels_u16 = np.clip(item.labels, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(image_dir / "structural_region_labels.png"), labels_u16)
    save_png(image_dir / "structural_regions_color.png", labels_to_rgb(item.labels))

    depth_rgb = np.repeat(to_uint8(item.depth_norm)[..., None], 3, axis=2)
    save_png(image_dir / "structural_regions_on_depth.png", overlay_labels(depth_rgb, item.labels))
    source_rgb = load_source_rgb(item.source_image, (h, w))
    if source_rgb is not None:
        save_png(image_dir / "structural_regions_on_image.png", overlay_labels(source_rgb, item.labels))
        roi_all = draw_roi_boxes(source_rgb, item.records, scale=None, width=3)
        save_png(image_dir / "scale_rois_on_image.png", roi_all)
        for scale in SCALE_NAMES:
            mask_overlay = overlay_mask(source_rgb, roi_scale_masks[scale], color_for_scale(scale), alpha=0.45)
            mask_overlay = draw_roi_boxes(mask_overlay, item.records, scale=scale, width=3)
            save_png(image_dir / f"{scale}_rois_on_image.png", mask_overlay)

    roi_depth = draw_roi_boxes(depth_rgb, item.records, scale=None, width=3)
    save_png(image_dir / "scale_rois_on_depth.png", roi_depth)

    if args.save_region_pngs:
        region_dir = image_dir / "region_masks"
        region_dir.mkdir(parents=True, exist_ok=True)
        for record in item.records:
            mask = (item.labels == record.region_id).astype(np.uint8) * 255
            save_png(region_dir / f"region_{record.region_id:04d}_{record.scale}.png", mask)

    region_payload = {
        "image_id": item.image_id,
        "meta_path": str(item.meta_path),
        "depth_path": str(item.depth_path),
        "source_image": str(item.source_image) if item.source_image else None,
        "height": h,
        "width": w,
        "region_count": len(item.records),
        "regions": [record_to_json(record) for record in item.records],
    }
    save_json(image_dir / "region_stats.json", region_payload)


def record_to_json(record: RegionRecord) -> Dict[str, Any]:
    return {
        "image_id": record.image_id,
        "region_id": record.region_id,
        "source": record.source,
        "area_pixels": record.area_pixels,
        "area_ratio": record.area_ratio,
        "scale": SCALE_DISPLAY[record.scale],
        "bbox_xyxy": list(record.bbox_xyxy),
        "eval_bbox_xyxy": list(record.eval_bbox_xyxy),
        "centroid_xy": [record.centroid_xy[0], record.centroid_xy[1]],
    }


def write_regions_csv(path: Path, records: Iterable[RegionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "region_id",
                "source",
                "area_pixels",
                "area_ratio",
                "scale",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "eval_x1",
                "eval_y1",
                "eval_x2",
                "eval_y2",
                "centroid_x",
                "centroid_y",
            ],
        )
        writer.writeheader()
        for record in records:
            x1, y1, x2, y2 = record.bbox_xyxy
            ex1, ey1, ex2, ey2 = record.eval_bbox_xyxy
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "region_id": record.region_id,
                    "source": record.source,
                    "area_pixels": record.area_pixels,
                    "area_ratio": f"{record.area_ratio:.10f}",
                    "scale": SCALE_DISPLAY[record.scale],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "eval_x1": ex1,
                    "eval_y1": ey1,
                    "eval_x2": ex2,
                    "eval_y2": ey2,
                    "centroid_x": f"{record.centroid_xy[0]:.3f}",
                    "centroid_y": f"{record.centroid_xy[1]:.3f}",
                }
            )


def summarize(results: Sequence[ImageResult], scale_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    records = [record for item in results for record in item.records]
    by_scale: Dict[str, Dict[str, Any]] = {}
    for scale in SCALE_NAMES:
        scale_records = [record for record in records if record.scale == scale]
        ratios = [record.area_ratio for record in scale_records]
        by_scale[SCALE_DISPLAY[scale]] = {
            "region_count": len(scale_records),
            "mean_area_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "median_area_ratio": float(np.median(ratios)) if ratios else 0.0,
            "mean_area_pixels": float(np.mean([r.area_pixels for r in scale_records])) if scale_records else 0.0,
        }

    all_ratios = [record.area_ratio for record in records]
    return {
        "benchmark": "Depth-derived Structural Scale Benchmark",
        "description": (
            "Regions are derived only from ground-truth depth discontinuities; "
            "they are structural regions, not objects, instances, or semantic regions."
        ),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "num_images": len(results),
        "num_regions": len(records),
        "min_area_ratio": args.min_area_ratio,
        "include_discontinuity_regions": args.include_discontinuity_regions,
        "discontinuity_region_filter": {
            "min_area_ratio": args.min_discontinuity_area_ratio,
            "max_area_ratio": args.max_discontinuity_area_ratio,
            "close_kernel": args.discontinuity_close_kernel,
            "dilate_kernel": args.discontinuity_dilate_kernel,
        },
        "evaluation_unit": "rectangular_roi",
        "roi_margin": args.roi_margin,
        "roi_note": (
            "area_ratio and scale are computed from exact depth-derived structural regions; "
            "metrics should be computed inside eval_bbox_xyxy / scale ROI masks."
        ),
        "threshold_method": args.threshold_method,
        "gradient_percentile": args.gradient_percentile if args.threshold_method == "percentile" else None,
        "morphology": {
            "closing_kernel": args.closing_kernel,
            "opening_kernel": args.opening_kernel,
        },
        "scale_config": scale_config,
        "scale_distribution": by_scale,
        "area_ratio": {
            "mean": float(np.mean(all_ratios)) if all_ratios else 0.0,
            "median": float(np.median(all_ratios)) if all_ratios else 0.0,
            "min": float(np.min(all_ratios)) if all_ratios else 0.0,
            "max": float(np.max(all_ratios)) if all_ratios else 0.0,
        },
    }


def plot_area_histogram(path: Path, records: Sequence[RegionRecord], scale_config: Dict[str, Any]) -> None:
    ratios = np.asarray([record.area_ratio for record in records], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if ratios.size:
        plt.hist(ratios * 100.0, bins=40, color="#4C72B0", alpha=0.85)
        for threshold in scale_config["thresholds"]:
            plt.axvline(threshold * 100.0, color="#C44E52", linestyle="--", linewidth=1)
    plt.xlabel("Structural region area ratio (%)")
    plt.ylabel("Region count")
    plt.title("Depth-derived Structural Region Scale Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_scale_bar(path: Path, summary: Dict[str, Any]) -> None:
    names = list(summary["scale_distribution"].keys())
    counts = [summary["scale_distribution"][name]["region_count"] for name in names]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(names, counts, color=["#8172B2", "#55A868", "#CCB974", "#C44E52"])
    plt.ylabel("Region count")
    plt.title("Structural Regions by Scale")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def color_for_scale(scale: str) -> Tuple[int, int, int]:
    return {
        "tiny": (129, 114, 178),
        "small": (85, 168, 104),
        "medium": (204, 185, 116),
        "large": (196, 78, 82),
    }[scale]


def resize_for_sheet(image: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    return Image.fromarray(image).resize(size, Image.BILINEAR)


def add_label(image: Image.Image, label: str, bg: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, image.width, 22], fill=bg)
    draw.text((5, 3), label, fill=(255, 255, 255), font=font)
    return labeled


def make_roi_contact_sheet(results: Sequence[ImageResult], output_dir: Path, rows: int) -> None:
    if rows <= 0:
        return
    selected = list(results[:rows])
    if not selected:
        return

    thumb = (160, 120)
    columns = ["RGB", "Depth", "Tiny ROI", "Small ROI", "Medium ROI", "Large ROI"]
    sheet = Image.new("RGB", (thumb[0] * len(columns), thumb[1] * len(selected)), color=(255, 255, 255))

    for row_idx, item in enumerate(selected):
        h, w = item.depth_norm.shape
        rgb = load_source_rgb(item.source_image, (h, w))
        if rgb is None:
            rgb = np.repeat(to_uint8(item.depth_norm)[..., None], 3, axis=2)
        depth_rgb = np.repeat(to_uint8(item.depth_norm)[..., None], 3, axis=2)
        roi_masks = scale_roi_masks(item)
        cells = [
            add_label(resize_for_sheet(rgb, thumb), f"{item.image_id} RGB"),
            add_label(resize_for_sheet(depth_rgb, thumb), "Depth"),
        ]
        for scale in SCALE_NAMES:
            over = overlay_mask(rgb, roi_masks[scale], color_for_scale(scale), alpha=0.50)
            over = draw_roi_boxes(over, item.records, scale=scale, width=3)
            cells.append(add_label(resize_for_sheet(over, thumb), f"{SCALE_DISPLAY[scale]} ROI", color_for_scale(scale)))
        for col_idx, cell in enumerate(cells):
            sheet.paste(cell, (col_idx * thumb[0], row_idx * thumb[1]))

    path = output_dir / "roi_scale_contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_manifest(path: Path, results: Sequence[ImageResult], args: argparse.Namespace) -> None:
    payload = {
        "input_dir": str(args.input_dir),
        "selected_meta_files": [str(item.meta_path) for item in results],
        "image_ids": [item.image_id for item in results],
    }
    save_json(path, payload)


def main() -> None:
    args = parse_args()
    meta_files = collect_meta_files(args.input_dir, args.meta_glob, args.offset, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] processing {len(meta_files)} depth samples from {args.input_dir}")
    results = [process_one(meta_path, args) for meta_path in meta_files]
    scale_config = assign_scales(results, args.scale_mode)

    all_records = [record for item in results for record in item.records]
    for item in results:
        save_image_result(item, args.output_dir, args)

    write_regions_csv(args.output_dir / "regions.csv", all_records)
    summary = summarize(results, scale_config, args)
    save_json(args.output_dir / "summary.json", summary)
    write_manifest(args.output_dir / "manifest.json", results, args)
    plot_area_histogram(args.output_dir / "area_ratio_histogram.png", all_records, scale_config)
    plot_scale_bar(args.output_dir / "scale_distribution.png", summary)
    make_roi_contact_sheet(results, args.output_dir, args.visualization_rows)

    print(f"[benchmark] wrote {len(results)} images and {len(all_records)} structural regions")
    print(f"[benchmark] output: {args.output_dir}")


if __name__ == "__main__":
    main()
