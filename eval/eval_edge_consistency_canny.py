#!/usr/bin/env python3
"""Evaluate edge-control consistency with blurred Canny edges.

This intentionally does NOT compare generated images against the input edge
condition. Instead, both the generated RGB image and the original GT RGB image
are processed with the same deterministic edge extractor:

  RGB -> grayscale -> GaussianBlur(k=11) -> Canny(70, 150)

The default is aligned with the user's edge condition recipe while avoiding the
random-threshold condition map as a direct ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


EDGE_ONLY_RE = re.compile(r"^(?P<stem>sa_\d+)_edge\.png$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class EvalItem:
    method: str
    stem: str
    gen_path: Path
    gt_rgb_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen_dirs", nargs="*", default=[], help="Explicit generated dirs to evaluate.")
    parser.add_argument("--names", nargs="*", default=[], help="Names for --gen_dirs, same length.")
    parser.add_argument("--baseline_root", default="outputs/baseline_eval")
    parser.add_argument("--sample_set", default="sa_000201_first2000")
    parser.add_argument(
        "--baseline_methods",
        nargs="*",
        default=["controlnet", "anycontrol", "unicontrolnet", "ctrl_adapter", "pixelponder", "ominicontrol", "relactrl"],
    )
    parser.add_argument("--image_root", default="t2i/data/blip/extracted_new/sa_000201")
    parser.add_argument("--output_root", default="outputs/edge_consistency_canny")
    parser.add_argument("--min_samples", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan gen dirs if no manifest is available.")
    parser.add_argument("--blur_kernel", type=int, default=11)
    parser.add_argument("--canny_low", type=int, default=70)
    parser.add_argument("--canny_high", type=int, default=150)
    parser.add_argument(
        "--ensemble_thresholds",
        default="",
        help=(
            "Optional semicolon list like '50,110;60,130;70,150;80,170;90,190'. "
            "If empty, uses only --canny_low/--canny_high."
        ),
    )
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--save_edge_maps", action="store_true", default=True)
    parser.add_argument("--no_save_edge_maps", action="store_false", dest="save_edge_maps")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def safe_name(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return text.strip("._") or "method"


def infer_name(path: Path) -> str:
    parts = [p for p in path.parts if p]
    if len(parts) >= 3 and parts[-2].startswith("all_modes_eval"):
        return f"{parts[-4]}_{parts[-2]}_{parts[-1]}"
    if len(parts) >= 3 and parts[-2] == "val":
        return f"{parts[-3]}_{parts[-1]}"
    return path.name


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "samples", "manifest"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported manifest format: {path}")


def find_gt_rgb(image_root: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def collect_from_manifest(method: str, sample_dir: Path, image_root: Path) -> list[EvalItem] | None:
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    items: list[EvalItem] = []
    for row in load_manifest(manifest_path):
        if row.get("control_mode") != "edge":
            continue
        output_path = row.get("output_path")
        if not output_path:
            continue
        gen_path = Path(output_path)
        match = EDGE_ONLY_RE.match(gen_path.name)
        if match is None or not gen_path.exists():
            continue
        stem = match.group("stem")
        gt_raw = row.get("rgb_path")
        gt_path = Path(gt_raw) if gt_raw else find_gt_rgb(image_root, stem)
        if gt_path is not None and not gt_path.exists():
            gt_path = find_gt_rgb(image_root, stem)
        items.append(EvalItem(method=method, stem=stem, gen_path=gen_path, gt_rgb_path=gt_path))
    return items


def collect_by_scan(method: str, gen_dir: Path, image_root: Path, recursive: bool) -> list[EvalItem]:
    iterator = gen_dir.rglob("*.png") if recursive else gen_dir.iterdir()
    items: list[EvalItem] = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        match = EDGE_ONLY_RE.match(path.name)
        if match is None:
            continue
        stem = match.group("stem")
        items.append(EvalItem(method=method, stem=stem, gen_path=path, gt_rgb_path=find_gt_rgb(image_root, stem)))
    return items


def collect_inputs(args: argparse.Namespace) -> dict[str, list[EvalItem]]:
    image_root = Path(args.image_root)
    methods: list[tuple[str, Path]] = []
    if args.names and len(args.names) != len(args.gen_dirs):
        raise ValueError("--names length must match --gen_dirs length")
    for i, raw in enumerate(args.gen_dirs):
        gen_dir = Path(raw).expanduser().resolve()
        methods.append((args.names[i] if args.names else infer_name(gen_dir), gen_dir))
    baseline_root = Path(args.baseline_root)
    for method in args.baseline_methods:
        sample_dir = baseline_root / method / args.sample_set
        if sample_dir.is_dir():
            methods.append((method, sample_dir.resolve()))

    out: dict[str, list[EvalItem]] = {}
    used: set[str] = set()
    for raw_name, gen_dir in methods:
        name = safe_name(raw_name)
        base = name
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        if not gen_dir.is_dir():
            print(f"[skip] {name}: not a directory: {gen_dir}")
            continue
        items = collect_from_manifest(name, gen_dir, image_root)
        if items is None:
            items = collect_by_scan(name, gen_dir, image_root, recursive=args.recursive)
        if args.max_samples and args.max_samples > 0:
            items = items[: args.max_samples]
        missing_gt = sum(1 for item in items if item.gt_rgb_path is None)
        if len(items) < args.min_samples:
            print(f"[skip] {name}: only {len(items)} strict edge images (< min_samples={args.min_samples})")
            continue
        if missing_gt:
            print(f"[warn] {name}: {missing_gt}/{len(items)} images have no GT RGB and will be skipped")
        out[name] = items
        print(f"[collect] {name}: {len(items)} strict edge images from {gen_dir}")
    if not out:
        raise RuntimeError("No methods to evaluate after filtering. Lower --min_samples or check paths.")
    return out


def parse_thresholds(args: argparse.Namespace) -> list[tuple[int, int]]:
    if not args.ensemble_thresholds.strip():
        return [(int(args.canny_low), int(args.canny_high))]
    pairs: list[tuple[int, int]] = []
    for raw_pair in args.ensemble_thresholds.split(";"):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        low, high = raw_pair.split(",", 1)
        pairs.append((int(low), int(high)))
    if not pairs:
        raise ValueError("--ensemble_thresholds did not contain any valid low,high pairs")
    return pairs


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def blurred_canny_soft(rgb: np.ndarray, blur_kernel: int, thresholds: list[tuple[int, int]]) -> np.ndarray:
    kernel = int(blur_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"blur_kernel must be a positive odd integer, got {blur_kernel}")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = []
    for low, high in thresholds:
        edge = cv2.Canny(gray, int(low), int(high)).astype(np.float32) / 255.0
        edges.append(edge)
    return np.mean(edges, axis=0).astype(np.float32)


def resize_soft(edge: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if tuple(edge.shape) == tuple(shape_hw):
        return edge
    return cv2.resize(edge, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def binarize(edge: np.ndarray) -> np.ndarray:
    return edge >= 0.5


def tolerance_precision_recall_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int) -> tuple[float, float, float]:
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    if pred_n == 0 and gt_n == 0:
        return 1.0, 1.0, 1.0
    if pred_n == 0:
        return 0.0, 0.0, 0.0
    if gt_n == 0:
        return 0.0, 0.0, 0.0
    tol = max(0, int(tolerance))
    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)
    precision = float((dist_to_gt[pred] <= tol).sum() / pred_n)
    recall = float((dist_to_pred[gt] <= tol).sum() / gt_n)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return math.nan
    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)
    return float(0.5 * (dist_to_gt[pred].mean() + dist_to_pred[gt].mean()))


def compute_metrics(pred_soft: np.ndarray, gt_soft: np.ndarray, tolerance: int) -> dict[str, float]:
    pred_bin = binarize(pred_soft)
    gt_bin = binarize(gt_soft)
    precision, recall, f1 = tolerance_precision_recall_f1(pred_bin, gt_bin, tolerance=tolerance)
    intersection = np.minimum(pred_soft, gt_soft).sum(dtype=np.float64)
    union = np.maximum(pred_soft, gt_soft).sum(dtype=np.float64)
    pred_density = float(pred_bin.mean())
    gt_density = float(gt_bin.mean())
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "soft_iou": float(intersection / union) if union > 0 else math.nan,
        "edge_mae": float(np.abs(pred_soft - gt_soft).mean()),
        "chamfer": chamfer_distance(pred_bin, gt_bin),
        "pred_density": pred_density,
        "gt_density": gt_density,
        "density_ratio": float(pred_density / gt_density) if gt_density > 0 else math.nan,
    }


def save_gray(path: Path, edge: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(edge, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def mean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def format_float(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["method", "n", "edge_f1", "precision", "recall", "chamfer", "soft_iou", "edge_mae", "density_ratio"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row["method"],
            str(row["n"]),
            format_float(row["edge_f1"]),
            format_float(row["edge_precision"]),
            format_float(row["edge_recall"]),
            format_float(row["chamfer"]),
            format_float(row["soft_iou"]),
            format_float(row["edge_mae"]),
            format_float(row["density_ratio"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def evaluate_method(
    method: str,
    items: list[EvalItem],
    args: argparse.Namespace,
    thresholds: list[tuple[int, int]],
    edge_root: Path,
    per_sample_file,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=method) if tqdm is not None else items
    for item in iterator:
        if item.gt_rgb_path is None or not item.gt_rgb_path.exists():
            continue
        pred_soft = blurred_canny_soft(read_rgb(item.gen_path), args.blur_kernel, thresholds)
        gt_soft = blurred_canny_soft(read_rgb(item.gt_rgb_path), args.blur_kernel, thresholds)
        pred_soft = resize_soft(pred_soft, gt_soft.shape)
        if args.save_edge_maps:
            save_gray(edge_root / method / "pred" / f"{item.stem}.edge.png", pred_soft)
            save_gray(edge_root / method / "gt" / f"{item.stem}.edge.png", gt_soft)
        scores = compute_metrics(pred_soft, gt_soft, tolerance=args.boundary_tolerance)
        record = {
            "method": method,
            "stem": item.stem,
            "gen_path": str(item.gen_path),
            "gt_rgb_path": str(item.gt_rgb_path),
            **scores,
        }
        records.append(record)
        per_sample_file.write(json.dumps(record, ensure_ascii=True) + "\n")
    if not records:
        raise RuntimeError(f"{method}: no evaluable records with GT RGB")
    summary = {"method": method, "n": len(records)}
    for key in ("edge_precision", "edge_recall", "edge_f1", "soft_iou", "edge_mae", "chamfer", "pred_density", "gt_density", "density_ratio"):
        summary[key] = mean_or_nan([r[key] for r in records])
    print(
        f"[done] {method}: n={summary['n']} edge_f1={summary['edge_f1']:.4f} "
        f"chamfer={summary['chamfer']:.4f} density_ratio={summary['density_ratio']:.4f}",
        flush=True,
    )
    return summary


def write_outputs(output_root: Path, args: argparse.Namespace, thresholds: list[tuple[int, int]], summaries: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    summaries = sorted(summaries, key=lambda row: (-(row["edge_f1"] if np.isfinite(row["edge_f1"]) else -1.0), row["method"]))
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "edge_consistency_canny_summary.json"
    csv_path = output_root / "edge_consistency_canny_summary.csv"
    md_path = output_root / "edge_consistency_canny_summary.md"
    manifest_path = output_root / "edge_consistency_canny_manifest.json"
    payload = {"args": vars(args), "thresholds": thresholds, "manifest": manifest, "summary": summaries}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "n",
            "edge_f1",
            "edge_precision",
            "edge_recall",
            "chamfer",
            "soft_iou",
            "edge_mae",
            "pred_density",
            "gt_density",
            "density_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})
    md_path.write_text(markdown_table(summaries), encoding="utf-8")
    print(f"[output] json={json_path}")
    print(f"[output] csv={csv_path}")
    print(f"[output] md={md_path}")
    print(f"[output] manifest={manifest_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    methods = collect_inputs(args)
    thresholds = parse_thresholds(args)
    manifest = {
        "edge_extractor": "RGB -> gray -> GaussianBlur -> Canny",
        "blur_kernel": args.blur_kernel,
        "thresholds": thresholds,
        "methods": {
            method: {
                "n_collected": len(items),
                "examples": [
                    {
                        "stem": item.stem,
                        "gen_path": str(item.gen_path),
                        "gt_rgb_path": str(item.gt_rgb_path) if item.gt_rgb_path else None,
                    }
                    for item in items[:5]
                ],
            }
            for method, items in methods.items()
        },
    }
    if args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        dry_path = output_root / "edge_consistency_canny_manifest.dry_run.json"
        dry_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[dry_run] manifest={dry_path}")
        return
    edge_root = output_root / "extracted_edges_k11_canny70_150"
    per_sample_path = output_root / "edge_consistency_canny_per_sample.jsonl"
    summaries: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with per_sample_path.open("w", encoding="utf-8") as per_sample_file:
        for method, items in methods.items():
            summaries.append(evaluate_method(method, items, args, thresholds, edge_root, per_sample_file))
    print(f"[output] per_sample={per_sample_path}")
    if args.save_edge_maps:
        print(f"[output] extracted_edges={edge_root}")
    write_outputs(output_root, args, thresholds, summaries, manifest)


if __name__ == "__main__":
    main()
