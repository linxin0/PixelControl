#!/usr/bin/env python3
"""Evaluate segmentation consistency with SAM2 re-labeling.

For each method, this script takes strict seg-only generated images named
`sa_xxxxxx_seg.png`, runs the same HuggingFace SAM2 mask-generation pipeline
used for the cached labels, and compares the predicted instance label map with
the ground-truth SAM2 condition label map.

Outputs:
  - summary JSON / CSV / Markdown table
  - per-sample JSONL
  - cached predicted SAM2 label maps to make reruns cheap
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    import torch
except Exception:  # pragma: no cover - lets metric-only helpers import without torch.
    torch = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SEG_ONLY_RE = re.compile(r"^(?P<stem>sa_\d+)_seg\.png$")

SAM2_PARAMS = {
    "points_per_batch": 128,
    "points_per_crop": 32,
    "crops_n_layers": 0,
    "crop_overlap_ratio": 0.3413333333333333,
    "crop_n_points_downscale_factor": 1,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "stability_score_offset": 1.0,
    "mask_threshold": 0.0,
    "crops_nms_thresh": 0.7,
}


@dataclass(frozen=True)
class EvalItem:
    method: str
    stem: str
    gen_path: Path
    gt_seg_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen_dirs", nargs="*", default=[], help="Explicit generated dirs to evaluate.")
    parser.add_argument("--names", nargs="*", default=[], help="Names for --gen_dirs, same length.")
    parser.add_argument(
        "--baseline_root",
        default="outputs/baseline_eval",
        help="Root containing baseline_eval/<method>/<sample_set>/ folders.",
    )
    parser.add_argument("--sample_set", default="sa_000201_first2000")
    parser.add_argument(
        "--baseline_methods",
        nargs="*",
        default=["controlnet", "anycontrol", "unicontrolnet", "ctrl_adapter", "pixelponder", "ominicontrol", "relactrl"],
        help="Baseline method folders to include when present.",
    )
    parser.add_argument(
        "--seg_root",
        default="t2i/data/blip_sam2_large_extracted/sa_000201",
        help="Ground-truth SAM2 label root. Usually contains {stem}.sam2_label.npy.",
    )
    parser.add_argument(
        "--sam2_model_dir",
        default="t2i/pretrained/sam2.1-hiera-large",
        help="Local SAM2.1-Hiera-Large model directory used by transformers pipeline.",
    )
    parser.add_argument("--output_root", default="outputs/seg_consistency_sam2")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, cpu, or pipeline integer device.")
    parser.add_argument("--min_samples", type=int, default=1000, help="Skip a method if matched strict seg images are fewer.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional cap per method for debugging.")
    parser.add_argument("--force_sam2", action="store_true", help="Regenerate predicted SAM2 labels even if cache exists.")
    parser.add_argument(
        "--sam2_batch_size",
        type=int,
        default=4,
        help="Number of generated images sent to the SAM2 pipeline at once on one GPU.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan gen dirs if no manifest is available.")
    parser.add_argument("--boundary_tolerance", type=int, default=2, help="Pixel tolerance for boundary F1.")
    parser.add_argument("--include_background", action="store_true", help="Include label 0 in mIoU/mAcc matching.")
    parser.add_argument("--no_cluster_metrics", action="store_true", help="Disable ARI/NMI if runtime is too high.")
    parser.add_argument("--sam2_only", action="store_true", help="Only generate cached SAM2 pred label npy files; skip metrics.")
    parser.add_argument("--dry_run", action="store_true", help="Only collect inputs and write a manifest; do not load SAM2.")
    parser.add_argument("--progress_every", type=int, default=25)
    for key, value in SAM2_PARAMS.items():
        arg_type = int if isinstance(value, int) else float
        parser.add_argument(f"--{key}", type=arg_type, default=value)
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


def find_gt_seg(seg_root: Path, stem: str) -> Path | None:
    candidates = (
        seg_root / f"{stem}.sam2_label.npy",
        seg_root / f"{stem}.sam2_label.png",
        seg_root / f"{stem}.npy",
        seg_root / f"{stem}.png",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_from_manifest(method: str, sample_dir: Path, seg_root: Path) -> list[EvalItem] | None:
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    items: list[EvalItem] = []
    for row in load_manifest(manifest_path):
        mode = row.get("control_mode")
        output_path = row.get("output_path")
        if mode != "seg" or not output_path:
            continue
        gen_path = Path(output_path)
        match = SEG_ONLY_RE.match(gen_path.name)
        if match is None or not gen_path.exists():
            continue
        stem = match.group("stem")
        gt_raw = row.get("seg_path")
        gt_path = Path(gt_raw) if gt_raw else find_gt_seg(seg_root, stem)
        if gt_path is not None and not gt_path.exists():
            gt_path = find_gt_seg(seg_root, stem)
        items.append(EvalItem(method=method, stem=stem, gen_path=gen_path, gt_seg_path=gt_path))
    return items


def collect_by_scan(method: str, gen_dir: Path, seg_root: Path, recursive: bool) -> list[EvalItem]:
    iterator = gen_dir.rglob("*.png") if recursive else gen_dir.iterdir()
    items: list[EvalItem] = []
    for path in sorted(iterator):
        if not path.is_file():
            continue
        match = SEG_ONLY_RE.match(path.name)
        if match is None:
            continue
        stem = match.group("stem")
        items.append(EvalItem(method=method, stem=stem, gen_path=path, gt_seg_path=find_gt_seg(seg_root, stem)))
    return items


def collect_inputs(args: argparse.Namespace) -> dict[str, list[EvalItem]]:
    seg_root = Path(args.seg_root)
    methods: list[tuple[str, Path]] = []
    if args.names and len(args.names) != len(args.gen_dirs):
        raise ValueError("--names length must match --gen_dirs length")
    for i, raw in enumerate(args.gen_dirs):
        gen_dir = Path(raw).expanduser().resolve()
        name = args.names[i] if args.names else infer_name(gen_dir)
        methods.append((name, gen_dir))
    baseline_root = Path(args.baseline_root)
    for method in args.baseline_methods:
        sample_dir = baseline_root / method / args.sample_set
        if sample_dir.is_dir():
            methods.append((method, sample_dir.resolve()))

    out: dict[str, list[EvalItem]] = {}
    used_names: set[str] = set()
    for raw_name, gen_dir in methods:
        name = safe_name(raw_name)
        base = name
        k = 1
        while name in used_names:
            k += 1
            name = f"{base}_{k}"
        used_names.add(name)
        if not gen_dir.is_dir():
            print(f"[skip] {name}: not a directory: {gen_dir}")
            continue
        items = collect_from_manifest(name, gen_dir, seg_root)
        if items is None:
            items = collect_by_scan(name, gen_dir, seg_root, recursive=args.recursive)
        if args.max_samples and args.max_samples > 0:
            items = items[: args.max_samples]
        missing_gt = sum(1 for item in items if item.gt_seg_path is None)
        if len(items) < args.min_samples:
            print(f"[skip] {name}: only {len(items)} strict seg images (< min_samples={args.min_samples})")
            continue
        if missing_gt:
            print(f"[warn] {name}: {missing_gt}/{len(items)} images have no GT seg label and will be skipped")
        out[name] = items
        print(f"[collect] {name}: {len(items)} strict seg images from {gen_dir}")
    if not out:
        raise RuntimeError("No methods to evaluate after filtering. Lower --min_samples or check paths.")
    return out


def pipeline_device(raw: str) -> int | str:
    text = str(raw).strip().lower()
    if text == "cpu":
        return -1
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    try:
        return int(text)
    except ValueError:
        return raw


def build_generator(args: argparse.Namespace):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from transformers import pipeline

    device = pipeline_device(args.device)
    print(f"[sam2] loading model={args.sam2_model_dir} device={args.device} pipeline_device={device}", flush=True)
    return pipeline("mask-generation", model=str(args.sam2_model_dir), device=device)


def tensor_or_array_to_bool(mask: Any) -> np.ndarray:
    if torch is not None and isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(bool)


def score_to_float(score: Any) -> float:
    if torch is not None and isinstance(score, torch.Tensor):
        return float(score.detach().cpu().item())
    return float(np.asarray(score).item())


def sam2_outputs_to_label(outputs: dict[str, Any], height: int, width: int) -> np.ndarray:
    stats: list[tuple[int, float, np.ndarray]] = []
    for mask, score in zip(outputs["masks"], outputs["scores"]):
        mask_np = tensor_or_array_to_bool(mask)
        area = int(mask_np.sum())
        if area > 0:
            stats.append((area, score_to_float(score), mask_np))
    stats.sort(key=lambda item: (-item[0], -item[1]))
    label_ids = np.zeros((height, width), dtype=np.uint16)
    for label_id, (_, _, mask_np) in enumerate(stats, start=1):
        label_ids[mask_np] = label_id
    return label_ids


def sam2_label_image(generator, image_path: Path, params: dict[str, Any]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    outputs = generator(image, **params)
    return sam2_outputs_to_label(outputs, image.height, image.width)


def sam2_label_batch(generator, image_paths: list[Path], params: dict[str, Any], batch_size: int) -> list[np.ndarray]:
    if int(batch_size) <= 1:
        return [sam2_label_image(generator, path, params) for path in image_paths]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        outputs = generator(images, batch_size=max(1, int(batch_size)), **params)
    except Exception as exc:
        print(f"[warn] SAM2 batch call failed ({exc!r}); falling back to per-image calls.", flush=True)
        return [sam2_label_image(generator, path, params) for path in image_paths]
    if isinstance(outputs, dict):
        outputs_list = [outputs]
    else:
        outputs_list = list(outputs)
    if len(outputs_list) != len(images):
        print(
            f"[warn] SAM2 batch returned {len(outputs_list)} outputs for {len(images)} images; "
            "falling back to per-image calls.",
            flush=True,
        )
        return [sam2_label_image(generator, path, params) for path in image_paths]
    return [
        sam2_outputs_to_label(output, image.height, image.width)
        for output, image in zip(outputs_list, images)
    ]


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def load_label(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            unique, inverse = np.unique(flat, axis=0, return_inverse=True)
            order = np.lexsort(unique.T[::-1])
            remap = np.zeros(len(unique), dtype=np.int64)
            remap[order] = np.arange(len(unique), dtype=np.int64)
            arr = remap[inverse].reshape(arr.shape[:2])
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label map from {path}, got shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


def resize_label_nearest(label: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if tuple(label.shape) == tuple(shape_hw):
        return label
    img = Image.fromarray(label.astype(np.int32), mode="I")
    resized = img.resize((shape_hw[1], shape_hw[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized).astype(np.int64, copy=False)


def matched_prediction(pred: np.ndarray, gt: np.ndarray, include_background: bool) -> tuple[np.ndarray, dict[str, Any]]:
    pred_labels = np.unique(pred)
    gt_labels = np.unique(gt)
    if not include_background:
        pred_labels = pred_labels[pred_labels != 0]
        gt_labels = gt_labels[gt_labels != 0]
    mapped = np.zeros_like(pred, dtype=np.int64)
    if len(pred_labels) == 0 or len(gt_labels) == 0:
        return mapped, {"matched_pairs": [], "n_pred_labels": int(len(pred_labels)), "n_gt_labels": int(len(gt_labels))}

    pred_index = {int(label): i for i, label in enumerate(pred_labels)}
    gt_index = {int(label): i for i, label in enumerate(gt_labels)}
    inter = np.zeros((len(pred_labels), len(gt_labels)), dtype=np.int64)
    mask = np.isin(pred, pred_labels) & np.isin(gt, gt_labels)
    if mask.any():
        pred_flat = pred[mask].ravel()
        gt_flat = gt[mask].ravel()
        for p, g in zip(pred_flat, gt_flat):
            inter[pred_index[int(p)], gt_index[int(g)]] += 1
    row_ind, col_ind = linear_sum_assignment(-inter)
    pairs: list[dict[str, int]] = []
    for r, c in zip(row_ind, col_ind):
        if inter[r, c] <= 0:
            continue
        pred_label = int(pred_labels[r])
        gt_label = int(gt_labels[c])
        mapped[pred == pred_label] = gt_label
        pairs.append({"pred": pred_label, "gt": gt_label, "intersection": int(inter[r, c])})
    return mapped, {"matched_pairs": pairs, "n_pred_labels": int(len(pred_labels)), "n_gt_labels": int(len(gt_labels))}


def segmentation_scores(mapped_pred: np.ndarray, gt: np.ndarray, include_background: bool) -> dict[str, float]:
    labels = np.unique(gt)
    if not include_background:
        labels = labels[labels != 0]
    if len(labels) == 0:
        return {"miou": math.nan, "pixel_acc": math.nan, "macc": math.nan}
    ious = []
    accs = []
    correct = mapped_pred == gt
    for label in labels:
        gt_mask = gt == label
        pred_mask = mapped_pred == label
        tp = np.logical_and(gt_mask, pred_mask).sum(dtype=np.float64)
        union = np.logical_or(gt_mask, pred_mask).sum(dtype=np.float64)
        gt_area = gt_mask.sum(dtype=np.float64)
        ious.append(float(tp / union) if union > 0 else math.nan)
        accs.append(float(tp / gt_area) if gt_area > 0 else math.nan)
    return {
        "miou": float(np.nanmean(ious)),
        "pixel_acc": float(correct.sum(dtype=np.float64) / correct.size),
        "macc": float(np.nanmean(accs)),
    }


def boundary_map(label: np.ndarray) -> np.ndarray:
    b = np.zeros(label.shape, dtype=bool)
    b[:-1, :] |= label[:-1, :] != label[1:, :]
    b[1:, :] |= label[1:, :] != label[:-1, :]
    b[:, :-1] |= label[:, :-1] != label[:, 1:]
    b[:, 1:] |= label[:, 1:] != label[:, :-1]
    return b


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int) -> float:
    pred_b = boundary_map(pred)
    gt_b = boundary_map(gt)
    n_pred = int(pred_b.sum())
    n_gt = int(gt_b.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    tol = max(0, int(tolerance))
    dist_to_gt = distance_transform_edt(~gt_b)
    dist_to_pred = distance_transform_edt(~pred_b)
    precision = float((dist_to_gt[pred_b] <= tol).sum() / max(n_pred, 1))
    recall = float((dist_to_pred[gt_b] <= tol).sum() / max(n_gt, 1))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def cluster_scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    return {
        "ari": float(adjusted_rand_score(gt_flat, pred_flat)),
        "nmi": float(normalized_mutual_info_score(gt_flat, pred_flat)),
    }


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
    headers = ["method", "n", "mIoU", "pixel_acc", "mAcc", "boundary_f1", "ARI", "NMI"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            row["method"],
            str(row["n"]),
            format_float(row["miou"]),
            format_float(row["pixel_acc"]),
            format_float(row["macc"]),
            format_float(row["boundary_f1"]),
            format_float(row.get("ari", math.nan)),
            format_float(row.get("nmi", math.nan)),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pred_label_path(pred_label_root: Path, method: str, stem: str) -> Path:
    return pred_label_root / method / f"{stem}.sam2_pred_label.npy"


def ensure_sam2_labels(
    method: str,
    items: list[EvalItem],
    generator,
    args: argparse.Namespace,
    params: dict[str, Any],
    pred_label_root: Path,
) -> None:
    missing = [
        item
        for item in items
        if item.gt_seg_path is not None
        and item.gt_seg_path.exists()
        and (args.force_sam2 or not pred_label_path(pred_label_root, method, item.stem).exists())
    ]
    if not missing:
        print(f"[sam2-cache] {method}: all labels already exist under {pred_label_root / method}", flush=True)
        return
    batch_size = max(1, int(args.sam2_batch_size))
    print(
        f"[sam2-cache] {method}: generating {len(missing)} labels "
        f"batch_size={batch_size} -> {pred_label_root / method}",
        flush=True,
    )
    iterator = range(0, len(missing), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"{method}:sam2", total=math.ceil(len(missing) / batch_size))
    for start in iterator:
        batch = missing[start : start + batch_size]
        labels = sam2_label_batch(generator, [item.gen_path for item in batch], params, batch_size=batch_size)
        for item, label in zip(batch, labels):
            atomic_save_npy(pred_label_path(pred_label_root, method, item.stem), label)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_method(
    method: str,
    items: list[EvalItem],
    args: argparse.Namespace,
    pred_label_root: Path,
    per_sample_file,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    iterator = tqdm(items, desc=f"{method}:metrics") if tqdm is not None else items
    t_start = time.time()
    for idx, item in enumerate(iterator, start=1):
        if item.gt_seg_path is None or not item.gt_seg_path.exists():
            continue
        cache_path = pred_label_path(pred_label_root, method, item.stem)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached SAM2 prediction: {cache_path}")
        pred = np.load(cache_path).astype(np.int64, copy=False)
        gt = load_label(item.gt_seg_path)
        pred = resize_label_nearest(pred, gt.shape)
        mapped, match_info = matched_prediction(pred, gt, include_background=args.include_background)
        scores = segmentation_scores(mapped, gt, include_background=args.include_background)
        scores["boundary_f1"] = boundary_f1(mapped, gt, tolerance=args.boundary_tolerance)
        if not args.no_cluster_metrics:
            scores.update(cluster_scores(pred, gt))
        else:
            scores.update({"ari": math.nan, "nmi": math.nan})
        record = {
            "method": method,
            "stem": item.stem,
            "gen_path": str(item.gen_path),
            "gt_seg_path": str(item.gt_seg_path),
            **scores,
            "n_pred_labels": match_info["n_pred_labels"],
            "n_gt_labels": match_info["n_gt_labels"],
            "n_matched_labels": len(match_info["matched_pairs"]),
        }
        records.append(record)
        per_sample_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        if args.progress_every > 0 and idx % args.progress_every == 0:
            per_sample_file.flush()
        if torch is not None and torch.cuda.is_available() and idx % 25 == 0:
            torch.cuda.empty_cache()
    elapsed = time.time() - t_start
    if not records:
        raise RuntimeError(f"{method}: no evaluable records with GT labels")
    summary = {
        "method": method,
        "n": len(records),
        "miou": mean_or_nan([r["miou"] for r in records]),
        "pixel_acc": mean_or_nan([r["pixel_acc"] for r in records]),
        "macc": mean_or_nan([r["macc"] for r in records]),
        "boundary_f1": mean_or_nan([r["boundary_f1"] for r in records]),
        "ari": mean_or_nan([r["ari"] for r in records]),
        "nmi": mean_or_nan([r["nmi"] for r in records]),
        "elapsed_sec": elapsed,
    }
    print(
        f"[done] {method}: n={summary['n']} "
        f"mIoU={summary['miou']:.4f} pixel_acc={summary['pixel_acc']:.4f} "
        f"mAcc={summary['macc']:.4f} boundary_f1={summary['boundary_f1']:.4f}",
        flush=True,
    )
    return summary


def write_outputs(output_root: Path, args: argparse.Namespace, summaries: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = sorted(summaries, key=lambda row: (-(row["miou"] if np.isfinite(row["miou"]) else -1.0), row["method"]))
    payload = {"args": vars(args), "manifest": manifest, "summary": summaries}
    json_path = output_root / "seg_consistency_sam2_summary.json"
    csv_path = output_root / "seg_consistency_sam2_summary.csv"
    md_path = output_root / "seg_consistency_sam2_summary.md"
    manifest_path = output_root / "seg_consistency_sam2_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "n", "miou", "pixel_acc", "macc", "boundary_f1", "ari", "nmi", "elapsed_sec"]
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
    pred_label_root = output_root / "sam2_pred_labels"
    methods = collect_inputs(args)
    params = {key: getattr(args, key) for key in SAM2_PARAMS}
    manifest = {
        "sam2_model_dir": args.sam2_model_dir,
        "sam2_params": params,
        "methods": {
            method: {
                "n_collected": len(items),
                "examples": [
                    {
                        "stem": item.stem,
                        "gen_path": str(item.gen_path),
                        "gt_seg_path": str(item.gt_seg_path) if item.gt_seg_path else None,
                    }
                    for item in items[:5]
                ],
            }
            for method, items in methods.items()
        },
    }
    if args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "seg_consistency_sam2_manifest.dry_run.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[dry_run] manifest={manifest_path}")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    generator = build_generator(args)
    for method, items in methods.items():
        ensure_sam2_labels(method, items, generator, args, params, pred_label_root)
    if args.sam2_only:
        manifest_path = output_root / "seg_consistency_sam2_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[sam2_only] labels saved under {pred_label_root}")
        print(f"[sam2_only] manifest={manifest_path}")
        return
    per_sample_path = output_root / "seg_consistency_sam2_per_sample.jsonl"
    summaries: list[dict[str, Any]] = []
    with per_sample_path.open("w", encoding="utf-8") as per_sample_file:
        for method, items in methods.items():
            summaries.append(evaluate_method(method, items, args, pred_label_root, per_sample_file))
    print(f"[output] per_sample={per_sample_path}")
    write_outputs(output_root, args, summaries, manifest)


if __name__ == "__main__":
    main()
