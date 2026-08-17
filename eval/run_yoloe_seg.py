#!/usr/bin/env python3
"""Run Ultralytics YOLOE-26x-seg on the first aligned SA-1B samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE


DEFAULT_META_DIR = Path("t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201")
DEFAULT_OUTPUT_DIR = Path("outputs/yoloe_segmentation/sa_000201_first50")
DEFAULT_MODEL = "yoloe-26x-seg.pt"
DEFAULT_LABELS = (
    "person, people, child, man, woman, animal, bird, dog, cat, horse, "
    "car, bus, truck, bicycle, motorcycle, boat, train, airplane, building, tower, "
    "house, castle, fortress, stone structure, structure, bridge, road, sidewalk, path, street, "
    "wall, fence, door, window, sign, "
    "traffic light, pole, bench, chair, table, umbrella, bag, backpack, suitcase, "
    "tree, bush, vegetation, plant, flower, grass, lawn, field, terrain, dirt, landscape, "
    "mountain, rock, water, sky"
)

CAPTION_LABEL_ALIASES = {
    "person": ("person", "people", "pedestrian", "pedestrians", "man", "woman", "child", "tourist", "tourists"),
    "people": ("people", "crowd", "group of people", "tourists", "pedestrians"),
    "building": ("building", "buildings", "architecture", "facade", "structure"),
    "structure": ("structure", "structures"),
    "stone structure": ("stone structure", "weathered stone", "stone"),
    "castle": ("castle",),
    "fortress": ("fortress",),
    "tower": ("tower", "leaning tower"),
    "cathedral": ("cathedral", "duomo", "church"),
    "lawn": ("lawn", "grass", "green"),
    "tree": ("tree", "trees"),
    "bush": ("bush", "bushes"),
    "vegetation": ("vegetation",),
    "sky": ("sky", "cloud", "clouds"),
    "road": ("road", "street", "lane"),
    "sidewalk": ("sidewalk", "pavement", "walkway"),
    "path": ("path", "trail"),
    "wall": ("wall", "fence"),
    "window": ("window", "windows"),
    "door": ("door", "entrance"),
    "sign": ("sign", "signage", "billboard"),
    "car": ("car", "cars", "vehicle", "vehicles"),
    "bus": ("bus", "buses"),
    "truck": ("truck", "trucks"),
    "bicycle": ("bicycle", "bike", "bikes"),
    "motorcycle": ("motorcycle", "motorbike"),
    "boat": ("boat", "ship"),
    "bench": ("bench", "benches"),
    "chair": ("chair", "chairs"),
    "table": ("table", "tables"),
    "umbrella": ("umbrella", "umbrellas"),
    "bag": ("bag", "backpack", "suitcase"),
    "flower": ("flower", "flowers"),
    "field": ("field", "open field"),
    "terrain": ("terrain", "rugged", "uneven"),
    "dirt": ("dirt",),
    "landscape": ("landscape",),
    "rock": ("rock", "rocks", "rocky", "stone"),
    "water": ("water", "river", "lake", "sea", "ocean"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOE-26x-seg on the first N aligned RGB images.")
    parser.add_argument("--meta_dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output_prefix", default="yoloe_26x_seg")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument(
        "--prompt_mode",
        choices=("caption_labels", "caption_plus_defaults", "all_labels"),
        default="caption_plus_defaults",
        help=(
            "caption_labels uses only labels extracted from each caption; "
            "caption_plus_defaults appends the default label bank; all_labels ignores captions."
        ),
    )
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--contact_sheet_cols", type=int, default=5)
    parser.add_argument("--contact_thumb_width", type=int, default=260)
    parser.add_argument(
        "--contact_sheet_page_size",
        type=int,
        default=200,
        help="Write one contact sheet per this many images. Use <=0 for a single full sheet.",
    )
    parser.add_argument("--quiet_labels", action="store_true", help="Do not print the full prompt label list per image.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_labels(labels: str) -> List[str]:
    parsed = [label.strip() for label in labels.split(",") if label.strip()]
    if not parsed:
        raise ValueError("--labels must contain at least one label")
    return parsed


def read_caption(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").strip().split())


def labels_from_caption(caption: str, fallback_labels: Sequence[str]) -> List[str]:
    caption_l = caption.lower()
    labels = []
    for label, aliases in CAPTION_LABEL_ALIASES.items():
        if any(alias in caption_l for alias in aliases):
            labels.append(label)
    return labels or list(fallback_labels)


def unique_labels(labels: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for label in labels:
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def build_prompt_labels(caption: str, fallback_labels: Sequence[str], prompt_mode: str) -> List[str]:
    if prompt_mode == "all_labels":
        return list(fallback_labels)
    caption_labels = labels_from_caption(caption, fallback_labels)
    if prompt_mode == "caption_plus_defaults":
        return unique_labels([*caption_labels, *fallback_labels])
    return caption_labels


def collect_samples(meta_dir: Path, offset: int, limit: int) -> List[Dict[str, Any]]:
    meta_files = sorted(meta_dir.glob("*.meta.json"))
    if not meta_files:
        raise RuntimeError(f"found 0 metadata files under {meta_dir}")
    selected = meta_files[offset:] if limit <= 0 else meta_files[offset : offset + limit]
    samples = []
    for meta_path in selected:
        meta = load_json(meta_path)
        image_path = Path(meta["source_image"])
        caption_path = Path(meta.get("source_caption", image_path.with_suffix(".txt")))
        if not image_path.is_file():
            raise FileNotFoundError(f"source image does not exist: {image_path}")
        if not caption_path.is_file():
            raise FileNotFoundError(f"source caption does not exist: {caption_path}")
        samples.append(
            {
                "image_id": meta_path.name[: -len(".meta.json")],
                "image_path": image_path,
                "caption_path": caption_path,
                "meta_path": meta_path,
            }
        )
    return samples


def set_model_classes(model: YOLOE, classes: Sequence[str], cache: Dict[Tuple[str, ...], Any]) -> None:
    key = tuple(classes)
    if key not in cache:
        cache[key] = model.get_text_pe(list(classes))
    model.set_classes(list(classes), cache[key])


def result_to_detections(result: Any) -> List[Dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clses = result.boxes.cls.detach().cpu().numpy().astype(int)
    polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)

    detections = []
    for idx, (box, conf, cls_idx) in enumerate(zip(boxes, confs, clses)):
        polygon = polygons[idx] if idx < len(polygons) else None
        detections.append(
            {
                "box": [round(float(x), 2) for x in box.tolist()],
                "score": float(conf),
                "label": result.names.get(int(cls_idx), str(cls_idx)),
                "class_id": int(cls_idx),
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon.tolist()]
                if polygon is not None
                else [],
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def save_mask_npz(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.masks is None or len(result.masks) == 0:
        np.savez_compressed(path, masks=np.zeros((0, 1, 1), dtype=bool))
        return
    masks = result.masks.data.detach().cpu().numpy() > 0.5
    np.savez_compressed(path, masks=masks)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "label", "score", "x1", "y1", "x2", "y2", "image_path", "caption_path", "prompt"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_contact_sheet(paths: Sequence[Path], output_path: Path, cols: int, thumb_width: int) -> None:
    if not paths:
        return
    cols = max(1, cols)
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / float(image.width)
        thumbs.append(image.resize((thumb_width, max(1, int(round(image.height * scale)))), Image.BILINEAR))
    thumb_h = max(im.height for im in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (thumb_width * cols, thumb_h * rows), color=(255, 255, 255))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_width, (idx // cols) * thumb_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def make_contact_sheets(
    paths: Sequence[Path],
    output_dir: Path,
    prefix: str,
    cols: int,
    thumb_width: int,
    page_size: int,
) -> List[str]:
    if not paths:
        return []
    output_paths = []
    if page_size <= 0:
        output_path = output_dir / f"{prefix}_detection_contact_sheet.jpg"
        make_contact_sheet(paths, output_path, cols=cols, thumb_width=thumb_width)
        return [str(output_path)]
    for page_idx, start in enumerate(range(0, len(paths), page_size), start=1):
        chunk = paths[start : start + page_size]
        output_path = output_dir / f"{prefix}_detection_contact_sheet_{page_idx:03d}.jpg"
        make_contact_sheet(chunk, output_path, cols=cols, thumb_width=thumb_width)
        output_paths.append(str(output_path))
    return output_paths


def build_summary(
    rows: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    contact_sheets: Sequence[str],
) -> Dict[str, Any]:
    label_counts = Counter(row["label"] for row in rows)
    detections_per_image = [sample["num_detections"] for sample in manifest["samples"]]
    return {
        "model": manifest["model"],
        "output_prefix": manifest["output_prefix"],
        "num_images": len(manifest["samples"]),
        "num_detections": len(rows),
        "images_with_detections": sum(1 for count in detections_per_image if count > 0),
        "images_without_detections": sum(1 for count in detections_per_image if count == 0),
        "mean_detections_per_image": float(np.mean(detections_per_image)) if detections_per_image else 0.0,
        "label_counts": dict(label_counts.most_common()),
        "contact_sheets": list(contact_sheets),
    }


def main() -> None:
    args = parse_args()
    fallback_labels = parse_labels(args.labels)
    samples = collect_samples(args.meta_dir, args.offset, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.output_prefix}] model={args.model} conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    model = YOLOE(args.model)
    embedding_cache: Dict[Tuple[str, ...], Any] = {}
    annotated_paths: List[Path] = []
    rows: List[Dict[str, Any]] = []
    manifest = {
        "model": args.model,
        "output_prefix": args.output_prefix,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "prompt_mode": args.prompt_mode,
        "contact_sheet_page_size": args.contact_sheet_page_size,
        "samples": [],
    }

    for idx, sample in enumerate(samples, start=1):
        image_id = sample["image_id"]
        caption = read_caption(sample["caption_path"])
        prompt_labels = build_prompt_labels(caption, fallback_labels, args.prompt_mode)
        if args.quiet_labels:
            print(f"[{args.output_prefix}] {idx:04d}/{len(samples):04d} {image_id} labels={len(prompt_labels)}")
        else:
            print(f"[{args.output_prefix}] {idx:04d}/{len(samples):04d} {image_id} labels={prompt_labels}")
        set_model_classes(model, prompt_labels, embedding_cache)
        result = model.predict(
            str(sample["image_path"]),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections = result_to_detections(result)

        image_dir = args.output_dir / "images" / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = image_dir / f"{args.output_prefix}_detections.jpg"
        plotted = result.plot()
        cv2.imwrite(str(annotated_path), plotted)
        save_mask_npz(image_dir / f"{args.output_prefix}_masks.npz", result)
        save_json(
            image_dir / "detections.json",
            {
                "image_id": image_id,
                "image_path": str(sample["image_path"]),
                "caption_path": str(sample["caption_path"]),
                "meta_path": str(sample["meta_path"]),
                "prompt_labels": prompt_labels,
                "detections": detections,
            },
        )
        annotated_paths.append(annotated_path)
        manifest["samples"].append(
            {
                "image_id": image_id,
                "image_path": str(sample["image_path"]),
                "caption_path": str(sample["caption_path"]),
                "annotated_path": str(annotated_path),
                "prompt_labels": prompt_labels,
                "num_detections": len(detections),
            }
        )
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            rows.append(
                {
                    "image_id": image_id,
                    "label": det["label"],
                    "score": f"{det['score']:.6f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "image_path": str(sample["image_path"]),
                    "caption_path": str(sample["caption_path"]),
                    "prompt": ", ".join(prompt_labels),
                }
            )

    write_csv(args.output_dir / "detections.csv", rows)
    contact_sheets = make_contact_sheets(
        annotated_paths,
        args.output_dir,
        args.output_prefix,
        cols=args.contact_sheet_cols,
        thumb_width=args.contact_thumb_width,
        page_size=args.contact_sheet_page_size,
    )
    manifest["contact_sheets"] = contact_sheets
    save_json(args.output_dir / "manifest.json", manifest)
    save_json(args.output_dir / "summary.json", build_summary(rows, manifest, contact_sheets))
    print(f"[{args.output_prefix}] wrote {len(samples)} images, {len(rows)} detections")
    print(f"[{args.output_prefix}] output: {args.output_dir}")


if __name__ == "__main__":
    main()
