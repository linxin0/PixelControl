#!/usr/bin/env python3
"""Save DA3-predicted depth maps for pure depth-control outputs.

Input images are strict generated depth-only RGB files:
  sa_xxxxxx_depth.png

For each method, this script runs Depth Anything 3 on those generated images and
saves:
  <output_root>/<method>/depth_npy/sa_xxxxxx.depth.npy
  <output_root>/<method>/depth_png/sa_xxxxxx.depth.png

The PNG is only a per-image min-max visualization. The NPY stores the resized
float32 DA3 relative depth map.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEPTH_ONLY_RE = re.compile(r"^(?P<stem>sa_\d+)_depth\.png$")


@dataclass(frozen=True)
class MethodInput:
    name: str
    source_dir: Path
    files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        default="outputs/depth_only_outputs_first2000_compare",
        help="Root with <method>/depth/sa_xxxxxx_depth.png folders.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/depth_only_da3_maps_first2000_compare",
    )
    parser.add_argument("--methods", nargs="*", default=[], help="Optional method allow-list.")
    parser.add_argument("--da3_src", default="t2i/third_party/depth-anything-3/src")
    parser.add_argument("--da3_model_dir", default="t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", default="upper_bound")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_false", dest="skip_existing")
    parser.add_argument("--save_png", action="store_true", default=True)
    parser.add_argument("--no_save_png", action="store_false", dest="save_png")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_inputs(args: argparse.Namespace) -> list[MethodInput]:
    root = Path(args.input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"input_root is not a directory: {root}")
    include = set(args.methods) if args.methods else None
    methods: list[MethodInput] = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = method_dir.name
        if include is not None and name not in include:
            continue
        depth_dir = method_dir / "depth"
        if not depth_dir.is_dir():
            continue
        files = [p for p in sorted(depth_dir.iterdir()) if p.is_file() and DEPTH_ONLY_RE.match(p.name)]
        if args.max_samples and args.max_samples > 0:
            files = files[: args.max_samples]
        if not files:
            continue
        methods.append(MethodInput(name=name, source_dir=depth_dir, files=files))
        print(f"[collect] {name}: {len(files)} strict depth images from {depth_dir}")
    if not methods:
        raise RuntimeError("No strict depth images found.")
    return methods


def load_da3(da3_src: str, model_dir: str, device: torch.device):
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
    # DA3's API imports export helpers unconditionally. Depth-map saving never
    # calls export, so stub it to avoid optional heavy deps such as moviepy,
    # pycolmap, plyfile, open3d, and trimesh.
    export_stub = types.ModuleType("depth_anything_3.utils.export")

    def _unavailable_export(*args, **kwargs):
        raise RuntimeError("DA3 export is unavailable in save_depth_only_da3_maps.py")

    export_stub.export = _unavailable_export
    sys.modules["depth_anything_3.utils.export"] = export_stub
    try:
        import moviepy.editor  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        try:
            import moviepy  # type: ignore

            sys.modules["moviepy.editor"] = moviepy
        except ModuleNotFoundError:
            moviepy_stub = types.ModuleType("moviepy")
            editor_stub = types.ModuleType("moviepy.editor")

            class _UnavailableImageSequenceClip:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("moviepy is unavailable; DA3 video export is not supported in this run")

            editor_stub.ImageSequenceClip = _UnavailableImageSequenceClip
            moviepy_stub.editor = editor_stub
            sys.modules["moviepy"] = moviepy_stub
            sys.modules["moviepy.editor"] = editor_stub
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_dir).to(device)
    model.eval()
    return model


def save_depth_png(path: Path, depth: np.ndarray) -> None:
    lo = float(np.nanmin(depth))
    hi = float(np.nanmax(depth))
    vis = (depth - lo) / max(hi - lo, 1e-6)
    arr = (np.clip(vis, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@torch.no_grad()
def predict_one(model, image_path: Path, args: argparse.Namespace) -> np.ndarray:
    result = model.inference(
        [str(image_path)],
        process_res=int(args.process_res),
        process_res_method=str(args.process_res_method),
        use_ray_pose=False,
    )
    depth = torch.from_numpy(result.depth[0].astype(np.float32)).unsqueeze(0).unsqueeze(0)
    depth = F.interpolate(
        depth,
        size=(int(args.resolution), int(args.resolution)),
        mode="bilinear",
        align_corners=False,
    )
    return depth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def output_paths(output_root: Path, method: str, src_path: Path) -> tuple[Path, Path]:
    match = DEPTH_ONLY_RE.match(src_path.name)
    if match is None:
        raise ValueError(f"Not a strict depth-only filename: {src_path.name}")
    stem = match.group("stem")
    npy_path = output_root / method / "depth_npy" / f"{stem}.depth.npy"
    png_path = output_root / method / "depth_png" / f"{stem}.depth.png"
    return npy_path, png_path


def main() -> None:
    args = parse_args()
    methods = discover_inputs(args)
    output_root = Path(args.output_root)
    manifest = {
        "input_root": args.input_root,
        "output_root": str(output_root),
        "da3_src": args.da3_src,
        "da3_model_dir": args.da3_model_dir,
        "resolution": args.resolution,
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "methods": [
            {
                "method": item.name,
                "source_dir": str(item.source_dir),
                "count": len(item.files),
                "examples": [p.name for p in item.files[:5]],
            }
            for item in methods
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest_da3_depth_maps.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[manifest] {manifest_path}")
    if args.dry_run:
        print("[dry-run] not running DA3")
        return

    device = torch.device(args.device)
    model = load_da3(args.da3_src, args.da3_model_dir, device)
    t0 = time.time()
    total_done = 0
    total_skipped = 0
    for item in methods:
        iterator = tqdm(item.files, desc=item.name) if tqdm is not None else item.files
        done = skipped = 0
        for src_path in iterator:
            npy_path, png_path = output_paths(output_root, item.name, src_path)
            if args.skip_existing and npy_path.exists() and (png_path.exists() or not args.save_png):
                skipped += 1
                continue
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            depth = predict_one(model, src_path, args)
            np.save(npy_path, depth)
            if args.save_png:
                save_depth_png(png_path, depth)
            done += 1
            if torch.cuda.is_available() and (done + skipped) % 25 == 0:
                torch.cuda.empty_cache()
        total_done += done
        total_skipped += skipped
        print(f"[done] {item.name}: saved={done} skipped={skipped} output={output_root / item.name}", flush=True)
    print(
        f"[summary] saved={total_done} skipped={total_skipped} "
        f"elapsed={(time.time() - t0) / 60.0:.1f}min output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
