#!/usr/bin/env python3
"""Evaluate depth-only baseline outputs with the existing DA3 consistency script.

This wrapper scans `outputs/baseline_eval/<method>/<sample_set>/` for images
named exactly `{stem}_depth.png`, creates a staging directory with symlinks named
`{stem}.png`, then calls PixelGen's `eval_depth_consistency_da3.py`.

It intentionally excludes multi-condition outputs such as:
  - `{stem}_depth_seg.png`
  - `{stem}_depth_edge.png`
  - `{stem}_depth_seg_edge.png`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEPTH_ONLY_RE = re.compile(r"^(?P<stem>.+)_depth\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DA3 depth consistency for baseline_eval depth-only images.")
    parser.add_argument(
        "--baseline_root",
        default="outputs/baseline_eval",
        help="Root containing baseline_eval/<method>/<sample_set>/ outputs.",
    )
    parser.add_argument(
        "--sample_set",
        default="sa_000201_first2000",
        help="Leaf folder to evaluate for each method. Default: sa_000201_first2000.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method names to evaluate. Default: all methods with the requested sample_set.",
    )
    parser.add_argument(
        "--exclude_methods",
        nargs="*",
        default=[
            "controlnet_smoke",
            "controlnet_smoke_multicontrol",
            "anycontrol_smoke",
            "ctrl_adapter_deco_smoke",
            "ctrl_adapter_deco_smoke_single_modes",
            "ctrl_adapter_device_remap_smoke",
            "ominicontrol_smoke",
            "unicontrolnet_smoke",
            "pixelponder_debug",
            "pixelponder_probe",
            "relactrl_canny",
        ],
        help="Method folders to skip.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/depth_consistency_da3_baseline_depth_only",
        help="Where staging dirs, manifest, and JSON metrics are written.",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="Final metrics JSON. Default: <output_root>/depth_consistency_da3_<sample_set>.json",
    )
    parser.add_argument(
        "--eval_script",
        default="t2i/third_party/PixelGen/scripts/eval_depth_consistency_da3.py",
        help="Existing PixelGen DA3 evaluation script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call eval_script.",
    )
    parser.add_argument(
        "--image_root",
        default="t2i/data/blip/extracted_new/sa_000201",
        help="Eval image/caption root.",
    )
    parser.add_argument(
        "--depth_root",
        default="t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201",
        help="GT depth condition root.",
    )
    parser.add_argument(
        "--da3_src",
        default="t2i/third_party/depth-anything-3/src",
        help="DA3 src path.",
    )
    parser.add_argument(
        "--da3_model_dir",
        default="t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1",
        help="DA3 model dir.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device passed to the DA3 eval script.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional max samples passed to eval script.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into staging instead of symlinking. Slower and uses more disk.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only build staging dirs and print the eval command.",
    )
    return parser.parse_args()


def method_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    root = Path(args.baseline_root)
    if not root.is_dir():
        raise FileNotFoundError(f"baseline_root does not exist: {root}")
    include = set(args.methods) if args.methods else None
    exclude = set(args.exclude_methods or [])
    out: list[tuple[str, Path]] = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        if include is not None and method not in include:
            continue
        if method in exclude or "smoke" in method or method.endswith("_debug") or method.endswith("_probe"):
            continue
        sample_dir = method_dir / args.sample_set
        if sample_dir.is_dir():
            out.append((method, sample_dir))
    return out


def collect_depth_only_images(src_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        match = DEPTH_ONLY_RE.match(path.name)
        if match is None:
            continue
        stem = match.group("stem")
        images[stem] = path
    return images


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def stage_method(method: str, src_dir: Path, stage_root: Path, copy_files: bool) -> dict:
    images = collect_depth_only_images(src_dir)
    stage_dir = stage_root / method
    reset_dir(stage_dir)
    for stem, src in images.items():
        dst = stage_dir / f"{stem}.png"
        if copy_files:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
    return {
        "method": method,
        "source_dir": str(src_dir),
        "stage_dir": str(stage_dir),
        "n_depth_only_images": len(images),
        "example_stems": list(images.keys())[:5],
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    stage_root = output_root / "staged" / args.sample_set
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    discovered = method_dirs(args)
    if not discovered:
        raise RuntimeError(
            f"No method dirs found under {args.baseline_root} with sample_set={args.sample_set}. "
            "Use --methods or --sample_set to adjust."
        )

    staged = []
    for method, src_dir in discovered:
        item = stage_method(method, src_dir, stage_root, copy_files=args.copy)
        if item["n_depth_only_images"] == 0:
            print(f"[skip] {method}: no *_depth.png in {src_dir}")
            continue
        print(f"[stage] {method}: {item['n_depth_only_images']} depth-only images -> {item['stage_dir']}")
        staged.append(item)

    if not staged:
        raise RuntimeError("No depth-only images found after staging.")

    manifest_path = output_root / f"manifest_depth_only_{args.sample_set}.json"
    manifest = {
        "baseline_root": args.baseline_root,
        "sample_set": args.sample_set,
        "depth_only_pattern": "{stem}_depth.png",
        "excluded_patterns": ["{stem}_depth_seg.png", "{stem}_depth_edge.png", "{stem}_depth_seg_edge.png"],
        "staged": staged,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)
    print(f"[manifest] {manifest_path}")

    output_json = Path(args.output_json) if args.output_json else output_root / f"depth_consistency_da3_{args.sample_set}.json"
    gen_dirs = [item["stage_dir"] for item in staged]
    cmd = [
        args.python,
        args.eval_script,
        "--gen_dirs",
        *gen_dirs,
        "--image_root",
        args.image_root,
        "--depth_root",
        args.depth_root,
        "--da3_src",
        args.da3_src,
        "--da3_model_dir",
        args.da3_model_dir,
        "--device",
        args.device,
        "--output_json",
        str(output_json),
    ]
    if args.max_samples > 0:
        cmd += ["--max_samples", str(args.max_samples)]

    print("[eval-cmd]")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if args.dry_run:
        print("[dry-run] staging complete; not running DA3 eval.")
        return

    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    subprocess.run(cmd, check=True, env=env)
    print(f"[done] wrote {output_json}")


if __name__ == "__main__":
    main()
