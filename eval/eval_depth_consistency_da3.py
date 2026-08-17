#!/usr/bin/env python3
"""Self-contained depth-consistency evaluation via DepthAnything-3 (DA3).

For depth-control outputs we cannot compare against the *input* depth condition
map directly (it has its own normalization). Instead we re-run DA3 on the
generated RGB image and compare its predicted depth to the ground-truth DA3
depth that was produced for the real image (the same model/release used to
build the training targets). This is the depth analogue of the SAM2 seg-cycle
and the blurred-Canny edge metric: a learned extractor closes the loop.

Metrics (scale-invariant; per-image (a,b) least-squares fit of pred to GT)
    SI-RMSE   lower is better
    AbsRel    lower is better (masked to GT > absrel_min_gt)
    delta1/2/3 higher is better (% pixels with max(p/g, g/p) < 1.25^k)
    pearson_r higher is better (linear correlation of pred vs GT)

Generated files may be ``sa_xxxxxx_depth.png`` (pass --control_suffix depth) or
``sa_xxxxxx.png`` (no suffix). GT depth is loaded from
``<depth_root>/<stem>.depth.npy`` (or .npy/.depth.png variants).

Run in the `deco` conda env (DA3 + omegaconf + addict + imageio):
    conda activate deco
    python eval/eval_depth_consistency_da3.py \
      --gen_dir outputs/infer_depth --name ours_depth --control_suffix depth \
      --depth_root t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201 \
      --output_json outputs/depth_consistency_ours.json --save_maps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEPTH_EXTS = (".depth.npy", ".npy", ".depth.png", ".png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA3 depth consistency.")
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--control_suffix", default="depth",
                   help="Match sa_xxxxxx_<suffix>.png; empty for sa_xxxxxx.png.")
    p.add_argument("--depth_root",
                   default="t2i/data/blip_depth_da3_nested_giant_large_1_1/sa_000201")
    p.add_argument("--da3_src", default="t2i/third_party/depth-anything-3/src")
    p.add_argument("--da3_model_dir", default="t2i/pretrained/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--process_res", type=int, default=504)
    p.add_argument("--process_res_method", default="upper_bound_resize")
    p.add_argument("--absrel_min_gt", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--save_maps", action="store_true",
                   help="Also save predicted DA3 maps as .npy + visualization .png.")
    p.add_argument("--maps_out", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_json", default="outputs/depth_consistency_da3.json")
    return p.parse_args()


def gen_pattern(suffix: str) -> re.Pattern:
    if suffix:
        return re.compile(rf"^(?P<stem>sa_\d+)_{re.escape(suffix)}\.png$")
    return re.compile(r"^(?P<stem>sa_\d+)\.png$")


def index_gen(gen_dir: str, suffix: str) -> Dict[str, str]:
    pat = gen_pattern(suffix)
    return {m.group("stem"): os.path.join(gen_dir, n)
            for n in sorted(os.listdir(gen_dir)) if (m := pat.match(n))}


def find_gt_depth(depth_root: str, stem: str) -> Optional[str]:
    for ext in DEPTH_EXTS:
        p = os.path.join(depth_root, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_gt_depth(path: str, resolution: int) -> torch.Tensor:
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        with Image.open(path) as im:
            d = np.asarray(im.convert("F"), dtype=np.float32)
    if d.ndim == 3:
        d = d.mean(-1)
    t = torch.from_numpy(d)[None, None]
    t = F.interpolate(t, size=(resolution, resolution), mode="bilinear", align_corners=False)[0, 0]
    lo, hi = float(t.min()), float(t.max())
    return (t - lo) / max(hi - lo, 1e-6)


def load_da3(da3_src: str, model_dir: str, device: torch.device):
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
    # DA3 imports export helpers (moviepy etc) unconditionally; stub them.
    export_stub = types.ModuleType("depth_anything_3.utils.export")
    export_stub.export = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("export unavailable"))
    sys.modules["depth_anything_3.utils.export"] = export_stub
    if "moviepy.editor" not in sys.modules:
        mv = types.ModuleType("moviepy"); ed = types.ModuleType("moviepy.editor")
        ed.ImageSequenceClip = object
        mv.editor = ed
        sys.modules.setdefault("moviepy", mv)
        sys.modules["moviepy.editor"] = ed
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(model_dir).to(device).eval()


@torch.no_grad()
def predict_depth(model, image_path: str, resolution: int, process_res: int, method: str) -> torch.Tensor:
    res = model.inference([image_path], process_res=process_res,
                          process_res_method=method, use_ray_pose=False)
    d = torch.from_numpy(res.depth[0].astype(np.float32))[None, None]
    d = F.interpolate(d, size=(resolution, resolution), mode="bilinear", align_corners=False)[0, 0]
    return d


def scale_shift_fit(pred: torch.Tensor, gt: torch.Tensor):
    p, g = pred.reshape(-1), gt.reshape(-1)
    n = p.numel()
    sp, sp2, sg, spg = p.sum(), (p * p).sum(), g.sum(), (p * g).sum()
    det = (n * sp2 - sp * sp).clamp_min(1e-8)
    a = (n * spg - sp * sg) / det
    b = (sg - a * sp) / n
    return a, b


def metrics_one(pred: torch.Tensor, gt: torch.Tensor, absrel_min_gt: float) -> Dict[str, float]:
    a, b = scale_shift_fit(pred, gt)
    pa = a * pred + b
    diff = pa - gt
    si_rmse = torch.sqrt((diff * diff).mean()).item()
    mask = gt > absrel_min_gt
    denom = mask.sum().clamp_min(1)
    absrel = ((diff.abs() / gt.clamp_min(1e-3)) * mask).sum() / denom
    pa_c = pa.clamp_min(1e-3); gt_c = gt.clamp_min(1e-3)
    ratio = torch.maximum(pa_c / gt_c, gt_c / pa_c)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()
    pv, gv = pa.reshape(-1), gt.reshape(-1)
    pv = pv - pv.mean(); gv = gv - gv.mean()
    pear = (pv * gv).sum() / (pv.norm() * gv.norm()).clamp_min(1e-8)
    return {"si_rmse": si_rmse, "absrel": float(absrel), "delta1": d1,
            "delta2": d2, "delta3": d3, "pearson_r": float(pear)}


def save_vis(path: Path, depth: np.ndarray) -> None:
    lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
    arr = (np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "L").save(path)


def main() -> None:
    args = parse_args()
    name = args.name or os.path.basename(args.gen_dir.rstrip("/"))
    device = torch.device(args.device)
    gen_index = index_gen(args.gen_dir, args.control_suffix)
    pairs = [(s, gp, find_gt_depth(args.depth_root, s)) for s, gp in gen_index.items()]
    pairs = [(s, gp, dp) for s, gp, dp in pairs if dp is not None]
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError(f"no (gen, GT-depth) pairs for {args.gen_dir}")
    print(f"[depth] {name}: {len(pairs)} paired images")

    model = load_da3(args.da3_src, args.da3_model_dir, device)
    maps_out = Path(args.maps_out) if args.maps_out else Path(args.output_json).parent / f"{name}_da3_maps"

    accum: List[Dict[str, float]] = []
    for i, (stem, gen_path, gt_path) in enumerate(pairs):
        pred = predict_depth(model, gen_path, args.resolution, args.process_res, args.process_res_method).cpu()
        gt = load_gt_depth(gt_path, args.resolution)
        accum.append(metrics_one(pred, gt, args.absrel_min_gt))
        if args.save_maps:
            np.save(maps_out / "depth_npy" / f"{stem}.depth.npy", pred.numpy().astype(np.float32))
            save_vis(maps_out / "depth_png" / f"{stem}.depth.png", pred.numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    keys = list(accum[0].keys())
    agg = {k: float(np.mean([r[k] for r in accum])) for k in keys}
    blob = {"name": name, "gen_dir": args.gen_dir, "depth_root": args.depth_root,
            "n_samples": len(accum), "da3_model_dir": args.da3_model_dir,
            "metrics": agg}
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2))
    print(f"[depth] {name}: " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    print(f"[depth] wrote {out}")
    if args.save_maps:
        print(f"[depth] DA3 maps -> {maps_out}")


if __name__ == "__main__":
    main()
