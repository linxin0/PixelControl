#!/usr/bin/env python3
"""Self-contained visual-quality evaluation (FID / CLIP-text / CLIP-img / LPIPS).

This is a standalone re-implementation of the visual-quality half of the
PixelGen ``eval_depth_metrics.py`` engine, with the DepthAnything dependency
removed so it runs anywhere with only ``torch / torchvision / transformers /
pytorch_fid / lpips`` installed.

It compares a folder of generated RGB images against the paired ground-truth
RGB images + captions of the evaluation set, and reports:

  * FID            (Inception-V3 2048-d features)       -- lower is better
  * CLIP-text      cos(CLIP_img(gen), CLIP_text(caption)) -- higher is better
  * CLIP-img       cos(CLIP_img(gen), CLIP_img(real))     -- higher is better
  * LPIPS (alex)   perceptual distance gen vs real        -- lower is better

Matching convention
-------------------
For each generated file the *stem* ``sa_XXXXXX`` is used to look up the GT
image ``<image_root>/<stem>.{jpg,jpeg,png}`` and caption ``<image_root>/<stem>.txt``.
Generated files may be named either ``<stem>.png`` or, for single-control
outputs, ``<stem>_<suffix>.png`` (e.g. ``sa_000201_seg.png``) -- pass the
suffix via ``--control_suffix seg``.

Examples
--------
# Our three-control run, segmentation-only outputs (sa_xxxxxx_seg.png):
python eval/eval_visual_quality.py \
  --gen_dir /.../val/all_modes_eval2000/iter_10000 \
  --name ours_seg --control_suffix seg \
  --image_root t2i/data/blip/extracted_new/sa_000201 \
  --clip_model t2i/pretrained/clip-vit-large-patch14 \
  --metrics fid clip_img lpips \
  --output_json outputs/vq_ours_seg.json

# Edge-only checkpoint outputs (sa_xxxxxx.png produced by infer):
python eval/eval_visual_quality.py \
  --gen_dir /.../val/edge_eval2000/iter_12000 --name edge_iter12000 \
  --metrics fid clip_img lpips --output_json outputs/vq_edge.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual quality: FID / CLIP / LPIPS.")
    p.add_argument("--gen_dir", action="append", default=[], dest="gen_dirs",
                   help="Generated image folder. Repeatable to compare many runs.")
    p.add_argument("--name", action="append", default=[], dest="names",
                   help="Display name per --gen_dir (must match count if given).")
    p.add_argument("--control_suffix", default="",
                   help="If set, only match files named sa_xxxxxx_<suffix>.png.")
    p.add_argument("--image_root",
                   default="t2i/data/blip/extracted_new/sa_000201",
                   help="GT eval folder containing <stem>.txt + <stem>.<img ext>.")
    p.add_argument("--clip_model",
                   default="t2i/pretrained/clip-vit-large-patch14")
    p.add_argument("--metrics", nargs="+",
                   choices=["fid", "clip_text", "clip_img", "lpips"],
                   default=["fid", "clip_text", "clip_img", "lpips"])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_json", default="outputs/visual_quality.json")
    return p.parse_args()


def find_with_exts(root: str, stem: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        path = os.path.join(root, stem + ext)
        if os.path.exists(path):
            return path
    return None


def gen_pattern(control_suffix: str) -> re.Pattern:
    if control_suffix:
        return re.compile(rf"^(?P<stem>sa_\d+)_{re.escape(control_suffix)}\.png$")
    return re.compile(r"^(?P<stem>sa_\d+)\.png$")


def index_gen_dir(gen_dir: str, control_suffix: str) -> Dict[str, str]:
    pat = gen_pattern(control_suffix)
    out: Dict[str, str] = {}
    for name in sorted(os.listdir(gen_dir)):
        m = pat.match(name)
        if m:
            out[m.group("stem")] = os.path.join(gen_dir, name)
    return out


class _PathDataset(Dataset):
    def __init__(self, paths: List[str], resolution: int):
        self.paths = paths
        self.tx = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            return self.tx(im.convert("RGB"))


def _build_inception(device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([block_idx]).to(device).eval()


@torch.no_grad()
def _inception_acts(model, paths, resolution, batch_size, device, n_workers=4) -> np.ndarray:
    dl = DataLoader(_PathDataset(paths, resolution), batch_size=batch_size,
                    num_workers=n_workers, pin_memory=True, shuffle=False)
    feats = []
    for x in dl:
        out = model(x.to(device, non_blocking=True))[0]
        feats.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _fid(acts_a, acts_b) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu_a, sig_a = acts_a.mean(0), np.cov(acts_a, rowvar=False)
    mu_b, sig_b = acts_b.mean(0), np.cov(acts_b, rowvar=False)
    return float(calculate_frechet_distance(mu_a, sig_a, mu_b, sig_b))


def _build_clip(model_name, device):
    from transformers import CLIPModel, CLIPProcessor
    return (CLIPModel.from_pretrained(model_name).to(device).eval(),
            CLIPProcessor.from_pretrained(model_name))


def _as_tensor(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"cannot coerce CLIP output {type(out).__name__}")


@torch.no_grad()
def _clip_img(model, proc, paths, batch_size, device) -> torch.Tensor:
    embeds = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        e = F.normalize(_as_tensor(model.get_image_features(**inp)), dim=-1)
        embeds.append(e.cpu())
    return torch.cat(embeds, 0)


@torch.no_grad()
def _clip_text(model, proc, caps, batch_size, device) -> torch.Tensor:
    embeds = []
    for i in range(0, len(caps), batch_size):
        inp = proc(text=caps[i:i + batch_size], return_tensors="pt",
                   padding=True, truncation=True, max_length=77).to(device)
        e = F.normalize(_as_tensor(model.get_text_features(**inp)), dim=-1)
        embeds.append(e.cpu())
    return torch.cat(embeds, 0)


def main() -> None:
    args = parse_args()
    if args.names and len(args.names) != len(args.gen_dirs):
        raise ValueError("--name count must match --gen_dir count")
    if not args.gen_dirs:
        raise ValueError("at least one --gen_dir is required")
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    metrics = set(args.metrics)
    need_clip = "clip_text" in metrics or "clip_img" in metrics
    need_incep = "fid" in metrics
    need_lpips = "lpips" in metrics

    clip_model = clip_proc = inception = lpips_model = None
    if need_clip:
        print(f"[vq] loading CLIP {args.clip_model}")
        clip_model, clip_proc = _build_clip(args.clip_model, device)
    if need_incep:
        print("[vq] loading InceptionV3")
        inception = _build_inception(device)
    if need_lpips:
        print("[vq] loading LPIPS(alex)")
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    # Cache real-image features keyed by the union of matched stems per gen_dir.
    results = {}
    for i, gen_dir in enumerate(args.gen_dirs):
        name = args.names[i] if args.names else os.path.basename(gen_dir.rstrip("/"))
        gen_index = index_gen_dir(gen_dir, args.control_suffix)
        # keep only stems that also have a GT image (+caption for clip_text)
        stems = []
        for stem in sorted(gen_index):
            ip = find_with_exts(args.image_root, stem)
            cp = os.path.join(args.image_root, stem + ".txt")
            if ip is None:
                continue
            if "clip_text" in metrics and not os.path.exists(cp):
                continue
            stems.append(stem)
        if args.max_samples > 0:
            stems = stems[: args.max_samples]
        if not stems:
            print(f"[vq][skip] {name}: 0 matched stems in {gen_dir}")
            continue

        gen_paths = [gen_index[s] for s in stems]
        real_paths = [find_with_exts(args.image_root, s) for s in stems]
        caps = []
        if "clip_text" in metrics:
            for s in stems:
                with open(os.path.join(args.image_root, s + ".txt"), encoding="utf-8", errors="ignore") as f:
                    caps.append(f.read().strip())

        rec = {"n_samples": len(stems), "gen_dir": gen_dir}
        print(f"\n===== {name} ({len(stems)} samples) =====")

        if need_clip:
            gen_e = _clip_img(clip_model, clip_proc, gen_paths, args.batch_size, device)
            if "clip_img" in metrics:
                real_e = _clip_img(clip_model, clip_proc, real_paths, args.batch_size, device)
                rec["clip_img"] = float((gen_e * real_e).sum(-1).mean())
            if "clip_text" in metrics:
                text_e = _clip_text(clip_model, clip_proc, caps, args.batch_size, device)
                rec["clip_text"] = float((gen_e * text_e).sum(-1).mean())

        if need_incep:
            real_acts = _inception_acts(inception, real_paths, args.resolution,
                                        args.batch_size, device, args.num_workers)
            gen_acts = _inception_acts(inception, gen_paths, args.resolution,
                                       args.batch_size, device, args.num_workers)
            rec["fid"] = _fid(real_acts, gen_acts)

        if need_lpips:
            dl_g = DataLoader(_PathDataset(gen_paths, args.resolution), batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
            dl_r = DataLoader(_PathDataset(real_paths, args.resolution), batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
            vals = []
            with torch.no_grad():
                for xg, xr in zip(dl_g, dl_r):
                    xg = (xg * 2 - 1).to(device)
                    xr = (xr * 2 - 1).to(device)
                    vals.append(lpips_model(xg, xr).view(-1).cpu())
            rec["lpips"] = float(torch.cat(vals).mean())

        results[name] = rec
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float)))

    blob = {"metadata": {"image_root": args.image_root, "resolution": args.resolution,
                         "clip_model": args.clip_model if need_clip else None,
                         "control_suffix": args.control_suffix,
                         "metrics": list(metrics)},
            "results": results}
    with open(args.output_json, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\n[vq] wrote {args.output_json}")

    cols = [m for m in ["fid", "clip_text", "clip_img", "lpips"] if m in metrics]
    w = 10
    header = f"{'run':>22s} | " + " | ".join(f"{c:>{w}s}" for c in cols)
    print("\n" + header + "\n" + "-" * len(header))
    for tag, rec in results.items():
        print(f"{tag:>22s} | " + " | ".join(f"{rec.get(c, float('nan')):>{w}.4f}" for c in cols))
    print("\nDirections: FID lower better | CLIP-text/img higher better | LPIPS lower better")


if __name__ == "__main__":
    main()
