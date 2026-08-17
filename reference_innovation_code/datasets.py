"""Minimal extracted datasets for single-control and three-control training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, Resize
from torchvision.transforms.functional import to_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
DEPTH_EXTS = (".depth.npy", ".npy", ".depth.png", ".png", ".depth.jpg", ".jpg", ".depth.jpeg", ".jpeg")
SEG_EXTS = (".sam2_label.npy", ".sam2_label.png", ".png", ".npy")
EDGE_EXTS = (".edge.npy", ".npy", ".edge.png", ".png", ".edge.jpg", ".jpg", ".edge.jpeg", ".jpeg")


def strip_image_ext(filename: str) -> str:
    for ext in IMAGE_EXTS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]


def find_with_exts(root: str | Path, stem: str, exts: Sequence[str]) -> str | None:
    root = str(root)
    for ext in exts:
        path = os.path.join(root, stem + ext)
        if os.path.exists(path):
            return path
    return None


def read_caption(path: str, default: str = "") -> str:
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return text or default


def _resize_crop_1ch(x: np.ndarray, target_size: int, mode: str) -> torch.Tensor:
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    h, w = x_t.shape[-2:]
    short = min(h, w)
    scale = float(target_size) / float(short)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    x_t = F.interpolate(x_t, size=(new_h, new_w), mode=mode, align_corners=False if mode == "bilinear" else None)
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    return x_t[:, :, top:top + target_size, left:left + target_size].squeeze(0)


def load_depth_to_tensor(path: str, target_size: int, normalize: bool = True, invert_depth: bool = False) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path).astype(np.float32)
    elif ext == ".npz":
        archive = np.load(path)
        depth = archive[list(archive.keys())[0]].astype(np.float32)
    else:
        with Image.open(path) as im:
            im = im.convert("I") if im.mode in ("I", "I;16") else im.convert("L")
            depth = np.asarray(im, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth.mean(axis=-1)
    out = _resize_crop_1ch(depth, target_size, mode="bilinear")
    if normalize:
        lo, hi = out.min(), out.max()
        out = (out - lo) / (hi - lo).clamp_min(1e-6)
    if invert_depth:
        out = 1.0 - out
    return out.clamp_(0.0, 1.0)


def load_seg_to_tensor(path: str, target_size: int, normalize: bool = True) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        seg = np.load(path)
    else:
        with Image.open(path) as im:
            seg = np.asarray(im.convert("L"))
    if seg.ndim == 3:
        seg = seg[..., 0]
    out = _resize_crop_1ch(seg.astype(np.float32), target_size, mode="nearest")
    if normalize:
        max_id = out.max()
        if max_id.item() > 0:
            out = out / max_id
        out = out.clamp_(0.0, 1.0)
    return out


def load_edge_to_tensor(path: str, target_size: int) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        edge = np.load(path).astype(np.float32)
    else:
        with Image.open(path) as im:
            edge = np.asarray(im.convert("L"), dtype=np.float32)
    if edge.ndim == 3:
        edge = edge.mean(axis=-1)
    out = _resize_crop_1ch(edge, target_size, mode="bilinear")
    lo, hi = out.min(), out.max()
    out = (out - lo) / (hi - lo).clamp_min(1e-6)
    return out.clamp_(0.0, 1.0)


def subdir_range(start: int, end: int) -> list[str]:
    return [f"sa_{i:06d}" for i in range(int(start), int(end) + 1)]


class PixelThreeControlDataset(Dataset):
    """Paired RGB/caption/depth/seg/edge dataset.

    Returns a dict ready for a PixelDiT-like training loop. The loop can sample
    active modes and zero inactive channels using `apply_multi_control_mode`.
    """

    def __init__(
        self,
        image_root: str,
        depth_root: str,
        seg_root: str,
        edge_root: str,
        resolution: int = 512,
        subdirs: Optional[Sequence[str]] = None,
        cache_index_path: str | None = None,
        max_retries: int = 20,
        seg_normalize: bool = True,
        require_caption: bool = True,
    ):
        self.image_root = image_root
        self.depth_root = depth_root
        self.seg_root = seg_root
        self.edge_root = edge_root
        self.resolution = int(resolution)
        self.subdirs = list(subdirs) if subdirs is not None else None
        self.max_retries = int(max_retries)
        self.seg_normalize = bool(seg_normalize)
        self.require_caption = bool(require_caption)
        self.samples: list[dict] = []
        if cache_index_path and os.path.exists(cache_index_path):
            self.samples = json.load(open(cache_index_path, "r", encoding="utf-8"))
        else:
            self._build_index()
            if cache_index_path:
                Path(cache_index_path).parent.mkdir(parents=True, exist_ok=True)
                json.dump(self.samples, open(cache_index_path, "w", encoding="utf-8"))
        self.resize = Resize(self.resolution)
        self.center_crop = CenterCrop(self.resolution)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _iter_subdirs(self):
        if self.subdirs is not None:
            return self.subdirs
        return sorted(p.name for p in Path(self.image_root).iterdir() if p.is_dir())

    def _build_index(self):
        for sub in self._iter_subdirs():
            image_dir = Path(self.image_root) / sub
            depth_dir = Path(self.depth_root) / sub
            seg_dir = Path(self.seg_root) / sub
            edge_dir = Path(self.edge_root) / sub
            if not image_dir.is_dir() or not depth_dir.is_dir() or not seg_dir.is_dir() or not edge_dir.is_dir():
                continue
            for cap in sorted(image_dir.glob("*.txt")):
                stem = cap.stem
                image_path = find_with_exts(image_dir, stem, IMAGE_EXTS)
                depth_path = find_with_exts(depth_dir, stem, DEPTH_EXTS)
                seg_path = find_with_exts(seg_dir, stem, SEG_EXTS)
                edge_path = find_with_exts(edge_dir, stem, EDGE_EXTS)
                if image_path and depth_path and seg_path and edge_path:
                    self.samples.append(
                        {
                            "stem": stem,
                            "image_path": image_path,
                            "caption_path": str(cap),
                            "depth_path": depth_path,
                            "seg_path": seg_path,
                            "edge_path": edge_path,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def _build_item(self, idx: int):
        sample = self.samples[idx]
        pil = Image.open(sample["image_path"]).convert("RGB")
        pil = self.center_crop(self.resize(pil))
        image_01 = to_tensor(pil)
        image_m11 = self.normalize(image_01)
        depth = load_depth_to_tensor(sample["depth_path"], self.resolution)
        seg = load_seg_to_tensor(sample["seg_path"], self.resolution, normalize=self.seg_normalize)
        edge = load_edge_to_tensor(sample["edge_path"], self.resolution)
        control = torch.cat([depth, seg, edge], dim=0)
        return {
            "image": image_m11,
            "caption": read_caption(sample["caption_path"]),
            "control": control,
            "control_keep": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "control_mode": "depth_seg_edge",
            "depth": depth,
            "seg": seg,
            "edge": edge,
            **sample,
        }

    def __getitem__(self, idx: int):
        cur = int(idx)
        for _ in range(self.max_retries):
            try:
                return self._build_item(cur)
            except Exception as exc:
                nxt = random.randint(0, len(self.samples) - 1)
                print(f"[PixelThreeControlDataset] bad sample idx={cur}: {exc!r}; retry idx={nxt}")
                cur = nxt
        raise RuntimeError(f"failed to load valid sample after {self.max_retries} retries")


class PixelSingleControlDataset(Dataset):
    """Single-control depth/seg/edge dataset for baseline training."""

    def __init__(
        self,
        image_root: str,
        control_root: str,
        control_type: str,
        resolution: int = 512,
        subdirs: Optional[Sequence[str]] = None,
        seg_normalize: bool = True,
    ):
        if control_type not in {"depth", "seg", "edge"}:
            raise ValueError("control_type must be depth, seg, or edge")
        self.image_root = image_root
        self.control_root = control_root
        self.control_type = control_type
        self.resolution = int(resolution)
        self.subdirs = list(subdirs) if subdirs is not None else None
        self.seg_normalize = bool(seg_normalize)
        self.samples: list[dict] = []
        self._build_index()
        self.resize = Resize(self.resolution)
        self.center_crop = CenterCrop(self.resolution)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _iter_subdirs(self):
        if self.subdirs is not None:
            return self.subdirs
        return sorted(p.name for p in Path(self.image_root).iterdir() if p.is_dir())

    def _find_control(self, control_dir: Path, stem: str):
        if self.control_type == "depth":
            return find_with_exts(control_dir, stem, DEPTH_EXTS)
        if self.control_type == "seg":
            return find_with_exts(control_dir, stem, SEG_EXTS)
        return find_with_exts(control_dir, stem, EDGE_EXTS)

    def _build_index(self):
        for sub in self._iter_subdirs():
            image_dir = Path(self.image_root) / sub
            control_dir = Path(self.control_root) / sub
            if not image_dir.is_dir() or not control_dir.is_dir():
                continue
            for cap in sorted(image_dir.glob("*.txt")):
                stem = cap.stem
                image_path = find_with_exts(image_dir, stem, IMAGE_EXTS)
                control_path = self._find_control(control_dir, stem)
                if image_path and control_path:
                    self.samples.append(
                        {"stem": stem, "image_path": image_path, "caption_path": str(cap), "control_path": control_path}
                    )

    def __len__(self):
        return len(self.samples)

    def _load_control(self, path: str):
        if self.control_type == "depth":
            return load_depth_to_tensor(path, self.resolution)
        if self.control_type == "seg":
            return load_seg_to_tensor(path, self.resolution, normalize=self.seg_normalize)
        return load_edge_to_tensor(path, self.resolution)

    def __getitem__(self, idx: int):
        sample = self.samples[int(idx)]
        pil = Image.open(sample["image_path"]).convert("RGB")
        pil = self.center_crop(self.resize(pil))
        image_m11 = self.normalize(to_tensor(pil))
        control = self._load_control(sample["control_path"])
        return {
            "image": image_m11,
            "caption": read_caption(sample["caption_path"]),
            "control": control,
            "control_keep": torch.tensor([1.0], dtype=torch.float32),
            "control_mode": self.control_type,
            self.control_type: control,
            **sample,
        }
