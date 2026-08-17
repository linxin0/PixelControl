"""Depth / depth+seg control datasets for PixelDiT.

These wrap the on-disk layout used by the original PixelGen depth-control
experiments (``data/blip/extracted``, ``blip_depth_*``,
``blip_sam2_large_extracted``) and re-emit samples in the **8-tuple shape**
expected by PixelDiT's training loop (see ``pixdit_datasets.py``):

    (img, txt_fea, attention_mask, data_info, idx, caption_type, dataindex_info, clipscore)

Control tensors (``depth`` / ``seg`` / ``control`` / ``control_keep`` /
``control_mode``) are smuggled inside ``data_info`` because the baseline
training loop already forwards ``data_info`` into ``model_kwargs``.

This module is a near-verbatim port of the original PixelGen
``src/data/dataset/{depth_condition_dataset,multi_condition_dataset}.py``
with PixelDiT-side I/O contract changes only.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, Resize
from torchvision.transforms.functional import to_tensor

# Keep truncated-image detection ON (do NOT silently load partial buffers):
# we would rather a corrupt control map raise so the dataset can *skip* that
# sample and draw a different one, instead of feeding garbage into training.
ImageFile.LOAD_TRUNCATED_IMAGES = False

from diffusion.data.builder import DATASETS


IMAGE_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
DEPTH_EXTS = (
    ".depth.npy", ".npy", ".depth.png", ".png", ".depth.jpg", ".jpg",
    ".depth.jpeg", ".jpeg",
)
SEG_EXTS = (".sam2_label.npy", ".sam2_label.png", ".png", ".npy")
EDGE_EXTS = (".edge.npy", ".npy", ".edge.png", ".png", ".edge.jpg", ".jpg", ".edge.jpeg", ".jpeg")


# --------------------------------------------------------------------------
# Low-level helpers (ported from PixelGen).
# --------------------------------------------------------------------------
def _is_image_file(filename: str) -> bool:
    return filename.endswith(IMAGE_EXTS)


def _strip_image_ext(filename: str) -> str:
    for ext in IMAGE_EXTS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]


def _find_depth_path(image_relpath: str, depth_root: str):
    base_no_ext = _strip_image_ext(image_relpath)
    for ext in DEPTH_EXTS:
        cand = os.path.join(depth_root, base_no_ext + ext)
        if os.path.exists(cand):
            return cand
    return None


def _find_seg_path(stem: str, seg_dir: str):
    for ext in SEG_EXTS:
        cand = os.path.join(seg_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _find_edge_path(stem: str, edge_dir: str):
    for ext in EDGE_EXTS:
        cand = os.path.join(edge_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def load_depth_to_tensor(
    depth_path: str,
    target_size: int,
    normalize: bool = True,
    repeat_to_3ch: bool = False,
    invert_depth: bool = False,
) -> torch.Tensor:
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    elif ext == ".npz":
        archive = np.load(depth_path)
        depth = archive[list(archive.keys())[0]].astype(np.float32)
    else:
        with Image.open(depth_path) as im:
            im = im.convert("I") if im.mode in ("I", "I;16") else im.convert("L")
            depth = np.asarray(im, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth.mean(axis=-1)
    assert depth.ndim == 2, f"Unexpected depth ndim={depth.ndim} for {depth_path}"
    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    H, W = depth_t.shape[-2:]
    short = min(H, W)
    scale = float(target_size) / float(short)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    depth_t = F.interpolate(depth_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    depth_t = depth_t[:, :, top: top + target_size, left: left + target_size]
    depth_t = depth_t.squeeze(0)
    if normalize:
        d_min = depth_t.min()
        d_max = depth_t.max()
        if (d_max - d_min).item() > 1e-6:
            depth_t = (depth_t - d_min) / (d_max - d_min)
        else:
            depth_t = torch.zeros_like(depth_t)
        depth_t = depth_t.clamp_(0.0, 1.0)
    if invert_depth:
        depth_t = 1.0 - depth_t
    if repeat_to_3ch:
        depth_t = depth_t.repeat(3, 1, 1)
    return depth_t


_SOBEL_KX = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)
_SOBEL_KY = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def compute_edge_from_rgb(rgb_01: torch.Tensor) -> torch.Tensor:
    """Sobel-magnitude edge map ``[1, H, W]`` in [0, 1] from RGB tensor.

    ``rgb_01`` is expected to be a ``[3, H, W]`` tensor with values in
    [0, 1]. The output is per-image min-max normalized so the dynamic
    range matches the depth / seg control channels.
    """
    if rgb_01.ndim != 3:
        raise ValueError(f"compute_edge_from_rgb expects [3,H,W], got {tuple(rgb_01.shape)}")
    gray = rgb_01.float().mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
    kx = _SOBEL_KX.to(device=gray.device, dtype=gray.dtype)
    ky = _SOBEL_KY.to(device=gray.device, dtype=gray.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(0)  # [1,H,W]
    e_min = edge.min()
    e_max = edge.max()
    if (e_max - e_min).item() > 1e-6:
        edge = (edge - e_min) / (e_max - e_min)
    else:
        edge = torch.zeros_like(edge)
    return edge.clamp_(0.0, 1.0).to(rgb_01.dtype)


def load_edge_from_disk(edge_path: str, target_size: int) -> torch.Tensor:
    """Load a pre-computed edge map ``[1, H, W]`` in [0,1] from disk.

    Accepts ``.npy`` / image files. Used by the eval dataset when a
    pre-computed edge directory is provided; the training dataset
    typically falls back to ``compute_edge_from_rgb``.
    """
    ext = os.path.splitext(edge_path)[1].lower()
    if ext == ".npy":
        edge = np.load(edge_path).astype(np.float32)
    else:
        with Image.open(edge_path) as im:
            edge = np.asarray(im.convert("L"), dtype=np.float32)
    if edge.ndim == 3:
        edge = edge.mean(axis=-1)
    edge_t = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0)
    h, w = edge_t.shape[-2:]
    short = min(h, w)
    scale = float(target_size) / float(short)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    edge_t = F.interpolate(edge_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    edge_t = edge_t[:, :, top:top + target_size, left:left + target_size]
    edge_t = edge_t.squeeze(0)
    e_min = edge_t.min()
    e_max = edge_t.max()
    if (e_max - e_min).item() > 1e-6:
        edge_t = (edge_t - e_min) / (e_max - e_min)
    else:
        edge_t = torch.zeros_like(edge_t)
    return edge_t.clamp_(0.0, 1.0)


def load_seg_to_tensor(seg_path: str, target_size: int, normalize: bool = True) -> torch.Tensor:
    ext = os.path.splitext(seg_path)[1].lower()
    if ext == ".npy":
        seg = np.load(seg_path)
    else:
        with Image.open(seg_path) as im:
            seg = np.asarray(im.convert("L"))
    if seg.ndim == 3:
        seg = seg[..., 0]
    assert seg.ndim == 2, f"Unexpected seg ndim={seg.ndim} for {seg_path}"
    seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    h, w = seg_t.shape[-2:]
    short = min(h, w)
    scale = float(target_size) / float(short)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    seg_t = F.interpolate(seg_t, size=(new_h, new_w), mode="nearest")
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    seg_t = seg_t[:, :, top: top + target_size, left: left + target_size]
    seg_t = seg_t.squeeze(0)
    if normalize:
        max_id = seg_t.max()
        if max_id.item() > 0:
            seg_t = seg_t / max_id
        seg_t = seg_t.clamp_(0.0, 1.0)
    return seg_t


def _read_caption(caption_path: str, default: str = "") -> str:
    if caption_path and os.path.exists(caption_path):
        with open(caption_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return default


# --------------------------------------------------------------------------
# Datasets.
# --------------------------------------------------------------------------
@DATASETS.register_module(name="PixelDepthControlDataset")
class PixelDepthControlDataset(Dataset):
    """Depth-controlled paired RGB/caption/depth dataset for PixelDiT.

    Yields PixelDiT's 8-tuple. Depth (shape ``[1, H, W]`` or ``[3, H, W]`` in
    [0,1]) is forwarded inside ``data_info["depth"]`` and also exposed as
    ``data_info["control"]`` for the model wrapper.
    """

    def __init__(
        self,
        image_root: str,
        depth_root: str,
        resolution: int = 512,
        depth_repeat_to_3ch: bool = False,
        invert_depth: bool = False,
        default_caption: str = "",
        require_caption: bool = True,
        max_samples: int = -1,
        cache_index_path: Optional[str] = None,
        subdirs: Optional[Sequence[str]] = None,
        early_stop_count: int = -1,
        max_length: int = 300,
        # ignored kwargs from build_dataset (transform, config, etc.) absorbed in **_
        **_,
    ):
        super().__init__()
        self.image_root = image_root
        self.depth_root = depth_root
        self.resolution = int(resolution)
        self.depth_repeat_to_3ch = bool(depth_repeat_to_3ch)
        self.invert_depth = bool(invert_depth)
        self.default_caption = str(default_caption)
        self.require_caption = bool(require_caption)
        self.subdirs = list(subdirs) if subdirs is not None else None
        self.early_stop_count = int(early_stop_count)
        self.max_length = int(max_length)

        self.image_paths: List[str] = []
        self.depth_paths: List[str] = []
        self.caption_paths: List[str] = []

        if cache_index_path is not None and os.path.exists(cache_index_path):
            self._load_index_cache(cache_index_path)
        else:
            self._build_index()
            if cache_index_path is not None:
                try:
                    self._save_index_cache(cache_index_path)
                except Exception as e:
                    print(f"[PixelDepthControlDataset] Failed to save cache: {e}")

        if max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.depth_paths = self.depth_paths[:max_samples]
            self.caption_paths = self.caption_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"PixelDepthControlDataset found 0 paired samples under "
                f"image_root={image_root} / depth_root={depth_root}."
            )
        self.ori_imgs_nums = len(self.image_paths)
        print(
            f"[PixelDepthControlDataset] paired samples: {len(self.image_paths)} "
            f"(resolution={self.resolution}, repeat_to_3ch={self.depth_repeat_to_3ch})"
        )

        self.resize = Resize(self.resolution)
        self.center_crop = CenterCrop(self.resolution)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # ---- index build / cache (mirror of PixelGen DepthConditionDataset) ----
    def _iter_candidate_subdirs(self) -> List[str]:
        if self.subdirs is not None:
            return list(self.subdirs)
        return sorted(
            name for name in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, name))
        )

    def _append_pair(self, image_path: str, depth_path: str, caption_path: str) -> None:
        self.image_paths.append(image_path)
        self.depth_paths.append(depth_path)
        self.caption_paths.append(caption_path if os.path.exists(caption_path) else "")

    def _build_index_mirrored_subdirs(self) -> None:
        matched = 0
        for subdir in self._iter_candidate_subdirs():
            image_dir = os.path.join(self.image_root, subdir)
            depth_dir = os.path.join(self.depth_root, subdir)
            if not os.path.isdir(image_dir):
                continue
            if not os.path.isdir(depth_dir):
                print(f"[PixelDepthControlDataset] WARN missing mirrored depth dir: {depth_dir}")
                continue
            subdir_pairs = 0
            for entry in os.scandir(image_dir):
                if not entry.is_file() or not entry.name.endswith(".txt"):
                    continue
                stem = entry.name[:-4]
                caption_path = entry.path
                image_path = None
                for ext in IMAGE_EXTS:
                    cand = os.path.join(image_dir, stem + ext)
                    if os.path.exists(cand):
                        image_path = cand
                        break
                if image_path is None:
                    continue
                depth_path = None
                for ext in DEPTH_EXTS:
                    cand = os.path.join(depth_dir, stem + ext)
                    if os.path.exists(cand):
                        depth_path = cand
                        break
                if depth_path is None:
                    continue
                self._append_pair(image_path, depth_path, caption_path)
                matched += 1
                subdir_pairs += 1
                if (self.early_stop_count > 0
                        and len(self.image_paths) >= self.early_stop_count):
                    return
            print(f"[PixelDepthControlDataset] indexed subdir {subdir}: {subdir_pairs} pairs", flush=True)

    def _build_index(self) -> None:
        # Fast path: mirrored subdir layout (matches PixelGen on-disk layout).
        candidate_subdirs = self._iter_candidate_subdirs()
        if self.subdirs is not None or (len(candidate_subdirs) > 0 and all(
            os.path.isdir(os.path.join(self.depth_root, subdir))
            for subdir in candidate_subdirs[:5]
        )):
            return self._build_index_mirrored_subdirs()
        # Fallback: full recursive walk.
        matched = 0
        for dirpath, _, files in os.walk(self.image_root):
            for fname in files:
                if not _is_image_file(fname):
                    continue
                image_path = os.path.join(dirpath, fname)
                relpath = os.path.relpath(image_path, self.image_root)
                depth_path = _find_depth_path(relpath, self.depth_root)
                if depth_path is None:
                    continue
                caption_path = _strip_image_ext(image_path) + ".txt"
                if self.require_caption and not os.path.exists(caption_path):
                    continue
                self._append_pair(image_path, depth_path, caption_path)
                matched += 1
                if (self.early_stop_count > 0
                        and len(self.image_paths) >= self.early_stop_count):
                    return

    def _save_index_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "image_root": self.image_root,
            "depth_root": self.depth_root,
            "image_paths": self.image_paths,
            "depth_paths": self.depth_paths,
            "caption_paths": self.caption_paths,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_index_cache(self, cache_path: str) -> None:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (payload.get("image_root") != self.image_root
                or payload.get("depth_root") != self.depth_root):
            print("[PixelDepthControlDataset] index cache root mismatch, rebuilding...")
            self._build_index()
            return
        self.image_paths = payload["image_paths"]
        self.depth_paths = payload["depth_paths"]
        self.caption_paths = payload.get("caption_paths", [])
        if len(self.caption_paths) != len(self.image_paths):
            self.caption_paths = [
                _strip_image_ext(image_path) + ".txt" for image_path in self.image_paths
            ]
        print(f"[PixelDepthControlDataset] loaded index cache from {cache_path}")

    # ---- PixelDiT-side helpers ----
    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Optional shim for AspectRatioBatchSampler. We use single fixed res."""
        return {
            "height": self.resolution,
            "width": self.resolution,
            "key": str(idx),
            "closest_ratio": "1.0",
            "version": "high_quality",
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor, str]:
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        caption_path = self.caption_paths[idx] if idx < len(self.caption_paths) else ""
        caption = _read_caption(caption_path, default=self.default_caption)
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = self.resize(pil_image)
        pil_image = self.center_crop(pil_image)
        raw_image = to_tensor(pil_image)
        normalized_image = self.normalize(raw_image)
        depth = load_depth_to_tensor(
            depth_path,
            target_size=self.resolution,
            normalize=True,
            repeat_to_3ch=self.depth_repeat_to_3ch,
            invert_depth=self.invert_depth,
        )
        expected_chans = 3 if self.depth_repeat_to_3ch else 1
        assert normalized_image.shape == (3, self.resolution, self.resolution)
        assert depth.shape == (expected_chans, self.resolution, self.resolution)
        return normalized_image, caption, depth, image_path

    def __getitem__(self, idx: int):
        img, caption, depth, image_path = self._sample(idx)
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": 1.0,
            "image_path": image_path,
            "depth": depth,            # [C,H,W] in [0,1], used by cycle loss + sampler
            "control": depth,          # forwarded to the control trainer
            "control_keep": torch.tensor([1.0], dtype=torch.float32),
            "control_mode": "depth",
        }
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        dataindex_info = {"index": idx, "shard": "control_depth", "shardindex": idx}
        return (
            img,                       # [0] image tensor in [-1,1]
            caption,                   # [1] raw caption string (encoded later by tokenizer)
            attention_mask,            # [2] [1,1,L]
            data_info,                 # [3] dict with control fields
            idx,                       # [4]
            "prompt",                  # [5] caption type label
            dataindex_info,            # [6]
            "0.0",                     # [7] dummy clipscore
        )


@DATASETS.register_module(name="PixelSingleControlDataset")
class PixelSingleControlDataset(Dataset):
    """Generic single-control paired dataset for depth / seg / edge.

    It follows the same 8-tuple PixelDiT contract as ``PixelDepthControlDataset``
    but lets a non-depth condition train through the original single-control
    branch.  This is useful for segmentation-only and edge-only baselines that
    should be trained "like depth", without the three-branch gated model.
    """

    def __init__(
        self,
        image_root: str,
        control_root: str,
        control_type: str,
        resolution: int = 512,
        control_normalize: bool = True,
        depth_repeat_to_3ch: bool = False,
        invert_depth: bool = False,
        default_caption: str = "",
        require_caption: bool = True,
        max_samples: int = -1,
        cache_index_path: Optional[str] = None,
        subdirs: Optional[Sequence[str]] = None,
        subdir_range: Optional[Sequence[int]] = None,
        early_stop_count: int = -1,
        max_length: int = 300,
        **_,
    ):
        super().__init__()
        control_type = str(control_type).lower()
        if control_type not in {"depth", "seg", "edge"}:
            raise ValueError(f"control_type must be one of depth/seg/edge, got {control_type}")
        self.image_root = image_root
        self.control_root = control_root
        self.control_type = control_type
        self.resolution = int(resolution)
        self.control_normalize = bool(control_normalize)
        self.depth_repeat_to_3ch = bool(depth_repeat_to_3ch)
        self.invert_depth = bool(invert_depth)
        self.default_caption = str(default_caption)
        self.require_caption = bool(require_caption)
        self.subdirs = self._resolve_subdirs(subdirs, subdir_range)
        self.early_stop_count = int(early_stop_count)
        self.max_length = int(max_length)

        self.image_paths: List[str] = []
        self.control_paths: List[str] = []
        self.caption_paths: List[str] = []

        if cache_index_path is not None and os.path.exists(cache_index_path):
            self._load_index_cache(cache_index_path)
        else:
            self._build_index()
            if cache_index_path is not None:
                try:
                    self._save_index_cache(cache_index_path)
                except Exception as e:
                    print(f"[PixelSingleControlDataset] Failed to save cache: {e}")

        if max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.control_paths = self.control_paths[:max_samples]
            self.caption_paths = self.caption_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"PixelSingleControlDataset found 0 paired samples under "
                f"image_root={image_root} / control_root={control_root} / type={control_type}."
            )
        self.ori_imgs_nums = len(self.image_paths)
        print(
            f"[PixelSingleControlDataset] paired {self.control_type} samples: "
            f"{len(self.image_paths)} (resolution={self.resolution}, control_root={self.control_root})"
        )

        self.resize = Resize(self.resolution)
        self.center_crop = CenterCrop(self.resolution)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @staticmethod
    def _resolve_subdirs(subdirs: Optional[Sequence[str]], subdir_range: Optional[Sequence[int]]) -> Optional[List[str]]:
        if subdirs is not None:
            return list(subdirs)
        if subdir_range is None:
            return None
        if len(subdir_range) != 2:
            raise ValueError(f"subdir_range must be [start, end], got {subdir_range}")
        start, end = int(subdir_range[0]), int(subdir_range[1])
        if end < start:
            raise ValueError(f"subdir_range end must be >= start, got {subdir_range}")
        return [f"sa_{i:06d}" for i in range(start, end + 1)]

    def _iter_candidate_subdirs(self) -> List[str]:
        if self.subdirs is not None:
            return list(self.subdirs)
        return sorted(
            name for name in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, name))
        )

    def _find_control_path(self, stem: str, control_dir: str):
        if self.control_type == "depth":
            for ext in DEPTH_EXTS:
                cand = os.path.join(control_dir, stem + ext)
                if os.path.exists(cand):
                    return cand
        if self.control_type == "seg":
            return _find_seg_path(stem, control_dir)
        return _find_edge_path(stem, control_dir)

    def _append_pair(self, image_path: str, control_path: str, caption_path: str) -> None:
        self.image_paths.append(image_path)
        self.control_paths.append(control_path)
        self.caption_paths.append(caption_path if os.path.exists(caption_path) else "")

    def _build_index(self) -> None:
        for subdir in self._iter_candidate_subdirs():
            image_dir = os.path.join(self.image_root, subdir)
            control_dir = os.path.join(self.control_root, subdir)
            if not os.path.isdir(image_dir):
                continue
            if not os.path.isdir(control_dir):
                print(f"[PixelSingleControlDataset] WARN missing mirrored control dir: {control_dir}")
                continue
            subdir_pairs = 0
            for entry in os.scandir(image_dir):
                if not entry.is_file() or not entry.name.endswith(".txt"):
                    continue
                stem = entry.name[:-4]
                image_path = None
                for ext in IMAGE_EXTS:
                    cand = os.path.join(image_dir, stem + ext)
                    if os.path.exists(cand):
                        image_path = cand
                        break
                if image_path is None:
                    continue
                control_path = self._find_control_path(stem, control_dir)
                if control_path is None:
                    continue
                self._append_pair(image_path, control_path, entry.path)
                subdir_pairs += 1
                if self.early_stop_count > 0 and len(self.image_paths) >= self.early_stop_count:
                    return
            print(
                f"[PixelSingleControlDataset] indexed subdir {subdir}: "
                f"{subdir_pairs} {self.control_type} pairs",
                flush=True,
            )

    def _save_index_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "image_root": self.image_root,
            "control_root": self.control_root,
            "control_type": self.control_type,
            "image_paths": self.image_paths,
            "control_paths": self.control_paths,
            "caption_paths": self.caption_paths,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_index_cache(self, cache_path: str) -> None:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("image_root") != self.image_root
            or payload.get("control_root") != self.control_root
            or payload.get("control_type") != self.control_type
            or "control_paths" not in payload
        ):
            print("[PixelSingleControlDataset] index cache mismatch, rebuilding...")
            self._build_index()
            return
        self.image_paths = payload["image_paths"]
        self.control_paths = payload["control_paths"]
        self.caption_paths = payload.get("caption_paths", [])
        if len(self.caption_paths) != len(self.image_paths):
            self.caption_paths = [
                _strip_image_ext(image_path) + ".txt" for image_path in self.image_paths
            ]
        print(f"[PixelSingleControlDataset] loaded index cache from {cache_path}")

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        return {
            "height": self.resolution,
            "width": self.resolution,
            "key": str(idx),
            "closest_ratio": "1.0",
            "version": "high_quality",
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_control(self, control_path: str) -> torch.Tensor:
        if self.control_type == "depth":
            return load_depth_to_tensor(
                control_path,
                target_size=self.resolution,
                normalize=self.control_normalize,
                repeat_to_3ch=self.depth_repeat_to_3ch,
                invert_depth=self.invert_depth,
            )
        if self.control_type == "seg":
            return load_seg_to_tensor(
                control_path,
                target_size=self.resolution,
                normalize=self.control_normalize,
            )
        return load_edge_from_disk(control_path, target_size=self.resolution)

    def _build_item(self, idx: int):
        image_path = self.image_paths[idx]
        control_path = self.control_paths[idx]
        caption_path = self.caption_paths[idx] if idx < len(self.caption_paths) else ""
        caption = _read_caption(caption_path, default=self.default_caption)
        pil_image = Image.open(image_path).convert("RGB")
        pil_image = self.resize(pil_image)
        pil_image = self.center_crop(pil_image)
        raw_image = to_tensor(pil_image)
        normalized_image = self.normalize(raw_image)
        control = self._load_control(control_path)
        assert normalized_image.shape == (3, self.resolution, self.resolution)
        assert control.shape == (1, self.resolution, self.resolution)

        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": 1.0,
            "image_path": image_path,
            "control": control,
            "control_keep": torch.tensor([1.0], dtype=torch.float32),
            "control_mode": self.control_type,
            f"{self.control_type}_path": control_path,
        }
        if self.control_type == "depth":
            data_info["depth"] = control
        elif self.control_type == "seg":
            data_info["seg"] = control
        else:
            data_info["edge"] = control

        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        dataindex_info = {"index": idx, "shard": f"control_{self.control_type}", "shardindex": idx}
        return (
            normalized_image,
            caption,
            attention_mask,
            data_info,
            idx,
            "prompt",
            dataindex_info,
            "0.0",
        )

    def __getitem__(self, idx: int):
        max_retries = 20 if self.control_type == "edge" else 1
        cur = int(idx)
        for _ in range(max_retries):
            try:
                return self._build_item(cur)
            except Exception as exc:  # noqa: BLE001 - corrupt sample, skip it
                if self.control_type != "edge":
                    raise
                nxt = random.randint(0, len(self) - 1)
                control_path = self.control_paths[cur] if cur < len(self.control_paths) else "?"
                print(
                    f"[PixelSingleControlDataset] sample idx={cur} failed "
                    f"({self.control_type}={control_path}): {exc!r}; skipping -> random idx={nxt}"
                )
                cur = nxt
        raise RuntimeError(
            f"[PixelSingleControlDataset] failed to load a valid sample after "
            f"{max_retries} random retries (last idx={cur})."
        )


class PixelSingleControlEvalDataset(Dataset):
    """Single-control validation dataset for depth / seg / edge."""

    def __init__(
        self,
        image_root: str,
        control_root: str,
        control_type: str,
        resolution: int = 512,
        max_samples: int = 500,
        control_normalize: bool = True,
        invert_depth: bool = False,
        seed_offset: int = 0,
    ):
        super().__init__()
        control_type = str(control_type).lower()
        if control_type not in {"depth", "seg", "edge"}:
            raise ValueError(f"control_type must be one of depth/seg/edge, got {control_type}")
        self.image_root = image_root
        self.control_root = control_root
        self.control_type = control_type
        self.resolution = int(resolution)
        self.max_samples = int(max_samples)
        self.control_normalize = bool(control_normalize)
        self.invert_depth = bool(invert_depth)
        self.seed_offset = int(seed_offset)
        self.samples: List[Dict[str, str]] = []
        self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"PixelSingleControlEvalDataset found 0 paired samples under "
                f"image_root={image_root}, control_root={control_root}, type={control_type}"
            )
        self.ori_imgs_nums = len(self.samples)
        print(
            f"[PixelSingleControlEvalDataset] paired {self.control_type} eval samples: "
            f"{len(self.samples)} (resolution={self.resolution})"
        )

    def _find_image_path(self, stem: str):
        for ext in IMAGE_EXTS:
            candidate = os.path.join(self.image_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _find_control_path(self, stem: str):
        if self.control_type == "depth":
            for ext in DEPTH_EXTS:
                candidate = os.path.join(self.control_root, stem + ext)
                if os.path.exists(candidate):
                    return candidate
            return None
        if self.control_type == "seg":
            return _find_seg_path(stem, self.control_root)
        return _find_edge_path(stem, self.control_root)

    def _build_index(self):
        count = 0
        for entry in sorted(os.scandir(self.image_root), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue
            stem = entry.name[:-4]
            image_path = self._find_image_path(stem)
            control_path = self._find_control_path(stem)
            if image_path is None or control_path is None:
                continue
            self.samples.append(
                {
                    "stem": stem,
                    "caption_path": entry.path,
                    "image_path": image_path,
                    "control_path": control_path,
                }
            )
            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def _load_control(self, control_path: str) -> torch.Tensor:
        if self.control_type == "depth":
            return load_depth_to_tensor(
                control_path,
                target_size=self.resolution,
                normalize=self.control_normalize,
                repeat_to_3ch=False,
                invert_depth=self.invert_depth,
            )
        if self.control_type == "seg":
            return load_seg_to_tensor(
                control_path,
                target_size=self.resolution,
                normalize=self.control_normalize,
            )
        return load_edge_from_disk(control_path, target_size=self.resolution)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        caption = _read_caption(sample["caption_path"], default="")
        control = self._load_control(sample["control_path"])
        assert control.shape == (1, self.resolution, self.resolution)
        return {
            "stem": f"{sample['stem']}_{self.control_type}",
            "caption": caption,
            "image_path": sample["image_path"],
            f"{self.control_type}_path": sample["control_path"],
            "control_mode": self.control_type,
            "seed_index": idx + self.seed_offset,
            "control": control,
            "control_keep": torch.tensor([1.0], dtype=torch.float32),
        }


class PixelDepthEvalDataset(Dataset):
    """PixelGen-compatible depth eval dataset.

    Mirrors PixelGen's ``DepthConditionEvalDataset``: scan a leaf image
    directory by sorted ``*.txt`` filenames, require paired image/depth, then
    keep the first ``max_samples`` complete triples.
    """

    def __init__(
        self,
        image_root: str,
        depth_root: str,
        resolution: int = 512,
        max_samples: int = 500,
        depth_repeat_to_3ch: bool = False,
        invert_depth: bool = False,
    ):
        super().__init__()
        self.image_root = image_root
        self.depth_root = depth_root
        self.resolution = int(resolution)
        self.max_samples = int(max_samples)
        self.depth_repeat_to_3ch = bool(depth_repeat_to_3ch)
        self.invert_depth = bool(invert_depth)
        self.samples: List[Dict[str, str]] = []
        self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"PixelDepthEvalDataset found 0 paired samples under "
                f"image_root={image_root}, depth_root={depth_root}"
            )
        self.ori_imgs_nums = len(self.samples)
        print(
            f"[PixelDepthEvalDataset] paired eval samples: {len(self.samples)} "
            f"(resolution={self.resolution})"
        )

    def _find_image_path(self, stem: str):
        for ext in IMAGE_EXTS:
            candidate = os.path.join(self.image_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _find_depth_path(self, stem: str):
        for ext in DEPTH_EXTS:
            candidate = os.path.join(self.depth_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_index(self):
        count = 0
        for entry in sorted(os.scandir(self.image_root), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue
            stem = entry.name[:-4]
            image_path = self._find_image_path(stem)
            depth_path = self._find_depth_path(stem)
            if image_path is None or depth_path is None:
                continue
            self.samples.append(
                {
                    "stem": stem,
                    "caption_path": entry.path,
                    "image_path": image_path,
                    "depth_path": depth_path,
                }
            )
            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        caption = _read_caption(sample["caption_path"], default="")
        depth = load_depth_to_tensor(
            sample["depth_path"],
            target_size=self.resolution,
            normalize=True,
            repeat_to_3ch=self.depth_repeat_to_3ch,
            invert_depth=self.invert_depth,
        )
        expected_chans = 3 if self.depth_repeat_to_3ch else 1
        assert depth.shape == (expected_chans, self.resolution, self.resolution)
        return {
            "stem": sample["stem"],
            "caption": caption,
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
            "control": depth,
            "control_keep": torch.tensor([1.0], dtype=torch.float32),
        }


class PixelMultiControlEvalDataset(Dataset):
    """Depth+seg validation dataset expanded over depth/seg/depth_seg modes."""

    CONTROL_MODES = ("depth", "seg", "depth_seg")

    def __init__(
        self,
        image_root: str,
        depth_root: str,
        seg_root: str,
        resolution: int = 512,
        max_samples: int = 500,
        invert_depth: bool = False,
        seg_normalize: bool = True,
        seed_offset: int = 0,
        control_modes: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.image_root = image_root
        self.depth_root = depth_root
        self.seg_root = seg_root
        self.resolution = int(resolution)
        self.max_samples = int(max_samples)
        self.invert_depth = bool(invert_depth)
        self.seg_normalize = bool(seg_normalize)
        self.seed_offset = int(seed_offset)
        if control_modes is None:
            self.control_modes = tuple(self.CONTROL_MODES)
        else:
            allowed = set(self.CONTROL_MODES)
            self.control_modes = tuple(str(m) for m in control_modes)
            invalid = [m for m in self.control_modes if m not in allowed]
            if invalid:
                raise ValueError(f"Unsupported PixelThreeControlEvalDataset control_modes={invalid}")
            if not self.control_modes:
                raise ValueError("PixelThreeControlEvalDataset control_modes must not be empty")
        self.samples: List[Dict[str, str]] = []
        self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"PixelMultiControlEvalDataset found 0 paired samples under "
                f"image_root={image_root}, depth_root={depth_root}, seg_root={seg_root}"
            )
        self.ori_imgs_nums = len(self.samples)
        print(
            f"[PixelMultiControlEvalDataset] paired eval samples: {len(self.samples)} "
            f"x {len(self.CONTROL_MODES)} modes = {len(self)} items "
            f"(resolution={self.resolution})"
        )

    def _find_image_path(self, stem: str):
        for ext in IMAGE_EXTS:
            candidate = os.path.join(self.image_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _find_depth_path(self, stem: str):
        for ext in DEPTH_EXTS:
            candidate = os.path.join(self.depth_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_index(self):
        count = 0
        for entry in sorted(os.scandir(self.image_root), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue
            stem = entry.name[:-4]
            image_path = self._find_image_path(stem)
            depth_path = self._find_depth_path(stem)
            seg_path = _find_seg_path(stem, self.seg_root)
            if image_path is None or depth_path is None or seg_path is None:
                continue
            self.samples.append(
                {
                    "stem": stem,
                    "caption_path": entry.path,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "seg_path": seg_path,
                }
            )
            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def __len__(self):
        return len(self.samples) * len(self.CONTROL_MODES)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.CONTROL_MODES)
        mode = self.CONTROL_MODES[idx % len(self.CONTROL_MODES)]
        sample = self.samples[sample_idx]
        caption = _read_caption(sample["caption_path"], default="")
        depth = load_depth_to_tensor(
            sample["depth_path"],
            target_size=self.resolution,
            normalize=True,
            repeat_to_3ch=False,
            invert_depth=self.invert_depth,
        )
        seg = load_seg_to_tensor(
            sample["seg_path"],
            target_size=self.resolution,
            normalize=self.seg_normalize,
        )
        zero_depth = torch.zeros_like(depth)
        zero_seg = torch.zeros_like(seg)
        if mode == "depth":
            control = torch.cat([depth, zero_seg], dim=0)
            control_keep = torch.tensor([1.0, 0.0], dtype=torch.float32)
        elif mode == "seg":
            control = torch.cat([zero_depth, seg], dim=0)
            control_keep = torch.tensor([0.0, 1.0], dtype=torch.float32)
        elif mode == "depth_seg":
            control = torch.cat([depth, seg], dim=0)
            control_keep = torch.tensor([1.0, 1.0], dtype=torch.float32)
        else:
            raise ValueError(f"unknown eval control mode: {mode}")
        return {
            "stem": f"{sample['stem']}_{mode}",
            "caption": caption,
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
            "seg_path": sample["seg_path"],
            "control_mode": mode,
            "seed_index": sample_idx + self.seed_offset,
            "control": control,
            "control_keep": control_keep,
        }


@DATASETS.register_module(name="PixelMultiControlDataset")
class PixelMultiControlDataset(PixelDepthControlDataset):
    """Depth + SAM2-seg paired dataset for multi-control training.

    Adds ``seg`` reads on top of ``PixelDepthControlDataset``. Per-sample control
    mode is sampled batch-wise by the trainer, not here, but the dataset prepares
    a stacked 2-channel ``control`` tensor for downstream convenience.
    """

    def __init__(
        self,
        *args,
        seg_root: str,
        seg_normalize: bool = True,
        enable_control_dropout: bool = True,
        control_modes: Sequence[str] = ("depth", "seg", "depth_seg"),
        control_probs: Sequence[float] = (0.3, 0.3, 0.4),
        **kwargs,
    ):
        self.seg_root = seg_root
        self.seg_normalize = bool(seg_normalize)
        self.enable_control_dropout = bool(enable_control_dropout)
        self.control_modes = list(control_modes)
        self.control_probs = [float(p) for p in control_probs]
        if len(self.control_modes) != len(self.control_probs):
            raise ValueError("control_modes and control_probs must have the same length")
        # Validate against any subset of {depth, seg, edge} joined by '_'.
        # This covers both the legacy depth+seg-only mix and the new
        # depth+seg+edge mix used by ``PixelThreeControlDataset``.
        _CONTROL_TOKENS = {"depth", "seg", "edge"}
        for m in self.control_modes:
            tokens = m.split("_")
            if len(tokens) == 0 or any(tok not in _CONTROL_TOKENS for tok in tokens):
                raise ValueError(
                    f"unsupported control_modes={self.control_modes}; each mode must be "
                    f"a '_'-joined subset of {_CONTROL_TOKENS} (e.g. 'depth', 'depth_seg', "
                    f"'depth_seg_edge')"
                )
        prob_sum = sum(self.control_probs)
        if prob_sum <= 0:
            raise ValueError(f"control_probs must sum to > 0, got {self.control_probs}")
        self.control_probs = [p / prob_sum for p in self.control_probs]
        self.seg_paths: List[str] = []
        super().__init__(*args, **kwargs)
        if len(self.seg_paths) != len(self.image_paths):
            self.seg_paths = self.seg_paths[: len(self.image_paths)]
        print(
            f"[PixelMultiControlDataset] paired depth+seg samples: {len(self.image_paths)} "
            f"(seg_root={self.seg_root})"
        )

    def _append_pair(self, image_path, depth_path, caption_path):
        rel_dir = os.path.relpath(os.path.dirname(image_path), self.image_root)
        seg_dir = os.path.join(self.seg_root, rel_dir)
        stem = _strip_image_ext(os.path.basename(image_path))
        seg_path = _find_seg_path(stem, seg_dir)
        if seg_path is None:
            return
        self.image_paths.append(image_path)
        self.depth_paths.append(depth_path)
        self.seg_paths.append(seg_path)
        self.caption_paths.append(caption_path if os.path.exists(caption_path) else "")

    def _save_index_cache(self, cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "image_root": self.image_root,
            "depth_root": self.depth_root,
            "seg_root": self.seg_root,
            "image_paths": self.image_paths,
            "depth_paths": self.depth_paths,
            "seg_paths": self.seg_paths,
            "caption_paths": self.caption_paths,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_index_cache(self, cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("image_root") != self.image_root
            or payload.get("depth_root") != self.depth_root
            or payload.get("seg_root") != self.seg_root
            or "seg_paths" not in payload
        ):
            print("[PixelMultiControlDataset] index cache mismatch, rebuilding...")
            self._build_index()
            return
        self.image_paths = payload["image_paths"]
        self.depth_paths = payload["depth_paths"]
        self.seg_paths = payload["seg_paths"]
        self.caption_paths = payload.get("caption_paths", [])
        if len(self.caption_paths) != len(self.image_paths):
            self.caption_paths = [
                _strip_image_ext(image_path) + ".txt" for image_path in self.image_paths
            ]
        print(f"[PixelMultiControlDataset] loaded index cache from {cache_path}")

    def __getitem__(self, idx: int):
        img, caption, depth, image_path = self._sample(idx)
        seg_path = self.seg_paths[idx]
        seg = load_seg_to_tensor(seg_path, target_size=self.resolution, normalize=self.seg_normalize)
        assert seg.shape == (1, self.resolution, self.resolution)
        # Pre-stack as 2ch transport tensor. The trainer will optionally zero
        # one channel + flip control_keep depending on the sampled mode.
        control = torch.cat([depth, seg], dim=0)
        control_keep = torch.tensor([1.0, 1.0], dtype=torch.float32)
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": 1.0,
            "image_path": image_path,
            "depth": depth,
            "seg": seg,
            "control": control,         # [2, H, W]
            "control_keep": control_keep,
            "control_mode": "depth_seg",
            "seg_path": seg_path,
        }
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        dataindex_info = {"index": idx, "shard": "control_multi", "shardindex": idx}
        return (
            img,
            caption,
            attention_mask,
            data_info,
            idx,
            "prompt",
            dataindex_info,
            "0.0",
        )


# --------------------------------------------------------------------------
# Three-control (depth + seg + edge) dataset.
# --------------------------------------------------------------------------
@DATASETS.register_module(name="PixelThreeControlDataset")
class PixelThreeControlDataset(PixelMultiControlDataset):
    """Depth + SAM2-seg + Sobel-edge paired dataset for 3-branch control.

    Mirrors :class:`PixelMultiControlDataset` but adds an edge channel
    loaded from ``edge_root``. Samples are indexed only when RGB/caption,
    depth, seg, and edge are all present. The training loop samples one of seven
    control-mode dropout patterns (``depth`` / ``seg`` / ``edge`` /
    ``depth_seg`` / ``depth_edge`` / ``seg_edge`` / ``depth_seg_edge``)
    per step; this dataset always emits the full 3-channel control
    tensor so the trainer is free to zero out the inactive channels.
    """

    def __init__(self, *args, edge_root: str, **kwargs):
        self.edge_root = edge_root
        self.edge_paths: List[str] = []
        super().__init__(*args, **kwargs)
        if len(self.edge_paths) != len(self.image_paths):
            self.edge_paths = self.edge_paths[: len(self.image_paths)]
        print(
            f"[PixelThreeControlDataset] paired depth+seg+edge samples: {len(self.image_paths)} "
            f"(edge_root={self.edge_root})"
        )

    def _append_pair(self, image_path, depth_path, caption_path):
        rel_dir = os.path.relpath(os.path.dirname(image_path), self.image_root)
        seg_dir = os.path.join(self.seg_root, rel_dir)
        edge_dir = os.path.join(self.edge_root, rel_dir)
        stem = _strip_image_ext(os.path.basename(image_path))
        seg_path = _find_seg_path(stem, seg_dir)
        edge_path = _find_edge_path(stem, edge_dir)
        if seg_path is None or edge_path is None:
            return
        self.image_paths.append(image_path)
        self.depth_paths.append(depth_path)
        self.seg_paths.append(seg_path)
        self.edge_paths.append(edge_path)
        self.caption_paths.append(caption_path if os.path.exists(caption_path) else "")

    def _save_index_cache(self, cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {
            "image_root": self.image_root,
            "depth_root": self.depth_root,
            "seg_root": self.seg_root,
            "edge_root": self.edge_root,
            "image_paths": self.image_paths,
            "depth_paths": self.depth_paths,
            "seg_paths": self.seg_paths,
            "edge_paths": self.edge_paths,
            "caption_paths": self.caption_paths,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_index_cache(self, cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("image_root") != self.image_root
            or payload.get("depth_root") != self.depth_root
            or payload.get("seg_root") != self.seg_root
            or payload.get("edge_root") != self.edge_root
            or "seg_paths" not in payload
            or "edge_paths" not in payload
        ):
            print("[PixelThreeControlDataset] index cache mismatch, rebuilding...")
            self._build_index()
            return
        self.image_paths = payload["image_paths"]
        self.depth_paths = payload["depth_paths"]
        self.seg_paths = payload["seg_paths"]
        self.edge_paths = payload["edge_paths"]
        self.caption_paths = payload.get("caption_paths", [])
        if len(self.caption_paths) != len(self.image_paths):
            self.caption_paths = [
                _strip_image_ext(image_path) + ".txt" for image_path in self.image_paths
            ]
        print(f"[PixelThreeControlDataset] loaded index cache from {cache_path}")

    def __getitem__(self, idx: int):
        # Robust to corrupt/truncated control maps: if any modality of this
        # sample fails to load, skip it and draw another random index instead
        # of crashing the whole run (bounded retries to avoid infinite loops).
        max_retries = 20
        cur = int(idx)
        for attempt in range(max_retries):
            try:
                return self._build_item(cur)
            except Exception as exc:  # noqa: BLE001 - corrupt sample, skip it
                nxt = random.randint(0, len(self) - 1)
                edge_path = self.edge_paths[cur] if cur < len(self.edge_paths) else "?"
                print(f"[PixelThreeControlDataset] sample idx={cur} failed "
                      f"(edge={edge_path}): {exc!r}; skipping -> random idx={nxt}")
                cur = nxt
        # Give up after too many corrupt draws.
        raise RuntimeError(
            f"[PixelThreeControlDataset] failed to load a valid sample after "
            f"{max_retries} random retries (last idx={cur})."
        )

    def _build_item(self, idx: int):
        img, caption, depth, image_path = self._sample(idx)
        seg_path = self.seg_paths[idx]
        edge_path = self.edge_paths[idx]
        seg = load_seg_to_tensor(seg_path, target_size=self.resolution, normalize=self.seg_normalize)
        assert seg.shape == (1, self.resolution, self.resolution)
        edge = load_edge_from_disk(edge_path, target_size=self.resolution)
        assert edge.shape == (1, self.resolution, self.resolution)

        control = torch.cat([depth, seg, edge], dim=0)  # [3, H, W]
        control_keep = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        data_info = {
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": 1.0,
            "image_path": image_path,
            "depth": depth,
            "seg": seg,
            "edge": edge,
            "control": control,
            "control_keep": control_keep,
            "control_mode": "depth_seg_edge",
            "seg_path": seg_path,
            "edge_path": edge_path,
        }
        attention_mask = torch.ones(1, 1, self.max_length, dtype=torch.int16)
        dataindex_info = {"index": idx, "shard": "control_three", "shardindex": idx}
        return (
            img,
            caption,
            attention_mask,
            data_info,
            idx,
            "prompt",
            dataindex_info,
            "0.0",
        )


class PixelThreeControlEvalDataset(Dataset):
    """Depth + seg + edge validation dataset expanded over 7 control modes.

    Each underlying image is materialized 7 times, one per control mode in
    ``CONTROL_MODES``. The control tensor is always ``[3, H, W]`` (depth,
    seg, edge) with inactive channels zeroed and ``control_keep`` reflecting
    the sampled mode.
    """

    CONTROL_MODES = (
        "depth",
        "seg",
        "edge",
        "depth_seg",
        "depth_edge",
        "seg_edge",
        "depth_seg_edge",
    )

    def __init__(
        self,
        image_root: str,
        depth_root: str,
        seg_root: str,
        edge_root: Optional[str] = None,
        resolution: int = 512,
        max_samples: int = 500,
        invert_depth: bool = False,
        seg_normalize: bool = True,
        seed_offset: int = 0,
        control_modes: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.image_root = image_root
        self.depth_root = depth_root
        self.seg_root = seg_root
        self.edge_root = edge_root
        self.resolution = int(resolution)
        self.max_samples = int(max_samples)
        self.invert_depth = bool(invert_depth)
        self.seg_normalize = bool(seg_normalize)
        self.seed_offset = int(seed_offset)
        if control_modes is None:
            self.control_modes = tuple(self.CONTROL_MODES)
        else:
            allowed = set(self.CONTROL_MODES)
            self.control_modes = tuple(str(m) for m in control_modes)
            invalid = [m for m in self.control_modes if m not in allowed]
            if invalid:
                raise ValueError(
                    f"Unsupported PixelThreeControlEvalDataset control_modes={invalid}"
                )
            if not self.control_modes:
                raise ValueError("PixelThreeControlEvalDataset control_modes must not be empty")
        self.samples: List[Dict[str, str]] = []
        self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"PixelThreeControlEvalDataset found 0 paired samples under "
                f"image_root={image_root}, depth_root={depth_root}, seg_root={seg_root}"
            )
        self.ori_imgs_nums = len(self.samples)
        print(
            f"[PixelThreeControlEvalDataset] paired eval samples: {len(self.samples)} "
            f"x {len(self.control_modes)} modes = {len(self)} items "
            f"control_modes={self.control_modes} "
            f"(resolution={self.resolution}, edge_root={self.edge_root or 'sobel(rgb)'})"
        )
        from torchvision.transforms import CenterCrop as _CC, Resize as _Rz
        self._resize = _Rz(self.resolution)
        self._center_crop = _CC(self.resolution)

    def _find_image_path(self, stem: str):
        for ext in IMAGE_EXTS:
            candidate = os.path.join(self.image_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _find_depth_path(self, stem: str):
        for ext in DEPTH_EXTS:
            candidate = os.path.join(self.depth_root, stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_index(self):
        count = 0
        for entry in sorted(os.scandir(self.image_root), key=lambda e: e.name):
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue
            stem = entry.name[:-4]
            image_path = self._find_image_path(stem)
            depth_path = self._find_depth_path(stem)
            seg_path = _find_seg_path(stem, self.seg_root)
            if image_path is None or depth_path is None or seg_path is None:
                continue
            self.samples.append({
                "stem": stem,
                "caption_path": entry.path,
                "image_path": image_path,
                "depth_path": depth_path,
                "seg_path": seg_path,
            })
            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def __len__(self):
        return len(self.samples) * len(self.control_modes)

    @staticmethod
    def _mode_to_keep(mode: str) -> torch.Tensor:
        order = {"depth": 0, "seg": 1, "edge": 2}
        keep = torch.zeros(3, dtype=torch.float32)
        for tag in mode.split("_"):
            if tag in order:
                keep[order[tag]] = 1.0
        return keep

    def __getitem__(self, idx):
        sample_idx = idx // len(self.control_modes)
        mode = self.control_modes[idx % len(self.control_modes)]
        sample = self.samples[sample_idx]
        caption = _read_caption(sample["caption_path"], default="")
        depth = load_depth_to_tensor(
            sample["depth_path"],
            target_size=self.resolution,
            normalize=True,
            repeat_to_3ch=False,
            invert_depth=self.invert_depth,
        )
        seg = load_seg_to_tensor(
            sample["seg_path"],
            target_size=self.resolution,
            normalize=self.seg_normalize,
        )
        # Edge: prefer disk if a dir is given; otherwise compute from the RGB.
        edge: Optional[torch.Tensor] = None
        if self.edge_root:
            edge_path = _find_edge_path(sample["stem"], self.edge_root)
            if edge_path is not None:
                try:
                    edge = load_edge_from_disk(edge_path, target_size=self.resolution)
                except Exception as exc:  # noqa: BLE001 - corrupt/truncated edge file
                    print(f"[PixelThreeControlEvalDataset] edge load failed for {edge_path} "
                          f"({exc}); recomputing edge from RGB.")
                    edge = None
        if edge is None:
            pil_image = Image.open(sample["image_path"]).convert("RGB")
            pil_image = self._resize(pil_image)
            pil_image = self._center_crop(pil_image)
            rgb_01 = to_tensor(pil_image)
            edge = compute_edge_from_rgb(rgb_01)
        assert depth.shape == (1, self.resolution, self.resolution)
        assert seg.shape == (1, self.resolution, self.resolution)
        assert edge.shape == (1, self.resolution, self.resolution)

        keep = self._mode_to_keep(mode)
        zero = torch.zeros_like(depth)
        ch_depth = depth if keep[0] > 0 else zero
        ch_seg = seg if keep[1] > 0 else zero
        ch_edge = edge if keep[2] > 0 else zero
        control = torch.cat([ch_depth, ch_seg, ch_edge], dim=0)
        return {
            "stem": f"{sample['stem']}_{mode}",
            "caption": caption,
            "image_path": sample["image_path"],
            "depth_path": sample["depth_path"],
            "seg_path": sample["seg_path"],
            "control_mode": mode,
            "seed_index": sample_idx + self.seed_offset,
            "control": control,
            "control_keep": keep,
        }
