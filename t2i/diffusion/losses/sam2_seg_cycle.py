"""SAM2-V2 segmentation-target cycle loss.

Ported from PixelGen ``src/losses/sam2_seg_cycle.py``. The training dataset
supplies the cached label map produced by Segment Anything V2 (SAM2); the
expensive full SAM2 re-estimation of generated images remains in
``eval/eval_seg_consistency_sam2.py``. The differentiable training surrogate
compares the generated image's grayscale structural edge map with that cached
SAM2 target at the MPCL pyramid scales. This avoids invoking the non-differentiable
HuggingFace mask-generation pipeline inside the denoising backward pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SAM2SegCycleLoss(nn.Module):
    def __init__(
        self,
        loss_res: int = 128,
        smooth_l1_beta: float = 0.05,
        enable_pyramid_cycle_loss: bool = True,
        cycle_scales=(512, 256, 128, 64),
        cycle_scale_weights=(0.75, 0.5, 0.5, 0.25),
        enable_coarse_to_fine_cycle: bool = True,
        enable_fine_sobel_weight: bool = True,
        alpha_fine_sobel: float = 0.3,
        enable_fine_grad_loss: bool = True,
        lambda_fine_grad: float = 0.01,
        enable_fine_debug_viz: bool = True,
        fine_debug_dir: str = "./outputs/multicontrol_v1_seg_fine_debug",
        fine_debug_every: int = 100,
        fine_debug_max_images: int = 4,
        verifier: str = "segment_anything_v2",
    ):
        super().__init__()
        self.loss_res = int(loss_res)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.enable_pyramid_cycle_loss = bool(enable_pyramid_cycle_loss)
        self.cycle_scales = [int(s) for s in cycle_scales]
        self.cycle_scale_weights = [float(w) for w in cycle_scale_weights]
        if len(self.cycle_scales) != len(self.cycle_scale_weights):
            raise ValueError("cycle_scales and cycle_scale_weights must have same length")
        self.enable_coarse_to_fine_cycle = bool(enable_coarse_to_fine_cycle)
        self.enable_fine_sobel_weight = bool(enable_fine_sobel_weight)
        self.alpha_fine_sobel = float(alpha_fine_sobel)
        self.enable_fine_grad_loss = bool(enable_fine_grad_loss)
        self.lambda_fine_grad = float(lambda_fine_grad)
        self.enable_fine_debug_viz = bool(enable_fine_debug_viz)
        self.fine_debug_dir = str(fine_debug_dir)
        self.fine_debug_every = max(1, int(fine_debug_every))
        self.fine_debug_max_images = max(1, int(fine_debug_max_images))
        self.verifier = str(verifier)
        if self.verifier not in {"segment_anything_v2", "sam2", "cached_sam2"}:
            raise ValueError(
                "SAM2SegCycleLoss expects verifier='segment_anything_v2' "
                f"(or alias sam2/cached_sam2), got {self.verifier!r}"
            )
        self.register_buffer("_fine_debug_calls", torch.zeros((), dtype=torch.long), persistent=False)
        print(
            "[SAM2SegCycleLoss] "
            f"loss_res={self.loss_res} pyramid={self.enable_pyramid_cycle_loss} "
            f"scales={self.cycle_scales} weights={self.cycle_scale_weights} "
            f"coarse_to_fine={self.enable_coarse_to_fine_cycle} verifier={self.verifier}"
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    @staticmethod
    def _sobel_xy(x: torch.Tensor):
        kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return gx, gy

    @classmethod
    def _edge_01(cls, x: torch.Tensor) -> torch.Tensor:
        gx, gy = cls._sobel_xy(x.float())
        edge = torch.sqrt(gx.square() + gy.square() + 1e-8)
        flat = edge.flatten(1)
        lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
        hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        return ((edge - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _to_gray_01(gen_image_m11: torch.Tensor) -> torch.Tensor:
        gen_01 = (gen_image_m11.float() + 1.0) * 0.5
        return (
            0.299 * gen_01[:, 0:1]
            + 0.587 * gen_01[:, 1:2]
            + 0.114 * gen_01[:, 2:3]
        )

    @staticmethod
    def _resize_image(x: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False)

    @staticmethod
    def _resize_seg(seg: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(seg.float(), size=(scale, scale), mode="nearest")

    def _plain_cycle_at_scale(self, gen_image_m11: torch.Tensor, target_seg_01: torch.Tensor, scale: int) -> torch.Tensor:
        gen_s = self._resize_image(gen_image_m11, scale)
        seg_s = self._resize_seg(target_seg_01, scale)
        gen_edge = self._edge_01(self._to_gray_01(gen_s))
        seg_edge = self._edge_01(seg_s)
        return F.smooth_l1_loss(gen_edge, seg_edge, beta=self.smooth_l1_beta, reduction="mean")

    def _fine_cycle_512(self, gen_image_m11: torch.Tensor, target_seg_01: torch.Tensor) -> torch.Tensor:
        gen_512 = self._resize_image(gen_image_m11, 512)
        seg_512 = self._resize_seg(target_seg_01, 512)
        gray_512 = self._to_gray_01(gen_512)
        gen_edge = self._edge_01(gray_512)
        seg_edge = self._edge_01(seg_512)
        per_pixel = F.smooth_l1_loss(gen_edge, seg_edge, beta=self.smooth_l1_beta, reduction="none")
        if self.enable_fine_sobel_weight and self.alpha_fine_sobel > 0:
            weight_map = 1.0 + self.alpha_fine_sobel * seg_edge
        else:
            weight_map = torch.ones_like(seg_edge)
        fine_loss = (per_pixel * weight_map).mean()
        if self.enable_fine_grad_loss and self.lambda_fine_grad > 0:
            pred_gx, pred_gy = self._sobel_xy(gen_edge)
            gt_gx, gt_gy = self._sobel_xy(seg_edge)
            grad_loss = 0.5 * (
                F.smooth_l1_loss(pred_gx, gt_gx, beta=self.smooth_l1_beta, reduction="mean")
                + F.smooth_l1_loss(pred_gy, gt_gy, beta=self.smooth_l1_beta, reduction="mean")
            )
            fine_loss = fine_loss + self.lambda_fine_grad * grad_loss
        self._maybe_save_fine_debug(seg_edge, weight_map)
        return fine_loss

    def _maybe_save_fine_debug(self, seg_edge: torch.Tensor, weight_map: torch.Tensor) -> None:
        if not self.enable_fine_debug_viz:
            return
        self._fine_debug_calls.add_(1)
        call_id = int(self._fine_debug_calls.item())
        if call_id % self.fine_debug_every != 0:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        out_dir = Path(self.fine_debug_dir)
        os.makedirs(out_dir, exist_ok=True)
        n = min(self.fine_debug_max_images, seg_edge.shape[0])
        edge = seg_edge[:n].detach().float().cpu()
        weight = weight_map[:n].detach().float().cpu()
        weight_vis = ((weight - 1.0) / max(self.alpha_fine_sobel, 1e-6)).clamp(0.0, 1.0)
        for i in range(n):
            self._save_gray(edge[i, 0], out_dir / f"call_{call_id:06d}_idx_{i}_seg_edge.png")
            self._save_gray(weight_vis[i, 0], out_dir / f"call_{call_id:06d}_idx_{i}_weight_map.png")

    @staticmethod
    def _save_gray(x: torch.Tensor, path: Path) -> None:
        arr = (x.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)

    def forward(self, gen_image_m11: torch.Tensor, target_seg_01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = gen_image_m11.shape
        assert c == 3, f"expected generated RGB image, got {c} channels"
        assert target_seg_01.shape[:2] == (b, 1), f"expected target seg [B,1,H,W], got {tuple(target_seg_01.shape)}"
        if not self.enable_pyramid_cycle_loss or not self.enable_coarse_to_fine_cycle:
            return self._plain_cycle_at_scale(gen_image_m11, target_seg_01, self.loss_res)
        total = gen_image_m11.new_zeros(())
        for scale, weight in zip(self.cycle_scales, self.cycle_scale_weights):
            if weight == 0:
                continue
            scale_i = int(scale)
            if scale_i == 512:
                loss_s = self._fine_cycle_512(gen_image_m11, target_seg_01)
            else:
                loss_s = self._plain_cycle_at_scale(gen_image_m11, target_seg_01, scale_i)
            total = total + float(weight) * loss_s
        return total
