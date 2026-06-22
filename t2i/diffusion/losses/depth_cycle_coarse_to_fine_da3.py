"""Coarse-to-fine DA3 pyramid cycle loss (v12 key innovation).

Ported from PixelGen ``src/losses/depth_cycle_coarse_to_fine_da3.py``.

Design recap:
  * 64 / 128 / 256: plain SmoothL1 depth cycle (global / object geometry).
  * 512: boundary-aware SmoothL1 weighted by target-depth Sobel structure.
  * Optional weak 512-scale gradient consistency.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .depth_cycle_pyramid_da3 import DA3PyramidDepthCycleLoss


class DA3CoarseToFinePyramidDepthCycleLoss(DA3PyramidDepthCycleLoss):
    def __init__(
        self,
        enable_coarse_to_fine_cycle: bool = True,
        enable_fine_sobel_weight: bool = True,
        alpha_fine_sobel: float = 0.3,
        enable_fine_grad_loss: bool = True,
        lambda_fine_grad: float = 0.01,
        enable_fine_debug_viz: bool = True,
        fine_debug_dir: str = "./outputs/depth_control_v1_fine_debug",
        fine_debug_every: int = 100,
        fine_debug_max_images: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_coarse_to_fine_cycle = bool(enable_coarse_to_fine_cycle)
        self.enable_fine_sobel_weight = bool(enable_fine_sobel_weight)
        self.alpha_fine_sobel = float(alpha_fine_sobel)
        self.enable_fine_grad_loss = bool(enable_fine_grad_loss)
        self.lambda_fine_grad = float(lambda_fine_grad)
        self.enable_fine_debug_viz = bool(enable_fine_debug_viz)
        self.fine_debug_dir = str(fine_debug_dir)
        self.fine_debug_every = max(1, int(fine_debug_every))
        self.fine_debug_max_images = max(1, int(fine_debug_max_images))
        self.register_buffer("_fine_debug_calls", torch.zeros((), dtype=torch.long), persistent=False)
        print(
            "[DA3CoarseToFinePyramidDepthCycleLoss] "
            f"enable={self.enable_coarse_to_fine_cycle} "
            f"fine_sobel={self.enable_fine_sobel_weight} "
            f"alpha_fine_sobel={self.alpha_fine_sobel} "
            f"fine_grad={self.enable_fine_grad_loss} "
            f"lambda_fine_grad={self.lambda_fine_grad}"
        )

    @staticmethod
    def _sobel_xy(depth: torch.Tensor):
        depth = depth.float().unsqueeze(1)
        kx = depth.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = depth.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        gx = F.conv2d(depth, kx, padding=1).squeeze(1)
        gy = F.conv2d(depth, ky, padding=1).squeeze(1)
        return gx, gy

    @classmethod
    def _sobel_map_01(cls, depth: torch.Tensor) -> torch.Tensor:
        gx, gy = cls._sobel_xy(depth)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-8)
        flat = grad.flatten(1)
        lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
        hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
        return ((grad - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

    def _aligned_depths_at_scale(self, pred_depth: torch.Tensor, gt_depth_01: torch.Tensor, scale: int):
        pred_s = self._resize_depth(pred_depth, scale)
        gt_s = F.interpolate(
            gt_depth_01,
            size=(scale, scale),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if self.pred_norm == "minmax":
            pred_s = self._minmax_per_image(pred_s)
        if self.gt_norm == "minmax":
            gt_s = self._minmax_per_image(gt_s)
        if self.affine_align:
            a, b_shift = self._scale_shift_fit(pred_s.float(), gt_s.float())
            pred_s = (a * pred_s.float() + b_shift).clamp(0.0, 1.0)
        return pred_s.float(), gt_s.float()

    def _plain_cycle_at_scale(self, pred_depth: torch.Tensor, gt_depth_01: torch.Tensor, scale: int) -> torch.Tensor:
        pred_s, gt_s = self._aligned_depths_at_scale(pred_depth, gt_depth_01, scale)
        return F.smooth_l1_loss(pred_s, gt_s, beta=self.smooth_l1_beta, reduction="mean")

    def _fine_cycle_512(self, pred_depth: torch.Tensor, gt_depth_01: torch.Tensor) -> torch.Tensor:
        pred_512, gt_512 = self._aligned_depths_at_scale(pred_depth, gt_depth_01, 512)
        per_pixel = F.smooth_l1_loss(pred_512, gt_512, beta=self.smooth_l1_beta, reduction="none")
        sobel_map = self._sobel_map_01(gt_512)
        if self.enable_fine_sobel_weight and self.alpha_fine_sobel > 0:
            weight_map = 1.0 + self.alpha_fine_sobel * sobel_map
        else:
            weight_map = torch.ones_like(sobel_map)
        fine_loss = (per_pixel * weight_map).mean()
        if self.enable_fine_grad_loss and self.lambda_fine_grad > 0:
            pred_gx, pred_gy = self._sobel_xy(pred_512)
            gt_gx, gt_gy = self._sobel_xy(gt_512)
            grad_loss = 0.5 * (
                F.smooth_l1_loss(pred_gx, gt_gx, beta=self.smooth_l1_beta, reduction="mean")
                + F.smooth_l1_loss(pred_gy, gt_gy, beta=self.smooth_l1_beta, reduction="mean")
            )
            fine_loss = fine_loss + self.lambda_fine_grad * grad_loss
        self._maybe_save_fine_debug(sobel_map, weight_map)
        return fine_loss

    def _maybe_save_fine_debug(self, sobel_map: torch.Tensor, weight_map: torch.Tensor) -> None:
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
        n = min(self.fine_debug_max_images, sobel_map.shape[0])
        sobel = sobel_map[:n].detach().float().cpu()
        weight = weight_map[:n].detach().float().cpu()
        weight_vis = ((weight - 1.0) / max(self.alpha_fine_sobel, 1e-6)).clamp(0.0, 1.0)
        for i in range(n):
            self._save_gray(sobel[i], out_dir / f"call_{call_id:06d}_idx_{i}_sobel_map.png")
            self._save_gray(weight_vis[i], out_dir / f"call_{call_id:06d}_idx_{i}_weight_map.png")

    @staticmethod
    def _save_gray(x: torch.Tensor, path: Path) -> None:
        arr = (x.clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)

    def forward(self, gen_image_m11: torch.Tensor, gt_depth_01: torch.Tensor) -> torch.Tensor:
        if not self.enable_coarse_to_fine_cycle:
            return super().forward(gen_image_m11, gt_depth_01)
        b, c, h, w = gen_image_m11.shape
        assert c == 3, f"expected 3-channel image, got {c}"
        assert gt_depth_01.shape[:2] == (b, 1), f"expected gt depth [B,1,H,W], got {tuple(gt_depth_01.shape)}"
        resized = self._resize_for_da3(gen_image_m11)
        da3_input = self._to_imagenet(resized)
        pred_depth = self._run_da3_depth(da3_input)
        total = pred_depth.new_zeros(())
        for scale, weight in zip(self.cycle_scales, self.cycle_scale_weights):
            if weight == 0:
                continue
            scale_i = int(scale)
            if scale_i == 512:
                loss_s = self._fine_cycle_512(pred_depth, gt_depth_01)
            else:
                loss_s = self._plain_cycle_at_scale(pred_depth, gt_depth_01, scale_i)
            total = total + float(weight) * loss_s
        return total
