"""Minimal extracted cycle losses for the user's multi-control method."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiConditionCycleLoss(nn.Module):
    """Dispatch cycle losses by active control mode."""

    def __init__(
        self,
        depth_cycle_loss: nn.Module | None = None,
        seg_cycle_loss: nn.Module | None = None,
        edge_cycle_loss: nn.Module | None = None,
        depth_weight: float = 1.0,
        seg_weight: float = 1.0,
        edge_weight: float = 1.0,
    ):
        super().__init__()
        self.depth_cycle_loss = depth_cycle_loss
        self.seg_cycle_loss = seg_cycle_loss
        self.edge_cycle_loss = edge_cycle_loss
        self.depth_weight = float(depth_weight)
        self.seg_weight = float(seg_weight)
        self.edge_weight = float(edge_weight)

    def forward(
        self,
        gen_image_m11: torch.Tensor,
        depth_01: torch.Tensor | None = None,
        seg_01: torch.Tensor | None = None,
        gt_image_m11: torch.Tensor | None = None,
        control_mode: str = "depth_seg",
    ) -> torch.Tensor:
        total = gen_image_m11.new_zeros(())
        tokens = set(str(control_mode).split("_"))
        if "depth" in tokens and self.depth_cycle_loss is not None and self.depth_weight != 0.0:
            if depth_01 is None:
                raise ValueError("depth cycle requested but depth_01 is None")
            total = total + self.depth_weight * self.depth_cycle_loss(gen_image_m11, depth_01)
        if "seg" in tokens and self.seg_cycle_loss is not None and self.seg_weight != 0.0:
            if seg_01 is None:
                raise ValueError("seg cycle requested but seg_01 is None")
            total = total + self.seg_weight * self.seg_cycle_loss(gen_image_m11, seg_01)
        if "edge" in tokens and self.edge_cycle_loss is not None and self.edge_weight != 0.0:
            if gt_image_m11 is None:
                raise ValueError("edge cycle requested but gt_image_m11 is None")
            total = total + self.edge_weight * self.edge_cycle_loss(gen_image_m11, gt_image_m11)
        return total


class SoftCannyImagePyramidCycleLoss(nn.Module):
    """Differentiable edge consistency loss.

    Generated RGB and GT RGB are both converted to soft edge maps with the same
    sampled threshold. This avoids directly matching noisy/random offline Canny
    labels.
    """

    def __init__(
        self,
        loss_res: int = 128,
        smooth_l1_beta: float = 0.05,
        enable_pyramid_cycle_loss: bool = True,
        cycle_scales=(512, 256, 128, 64),
        cycle_scale_weights=(0.75, 0.5, 0.5, 0.25),
        gaussian_kernel: int = 11,
        threshold_min: float = 0.2745,
        threshold_max: float = 0.5882,
        temperature: float = 0.03,
    ):
        super().__init__()
        self.loss_res = int(loss_res)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.enable_pyramid_cycle_loss = bool(enable_pyramid_cycle_loss)
        self.cycle_scales = [int(s) for s in cycle_scales]
        self.cycle_scale_weights = [float(w) for w in cycle_scale_weights]
        if len(self.cycle_scales) != len(self.cycle_scale_weights):
            raise ValueError("cycle_scales and cycle_scale_weights must have same length")
        self.gaussian_kernel = int(gaussian_kernel)
        if self.gaussian_kernel <= 0 or self.gaussian_kernel % 2 == 0:
            raise ValueError(f"gaussian_kernel must be a positive odd int, got {gaussian_kernel}")
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.temperature = float(temperature)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # Stateless loss. Returning empty state keeps checkpoints small.
        if destination is None:
            destination = {}
        return destination

    @staticmethod
    def _to_gray_01(image_m11: torch.Tensor) -> torch.Tensor:
        image_01 = (image_m11.float() + 1.0) * 0.5
        return 0.299 * image_01[:, 0:1] + 0.587 * image_01[:, 1:2] + 0.114 * image_01[:, 2:3]

    @staticmethod
    def _sobel_xy(x: torch.Tensor):
        kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        k = self.gaussian_kernel
        sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords.square()) / (2 * sigma * sigma))
        g = g / g.sum().clamp_min(1e-8)
        kernel = (g[:, None] * g[None, :]).view(1, 1, k, k)
        return F.conv2d(x, kernel, padding=k // 2)

    def _sample_threshold(self, batch: int, device, dtype) -> torch.Tensor:
        lo, hi = sorted((self.threshold_min, self.threshold_max))
        t = torch.rand((batch, 1, 1, 1), device=device, dtype=dtype)
        return lo + (hi - lo) * t

    def _soft_canny_01(self, image_m11: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        gray = self._gaussian_blur(self._to_gray_01(image_m11))
        gx, gy = self._sobel_xy(gray)
        mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
        flat = mag.flatten(1)
        lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
        hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        mag = ((mag - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        return torch.sigmoid((mag - threshold) / max(self.temperature, 1e-6))

    @staticmethod
    def _resize_image(x: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False)

    def _cycle_at_scale(
        self,
        gen_image_m11: torch.Tensor,
        gt_image_m11: torch.Tensor,
        scale: int,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        gen_s = self._resize_image(gen_image_m11, scale)
        gt_s = self._resize_image(gt_image_m11, scale)
        gen_edge = self._soft_canny_01(gen_s, threshold)
        gt_edge = self._soft_canny_01(gt_s, threshold)
        return F.smooth_l1_loss(gen_edge, gt_edge.detach(), beta=self.smooth_l1_beta, reduction="mean")

    def forward(self, gen_image_m11: torch.Tensor, gt_image_m11: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = gen_image_m11.shape
        if c != 3 or gt_image_m11.shape[:2] != (b, 3):
            raise ValueError("SoftCannyImagePyramidCycleLoss expects generated and GT RGB tensors [B,3,H,W]")
        threshold = self._sample_threshold(b, gen_image_m11.device, gen_image_m11.dtype)
        if not self.enable_pyramid_cycle_loss:
            return self._cycle_at_scale(gen_image_m11, gt_image_m11, self.loss_res, threshold)
        total = gen_image_m11.new_zeros(())
        for scale, weight in zip(self.cycle_scales, self.cycle_scale_weights):
            if weight != 0:
                total = total + float(weight) * self._cycle_at_scale(gen_image_m11, gt_image_m11, scale, threshold)
        return total


EdgePyramidCycleLoss = SoftCannyImagePyramidCycleLoss
