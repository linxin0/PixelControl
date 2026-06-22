"""DA3 cycle / consistency feedback loss (v6 family).

Ported from PixelGen ``src/losses/depth_cycle_da3.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .depth_consistency_da3 import DA3ConsistencyLoss


class DA3DepthCycleLoss(DA3ConsistencyLoss):
    def __init__(
        self,
        loss_res: int = 128,
        affine_align: bool = True,
        smooth_l1_beta: float = 0.05,
        enable_region_balanced_cycle: bool = False,
        region_patch_size: int = 32,
        region_valid_thresh: float = 0.02,
        far_depth_thresh: float = 0.85,
        min_valid_depth_ratio: float = 0.5,
        lambda_region_cycle: float = 0.1,
        region_cycle_warmup_steps: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_res = int(loss_res)
        self.affine_align = bool(affine_align)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.enable_region_balanced_cycle = bool(enable_region_balanced_cycle)
        self.region_patch_size = int(region_patch_size)
        self.region_valid_thresh = float(region_valid_thresh)
        self.far_depth_thresh = float(far_depth_thresh)
        self.min_valid_depth_ratio = float(min_valid_depth_ratio)
        self.lambda_region_cycle = float(lambda_region_cycle)
        self.region_cycle_warmup_steps = int(region_cycle_warmup_steps)
        self.register_buffer("_region_cycle_calls", torch.zeros((), dtype=torch.long), persistent=False)
        print(
            f"[DA3DepthCycleLoss] loss_res={self.loss_res} "
            f"affine_align={self.affine_align} beta={self.smooth_l1_beta} "
            f"region_balanced={self.enable_region_balanced_cycle}"
        )

    @staticmethod
    def _scale_shift_fit(pred: torch.Tensor, gt: torch.Tensor):
        n = pred.shape[0]
        p = pred.reshape(n, -1)
        g = gt.reshape(n, -1)
        sum_p = p.sum(dim=1)
        sum_p2 = (p * p).sum(dim=1)
        sum_g = g.sum(dim=1)
        sum_pg = (p * g).sum(dim=1)
        count = p.shape[1]
        det = (count * sum_p2 - sum_p * sum_p).clamp_min(1e-8)
        a = (count * sum_pg - sum_p * sum_g) / det
        b = (sum_g - a * sum_p) / count
        return a.view(n, 1, 1), b.view(n, 1, 1)

    @staticmethod
    def _sobel_gradient_01(condition: torch.Tensor) -> torch.Tensor:
        condition = condition.float().unsqueeze(1)
        kx = condition.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        ky = condition.new_tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        gx = F.conv2d(condition, kx, padding=1)
        gy = F.conv2d(condition, ky, padding=1)
        grad = torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(1)
        flat = grad.flatten(1)
        g_min = flat.amin(dim=1, keepdim=True).view(-1, 1, 1)
        g_max = flat.amax(dim=1, keepdim=True).view(-1, 1, 1)
        return ((grad - g_min) / (g_max - g_min).clamp_min(1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _normalize_depth_01(depth: torch.Tensor) -> torch.Tensor:
        return depth.float().clamp(0.0, 1.0)

    def _region_balanced_loss(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        patch = self.region_patch_size
        if patch <= 0:
            raise ValueError(f"region_patch_size must be > 0, got {patch}")
        h, w = gt_depth.shape[-2:]
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch
        err = (pred_depth.float() - gt_depth.float()).abs().unsqueeze(1)
        grad = self._sobel_gradient_01(gt_depth).unsqueeze(1)
        depth_norm = self._normalize_depth_01(gt_depth)
        valid_depth = (depth_norm < self.far_depth_thresh).float().unsqueeze(1)
        if pad_h or pad_w:
            err = F.pad(err, (0, pad_w, 0, pad_h), mode="replicate")
            grad = F.pad(grad, (0, pad_w, 0, pad_h), mode="replicate")
            valid_depth = F.pad(valid_depth, (0, pad_w, 0, pad_h), mode="replicate")
        err_patch = F.avg_pool2d(err, kernel_size=patch, stride=patch).flatten(1)
        edge_density = F.avg_pool2d(grad, kernel_size=patch, stride=patch).flatten(1)
        valid_depth_ratio = F.avg_pool2d(valid_depth, kernel_size=patch, stride=patch).flatten(1)
        valid = (edge_density > self.region_valid_thresh) & (valid_depth_ratio > self.min_valid_depth_ratio)
        valid_f = valid.to(err_patch.dtype)
        per_image = (err_patch * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        per_image = torch.where(valid.any(dim=1), per_image, torch.zeros_like(per_image))
        return per_image.mean()

    def _region_cycle_weight(self) -> float:
        if self.lambda_region_cycle <= 0:
            return 0.0
        if self.region_cycle_warmup_steps <= 0:
            return self.lambda_region_cycle
        self._region_cycle_calls.add_(1)
        scale = min(1.0, float(self._region_cycle_calls.item()) / float(self.region_cycle_warmup_steps))
        return self.lambda_region_cycle * scale

    def forward(self, gen_image_m11: torch.Tensor, gt_depth_01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = gen_image_m11.shape
        assert c == 3, f"expected 3-channel image, got {c}"
        assert gt_depth_01.shape[:2] == (b, 1), f"expected gt depth [B,1,H,W], got {tuple(gt_depth_01.shape)}"
        resized = self._resize_for_da3(gen_image_m11)
        da3_input = self._to_imagenet(resized)
        pred_depth = self._run_da3_depth(da3_input)
        pred_depth = F.interpolate(
            pred_depth.unsqueeze(1),
            size=(self.loss_res, self.loss_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        gt_depth = F.interpolate(
            gt_depth_01,
            size=(self.loss_res, self.loss_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if self.pred_norm == "minmax":
            pred_depth = self._minmax_per_image(pred_depth)
        if self.gt_norm == "minmax":
            gt_depth = self._minmax_per_image(gt_depth)
        if self.affine_align:
            a, b_shift = self._scale_shift_fit(pred_depth.float(), gt_depth.float())
            pred_depth = (a * pred_depth.float() + b_shift).clamp(0.0, 1.0)
        base_cycle = F.smooth_l1_loss(
            pred_depth.float(),
            gt_depth.float(),
            beta=self.smooth_l1_beta,
            reduction="mean",
        )
        if not self.enable_region_balanced_cycle:
            return base_cycle
        region_weight = self._region_cycle_weight()
        if region_weight <= 0:
            return base_cycle
        region_cycle = self._region_balanced_loss(pred_depth, gt_depth)
        return base_cycle + region_weight * region_cycle
