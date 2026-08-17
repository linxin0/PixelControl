"""Multi-scale DA3 pyramid depth cycle loss (v10 family).

Ported from PixelGen ``src/losses/depth_cycle_pyramid_da3.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .depth_cycle_da3 import DA3DepthCycleLoss


class DA3PyramidDepthCycleLoss(DA3DepthCycleLoss):
    def __init__(
        self,
        enable_pyramid_cycle_loss: bool = True,
        cycle_scales=(512, 256, 128, 64),
        cycle_scale_weights=(0.75, 0.5, 0.5, 0.25),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_pyramid_cycle_loss = bool(enable_pyramid_cycle_loss)
        self.cycle_scales = [int(s) for s in cycle_scales]
        self.cycle_scale_weights = [float(w) for w in cycle_scale_weights]
        if len(self.cycle_scales) != len(self.cycle_scale_weights):
            raise ValueError(
                "cycle_scales and cycle_scale_weights must have the same length, "
                f"got {self.cycle_scales} and {self.cycle_scale_weights}"
            )
        if any(s <= 0 for s in self.cycle_scales):
            raise ValueError(f"cycle_scales must be positive, got {self.cycle_scales}")
        print(
            f"[DA3PyramidDepthCycleLoss] enable={self.enable_pyramid_cycle_loss} "
            f"scales={self.cycle_scales} weights={self.cycle_scale_weights}"
        )

    @staticmethod
    def _resize_depth(depth: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(
            depth.unsqueeze(1),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def _cycle_at_scale(self, pred_depth: torch.Tensor, gt_depth_01: torch.Tensor, scale: int) -> torch.Tensor:
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
        return F.smooth_l1_loss(
            pred_s.float(),
            gt_s.float(),
            beta=self.smooth_l1_beta,
            reduction="mean",
        )

    def forward(self, gen_image_m11: torch.Tensor, gt_depth_01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = gen_image_m11.shape
        assert c == 3, f"expected 3-channel image, got {c}"
        assert gt_depth_01.shape[:2] == (b, 1), f"expected gt depth [B,1,H,W], got {tuple(gt_depth_01.shape)}"
        resized = self._resize_for_da3(gen_image_m11)
        da3_input = self._to_imagenet(resized)
        pred_depth = self._run_da3_depth(da3_input)
        if not self.enable_pyramid_cycle_loss:
            return self._cycle_at_scale(pred_depth, gt_depth_01, self.loss_res)
        total = pred_depth.new_zeros(())
        for scale, weight in zip(self.cycle_scales, self.cycle_scale_weights):
            if weight == 0:
                continue
            total = total + float(weight) * self._cycle_at_scale(pred_depth, gt_depth_01, scale)
        return total
