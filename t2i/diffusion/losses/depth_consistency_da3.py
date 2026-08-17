"""Frozen DA3 depth consistency loss.

Ported verbatim from the original PixelGen depth-control losses
(``src/losses/depth_consistency_da3.py``) with only import-path tweaks so it
can live under ``t2i/diffusion/losses``.

For full design rationale (in-place patch for nested DA3, ImageNet
normalization, SILog math), see the PixelGen docstring inline below.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patch_moviepy_for_da3():
    """DA3's ``api.py`` transitively imports ``moviepy.editor``. moviepy>=2
    removed that submodule but exposes the same names from the package root.
    We alias it so DA3 import chain completes.
    """
    try:
        import moviepy.editor  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    try:
        import moviepy  # type: ignore

        sys.modules["moviepy.editor"] = moviepy
    except ModuleNotFoundError:
        pass


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


class DA3ConsistencyLoss(nn.Module):
    """Frozen DA3 + SILog loss between generated-image depth and conditioning depth."""

    DA3_PATCH_SIZE = 14
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        da3_src: str = "./third_party/depth-anything-3/src",
        da3_model_dir: str = "./pretrained/DA3NESTED-GIANT-LARGE-1.1",
        process_res: int = 504,
        silog_lambda: float = 0.85,
        silog_eps: float = 1e-3,
        gt_norm: str = "minmax",
        pred_norm: str = "minmax",
        load_dtype: Optional[str] = None,
        normalize_input: bool = True,
    ):
        super().__init__()
        assert gt_norm in ("none", "minmax"), gt_norm
        assert pred_norm in ("none", "minmax"), pred_norm

        self.da3_src = da3_src
        self.da3_model_dir = da3_model_dir
        self.process_res = _round_to_multiple(int(process_res), self.DA3_PATCH_SIZE)
        self.silog_lambda = float(silog_lambda)
        self.silog_eps = float(silog_eps)
        self.gt_norm = gt_norm
        self.pred_norm = pred_norm
        self.normalize_input = bool(normalize_input)

        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)
        _patch_moviepy_for_da3()
        from depth_anything_3.api import DepthAnything3

        if not os.path.isdir(da3_model_dir):
            raise FileNotFoundError(f"DA3 checkpoint dir not found: {da3_model_dir}")
        print(f"[DA3ConsistencyLoss] loading DA3 from {da3_model_dir}")
        da3 = DepthAnything3.from_pretrained(da3_model_dir)
        da3.eval()
        for p in da3.parameters():
            p.requires_grad = False
        if load_dtype is not None:
            target = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[load_dtype]
            da3 = da3.to(target)
        self.da3 = da3

        inner = self.da3.model
        if hasattr(inner, "da3") and hasattr(inner, "da3_metric"):
            self._install_nested_inplace_patch(inner)
            print(
                "[DA3ConsistencyLoss] nested DA3 detected -> patched "
                "_apply_depth_alignment to be out-of-place."
            )

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        print(
            f"[DA3ConsistencyLoss] ready: process_res={self.process_res} "
            f"silog_lambda={self.silog_lambda} pred_norm={self.pred_norm} "
            f"gt_norm={self.gt_norm}"
        )

    @staticmethod
    def _install_nested_inplace_patch(nested_net) -> None:
        """Replace ``_apply_depth_alignment`` with an out-of-place version so
        autograd doesn't trip on version-bumped tensors during training.
        """
        from depth_anything_3.utils.alignment import (
            compute_alignment_mask,
            compute_sky_mask,
            least_squares_scale_scalar,
            sample_tensor_for_quantile,
        )

        def _apply_depth_alignment_oop(self, output, metric_output):
            non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
            assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"
            depth_conf_ns = output.depth_conf[non_sky_mask]
            depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
            median_conf = torch.quantile(depth_conf_sampled, 0.5)
            align_mask = compute_alignment_mask(
                output.depth_conf,
                non_sky_mask,
                output.depth,
                metric_output.depth,
                median_conf,
            )
            valid_depth = output.depth[align_mask]
            valid_metric_depth = metric_output.depth[align_mask]
            scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)
            output.depth = output.depth * scale_factor
            new_extrinsics = output.extrinsics.clone()
            new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] * scale_factor
            output.extrinsics = new_extrinsics
            output.is_metric = 1
            output.scale_factor = scale_factor.item()
            return output

        import types
        nested_net._apply_depth_alignment = types.MethodType(_apply_depth_alignment_oop, nested_net)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        return torch.nn.modules.module._IncompatibleKeys(missing_keys=[], unexpected_keys=[])

    def train(self, mode: bool = True):
        super().train(mode)
        self.da3.eval()
        return self

    def _to_imagenet(self, pixels_m11: torch.Tensor) -> torch.Tensor:
        pixels_01 = pixels_m11.add(1.0).mul(0.5).clamp_(0.0, 1.0)
        if not self.normalize_input:
            return pixels_01
        return (pixels_01 - self._imagenet_mean) / self._imagenet_std

    def _resize_for_da3(self, pixels_m11: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            pixels_m11,
            size=(self.process_res, self.process_res),
            mode="bilinear",
            align_corners=False,
        )

    def _run_da3_depth(self, da3_input: torch.Tensor) -> torch.Tensor:
        x = da3_input.unsqueeze(1)
        out = self.da3.model(
            x,
            None,
            None,
            [],
            False,
            False,
            "saddle_balanced",
        )
        depth = out["depth"]
        if depth.dim() == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.dim() == 4:
            depth = depth.squeeze(1)
        assert depth.dim() == 3, f"unexpected DA3 depth shape {tuple(depth.shape)}"
        return depth

    @staticmethod
    def _minmax_per_image(x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        rng = (hi - lo).clamp_min(1e-6)
        out = (flat - lo) / rng
        return out.reshape_as(x).clamp_(0.0, 1.0)

    def _silog(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        log_pred = torch.log(pred.clamp_min(self.silog_eps))
        log_gt = torch.log(gt.clamp_min(self.silog_eps))
        d = log_pred - log_gt
        per_image_sq_mean = d.pow(2).mean(dim=(1, 2))
        per_image_mean = d.mean(dim=(1, 2))
        inner = per_image_sq_mean - self.silog_lambda * per_image_mean.pow(2)
        return inner.clamp_min(0.0).mean()

    def forward(self, gen_image_m11: torch.Tensor, gt_depth_01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = gen_image_m11.shape
        assert c == 3, f"expected 3-channel image, got {c}"
        assert gt_depth_01.shape[0] == b, f"batch mismatch: gen={b} vs gt={gt_depth_01.shape[0]}"
        assert gt_depth_01.shape[1] == 1, f"expected 1-channel depth, got {gt_depth_01.shape[1]}"

        resized = self._resize_for_da3(gen_image_m11)
        da3_input = self._to_imagenet(resized)
        pred_depth = self._run_da3_depth(da3_input)
        pred_depth = F.interpolate(
            pred_depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        gt_depth = gt_depth_01.squeeze(1)
        if self.pred_norm == "minmax":
            pred_depth = self._minmax_per_image(pred_depth)
        if self.gt_norm == "minmax":
            gt_depth = self._minmax_per_image(gt_depth)
        return self._silog(pred_depth.float(), gt_depth.float())
