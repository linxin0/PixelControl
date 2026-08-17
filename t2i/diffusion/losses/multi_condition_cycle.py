"""Cycle loss wrapper for depth/seg multi-condition training.

Ported from PixelGen ``src/losses/multi_condition_cycle.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiConditionCycleLoss(nn.Module):
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
        print(
            f"[MultiConditionCycleLoss] depth_weight={self.depth_weight} "
            f"seg_weight={self.seg_weight} edge_weight={self.edge_weight} "
            f"has_edge={self.edge_cycle_loss is not None}"
        )

    def forward(
        self,
        gen_image_m11: torch.Tensor,
        depth_01: torch.Tensor | None = None,
        seg_01: torch.Tensor | None = None,
        gt_image_m11: torch.Tensor | None = None,
        control_mode: str = "depth_seg",
    ) -> torch.Tensor:
        total = gen_image_m11.new_zeros(())
        tokens = set(control_mode.split("_"))
        if "depth" in tokens and self.depth_cycle_loss is not None and self.depth_weight != 0.0:
            assert depth_01 is not None, "depth cycle requested but depth_01 is None"
            total = total + self.depth_weight * self.depth_cycle_loss(gen_image_m11, depth_01)
        if "seg" in tokens and self.seg_cycle_loss is not None and self.seg_weight != 0.0:
            assert seg_01 is not None, "seg cycle requested but seg_01 is None"
            total = total + self.seg_weight * self.seg_cycle_loss(gen_image_m11, seg_01)
        if "edge" in tokens and self.edge_cycle_loss is not None and self.edge_weight != 0.0:
            assert gt_image_m11 is not None, "edge cycle requested but gt_image_m11 is None"
            total = total + self.edge_weight * self.edge_cycle_loss(gen_image_m11, gt_image_m11)
        return total
