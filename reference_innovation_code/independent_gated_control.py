"""Minimal extracted code for independent control branches + layer-wise gate.

This file contains the useful model-side logic from the current PixelDiT
three-control implementation, stripped away from the full PixelDiT backbone.

How to use it:
1. Keep your original PixelDiT backbone.
2. Build one condition encoder per control: depth / seg / edge.
3. Feed each branch's per-layer condition tokens into
   :class:`IndependentBranchGatedFusion`.
4. Add the returned fused residual to the backbone hidden state at each
   injection layer.

The important behavior is exact:
- single condition: hard select active branch, gate ignored
- multi condition: masked softmax over active controls only
- inactive branch gradients can be masked after backward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTROL_NAMES = ("depth", "seg", "edge")
CONTROL_TOKEN_ORDER = CONTROL_NAMES


def _sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def sobel_structure_map(control: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Compute a patch-token structure map from a control image.

    Args:
        control: [B, C, H, W] control tensor in [0, 1].
        grid_hw: token grid shape, e.g. (H // patch, W // patch).

    Returns:
        [B, grid_h * grid_w, 1] normalized structure weights in [0, 1].
    """
    if control.ndim != 4:
        raise ValueError(f"expected control [B,C,H,W], got {tuple(control.shape)}")
    x = control.float().mean(dim=1, keepdim=True)
    kx, ky = _sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    flat = mag.flatten(1)
    lo = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
    mag = ((mag - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)
    mag = F.interpolate(mag, size=grid_hw, mode="bilinear", align_corners=False)
    return mag.flatten(2).transpose(1, 2).contiguous()


class StructureAwareGatedZeroAdapter(nn.Module):
    """Adapter used by each control branch.

    The original code changed the adapter to expose ``compute_residual`` so
    residuals can be fused across branches before adding to the backbone.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(()))

    def compute_residual(
        self,
        cond: torch.Tensor,
        structure_map: torch.Tensor | None = None,
        alpha_inject: float = 0.5,
    ) -> torch.Tensor:
        residual = self.gate * self.proj(self.norm(cond))
        if structure_map is not None and alpha_inject != 0.0:
            residual = residual * (1.0 + float(alpha_inject) * structure_map.to(residual.dtype))
        return residual

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        structure_map: torch.Tensor | None = None,
        alpha_inject: float = 0.5,
    ) -> torch.Tensor:
        return x + self.compute_residual(cond, structure_map, alpha_inject)


class IndependentBranchGatedFusion(nn.Module):
    """Layer-wise gated fusion for depth / seg / edge branch residuals.

    This module owns only adapters and gate logits. Encoders live outside so it
    can be attached to any PixelDiT-like backbone.

    Args:
        hidden_size: hidden dimension of backbone tokens.
        num_layers: number of injection layers.
        control_names: fixed order, default ("depth", "seg", "edge").
        init_gate_logits: layer-wise gate initialization, default favors depth
            and downweights edge.
        control_structure_inject: per-control flags for structure map injection.
        alpha_inject: residual multiplier strength for structure maps.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 14,
        control_names: Sequence[str] = CONTROL_NAMES,
        init_gate_logits: Sequence[float] = (0.5, 0.0, -0.5),
        control_structure_inject: Sequence[bool] = (True, True, False),
        alpha_inject: float = 2.0,
    ):
        super().__init__()
        self.control_names = tuple(control_names)
        self.num_controls = len(self.control_names)
        self.num_layers = int(num_layers)
        if self.control_names != CONTROL_NAMES:
            raise ValueError(f"minimal code expects control_names={CONTROL_NAMES}, got {self.control_names}")
        if len(init_gate_logits) != self.num_controls:
            raise ValueError("init_gate_logits length must equal number of controls")
        if len(control_structure_inject) != self.num_controls:
            raise ValueError("control_structure_inject length must equal number of controls")

        self.depth_adapters = nn.ModuleList([StructureAwareGatedZeroAdapter(hidden_size) for _ in range(num_layers)])
        self.seg_adapters = nn.ModuleList([StructureAwareGatedZeroAdapter(hidden_size) for _ in range(num_layers)])
        self.edge_adapters = nn.ModuleList([StructureAwareGatedZeroAdapter(hidden_size) for _ in range(num_layers)])
        init_row = torch.tensor(init_gate_logits, dtype=torch.float32)
        self.control_gate_logits = nn.Parameter(init_row.view(1, self.num_controls).repeat(num_layers, 1).clone())

        self.control_structure_inject = tuple(bool(x) for x in control_structure_inject)
        self.alpha_inject = float(alpha_inject)
        self.last_gate_weights: torch.Tensor | None = None

    @staticmethod
    def per_layer_per_sample_weights(gate_logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Return [L, B, 3] fusion weights.

        Single-control samples are hard-selected and do not use the gate.
        Multi-control samples use masked softmax over active controls.
        """
        if gate_logits.ndim != 2:
            raise ValueError(f"gate_logits must be [L,N], got {tuple(gate_logits.shape)}")
        if keep_mask.ndim != 2:
            raise ValueError(f"keep_mask must be [B,N], got {tuple(keep_mask.shape)}")
        num_layers, num_controls = gate_logits.shape
        batch, keep_controls = keep_mask.shape
        if num_controls != keep_controls:
            raise ValueError(f"gate controls {num_controls} != keep controls {keep_controls}")

        keep = (keep_mask > 0).to(gate_logits.dtype)
        active_count = keep.sum(dim=1)
        weights = gate_logits.new_zeros((num_layers, batch, num_controls))

        single = active_count == 1
        if single.any():
            weights[:, single, :] = keep[single].view(1, -1, num_controls)

        multi = active_count > 1
        if multi.any():
            logits = gate_logits[:, None, :].expand(num_layers, int(multi.sum()), num_controls)
            mask = keep[multi].bool().view(1, -1, num_controls)
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            weights[:, multi, :] = torch.softmax(logits, dim=-1)

        return weights

    def _adapters(self) -> list[nn.ModuleList]:
        return [self.depth_adapters, self.seg_adapters, self.edge_adapters]

    def fuse_layer(
        self,
        layer_idx: int,
        branch_tokens: Sequence[torch.Tensor | None],
        keep_mask: torch.Tensor,
        branch_structure_maps: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Fuse one injection layer.

        Args:
            layer_idx: injection layer index in [0, num_layers).
            branch_tokens: length-3 list. Each active branch has [B, T, D].
            keep_mask: [B, 3].
            branch_structure_maps: length-3 list of [B, T, 1] or None.

        Returns:
            fused residual [B, T, D].
        """
        if len(branch_tokens) != self.num_controls:
            raise ValueError("branch_tokens must have length 3")
        if branch_structure_maps is None:
            branch_structure_maps = [None] * self.num_controls
        weights = self.per_layer_per_sample_weights(self.control_gate_logits.to(keep_mask.dtype), keep_mask)
        self.last_gate_weights = weights.detach().float().cpu()

        fused = None
        adapters = self._adapters()
        for branch_idx, tokens in enumerate(branch_tokens):
            if tokens is None:
                continue
            struct = branch_structure_maps[branch_idx]
            if not self.control_structure_inject[branch_idx]:
                struct = None
            residual = adapters[branch_idx][layer_idx].compute_residual(tokens, struct, self.alpha_inject)
            w = weights[layer_idx, :, branch_idx].to(residual.dtype).view(-1, 1, 1)
            part = w * residual
            fused = part if fused is None else fused + part
        if fused is None:
            raise RuntimeError("no active branch tokens were provided")
        return fused

    def forward(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        branch_tokens: Sequence[torch.Tensor | None],
        keep_mask: torch.Tensor,
        branch_structure_maps: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        return hidden + self.fuse_layer(layer_idx, branch_tokens, keep_mask, branch_structure_maps)

    def param_groups(
        self,
        base_lr: float,
        weight_decay: float = 0.0,
        depth_branch_lr_scale: float = 1.0,
        seg_branch_lr_scale: float = 1.0,
        edge_branch_lr_scale: float = 1.0,
        gate_lr_scale: float = 1.0,
    ) -> list[dict]:
        return [
            {
                "name": "depth_branch",
                "params": list(self.depth_adapters.parameters()),
                "lr": base_lr * depth_branch_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "name": "seg_branch",
                "params": list(self.seg_adapters.parameters()),
                "lr": base_lr * seg_branch_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "name": "edge_branch",
                "params": list(self.edge_adapters.parameters()),
                "lr": base_lr * edge_branch_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "name": "gate",
                "params": [self.control_gate_logits],
                "lr": base_lr * gate_lr_scale,
                "weight_decay": weight_decay,
            },
        ]


def mode_to_keep(mode: str, num_controls: int = 3, *, device=None, dtype=None) -> torch.Tensor:
    tokens = set(str(mode).split("_"))
    keep = [
        1.0 if CONTROL_TOKEN_ORDER[i] in tokens else 0.0
        for i in range(num_controls)
    ]
    return torch.tensor(keep, device=device, dtype=dtype or torch.float32)


def apply_multi_control_mode(control: torch.Tensor, mode: str, num_controls: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero inactive control channels and return [B, num_controls] keep mask."""
    if control.ndim != 4 or control.shape[1] % num_controls != 0:
        raise ValueError(f"control shape {tuple(control.shape)} not divisible by num_controls={num_controls}")
    batch = control.shape[0]
    channels_per_control = control.shape[1] // num_controls
    keep = mode_to_keep(mode, num_controls, device=control.device, dtype=control.dtype)
    parts = []
    for i in range(num_controls):
        part = control[:, i * channels_per_control:(i + 1) * channels_per_control]
        parts.append(part if keep[i] > 0 else torch.zeros_like(part))
    control_out = torch.cat(parts, dim=1)
    keep_out = keep.view(1, num_controls).expand(batch, num_controls).contiguous()
    return control_out, keep_out


def sample_control_mode_ddp(
    modes: Sequence[str],
    probs: Sequence[float],
    enable_dropout: bool,
    device: torch.device,
) -> str:
    """Sample one control mode and broadcast it across DDP ranks."""
    if not enable_dropout:
        for fallback in ("depth_seg_edge", "depth_seg"):
            if fallback in modes:
                return fallback
        return modes[0]
    prob_t = torch.tensor(probs, dtype=torch.float32, device=device)
    prob_t = prob_t / prob_t.sum().clamp_min(1e-8)
    idx_t = torch.multinomial(prob_t, num_samples=1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(idx_t, src=0)
    return str(modes[int(idx_t.item())])


def mask_inactive_control_grads(model: nn.Module, control_mode: str) -> None:
    """Mask inactive branch grads for independent depth/seg/edge branches."""
    tokens = set(str(control_mode).split("_"))
    active = {
        "depth": "depth" in tokens,
        "seg": "seg" in tokens,
        "edge": "edge" in tokens,
    }
    gate_active = sum(active.values()) > 1
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "control_gate_logits" in name:
            if not gate_active:
                param.grad = None
        elif "depth_encoder" in name or "depth_adapters" in name:
            if not active["depth"]:
                param.grad = None
        elif "seg_encoder" in name or "seg_adapters" in name:
            if not active["seg"]:
                param.grad = None
        elif "edge_encoder" in name or "edge_adapters" in name:
            if not active["edge"]:
                param.grad = None


@dataclass(frozen=True)
class ControlSamplingConfig:
    modes: tuple[str, ...] = (
        "depth",
        "seg",
        "edge",
        "depth_seg",
        "depth_edge",
        "seg_edge",
        "depth_seg_edge",
    )
    probs: tuple[float, ...] = (0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19)

