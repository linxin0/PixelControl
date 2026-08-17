"""Depth condition encoder + gated zero-init adapters for PixDiT control.

Ported verbatim (interface-preserving) from PixelGen
``src/models/depth_condition_encoder_v2.py``. Keeps PixelGen depth-control v11/v12
checkpoints binary-compatible with the new PixDiT-side adapters.

The encoder produces ``[B, num_patches, hidden_size]`` tokens aligned to the
PixDiT patch grid; the adapters add them as zero-initialized residuals after
selected transformer blocks.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU. Stride>1 downsamples by 2."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        ngroups = min(8, out_ch)
        self.norm = nn.GroupNorm(ngroups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthEncoder(nn.Module):
    """Convolutional depth encoder producing patch-aligned tokens.

    Input  : depth map ``[B, in_channels, H, W]`` with H == W == input_size.
    Output : tokens ``[B, num_patches, hidden_size]`` where
             ``num_patches = (input_size / patch_size) ** 2``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 1536,
        patch_size: int = 16,
        base_channels: int = 64,
        max_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)

        n_stages = int(math.log2(max(1, patch_size)))
        layers = []
        cur_ch = in_channels
        next_ch = base_channels
        for _ in range(n_stages):
            layers.append(_ConvBlock(cur_ch, next_ch, stride=2))
            cur_ch = next_ch
            next_ch = min(next_ch * 2, max_channels)
        layers.append(_ConvBlock(cur_ch, cur_ch, stride=1))
        self.stem = nn.Sequential(*layers)
        self.proj = nn.Conv2d(cur_ch, hidden_size, kernel_size=1)

        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.stem(depth)
        H_in, W_in = depth.shape[-2:]
        target_h = max(1, H_in // self.patch_size)
        target_w = max(1, W_in // self.patch_size)
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.adaptive_avg_pool2d(x, (target_h, target_w))
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class GatedZeroAdapter(nn.Module):
    """Zero-initialized residual adapter with a per-layer learnable gate.

    final output: x + gate * proj(LN(cond))
    """

    def __init__(self, hidden_size: int, init_gate: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if float(init_gate) == 0.0:
            warnings.warn(
                "GatedZeroAdapter(init_gate=0.0) combined with zero-init proj "
                "creates a dead-start: both gradients are zero and the adapter "
                "will never learn. Use init_gate>0 (e.g., 1.0).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.gate = nn.Parameter(torch.full((1,), float(init_gate)))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.proj(self.norm(cond))


class StructureAwareGatedZeroAdapter(GatedZeroAdapter):
    """Gated adapter that can spatially modulate its residual before injection.

    structure_map: ``[B, N, 1]`` in [0, 1] (Sobel of the condition).
    """

    def compute_residual(
        self,
        cond: torch.Tensor,
        structure_map: torch.Tensor | None = None,
        alpha_inject: float = 0.5,
    ) -> torch.Tensor:
        """Return the (sobel-modulated) residual without adding it to ``x``.

        Exposed so callers that need to mix residuals from multiple parallel
        control branches (e.g. ``PixDiT_T2I_Control`` independent-branches
        path) can weight each branch's residual before summation.
        """
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


@torch.no_grad()
def sobel_structure_map(depth: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Return token-grid Sobel structure map ``[B, Ht*Wt, 1]`` in [0, 1].

    Depth tensor of shape ``[B, C, H, W]``; if C>1 we average to 1ch first.
    The map is computed in fp32 then cast back to the input dtype. This is a
    pure forward operator (no learnable parameters); kept under ``no_grad``
    because the Sobel kernel is constant data.
    """
    if depth.shape[1] > 1:
        d = depth.mean(dim=1, keepdim=True)
    else:
        d = depth
    d = d.float()
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=d.device,
        dtype=d.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=d.device,
        dtype=d.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(d, kx, padding=1)
    gy = F.conv2d(d, ky, padding=1)
    grad = torch.sqrt(gx.square() + gy.square() + 1e-8)
    flat = grad.flatten(1)
    g_min = flat.amin(dim=1, keepdim=True).view(-1, 1, 1, 1)
    g_max = flat.amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
    grad = (grad - g_min) / (g_max - g_min).clamp_min(1e-6)
    structure = F.interpolate(grad, size=target_hw, mode="bilinear", align_corners=False)
    return structure.flatten(2).transpose(1, 2).contiguous().to(depth.dtype)
