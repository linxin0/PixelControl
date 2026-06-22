"""PixDiTControlTrainer: PixDiTTrainer-compatible wrapper that adds depth
(single) and depth+seg (multi) control via the ``PixDiT_T2I_Control`` core.

Drop-in compatible with PixelDiT's existing training loop / inference call sites:
  * ``forward(x, timestep, y, mask=None, data_info=None, repa_tokens=None, **kwargs)``
  * ``forward_with_dpmsolver(x, timestep, y, mask=None, **kwargs)``

Control tensors are taken from ``kwargs`` (or, as a convenience for the baseline
PixDiT training loop that passes ``data_info`` through, from ``data_info``)
under keys:
  * ``control``       -> tensor [B, C, H, W] or list of such tensors
  * ``control_keep``  -> tensor [B, n_locals]  (optional)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from pixdit_core.pixeldit_t2i_control import PixDiT_T2I_Control  # type: ignore
except Exception as exc:  # pragma: no cover - explicit import error path
    raise ImportError(
        "Failed to import PixDiT_T2I_Control from pixdit_core.pixeldit_t2i_control. "
        "Check repo layout and PYTHONPATH."
    ) from exc


def _extract_control_from_kwargs(kwargs: dict, data_info: Any):
    control = kwargs.get("control", None)
    control_keep = kwargs.get("control_keep", None)
    if control is None and isinstance(data_info, dict):
        control = data_info.get("control", None)
    if control_keep is None and isinstance(data_info, dict):
        control_keep = data_info.get("control_keep", None)
    return control, control_keep


@MODELS.register_module()
class PixDiTControlTrainer(nn.Module):
    """Pixel-space PixDiT trainer with depth / depth+seg control.

    Compared to ``PixDiTTrainer`` the only extra knobs are the depth-encoder /
    adapter hyperparameters under ``extra`` (forwarded to ``PixDiT_T2I_Control``)
    plus an optional ``pretrained_ckpt`` to warm-start the core from a baseline
    PixDiT T2I checkpoint while keeping the new control modules randomly init.
    """

    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 3,
        image_size: int = 512,
        class_dropout_prob: float = 0.0,
        pred_sigma: bool = False,
        learn_sigma: bool = False,
        config=None,
        caption_channels: int = 2304,
        model_max_length: int = 300,
        extra: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        extra = dict(extra or {})

        # Backbone hyper-params identical to PixDiTTrainer
        patch_size = int(extra.get("patch_size", 32))
        num_groups = int(extra.get("num_groups", 24))
        hidden_size = int(extra.get("hidden_size", 1920))
        pixel_hidden_size = int(extra.get("pixel_hidden_size", extra.get("hidden_size_x", 32)))
        pixel_attn_hidden_size = int(extra.get("pixel_attn_hidden_size", hidden_size))
        pixel_num_groups = int(extra.get("pixel_num_groups", num_groups))
        total_depth = int(extra.get("depth", extra.get("num_blocks", 18)))
        patch_depth = int(extra.get("patch_depth", extra.get("num_cond_blocks", total_depth)))
        pixel_depth = int(extra.get("pixel_depth", max(total_depth - patch_depth, 1)))
        num_text_blocks = int(extra.get("num_text_blocks", 4))
        txt_embed_dim = int(extra.get("txt_embed_dim", caption_channels))
        txt_max_length = int(extra.get("txt_max_length", model_max_length))
        use_text_rope = bool(extra.get("use_text_rope", True))
        text_rope_theta = float(extra.get("text_rope_theta", 10000.0))
        repa_encoder_index = int(extra.get("repa_encoder_index", -1))
        use_pixel_abs_pos = bool(extra.get("use_pixel_abs_pos", True))

        # Control-specific knobs
        use_depth_condition = bool(extra.get("use_depth_condition", True))
        control_mode = str(extra.get("control_mode", "single"))
        depth_channels = int(extra.get("depth_channels", 1))
        n_local_controls = int(extra.get("n_local_controls", 1))
        depth_base_channels = int(extra.get("depth_base_channels", 64))
        depth_max_channels = int(extra.get("depth_max_channels", 512))
        inject_every = int(extra.get("inject_every", 1))
        inject_layer_indices = extra.get("inject_layer_indices", None)
        init_gate = float(extra.get("init_gate", 1.0))
        enable_structure_inject = bool(extra.get("enable_structure_inject", True))
        alpha_inject = float(extra.get("alpha_inject", 2.0))
        freeze_backbone = bool(extra.get("freeze_backbone", False))
        pretrained_ckpt = extra.get("pretrained_ckpt", None)
        load_strict = bool(extra.get("load_strict", False))
        load_prefix = str(extra.get("load_prefix", "core."))
        skip_pretrained_modules = extra.get("skip_pretrained_modules", None)
        if skip_pretrained_modules is not None:
            skip_pretrained_modules = tuple(str(v) for v in skip_pretrained_modules)
        # Independent-branches knobs (new). Defaults preserve the legacy
        # single-control behaviour (control_names is only consulted when
        # control_mode="multi").
        control_names = tuple(extra.get("control_names", ("depth", "seg", "edge")))
        control_structure_inject = extra.get("control_structure_inject", None)
        if control_structure_inject is not None:
            control_structure_inject = tuple(bool(v) for v in control_structure_inject)
        init_gate_logits = tuple(
            float(v) for v in extra.get("init_gate_logits", (0.5, 0.0, 0.0))
        )
        freeze_depth_branch = bool(extra.get("freeze_depth_branch", False))
        freeze_control_branches = extra.get("freeze_control_branches", None)
        if freeze_control_branches is not None:
            freeze_control_branches = tuple(str(v) for v in freeze_control_branches)
        # Cache for use by an external optimizer-building helper. Not consumed
        # inside this wrapper.
        self._depth_branch_lr_scale = float(extra.get("depth_branch_lr_scale", 1.0))
        self._seg_branch_lr_scale = float(extra.get("seg_branch_lr_scale", 1.0))
        self._edge_branch_lr_scale = float(extra.get("edge_branch_lr_scale", 1.0))
        self._gate_lr_scale = float(extra.get("gate_lr_scale", 1.0))

        self.core = PixDiT_T2I_Control(
            in_channels=in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks,
            patch_size=patch_size,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            use_text_rope=use_text_rope,
            text_rope_theta=text_rope_theta,
            repa_encoder_index=repa_encoder_index,
            use_pixel_abs_pos=use_pixel_abs_pos,
            use_depth_condition=use_depth_condition,
            control_mode=control_mode,
            depth_channels=depth_channels,
            n_local_controls=n_local_controls,
            depth_base_channels=depth_base_channels,
            depth_max_channels=depth_max_channels,
            inject_every=inject_every,
            inject_layer_indices=inject_layer_indices,
            init_gate=init_gate,
            enable_structure_inject=enable_structure_inject,
            alpha_inject=alpha_inject,
            freeze_backbone=freeze_backbone,
            control_names=control_names,
            control_structure_inject=control_structure_inject,
            init_gate_logits=init_gate_logits,
            freeze_depth_branch=freeze_depth_branch,
            freeze_control_branches=freeze_control_branches,
            pretrained_ckpt=pretrained_ckpt,
            load_strict=load_strict,
            load_prefix=load_prefix,
            skip_pretrained_modules=skip_pretrained_modules,
        )

        self.image_size = int(image_size)
        self.patch_size = patch_size
        self.pred_sigma = bool(pred_sigma)
        self.config = config
        self._txt_embed_dim = int(txt_embed_dim)
        if int(caption_channels) != self._txt_embed_dim:
            raise ValueError(
                f"caption_channels {caption_channels} != txt_embed_dim {self._txt_embed_dim}"
            )

        # REPA projector (same shape contract as PixDiTTrainer).
        projector_dim = 2048
        self._repa_projector = nn.Sequential(
            nn.Linear(self.core.hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, 768),
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def depth_branch_lr_scale(self) -> float:
        """LR scale for the depth control branch.

        Used by :func:`build_control_param_groups` (in ``train_control.py``)
        when constructing the optimizer so the loaded depth-control weights
        receive a smaller learning rate than the freshly-initialized
        seg/edge branches.
        """
        return self._depth_branch_lr_scale

    @property
    def seg_branch_lr_scale(self) -> float:
        return self._seg_branch_lr_scale

    @property
    def edge_branch_lr_scale(self) -> float:
        return self._edge_branch_lr_scale

    @property
    def gate_lr_scale(self) -> float:
        return self._gate_lr_scale

    def get_param_groups(
        self,
        base_lr: float,
        weight_decay: float = 0.0,
        depth_branch_lr_scale: Optional[float] = None,
        seg_branch_lr_scale: Optional[float] = None,
        edge_branch_lr_scale: Optional[float] = None,
        gate_lr_scale: Optional[float] = None,
    ):
        """Forward to ``self.core.get_param_groups`` honouring the cached
        per-branch LR scales.
        """
        return self.core.get_param_groups(
            base_lr=base_lr,
            depth_branch_lr_scale=(
                self.depth_branch_lr_scale
                if depth_branch_lr_scale is None else float(depth_branch_lr_scale)
            ),
            seg_branch_lr_scale=(
                self.seg_branch_lr_scale
                if seg_branch_lr_scale is None else float(seg_branch_lr_scale)
            ),
            edge_branch_lr_scale=(
                self.edge_branch_lr_scale
                if edge_branch_lr_scale is None else float(edge_branch_lr_scale)
            ),
            gate_lr_scale=(
                self.gate_lr_scale
                if gate_lr_scale is None else float(gate_lr_scale)
            ),
            weight_decay=weight_decay,
        )

    @property
    def last_gate_weights(self):
        """Last per-layer per-sample gate weights ``[num_inject, B, N]``
        (CPU fp32) cached during the most recent forward call. ``None``
        when the model is in legacy single-control mode or before any
        forward.
        """
        return getattr(self.core, "_last_gate_weights", None)

    def _process_text(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 4:
            y_proc = y.squeeze(1)
        elif y.dim() == 3:
            y_proc = y
        else:
            raise ValueError("PixDiTControlTrainer expects y of shape [B,1,L,C] or [B,L,C]")
        y_proc = y_proc.to(self.dtype)
        if y_proc.shape[-1] != self._txt_embed_dim:
            raise RuntimeError(
                f"PixDiTControlTrainer: text embedding dim {y_proc.shape[-1]} != "
                f"expected {self._txt_embed_dim}."
            )
        return y_proc

    def _to_dtype(self, control, control_keep):
        if isinstance(control, torch.Tensor):
            control = control.to(self.dtype)
        elif isinstance(control, (list, tuple)):
            control = [c.to(self.dtype) if isinstance(c, torch.Tensor) else c for c in control]
        if isinstance(control_keep, torch.Tensor):
            control_keep = control_keep.to(self.dtype)
        return control, control_keep

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        data_info: Any = None,
        repa_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y_proc = self._process_text(y)

        if hasattr(self.core, "last_repa_tokens"):
            self.core.last_repa_tokens = None

        control, control_keep = _extract_control_from_kwargs(kwargs, data_info)
        control, control_keep = self._to_dtype(control, control_keep)

        out = self.core(
            x,
            timestep,
            y_proc,
            s=None,
            mask=None,
            control=control,
            control_keep=control_keep,
        )

        repa_loss = None
        if repa_tokens is not None:
            repa_tokens = repa_tokens.to(self.dtype)
        if (
            repa_tokens is not None
            and getattr(self.core, "last_repa_tokens", None) is not None
        ):
            proj_tokens = self._repa_projector(self.core.last_repa_tokens)
            proj_tokens = F.normalize(proj_tokens, dim=-1)
            B, Td, C = repa_tokens.shape
            Bu, Tu, Cu = proj_tokens.shape
            h_u = int(Tu ** 0.5)
            h_d = int(Td ** 0.5)
            if h_u * h_u == Tu and h_d * h_d == Td:
                if Td > Tu:
                    dino_2d = repa_tokens.permute(0, 2, 1).reshape(B, C, h_d, h_d)
                    dino_resized = F.interpolate(dino_2d, size=(h_u, h_u), mode="bilinear", align_corners=False)
                    dino_resized = dino_resized.flatten(2).permute(0, 2, 1)
                    dino_resized = F.normalize(dino_resized, dim=-1)
                    repa_loss = -((proj_tokens * dino_resized).sum(dim=-1)).mean()
                elif Td < Tu:
                    usit_2d = proj_tokens.permute(0, 2, 1).reshape(Bu, Cu, h_u, h_u)
                    usit_resized = F.interpolate(usit_2d, size=(h_d, h_d), mode="bilinear", align_corners=False)
                    usit_resized = usit_resized.flatten(2).permute(0, 2, 1)
                    usit_resized = F.normalize(usit_resized, dim=-1)
                    repa_tokens = F.normalize(repa_tokens, dim=-1)
                    repa_loss = -((usit_resized * repa_tokens).sum(dim=-1)).mean()
                else:
                    repa_tokens = F.normalize(repa_tokens, dim=-1)
                    repa_loss = -((proj_tokens * repa_tokens).sum(dim=-1)).mean()
        return {"x": out, "repa_loss": repa_loss}

    def forward_with_dpmsolver(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        out = self.forward(x, timestep, y, mask=mask, **kwargs)
        if isinstance(out, dict):
            return out["x"]
        return out
