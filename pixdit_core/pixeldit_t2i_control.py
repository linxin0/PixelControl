"""PixDiT_T2I_Control: depth / depth+seg+edge control variant of PixDiT_T2I.

Two execution paths:
  * ``control_mode="single"`` (legacy depth-only): a single ``DepthEncoder`` +
    ``depth_adapters`` stack injecting after every patch block. This is the
    layout produced by the original PixelDiT depth-control v1 training run
    and the checkpoint format we still need to consume.
  * ``control_mode="multi"`` (new independent-branches layout): three
    completely independent control branches — depth / seg / edge. Each branch
    has its own ``DepthEncoder`` + ``StructureAwareGatedZeroAdapter`` stack
    that produces the per-layer residuals previously fused by the
    "shared encoder + mask-weighted average" approach.

    The branches are mixed *per-layer* and *per-sample* with a learnable
    layer-wise gate ``control_gate_logits`` of shape ``[num_inject, 3]``:

        - For samples with only one active control (``sum(keep[b]) == 1``),
          we hard-select the active branch — the learnable gate is NOT used
          and receives no gradient from that sample.  This keeps single-
          condition behaviour numerically identical to a per-branch
          depth-only / seg-only / edge-only model.
        - For samples with two or three active controls, we mask the gate
          logits to the active subset and softmax-normalize them, so the
          weights are exactly the active branches' relative contributions
          (summing to 1).

    Inactive control branches' encoders are skipped entirely when no sample
    in the batch needs them.

The pixel-level ``pixel_blocks`` pathway is left untouched so the high-
frequency refinement branch is fully under the trained backbone's control.

Control input contract for ``control_mode="multi"``:
  * ``control``       : ``[B, 3*depth_channels, H, W]`` ordered as
                        [depth | seg | edge] along the channel axis,
                        or a list/tuple of 3 tensors with the same order.
                        For backward compatibility a 1-channel tensor is
                        accepted as depth-only and a 2-channel tensor as
                        [depth, seg] with edge implicitly zeroed.
  * ``control_keep``  : ``[B, 3]`` binary mask over [depth, seg, edge].
                        For backward compatibility ``[B, 1]`` / ``[B, 2]``
                        masks are right-padded with zeros.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pixeldit_t2i import PixDiT_T2I
from .depth_condition_encoder import (
    DepthEncoder,
    StructureAwareGatedZeroAdapter,
    sobel_structure_map,
)


ControlInput = Union[torch.Tensor, Sequence[torch.Tensor], None]


# Canonical channel order for the independent-branches layout.
CONTROL_NAMES_DEFAULT = ("depth", "seg", "edge")


class PixDiT_T2I_Control(PixDiT_T2I):
    """PixDiT T2I backbone with depth (single) or depth+seg+edge (independent)
    control branches.
    """

    def __init__(
        self,
        *args,
        use_depth_condition: bool = True,
        control_mode: str = "single",
        depth_channels: int = 1,
        # Legacy knob retained for backward compat with the older fused path.
        # In the new ``multi`` mode it is implicitly len(control_names).
        n_local_controls: int = 1,
        depth_base_channels: int = 64,
        depth_max_channels: int = 512,
        inject_every: int = 1,
        inject_layer_indices: Optional[List[int]] = None,
        init_gate: float = 1.0,
        enable_structure_inject: bool = True,
        alpha_inject: float = 2.0,
        freeze_backbone: bool = False,
        # New independent-branches knobs.
        control_names: Sequence[str] = CONTROL_NAMES_DEFAULT,
        control_structure_inject: Optional[Sequence[bool]] = None,
        init_gate_logits: Sequence[float] = (0.5, 0.0, 0.0),
        freeze_depth_branch: bool = False,
        freeze_control_branches: Optional[Sequence[str]] = None,
        pretrained_ckpt: Optional[str] = None,
        load_strict: bool = False,
        load_prefix: str = "core.",
        skip_pretrained_modules: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert control_mode in ("single", "multi"), control_mode
        self.use_depth_condition = bool(use_depth_condition)
        self.control_mode = str(control_mode)
        self.depth_channels = int(depth_channels)
        self.n_local_controls = int(n_local_controls)
        self.enable_structure_inject = bool(enable_structure_inject)
        self.alpha_inject = float(alpha_inject)
        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_depth_branch = bool(freeze_depth_branch)
        self.freeze_control_branches = set(str(v) for v in (freeze_control_branches or ()))
        self.skip_pretrained_modules = tuple(str(s) for s in (skip_pretrained_modules or ()))
        self.control_names = tuple(str(n) for n in control_names)
        if control_structure_inject is None:
            control_structure_inject = (True,) * len(self.control_names)
        self.control_structure_inject = tuple(bool(v) for v in control_structure_inject)
        if len(self.control_structure_inject) != len(self.control_names):
            raise ValueError(
                "control_structure_inject must have the same length as control_names, "
                f"got {len(self.control_structure_inject)} vs {len(self.control_names)}"
            )
        if self.control_mode == "multi":
            if len(self.control_names) != 3:
                raise ValueError(
                    "control_mode=multi expects control_names of length 3, "
                    f"got {self.control_names}"
                )
        self.num_controls = len(self.control_names)

        if self.control_mode == "single" and self.n_local_controls != 1:
            raise ValueError("control_mode=single requires n_local_controls=1")

        if inject_layer_indices is None:
            inject_layer_indices = list(range(inject_every - 1, self.patch_depth, inject_every))
        inject_layer_indices = sorted({int(i) for i in inject_layer_indices})
        for idx in inject_layer_indices:
            assert 0 <= idx < self.patch_depth, (
                f"inject_layer_indices contains {idx} outside [0, {self.patch_depth})"
            )
        self.inject_layer_indices = inject_layer_indices
        self._block_to_adapter = [-1] * self.patch_depth
        for slot, idx in enumerate(self.inject_layer_indices):
            self._block_to_adapter[idx] = slot
        self.num_inject_layers = len(self.inject_layer_indices)

        # Cached per-forward debug info (gate weights actually used). The
        # trainer can read this to log gate behaviour without touching the
        # model graph. ``None`` outside a forward.
        self._last_gate_weights: Optional[torch.Tensor] = None

        if not self.use_depth_condition:
            # No control modules at all.
            self.depth_encoder = None
            self.depth_adapters = None
            self.seg_encoder = None
            self.seg_adapters = None
            self.edge_encoder = None
            self.edge_adapters = None
            self.control_gate_logits = None
        elif self.control_mode == "single":
            # Legacy depth-only: keep names compatible with old checkpoints.
            self.depth_encoder = DepthEncoder(
                in_channels=self.depth_channels,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                base_channels=int(depth_base_channels),
                max_channels=int(depth_max_channels),
            )
            self.depth_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(self.hidden_size, init_gate=init_gate)
                for _ in self.inject_layer_indices
            ])
            self.seg_encoder = None
            self.seg_adapters = None
            self.edge_encoder = None
            self.edge_adapters = None
            self.control_gate_logits = None
            print(
                f"[PixDiT_T2I_Control] mode=single (legacy) "
                f"depth_channels={self.depth_channels} "
                f"inject_sites={self.inject_layer_indices} "
                f"structure_inject={self.enable_structure_inject} "
                f"alpha_inject={self.alpha_inject} init_gate={init_gate}"
            )
        else:
            # New independent-branches multi-control: one encoder + one
            # adapter stack PER condition. Naming for the depth branch is
            # preserved verbatim so the legacy depth-only checkpoint
            # transfers without any key remapping.
            self.depth_encoder = DepthEncoder(
                in_channels=self.depth_channels,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                base_channels=int(depth_base_channels),
                max_channels=int(depth_max_channels),
            )
            self.depth_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(self.hidden_size, init_gate=init_gate)
                for _ in self.inject_layer_indices
            ])
            self.seg_encoder = DepthEncoder(
                in_channels=self.depth_channels,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                base_channels=int(depth_base_channels),
                max_channels=int(depth_max_channels),
            )
            self.seg_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(self.hidden_size, init_gate=init_gate)
                for _ in self.inject_layer_indices
            ])
            self.edge_encoder = DepthEncoder(
                in_channels=self.depth_channels,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                base_channels=int(depth_base_channels),
                max_channels=int(depth_max_channels),
            )
            self.edge_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(self.hidden_size, init_gate=init_gate)
                for _ in self.inject_layer_indices
            ])
            if len(init_gate_logits) != self.num_controls:
                raise ValueError(
                    f"init_gate_logits length {len(init_gate_logits)} != "
                    f"num_controls {self.num_controls}"
                )
            # Layer-wise scalar gate logits, broadcast over layers. Stored
            # in fp32 to keep gradients well-behaved under bf16 forwards.
            init_row = torch.tensor(list(init_gate_logits), dtype=torch.float32)
            self.control_gate_logits = nn.Parameter(
                init_row.view(1, self.num_controls).repeat(self.num_inject_layers, 1).clone()
            )
            print(
                f"[PixDiT_T2I_Control] mode=multi (independent branches) "
                f"control_names={self.control_names} "
                f"depth_channels={self.depth_channels} "
                f"inject_sites={self.inject_layer_indices} "
                f"structure_inject={self.enable_structure_inject} "
                f"per_control_structure_inject={self.control_structure_inject} "
                f"alpha_inject={self.alpha_inject} init_gate={init_gate} "
                f"init_gate_logits={list(init_gate_logits)} "
                f"freeze_depth_branch={self.freeze_depth_branch}"
            )

        if pretrained_ckpt is not None:
            self.load_pretrained_backbone(
                pretrained_ckpt,
                strict=load_strict,
                prefix=load_prefix,
                skip_modules=self.skip_pretrained_modules,
            )

        if self.freeze_backbone:
            self.apply_freeze()
        if self.control_mode == "multi":
            branches_to_freeze = set(self.freeze_control_branches)
            if self.freeze_depth_branch:
                branches_to_freeze.add("depth")
            if branches_to_freeze:
                self.apply_freeze_control_branches(branches_to_freeze)

    # ------------------------------------------------------------------
    # Parameter group helpers.
    # ------------------------------------------------------------------
    _DEPTH_BRANCH_NAMES = ("depth_encoder", "depth_adapters")
    _SEG_BRANCH_NAMES = ("seg_encoder", "seg_adapters")
    _EDGE_BRANCH_NAMES = ("edge_encoder", "edge_adapters")
    _GATE_NAMES = ("control_gate_logits",)

    @classmethod
    def _is_depth_branch_param(cls, name: str) -> bool:
        return any(tag in name for tag in cls._DEPTH_BRANCH_NAMES)

    @classmethod
    def _is_seg_branch_param(cls, name: str) -> bool:
        return any(tag in name for tag in cls._SEG_BRANCH_NAMES)

    @classmethod
    def _is_edge_branch_param(cls, name: str) -> bool:
        return any(tag in name for tag in cls._EDGE_BRANCH_NAMES)

    @classmethod
    def _is_gate_param(cls, name: str) -> bool:
        return any(tag in name for tag in cls._GATE_NAMES)

    @classmethod
    def _is_control_param_name(cls, name: str) -> bool:
        return (
            cls._is_depth_branch_param(name)
            or cls._is_seg_branch_param(name)
            or cls._is_edge_branch_param(name)
            or cls._is_gate_param(name)
        )

    def apply_freeze(self) -> None:
        """Freeze the backbone; train only the control modules."""
        n_frozen, n_trainable = 0, 0
        for n, p in self.named_parameters():
            if self.use_depth_condition and self._is_control_param_name(n):
                p.requires_grad = True
                n_trainable += p.numel()
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(
            f"[PixDiT_T2I_Control] freeze_backbone=True -> "
            f"frozen={n_frozen/1e6:.2f}M, trainable={n_trainable/1e6:.2f}M"
        )

    def apply_freeze_depth_branch(self) -> None:
        """Freeze just the depth branch (encoder + adapters).

        Useful for the independent-branches multi-control fine-tune where we
        want to preserve the loaded depth-control v1 capability while
        training seg/edge branches + the layer-wise gate.
        """
        n_frozen = 0
        for n, p in self.named_parameters():
            if self._is_depth_branch_param(n):
                p.requires_grad = False
                n_frozen += p.numel()
        print(
            f"[PixDiT_T2I_Control] freeze_depth_branch=True -> "
            f"frozen={n_frozen/1e6:.2f}M depth-branch params"
        )

    def apply_freeze_control_branches(self, branches: Sequence[str]) -> None:
        """Freeze selected independent control branches by name."""
        branch_set = {str(b) for b in branches}
        predicates = {
            "depth": self._is_depth_branch_param,
            "seg": self._is_seg_branch_param,
            "edge": self._is_edge_branch_param,
        }
        unknown = branch_set - set(predicates)
        if unknown:
            raise ValueError(f"Unknown control branches to freeze: {sorted(unknown)}")
        frozen = {name: 0 for name in branch_set}
        for n, p in self.named_parameters():
            for branch, pred in predicates.items():
                if branch in branch_set and pred(n):
                    p.requires_grad = False
                    frozen[branch] += p.numel()
                    break
        msg = ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(frozen.items()))
        print(f"[PixDiT_T2I_Control] freeze_control_branches={sorted(branch_set)} -> {msg}")

    def get_param_groups(
        self,
        base_lr: float,
        depth_branch_lr_scale: float = 1.0,
        seg_branch_lr_scale: float = 1.0,
        edge_branch_lr_scale: float = 1.0,
        gate_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ) -> List[dict]:
        """Return a list of optimizer param-group dicts.

        Splits the model parameters into:
          * ``backbone``     : everything that is not part of a control branch
          * ``depth_branch`` : depth encoder + adapters, scaled by
                               ``depth_branch_lr_scale`` (typically <= 1.0
                               to protect the loaded depth-control weights)
          * ``seg_branch``   : seg encoder + adapters, scaled by
                               ``seg_branch_lr_scale``
          * ``edge_branch``  : edge encoder + adapters, scaled by
                               ``edge_branch_lr_scale``
          * ``gate``         : the learnable per-layer fusion gate
                               scaled by ``gate_lr_scale``

        Groups for which every parameter is frozen are omitted so the
        optimizer doesn't fail on empty parameter lists.
        """
        groups = {
            "backbone": [],
            "depth_branch": [],
            "seg_branch": [],
            "edge_branch": [],
            "gate": [],
        }
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if self._is_gate_param(n):
                groups["gate"].append(p)
            elif self._is_depth_branch_param(n):
                groups["depth_branch"].append(p)
            elif self._is_seg_branch_param(n):
                groups["seg_branch"].append(p)
            elif self._is_edge_branch_param(n):
                groups["edge_branch"].append(p)
            else:
                groups["backbone"].append(p)
        out: List[dict] = []
        if groups["backbone"]:
            out.append({"params": groups["backbone"], "lr": float(base_lr),
                        "weight_decay": float(weight_decay), "name": "backbone"})
        if groups["depth_branch"]:
            out.append({"params": groups["depth_branch"],
                        "lr": float(base_lr) * float(depth_branch_lr_scale),
                        "weight_decay": float(weight_decay), "name": "depth_branch"})
        if groups["seg_branch"]:
            out.append({"params": groups["seg_branch"],
                        "lr": float(base_lr) * float(seg_branch_lr_scale),
                        "weight_decay": float(weight_decay), "name": "seg_branch"})
        if groups["edge_branch"]:
            out.append({"params": groups["edge_branch"],
                        "lr": float(base_lr) * float(edge_branch_lr_scale),
                        "weight_decay": float(weight_decay), "name": "edge_branch"})
        if groups["gate"]:
            out.append({"params": groups["gate"],
                        "lr": float(base_lr) * float(gate_lr_scale),
                        "weight_decay": float(weight_decay), "name": "gate"})
        return out

    # ------------------------------------------------------------------
    # Checkpoint loading.
    # ------------------------------------------------------------------
    def load_pretrained_backbone(
        self,
        ckpt_path: str,
        strict: bool = False,
        prefix: str = "core.",
        skip_modules: Optional[Sequence[str]] = None,
    ) -> None:
        """Load a baseline PixDiTTrainer checkpoint into the core.

        Supports both:
          * the original baseline T2I checkpoint (no control modules), and
          * the depth-control v1 checkpoint which carries
            ``core.depth_encoder.*`` + ``core.depth_adapters.*`` — these
            naturally match the new model's depth branch keys.

        Loads non-strict so the freshly-added seg/edge encoders, seg/edge
        adapters and ``control_gate_logits`` keep their fresh init.
        """
        import os

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        print(f"[PixDiT_T2I_Control] loading pretrained backbone from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if prefix:
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if len(sub) == 0:
                sub = dict(sd)
        else:
            sub = dict(sd)
        skip_modules = tuple(str(s) for s in (skip_modules or ()))
        own = self.state_dict()
        filtered, dropped, skipped = {}, [], []
        for k, v in sub.items():
            if skip_modules and any(k.startswith(skip) for skip in skip_modules):
                skipped.append(k)
                continue
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
            else:
                dropped.append(k)
        missing, unexpected = self.load_state_dict(filtered, strict=strict)
        n_depth = sum(1 for k in missing if self._is_depth_branch_param(k))
        n_seg = sum(1 for k in missing if self._is_seg_branch_param(k))
        n_edge = sum(1 for k in missing if self._is_edge_branch_param(k))
        n_gate = sum(1 for k in missing if self._is_gate_param(k))
        n_other = len(missing) - n_depth - n_seg - n_edge - n_gate
        print(
            f"[PixDiT_T2I_Control] loaded {len(filtered)} tensors; "
            f"missing(depth)={n_depth}, missing(seg)={n_seg}, "
            f"missing(edge)={n_edge}, missing(gate)={n_gate}, "
            f"missing(other)={n_other}, "
            f"dropped(shape mismatch / unknown)={len(dropped)}, "
            f"skipped(by config)={len(skipped)}, "
            f"unexpected={len(unexpected)}"
        )

    # ------------------------------------------------------------------
    # Control normalization.
    # ------------------------------------------------------------------
    def _normalize_independent_controls(
        self,
        control: ControlInput,
        control_keep: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor]:
        """Split ``control`` into a list of per-branch tensors plus a
        ``[B, num_controls]`` keep mask.

        Returns ``(per_branch_controls, keep_mask)`` where
        ``per_branch_controls`` is a list of length ``num_controls`` (each
        either a ``[B, depth_channels, H, W]`` tensor or ``None`` if that
        branch is inactive for the whole batch).
        """
        N = self.num_controls
        # Normalize ``control`` to a list of per-branch tensors.
        if isinstance(control, torch.Tensor):
            if control.ndim != 4:
                raise ValueError(
                    f"control tensor must be 4D, got shape {tuple(control.shape)}"
                )
            C = control.shape[1]
            chs = self.depth_channels
            if C == chs * N:
                per_branch = [
                    control[:, i * chs:(i + 1) * chs] for i in range(N)
                ]
            elif C == chs * 2:
                # Back-compat: legacy depth+seg only. Edge branch is zero.
                per_branch = [
                    control[:, 0:chs],
                    control[:, chs:2 * chs],
                    None,
                ]
            elif C == chs:
                # Back-compat: legacy depth only.
                per_branch = [control[:, 0:chs], None, None]
            else:
                raise ValueError(
                    f"control tensor has {C} channels but expected one of "
                    f"{{ {chs*N}, {chs*2}, {chs} }}"
                )
        elif isinstance(control, (list, tuple)):
            if not all((isinstance(c, torch.Tensor) or c is None) for c in control):
                raise TypeError("all local controls must be tensors or None")
            if len(control) != N:
                raise ValueError(
                    f"control list must have length {N}, got {len(control)}"
                )
            per_branch = list(control)
        else:
            raise TypeError(f"unsupported control type: {type(control).__name__}")

        # Coerce per-branch tensors to (B, depth_channels, H, W) + correct dtype.
        for i, c in enumerate(per_branch):
            if c is None:
                continue
            if c.shape[0] != batch_size:
                if c.shape[0] * 2 == batch_size:
                    c = torch.cat([c, c], dim=0)
                else:
                    raise ValueError(
                        f"local control batch ({c.shape[0]}) incompatible with image batch ({batch_size})"
                    )
            per_branch[i] = c.to(dtype=dtype)

        # Build / normalize the keep mask.
        if control_keep is None:
            keep_vals = [1.0 if c is not None else 0.0 for c in per_branch]
            keep_mask = torch.tensor(keep_vals, dtype=dtype, device=device).view(1, N).expand(batch_size, N).clone()
        else:
            if not isinstance(control_keep, torch.Tensor):
                control_keep = torch.tensor(control_keep, dtype=dtype, device=device)
            keep_mask = control_keep.to(device=device, dtype=dtype)
            if keep_mask.ndim == 1:
                keep_mask = keep_mask.unsqueeze(0).expand(batch_size, -1).contiguous()
            if keep_mask.shape[0] != batch_size:
                if keep_mask.shape[0] * 2 == batch_size:
                    keep_mask = torch.cat([keep_mask, keep_mask], dim=0)
                else:
                    raise ValueError(
                        f"control_keep batch ({keep_mask.shape[0]}) incompatible "
                        f"with image batch ({batch_size})"
                    )
            # Pad legacy 1-/2-wide masks with zeros to width = num_controls.
            if keep_mask.shape[1] < N:
                pad = torch.zeros(
                    batch_size, N - keep_mask.shape[1], dtype=keep_mask.dtype, device=keep_mask.device
                )
                keep_mask = torch.cat([keep_mask, pad], dim=1)
            elif keep_mask.shape[1] > N:
                keep_mask = keep_mask[:, :N]

        # Drop branches that no sample in the batch needs.
        batch_active = (keep_mask > 0).any(dim=0)  # [N] bool
        for i in range(N):
            if not bool(batch_active[i].item()):
                per_branch[i] = None

        # Hard-zero keep entries that we have no tensor for.
        for i in range(N):
            if per_branch[i] is None and batch_active[i].item():
                # A branch was requested by keep_mask but no control tensor
                # was supplied. Zero its keep column to avoid silent NaNs.
                keep_mask[:, i] = 0.0
        return per_branch, keep_mask

    @staticmethod
    def _per_layer_per_sample_weights(
        gate_logits: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-layer per-sample mixing weights ``[L, B, N]``.

        Implements the user spec: single-condition samples hard-select the
        active branch (so the learnable ``gate_logits`` get no gradient
        from them), while multi-condition samples use a softmax over the
        active subset.

        ``gate_logits``: ``[L, N]``
        ``keep_mask``  : ``[B, N]`` with values in {0, 1}
        """
        L, N = gate_logits.shape
        B = keep_mask.shape[0]
        assert keep_mask.shape[1] == N

        keep = keep_mask.to(gate_logits.dtype)
        n_active = keep.sum(dim=-1)  # [B]
        is_single = (n_active <= 1.0).view(1, B, 1)

        # Hard-select branch: weights == keep itself (1 only for the active
        # branch when n_active == 1; all zeros when n_active == 0).
        hard = keep.view(1, B, N).expand(L, B, N)

        # Masked softmax over active branches for the multi-condition path.
        logits_lb = gate_logits.view(L, 1, N).expand(L, B, N)
        active_bn = (keep > 0).view(1, B, N).expand(L, B, N)
        neg_inf = torch.full_like(logits_lb, float("-inf"))
        masked_logits = torch.where(active_bn, logits_lb, neg_inf)
        # When a sample row has no active branch, all entries are -inf and
        # softmax would NaN. Replace those rows with zeros first, then mask
        # them back to zero after the softmax.
        no_active = (n_active <= 0.0).view(1, B, 1).expand(L, B, N)
        masked_logits = torch.where(no_active, torch.zeros_like(masked_logits), masked_logits)
        soft = F.softmax(masked_logits, dim=-1)
        soft = soft * (~no_active).to(soft.dtype)

        w = torch.where(is_single, hard, soft)
        return w  # [L, B, N]

    def _compute_branch_features(
        self,
        per_branch_controls: List[Optional[torch.Tensor]],
        grid_hw: Tuple[int, int],
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Run each active branch encoder + structure-map computation."""
        encoders = (self.depth_encoder, self.seg_encoder, self.edge_encoder)
        feats: List[Optional[torch.Tensor]] = []
        structs: List[Optional[torch.Tensor]] = []
        for branch_idx, (c, enc) in enumerate(zip(per_branch_controls, encoders)):
            if c is None or enc is None:
                feats.append(None)
                structs.append(None)
                continue
            feats.append(enc(c))
            use_struct = (
                self.enable_structure_inject
                and self.alpha_inject != 0.0
                and self.control_structure_inject[branch_idx]
            )
            if use_struct:
                structs.append(sobel_structure_map(c, grid_hw))
            else:
                structs.append(None)
        return feats, structs

    # ------------------------------------------------------------------
    # Legacy (mode=single) helpers: kept verbatim so the existing
    # depth-control v1 training / inference paths still work.
    # ------------------------------------------------------------------
    def _normalize_local_controls_legacy(
        self,
        control: ControlInput,
        control_keep: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
    ):
        if control is None:
            return [], None
        if isinstance(control, torch.Tensor):
            if control.ndim != 4:
                raise ValueError(f"control tensor must be 4D, got shape {tuple(control.shape)}")
            controls = [control]
            if control_keep is None:
                control_keep = control.new_ones(control.shape[0], len(controls))
        elif isinstance(control, (list, tuple)):
            controls = list(control)
            if control_keep is None and len(controls) > 0:
                control_keep = controls[0].new_ones(controls[0].shape[0], len(controls))
        else:
            raise TypeError(f"unsupported control type: {type(control).__name__}")
        normalized = []
        for c in controls:
            if c.shape[0] != batch_size:
                if c.shape[0] * 2 == batch_size:
                    c = torch.cat([c, c], dim=0)
                else:
                    raise ValueError(
                        f"local control batch ({c.shape[0]}) incompatible with image batch ({batch_size})"
                    )
            normalized.append(c.to(dtype))
        if control_keep is not None:
            control_keep = control_keep.to(dtype=dtype)
            if control_keep.shape[0] != batch_size:
                if control_keep.shape[0] * 2 == batch_size:
                    control_keep = torch.cat([control_keep, control_keep], dim=0)
            if control_keep.ndim == 1:
                control_keep = control_keep.unsqueeze(0).expand(batch_size, -1)
        return normalized, control_keep

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        control: ControlInput = None,
        control_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)
        condition = torch.nn.functional.silu(t_emb)

        # --- Build per-branch features + per-layer weights -----------------
        single_legacy_feat: Optional[torch.Tensor] = None
        single_legacy_struct: Optional[torch.Tensor] = None
        per_branch_feats: List[Optional[torch.Tensor]] = [None, None, None]
        per_branch_structs: List[Optional[torch.Tensor]] = [None, None, None]
        gate_weights: Optional[torch.Tensor] = None  # [num_inject, B, N]
        keep_mask: Optional[torch.Tensor] = None  # [B, N]

        if self.use_depth_condition and control is not None:
            if self.control_mode == "single":
                local_controls, _ = self._normalize_local_controls_legacy(
                    control, control_keep, batch_size=B, dtype=x.dtype,
                )
                if local_controls:
                    c0 = local_controls[0]
                    single_legacy_feat = self.depth_encoder(c0)
                    if self.enable_structure_inject and self.alpha_inject != 0.0:
                        single_legacy_struct = sobel_structure_map(c0, (Hs, Ws))
            else:
                per_branch_controls, keep_mask = self._normalize_independent_controls(
                    control, control_keep, batch_size=B, dtype=x.dtype, device=x.device,
                )
                per_branch_feats, per_branch_structs = self._compute_branch_features(
                    per_branch_controls, (Hs, Ws),
                )
                if self.control_gate_logits is not None:
                    gate_weights = self._per_layer_per_sample_weights(
                        self.control_gate_logits.to(x.dtype), keep_mask
                    )  # [num_inject, B, N]
                    # Cache an FP32 detached copy for logging.
                    self._last_gate_weights = gate_weights.detach().float().cpu()

        if s is None:
            s0 = self.s_embedder(x_patches)
            pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
            attn_mask_joint = None
            if mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 3 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = (m == 0)
                    pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(B, 1, 1, Ltxt + L)
            self.last_repa_tokens = None
            s = s0
            for i in range(self.patch_depth):
                s, y_emb = self.patch_blocks[i](s, y_emb, condition, pos, pos_txt, attn_mask_joint)
                slot = self._block_to_adapter[i]
                if slot >= 0:
                    if self.control_mode == "single":
                        if single_legacy_feat is not None:
                            s = self.depth_adapters[slot](
                                s,
                                single_legacy_feat,
                                structure_map=single_legacy_struct,
                                alpha_inject=self.alpha_inject,
                            )
                    else:
                        if gate_weights is not None:
                            w_layer = gate_weights[slot]  # [B, 3]
                            residual = None
                            branch_adapters = (
                                self.depth_adapters,
                                self.seg_adapters,
                                self.edge_adapters,
                            )
                            for c_idx, (feat, struct, adapters) in enumerate(
                                zip(per_branch_feats, per_branch_structs, branch_adapters)
                            ):
                                if feat is None or adapters is None:
                                    continue
                                col = w_layer[:, c_idx]
                                r_c = adapters[slot].compute_residual(
                                    feat,
                                    structure_map=struct,
                                    alpha_inject=self.alpha_inject,
                                )
                                contribution = col.view(B, 1, 1).to(r_c.dtype) * r_c
                                residual = contribution if residual is None else residual + contribution
                            if residual is not None:
                                s = s + residual
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s
            s = torch.nn.functional.silu(t_emb + s)
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        batch_size_, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
