# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Depth-control / multi-control training entry for PixelDiT.
A focused, non-invasive sibling of ``t2i/train.py`` (the baseline T2I trainer)
that adds:
  * RGB + caption + depth (single-control) **or** RGB + caption + depth + seg
    (multi-control) dataset reads via the registered datasets in
    ``t2i/diffusion/data/datasets/control_datasets.py``.
  * A ``PixDiTControlTrainer`` model (registered in
    ``t2i/diffusion/model/control_trainer.py``) that injects depth/seg
    condition tokens after each patch-level MMDiT block, modulated by a Sobel
    structure map (sobel-weighted injection).
  * A DA3 coarse-to-fine pyramid depth cycle loss (and, in multi-control,
    a SAM2-seg cycle loss), applied on a small subbatch in a configurable
    timestep window, mirroring the PixelGen v12 / multicontrol_v1 design.
  * Batch-level control-mode dropout for multi-control:
    ``depth / seg / depth_seg`` sampled per step with given probabilities.

This file deliberately does NOT touch ``train.py`` or any baseline PixelDiT
code path. The new dataset / model classes self-register via this script's
imports, so a normal ``torchrun`` invocation of ``train_control.py`` with a
config that selects them is all that's needed.

Launcher: ``bash t2i/train_control.sh t2i/configs_t2i/pixeldit_depth_control_v1.yaml``
"""

import datetime
import gc
import getpass
import hashlib
import json
import math
import os
import os.path as osp
import re
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from PIL import Image
from termcolor import colored

# Make the repo root importable for ``pixdit_core``.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffusion import DPMS
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.data.datasets.control_datasets import (
    PixelDepthEvalDataset,
    PixelMultiControlEvalDataset,
    PixelSingleControlEvalDataset,
    PixelThreeControlEvalDataset,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import (
    AEConfig,
    BaseConfig,
    DataConfig,
    ModelConfig,
    PixDiTConfig,
    SchedulerConfig,
    TextEncoderConfig,
    TrainingConfig,
    model_init_config,
)
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import init_random_seed, set_random_seed
from diffusion.utils.optimizer import OPTIMIZER_REGISTRY, auto_scale_lr, build_optimizer

# Side-effect imports: register PixDiTControlTrainer + PixelDepthControlDataset
# + PixelMultiControlDataset into the global MODELS / DATASETS registries.
import diffusion.model.control_trainer  # noqa: F401
import diffusion.data.datasets.control_datasets  # noqa: F401

# Cycle / consistency losses (lazily instantiated from config).
from diffusion.losses import (
    DA3CoarseToFinePyramidDepthCycleLoss,
    DA3DepthCycleLoss,
    DA3PyramidDepthCycleLoss,
    EdgePyramidCycleLoss,
    MultiConditionCycleLoss,
    SAM2SegCycleLoss,
    SoftCannyImagePyramidCycleLoss,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _get_resume_option(resume_from: Any, key: str, default: Any) -> Any:
    if isinstance(resume_from, dict):
        return resume_from.get(key, default)
    return default


# ---------------------------------------------------------------------------
# Extended pyrallis config: PixDiTConfig + control + cycle_loss
# ---------------------------------------------------------------------------
@dataclass
class CycleLossConfig(BaseConfig):
    """Free-form cycle loss spec consumed by ``build_cycle_loss``.

    Set ``type=`` to one of:
      * ``DA3CoarseToFinePyramidDepthCycleLoss``
      * ``DA3PyramidDepthCycleLoss``
      * ``DA3DepthCycleLoss``
      * ``SAM2SegCycleLoss``
      * ``MultiConditionCycleLoss`` (then nested
        ``depth_cycle_loss`` / ``seg_cycle_loss`` dicts are required)
    All other fields are forwarded as kwargs to the class constructor.
    """

    type: Optional[str] = None
    init_args: Dict[str, Any] = field(default_factory=dict)
    # Explicit structure verifier used by this loss. This keeps the selected
    # DA3 / SAM2 / Soft-Canny paths visible in the experiment configuration.
    verifier: Optional[str] = None


@dataclass
class ControlConfig(BaseConfig):
    enabled: bool = False
    mode: str = "single"  # "single" or "multi"
    inject_t_min: float = 0.0
    inject_t_max: float = 1.0
    cycle_weight: float = 0.0
    cycle_t_min: float = 0.3
    cycle_t_max: float = 1.0
    cycle_subbatch_size: int = 2
    cycle_apply_every: int = 1
    # Multi-control batch-level mode dropout (ignored in single mode).
    enable_control_dropout: bool = True
    control_modes: List[str] = field(default_factory=lambda: ["depth", "seg", "depth_seg"])
    control_probs: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    # Number of independent control branches the model expects. 2 = legacy
    # depth+seg fused path, 3 = new independent depth/seg/edge branches.
    num_controls: int = 2
    # Channels per branch (kept as a single int because all current control
    # signals are single-channel maps).
    depth_channels: int = 1
    # Optimizer knob: scale the LR of the depth control branch relative to
    # the seg/edge branches (typically <= 1.0 to protect loaded weights).
    depth_branch_lr_scale: float = 1.0
    seg_branch_lr_scale: float = 1.0
    edge_branch_lr_scale: float = 1.0
    gate_lr_scale: float = 1.0
    # How often (in optimizer steps) to log the layer-wise gate weights when
    # running in multi-control / independent-branches mode. 0 disables.
    gate_log_every: int = 100
    verifier_backends: Dict[str, str] = field(
        default_factory=lambda: {
            "depth": "depth_anything_v3",
            "seg": "segment_anything_v2",
            "edge": "soft_canny",
        }
    )
    # Cycle loss spec (see CycleLossConfig). Keep this non-Optional because
    # pyrallis 0.3.x has trouble decoding Optional[nested dataclass].
    cycle_loss: CycleLossConfig = field(default_factory=CycleLossConfig)


@dataclass
class ValidationConfig(BaseConfig):
    enabled: bool = False
    every_n_steps: int = 250
    image_root: str = ""
    depth_root: str = ""
    seg_root: str = ""
    # Optional pre-computed edge map directory; when empty the inference
    # dataset computes a Sobel edge from the RGB image on the fly.
    edge_root: str = ""
    # 1 = depth eval only, 2 = depth+seg multi eval, 3 = depth+seg+edge eval.
    # Set 3 to enable the 7-mode three-control inference sweep.
    num_controls: int = 2
    resolution: int = 512
    max_samples: int = 16
    batch_size: int = 1
    num_sampling_steps: int = 50
    cfg_scale: float = 2.75
    seed: int = 2025
    save_dir: str = "val"
    negative_prompt: str = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
    depth_repeat_to_3ch: bool = False
    invert_depth: bool = False
    seg_normalize: bool = True
    require_caption: bool = True
    cache_index_path: Optional[str] = None
    subdirs: Optional[List[str]] = None
    control_modes: Optional[List[str]] = None


@dataclass
class PixDiTControlConfig(PixDiTConfig):
    control: ControlConfig = field(default_factory=ControlConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# Cycle loss factory.
# ---------------------------------------------------------------------------
_LOSS_REGISTRY = {
    "DA3CoarseToFinePyramidDepthCycleLoss": DA3CoarseToFinePyramidDepthCycleLoss,
    "DA3PyramidDepthCycleLoss": DA3PyramidDepthCycleLoss,
    "DA3DepthCycleLoss": DA3DepthCycleLoss,
    "SAM2SegCycleLoss": SAM2SegCycleLoss,
    "EdgePyramidCycleLoss": EdgePyramidCycleLoss,
    "SoftCannyImagePyramidCycleLoss": SoftCannyImagePyramidCycleLoss,
    "MultiConditionCycleLoss": MultiConditionCycleLoss,
}


def build_control_optimizer(model, optimizer_cfg, control_cfg: ControlConfig):
    """Build an optimizer that honours the model's control-branch param
    groups (if the model exposes ``get_param_groups``).

    Falls back to :func:`build_optimizer` when no scaling is requested
    (preserving the legacy behaviour exactly).
    """
    depth_scale = float(getattr(control_cfg, "depth_branch_lr_scale", 1.0) or 1.0)
    seg_scale = float(getattr(control_cfg, "seg_branch_lr_scale", 1.0) or 1.0)
    edge_scale = float(getattr(control_cfg, "edge_branch_lr_scale", 1.0) or 1.0)
    gate_scale = float(getattr(control_cfg, "gate_lr_scale", 1.0) or 1.0)
    inner = model.module if hasattr(model, "module") else model
    if (
        depth_scale == 1.0
        and seg_scale == 1.0
        and edge_scale == 1.0
        and gate_scale == 1.0
    ) or not hasattr(inner, "get_param_groups"):
        return build_optimizer(model, optimizer_cfg)

    cfg = dict(optimizer_cfg)
    opt_type = cfg.pop("type", None)
    cfg.pop("constructor", None)
    cfg.pop("paramwise_cfg", None)
    optimizer_cls = OPTIMIZER_REGISTRY.get(opt_type)
    if optimizer_cls is None:
        raise ValueError(
            f"Unknown optimizer type '{opt_type}'. Available: {sorted(OPTIMIZER_REGISTRY)}"
        )
    base_lr = float(cfg.get("lr", 0.0))
    wd = float(cfg.get("weight_decay", 0.0))
    param_groups = inner.get_param_groups(
        base_lr=base_lr,
        weight_decay=wd,
        depth_branch_lr_scale=depth_scale,
        seg_branch_lr_scale=seg_scale,
        edge_branch_lr_scale=edge_scale,
        gate_lr_scale=gate_scale,
    )
    optimizer = optimizer_cls(param_groups, **cfg)

    from diffusion.utils.logger import get_root_logger
    logger = get_root_logger()
    learnable_count = sum(p.requires_grad for p in inner.parameters())
    fix_count = sum((not p.requires_grad) for p in inner.parameters())
    group_info = "; ".join(
        f"[{g.get('name','?')}] n={len(g['params'])} lr={g.get('lr', base_lr):.3e}"
        for g in optimizer.param_groups
    )
    logger.info(
        f"{optimizer.__class__.__name__} optimizer: "
        f"depth_branch_lr_scale={depth_scale}, seg_branch_lr_scale={seg_scale}, "
        f"edge_branch_lr_scale={edge_scale}, gate_lr_scale={gate_scale}, "
        f"learnable={learnable_count}, frozen={fix_count}. Groups: {group_info}"
    )
    return optimizer


def build_cycle_loss(
    spec: Optional[CycleLossConfig],
    verifier_backends: Optional[Dict[str, str]] = None,
):
    if spec is None or spec.type is None:
        return None
    cls = _LOSS_REGISTRY.get(spec.type)
    if cls is None:
        raise ValueError(f"Unknown cycle loss type: {spec.type}")
    init_args = dict(spec.init_args or {})
    # Recursively build nested losses for MultiConditionCycleLoss.
    if spec.type == "MultiConditionCycleLoss":
        verifier_keys = {
            "depth_cycle_loss": "depth",
            "seg_cycle_loss": "seg",
            "edge_cycle_loss": "edge",
        }
        for key in ("depth_cycle_loss", "seg_cycle_loss", "edge_cycle_loss"):
            sub = init_args.get(key, None)
            if isinstance(sub, dict):
                backend_key = verifier_keys[key]
                verifier = sub.get("verifier")
                if verifier is None and verifier_backends:
                    verifier = verifier_backends.get(backend_key)
                sub_cfg = CycleLossConfig(
                    type=sub.get("type"),
                    init_args=sub.get("init_args", {}),
                    verifier=verifier,
                )
                init_args[key] = build_cycle_loss(sub_cfg, verifier_backends)
    if spec.verifier is not None:
        init_args.setdefault("verifier", spec.verifier)
    return cls(**init_args)


# ---------------------------------------------------------------------------
# Continuous-time flow-matching training step (with cycle loss).
# ---------------------------------------------------------------------------
def _time_shift(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Same flow-shift formula PixelDiT applies to its discrete sigmas
    (``new_sigma = shift * sigma / (1 + (shift-1) * sigma)``), but in
    continuous time. Equivalent for ``t ∈ [0, 1]``.
    """
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def _normalize_caption_to_text_embedding(
    caption_list, tokenizer, text_encoder, chi_prompt_str: Optional[str],
    model_max_length: int, device: torch.device,
):
    """Mirror of the in-training caption path in baseline ``train.py``.

    Returns ``(y, y_mask)`` of shapes ``[B, 1, L, C]`` / ``[B, 1, 1, L]``.
    """
    with torch.no_grad():
        if chi_prompt_str:
            prompt = [chi_prompt_str + i for i in caption_list]
            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt_str))
            max_length_all = num_sys_prompt_tokens + model_max_length - 2
        else:
            prompt = list(caption_list)
            max_length_all = model_max_length
        txt_tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(range(-model_max_length + 1, 0))
        y = text_encoder(
            txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask
        )[0][:, None][:, :, select_index]
        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
    return y, y_mask


def _collect_data_info(data_info: Any, start: int, end: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(data_info, dict):
        return out
    for key, value in data_info.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[start:end]
        elif isinstance(value, (list, tuple)):
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


@torch.no_grad()
def run_control_validation(
    *,
    model: torch.nn.Module,
    tokenizer,
    text_encoder,
    null_y: torch.Tensor,
    null_y_mask: torch.Tensor,
    validation_cfg: ValidationConfig,
    text_encoder_cfg: TextEncoderConfig,
    scheduler_cfg: SchedulerConfig,
    work_dir: str,
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
    logger,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    if not validation_cfg.enabled:
        return
    if not validation_cfg.image_root:
        logger.warning("Validation skipped: image_root is empty.")
        return

    val_dir = osp.join(work_dir, validation_cfg.save_dir, f"iter_{global_step}")
    os.makedirs(val_dir, exist_ok=True)

    n_val_controls = int(getattr(validation_cfg, "num_controls", 2) or 2)
    requested_modes = getattr(validation_cfg, "control_modes", None) or ["depth"]
    single_val_mode = str(requested_modes[0]) if n_val_controls == 1 else "depth"
    is_multi_val = bool(getattr(validation_cfg, "seg_root", ""))
    if n_val_controls == 1 and single_val_mode in {"seg", "edge"}:
        control_root = validation_cfg.seg_root if single_val_mode == "seg" else validation_cfg.edge_root
        if not control_root:
            logger.warning(f"Validation skipped: {single_val_mode}_root is empty.")
            return
        val_dataset = PixelSingleControlEvalDataset(
            image_root=validation_cfg.image_root,
            control_root=control_root,
            control_type=single_val_mode,
            resolution=validation_cfg.resolution,
            max_samples=validation_cfg.max_samples,
            control_normalize=(validation_cfg.seg_normalize if single_val_mode == "seg" else True),
            invert_depth=validation_cfg.invert_depth,
            seed_offset=int(validation_cfg.seed),
        )
    elif n_val_controls >= 3 and is_multi_val:
        val_dataset = PixelThreeControlEvalDataset(
            image_root=validation_cfg.image_root,
            depth_root=validation_cfg.depth_root,
            seg_root=validation_cfg.seg_root,
            edge_root=(getattr(validation_cfg, "edge_root", "") or None),
            resolution=validation_cfg.resolution,
            invert_depth=validation_cfg.invert_depth,
            seg_normalize=validation_cfg.seg_normalize,
            max_samples=validation_cfg.max_samples,
            seed_offset=int(validation_cfg.seed),
            control_modes=getattr(validation_cfg, "control_modes", None),
        )
    elif is_multi_val:
        val_dataset = PixelMultiControlEvalDataset(
            image_root=validation_cfg.image_root,
            depth_root=validation_cfg.depth_root,
            seg_root=validation_cfg.seg_root,
            resolution=validation_cfg.resolution,
            invert_depth=validation_cfg.invert_depth,
            seg_normalize=validation_cfg.seg_normalize,
            max_samples=validation_cfg.max_samples,
            seed_offset=int(validation_cfg.seed),
        )
    else:
        if not validation_cfg.depth_root:
            logger.warning("Validation skipped: depth_root is empty.")
            return
        val_dataset = PixelDepthEvalDataset(
            image_root=validation_cfg.image_root,
            depth_root=validation_cfg.depth_root,
            resolution=validation_cfg.resolution,
            depth_repeat_to_3ch=validation_cfg.depth_repeat_to_3ch,
            invert_depth=validation_cfg.invert_depth,
            max_samples=validation_cfg.max_samples,
        )

    unwrapped = model.module if hasattr(model, "module") else model
    was_training = unwrapped.training
    unwrapped.eval()

    chi_prompt_str = "\n".join(text_encoder_cfg.chi_prompt) if text_encoder_cfg.chi_prompt else None
    generator = torch.Generator(device=device).manual_seed(int(validation_cfg.seed))
    max_length = text_encoder_cfg.model_max_length
    flow_shift = scheduler_cfg.flow_shift
    bs = max(1, int(validation_cfg.batch_size))

    rank = int(rank)
    world_size = max(1, int(world_size))
    val_indices = list(range(rank, len(val_dataset), world_size))
    logger.info(
        f"Running control inference at step {global_step}: save to {val_dir} "
        f"(rank {rank}/{world_size}, items={len(val_indices)}/{len(val_dataset)})"
    )
    for start in range(0, len(val_indices), bs):
        idxs = val_indices[start:start + bs]
        samples = [val_dataset[i] for i in idxs]
        captions = [sample["caption"] for sample in samples]
        controls = torch.stack([sample["control"] for sample in samples], dim=0).to(device=device, dtype=dtype)
        control_keep = torch.stack([sample["control_keep"] for sample in samples], dim=0).to(device=device, dtype=dtype)
        stems = [sample["stem"] for sample in samples]

        y, y_mask = _normalize_caption_to_text_embedding(
            captions,
            tokenizer,
            text_encoder,
            chi_prompt_str,
            max_length,
            device,
        )
        y = y.to(dtype=dtype)
        y_mask = y_mask.to(device=device)
        n = len(captions)
        seed_indices = [sample.get("seed_index", None) for sample in samples]
        if all(seed_idx is not None for seed_idx in seed_indices):
            latents = []
            for seed_idx in seed_indices:
                sample_gen = torch.Generator(device=device).manual_seed(int(seed_idx))
                latents.append(
                    torch.randn(
                        3,
                        validation_cfg.resolution,
                        validation_cfg.resolution,
                        device=device,
                        dtype=dtype,
                        generator=sample_gen,
                    )
                )
            x = torch.stack(latents, dim=0)
        else:
            x = torch.randn(
                n,
                3,
                validation_cfg.resolution,
                validation_cfg.resolution,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        img_hw = torch.tensor(
            [[validation_cfg.resolution, validation_cfg.resolution]],
            dtype=torch.float32,
            device=device,
        ).repeat(n, 1)
        ar = torch.ones((n, 1), dtype=torch.float32, device=device)
        model_kwargs = {
            "data_info": {
                "img_hw": img_hw,
                "aspect_ratio": ar,
                "control": controls,
                "control_keep": control_keep,
            },
            "mask": y_mask,
            "control": controls,
            "control_keep": control_keep,
        }
        if null_y.dim() == 4:
            null_condition = null_y.to(device=device, dtype=dtype).expand(n, -1, -1, -1)
        else:
            null_condition = null_y.to(device=device, dtype=dtype).repeat(n, 1, 1)[:, None]

        dpm_solver = DPMS(
            unwrapped.forward_with_dpmsolver,
            condition=y,
            uncondition=null_condition,
            guidance_type="classifier-free",
            cfg_scale=float(validation_cfg.cfg_scale),
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
            interval_guidance=[0, 1],
        )
        gen = dpm_solver.sample(
            x,
            steps=int(validation_cfg.num_sampling_steps),
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )
        for i, sample in enumerate(gen):
            save_image(sample, osp.join(val_dir, f"{stems[i]}.png"), nrow=1, normalize=True, value_range=(-1, 1))
        del gen, dpm_solver, x, y, y_mask, controls, control_keep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if was_training:
        unwrapped.train()


_CONTROL_TOKEN_ORDER = ("depth", "seg", "edge")


def _sample_control_mode(
    mode_cfg_modes: List[str],
    mode_cfg_probs: List[float],
    enable_dropout: bool,
    device: torch.device,
) -> str:
    if not enable_dropout:
        # Default to the richest mode when dropout is off. Prefer the
        # all-controls-on entry if present.
        for fallback in ("depth_seg_edge", "depth_seg"):
            if fallback in mode_cfg_modes:
                return fallback
        return mode_cfg_modes[0]
    probs = torch.tensor(mode_cfg_probs, dtype=torch.float32, device=device)
    probs = probs / probs.sum().clamp_min(1e-8)
    idx_t = torch.multinomial(probs, num_samples=1)
    # All DDP ranks MUST agree on the control mode for every step. With
    # independent control branches, different ranks would otherwise activate
    # different branches (and run different cycle models), which desynchronizes
    # DDP gradient reduction and eventually hangs the collective -> SIGABRT.
    # Broadcast rank-0's choice so every rank trains the exact same branches.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(idx_t, src=0)
    idx = int(idx_t.item())
    return mode_cfg_modes[idx]


def _mode_to_keep(mode: str, num_controls: int) -> List[float]:
    tokens = set(mode.split("_"))
    return [
        1.0 if _CONTROL_TOKEN_ORDER[i] in tokens else 0.0
        for i in range(num_controls)
    ]


def _apply_multi_control_mode(
    control: torch.Tensor, mode: str, num_controls: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a ``[B, num_controls*C, H, W]`` stacked control tensor laid out
    as ``[depth | seg | edge]`` (truncated to ``num_controls``), zero the
    inactive channels for ``mode`` and emit a matching ``[B, num_controls]``
    keep mask.

    ``num_controls`` is 2 (legacy depth+seg) or 3 (new depth+seg+edge).
    """
    if control.ndim != 4 or control.shape[1] % num_controls != 0:
        raise ValueError(
            f"control shape {tuple(control.shape)} not divisible by num_controls={num_controls}"
        )
    chs = control.shape[1] // num_controls
    parts = []
    keep_list = []
    tokens = set(mode.split("_"))
    for i in range(num_controls):
        tag = _CONTROL_TOKEN_ORDER[i]
        seg_i = control[:, i * chs:(i + 1) * chs]
        if tag in tokens:
            parts.append(seg_i)
            keep_list.append(1.0)
        else:
            parts.append(torch.zeros_like(seg_i))
            keep_list.append(0.0)
    ctrl = torch.cat(parts, dim=1)
    keep = control.new_tensor(keep_list).view(1, num_controls).expand(control.shape[0], num_controls).contiguous()
    return ctrl, keep


def _mask_inactive_control_grads(model, control_mode: str) -> None:
    """Keep branch updates strictly aligned with the sampled control mode.

    Forward already skips inactive branches for normal one-mode batches, but
    this explicit gradient mask is the training invariant we want:

      * depth-only updates only depth branch.
      * seg-only updates only seg branch.
      * edge-only updates only edge branch.
      * multi-condition updates exactly the active branches plus the fusion gate.

    Backbone parameters are handled by ``freeze_backbone`` as before; this
    function only touches control-branch / gate gradients.
    """
    inner = model.module if hasattr(model, "module") else model
    has_independent_branches = any(
        ("seg_encoder" in name or "seg_adapters" in name or "edge_encoder" in name or "edge_adapters" in name)
        for name, _ in inner.named_parameters()
    )
    if not has_independent_branches:
        return

    tokens = set(str(control_mode).split("_"))
    branch_active = {
        "depth": "depth" in tokens,
        "seg": "seg" in tokens,
        "edge": "edge" in tokens,
    }
    n_active = sum(branch_active.values())
    gate_active = n_active > 1

    for name, param in inner.named_parameters():
        if param.grad is None:
            continue
        if "control_gate_logits" in name:
            if not gate_active:
                param.grad = None
        elif "depth_encoder" in name or "depth_adapters" in name:
            if not branch_active["depth"]:
                param.grad = None
        elif "seg_encoder" in name or "seg_adapters" in name:
            if not branch_active["seg"]:
                param.grad = None
        elif "edge_encoder" in name or "edge_adapters" in name:
            if not branch_active["edge"]:
                param.grad = None


def _compute_fm_and_cycle_loss(
    *,
    model,
    clean_images: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    data_info: Any,
    depth_batch: Optional[torch.Tensor],
    seg_batch: Optional[torch.Tensor],
    cycle_loss_module: Optional[torch.nn.Module],
    control_cfg: ControlConfig,
    scheduler_cfg: SchedulerConfig,
    step_counter: int,
    device: torch.device,
):
    """Run one training-step worth of work: FM loss + optional cycle loss.

    All math is done in continuous time with the PixelDiT flow-shift formula
    so we can recover ``pred_x_start`` cleanly for the cycle loss.
    """
    B = clean_images.shape[0]
    if scheduler_cfg.weighting_scheme in ("logit_normal",):
        u = compute_density_for_timestep_sampling(
            weighting_scheme=scheduler_cfg.weighting_scheme,
            batch_size=B,
            logit_mean=scheduler_cfg.logit_mean,
            logit_std=scheduler_cfg.logit_std,
            mode_scale=None,
        )
    else:
        u = torch.rand((B,), dtype=torch.float32)
    u = u.to(device=device, dtype=torch.float32)
    sigma_main = _time_shift(u, scheduler_cfg.flow_shift).clamp(1e-4, 1.0 - 1e-4)
    sigma = sigma_main.view(B, 1, 1, 1).to(dtype=clean_images.dtype)
    alpha = 1.0 - sigma
    noise = torch.randn_like(clean_images)
    x_t = alpha * clean_images + sigma * noise
    v_target = noise - clean_images
    t_for_model = (sigma_main * float(scheduler_cfg.train_sampling_steps)).to(dtype=clean_images.dtype)

    # Build (control, control_keep, mode) for this step.
    control = None
    control_keep = None
    mode = control_cfg.control_modes[0] if control_cfg.control_modes else "depth"
    if control_cfg.enabled and isinstance(data_info, dict) and "control" in data_info:
        control = data_info["control"]
        control_keep = data_info.get("control_keep", None)
        n_branches = int(getattr(control_cfg, "num_controls", 2) or 2)
        if (
            control_cfg.mode == "multi"
            and isinstance(control, torch.Tensor)
            and control.shape[1] >= n_branches
        ):
            mode = _sample_control_mode(
                control_cfg.control_modes,
                control_cfg.control_probs,
                control_cfg.enable_control_dropout,
                device=device,
            )
            control, control_keep = _apply_multi_control_mode(
                control, mode, num_controls=n_branches,
            )
        else:
            mode = data_info.get("control_mode", mode)
            if isinstance(mode, (list, tuple)):
                mode = str(mode[0])
        inject_mask = (
            (sigma_main >= float(control_cfg.inject_t_min))
            & (sigma_main <= float(control_cfg.inject_t_max))
        ).to(dtype=clean_images.dtype).view(B, 1)
        if control_keep is None:
            keep_width = max(n_branches, int(getattr(control_cfg, "n_local_controls", 1) or 1))
            control_keep = inject_mask.expand(B, keep_width).contiguous()
        else:
            if control_keep.ndim == 1:
                control_keep = control_keep.unsqueeze(0).expand(B, -1)
            control_keep = control_keep.to(device=device, dtype=clean_images.dtype) * inject_mask

    out = model(
        x_t,
        t_for_model,
        y=y,
        mask=y_mask,
        data_info=data_info,
        control=control,
        control_keep=control_keep,
        repa_tokens=None,
    )
    pred_v = out["x"] if isinstance(out, dict) else out
    fm_loss = (pred_v - v_target).pow(2).mean()
    total_loss = fm_loss

    cycle_loss_val = torch.zeros((), device=device)
    cycle_active = torch.zeros((), device=device)
    if (
        cycle_loss_module is not None
        and control_cfg.cycle_weight > 0
        and (step_counter % max(1, int(control_cfg.cycle_apply_every))) == 0
    ):
        mask_t = (sigma_main >= control_cfg.cycle_t_min) & (sigma_main <= control_cfg.cycle_t_max)
        idx = torch.nonzero(mask_t, as_tuple=False).squeeze(-1)
        cap = max(1, int(control_cfg.cycle_subbatch_size))
        if idx.numel() > cap:
            idx = idx[:cap]
        if idx.numel() > 0:
            pred_x_start_sub = (x_t.index_select(0, idx) - sigma.index_select(0, idx) * pred_v.index_select(0, idx))
            pred_x_start_sub = pred_x_start_sub.clamp(-1.0, 1.0).to(dtype=clean_images.dtype)
            if isinstance(cycle_loss_module, MultiConditionCycleLoss):
                cyc = cycle_loss_module(
                    pred_x_start_sub,
                    depth_01=(depth_batch.index_select(0, idx) if depth_batch is not None else None),
                    seg_01=(seg_batch.index_select(0, idx) if seg_batch is not None else None),
                    gt_image_m11=clean_images.index_select(0, idx),
                    control_mode=mode,
                )
            else:
                cyc = cycle_loss_module(pred_x_start_sub, depth_batch.index_select(0, idx))
            total_loss = fm_loss + float(control_cfg.cycle_weight) * cyc
            cycle_loss_val = cyc.detach()
            cycle_active = torch.ones((), device=device)

    return {
        "loss": total_loss,
        "fm_loss": fm_loss.detach(),
        "cycle_loss": cycle_loss_val,
        "cycle_active": cycle_active,
        "control_mode": mode,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
@pyrallis.wrap()
def main(cfg: PixDiTControlConfig) -> None:
    config = cfg
    args = cfg

    init_train = "DDP"  # FSDP is not exercised for control training; keep it simple
    training_start_time = time.time()

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to if args.report_to != "none" else None,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler, ddp_kwargs],
    )

    logger = get_root_logger(osp.join(config.work_dir, "train_control.log"))
    logger.info(accelerator.state)

    if getattr(config.train, "seed", None) is None:
        config.train.seed = init_random_seed(None)
    elif int(config.train.seed) < 0:
        config.train.seed = int(time.time())
    else:
        config.train.seed = init_random_seed(config.train.seed)
    set_random_seed(config.train.seed + int(os.environ.get("LOCAL_RANK", 0)))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)

    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for control training")
    logger.info(f"Control config: enabled={config.control.enabled} mode={config.control.mode} "
                f"cycle_weight={config.control.cycle_weight}")

    image_size = config.model.image_size
    latent_size = image_size  # pixel-space
    max_length = config.text_encoder.model_max_length

    # 1) Text encoder + null embed
    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        try:
            text_embed_dim = int(getattr(text_encoder.config, "hidden_size"))
        except Exception:
            text_embed_dim = int(getattr(config.text_encoder, "caption_channels", 4096))
    else:
        text_embed_dim = config.text_encoder.caption_channels

    chi_prompt_str = "\n".join(config.text_encoder.chi_prompt) if config.text_encoder.chi_prompt else None

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    safe_text_encoder_name = str(config.text_encoder.text_encoder_name).replace("/", "-")
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{safe_text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )

    null_y_train = None
    null_y_mask = None
    if not config.data.load_text_feat and tokenizer is not None and text_encoder is not None:
        with torch.no_grad():
            null_tokens_train = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb_train = text_encoder(
                null_tokens_train.input_ids, attention_mask=null_tokens_train.attention_mask
            )[0].detach()
            null_attention_mask_train = null_tokens_train.attention_mask
        null_y_train = null_token_emb_train[:, None]
        null_y_mask = null_attention_mask_train[:, None, None, :]
        if accelerator.is_main_process and not osp.exists(null_embed_path):
            torch.save(
                {
                    "uncond_prompt_embeds": null_token_emb_train,
                    "uncond_prompt_embeds_mask": null_attention_mask_train,
                },
                null_embed_path,
            )

    # 2) Build model
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        null_embed_path=null_embed_path,
        **model_kwargs,
    ).train()

    model_ema = deepcopy(model).eval() if config.train.ema_update else None
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    # 3) Build dataset (control-aware)
    if not isinstance(config.data.data_dir, list):
        config.data.data_dir = [config.data.data_dir] if config.data.data_dir else []
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]

    num_replicas = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    dataset_cfg = asdict(config.data)
    if isinstance(dataset_cfg.get("data"), dict):
        nested_data_cfg = dict(dataset_cfg.pop("data"))
        dataset_cfg.update(nested_data_cfg)

    dataset = build_dataset(
        dataset_cfg,
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=config.train.real_prompt_ratio,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
    )

    sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
    train_dataloader = build_dataloader(
        dataset,
        num_workers=config.train.num_workers,
        batch_size=config.train.train_batch_size,
        shuffle=False,
        sampler=sampler,
    )
    train_dataloader_len = len(train_dataloader)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 4) Optimizer / LR scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_control_optimizer(model, config.train.optimizer, config.control)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)

    # Resume full training state before wrapping with Accelerator.
    start_epoch = 0
    start_step = 0
    optimized_step = 0
    if config.resume_from:
        start_epoch, missing, unexpected, rng_state = load_checkpoint(
            checkpoint=config.resume_from,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            load_ema=False,
            resume_optimizer=bool(_get_resume_option(config.model.resume_from, "resume_optimizer", True)),
            resume_lr_scheduler=bool(_get_resume_option(config.model.resume_from, "resume_lr_scheduler", True)),
            null_embed_path=null_embed_path,
            FSDP=False,
        )
        match = re.search(r"_step_(\d+)", str(config.resume_from))
        optimized_step = int(match.group(1)) if match else 0
        start_step = optimized_step * max(1, int(config.train.gradient_accumulation_steps))
        if rng_state is not None:
            try:
                torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and "torch_cuda" in rng_state:
                    torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
                np.random.set_state(rng_state["numpy"])
                import random
                random.setstate(rng_state["python"])
                if "generator" in rng_state:
                    generator.set_state(rng_state["generator"])
            except Exception as e:
                logger.warning(f"Failed to restore RNG state from {config.resume_from}: {e}")
        logger.info(
            f"Resumed control training from {config.resume_from}: "
            f"start_epoch={start_epoch}, optimized_step={optimized_step}, "
            f"micro_step={start_step}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    # 5) Cycle loss
    cycle_loss_module = None
    if config.control.enabled and config.control.cycle_weight > 0:
        cycle_loss_module = build_cycle_loss(
            config.control.cycle_loss,
            getattr(config.control, "verifier_backends", None),
        )
        if cycle_loss_module is not None:
            cycle_loss_module = cycle_loss_module.to(accelerator.device)
            logger.info(
                f"Cycle loss: {cycle_loss_module.__class__.__name__} "
                f"weight={config.control.cycle_weight} "
                f"t_window=[{config.control.cycle_t_min}, {config.control.cycle_t_max}] "
                f"subbatch={config.control.cycle_subbatch_size} "
                f"apply_every={config.control.cycle_apply_every}"
            )

    # 6) Trackers
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    if accelerator.is_main_process and args.report_to and args.report_to != "none":
        try:
            accelerator.init_trackers(args.tracker_project_name, dict(vars(config)))
        except Exception as e:
            logger.warning(f"Tracker init failed, falling back to tb: {e}")
            accelerator.init_trackers(f"tb_{timestamp}")

    # 7) Prepare with accelerate
    model = accelerator.prepare(model)
    if model_ema is not None:
        model_ema = accelerator.prepare(model_ema)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # 8) Training loop
    global_step = start_step + 1
    log_buffer = LogBuffer()
    cycle_step_counter = 0
    resume_skip_batches = start_step % max(1, train_dataloader_len)
    resume_skip_samples = resume_skip_batches * int(config.train.train_batch_size)
    sampler_can_seek = hasattr(sampler, "set_start")

    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        try:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        except Exception:
            pass

        skipped_batches_this_epoch = 0
        fast_resume_skip = False
        if epoch == start_epoch + 1 and resume_skip_batches > 0:
            if sampler_can_seek:
                sampler.set_start(resume_skip_samples)
                skipped_batches_this_epoch = resume_skip_batches
                fast_resume_skip = True
                if accelerator.is_main_process:
                    logger.info(
                        f"Fast resume skip: start dataloader at batch {resume_skip_batches} "
                        f"({resume_skip_samples} samples per rank)"
                    )
            else:
                if accelerator.is_main_process:
                    logger.info(f"Resume skip: consuming {resume_skip_batches} dataloader batches")
        elif sampler_can_seek:
            sampler.set_start(0)

        time_start = time.time()
        for step, batch in enumerate(train_dataloader):
            if not fast_resume_skip and epoch == start_epoch + 1 and step < resume_skip_batches:
                continue
            img = batch[0].to(accelerator.device)
            captions = batch[1]
            data_info = batch[3]

            # Move control tensors onto device (collate may have stacked them).
            depth_batch = None
            seg_batch = None
            if isinstance(data_info, dict):
                if "depth" in data_info and isinstance(data_info["depth"], torch.Tensor):
                    depth_batch = data_info["depth"].to(accelerator.device, dtype=img.dtype)
                    data_info["depth"] = depth_batch
                if "seg" in data_info and isinstance(data_info["seg"], torch.Tensor):
                    seg_batch = data_info["seg"].to(accelerator.device, dtype=img.dtype)
                    data_info["seg"] = seg_batch
                if "control" in data_info and isinstance(data_info["control"], torch.Tensor):
                    data_info["control"] = data_info["control"].to(accelerator.device, dtype=img.dtype)
                if "control_keep" in data_info and isinstance(data_info["control_keep"], torch.Tensor):
                    data_info["control_keep"] = data_info["control_keep"].to(accelerator.device, dtype=img.dtype)

            # Text embedding.
            if load_text_feat:
                y = batch[1]
                y_mask = batch[2]
            else:
                y, y_mask = _normalize_caption_to_text_embedding(
                    captions, tokenizer, text_encoder, chi_prompt_str, max_length, accelerator.device,
                )

            # CFG dropout on text.
            p_drop = float(getattr(config.model, "class_dropout_prob", 0.0) or 0.0)
            if p_drop > 0 and isinstance(y, torch.Tensor):
                if null_y_train is not None and null_y_mask is not None:
                    null_y_local = null_y_train.to(device=y.device, dtype=y.dtype)
                    null_y_mask_local = null_y_mask.to(device=y_mask.device, dtype=y_mask.dtype)
                else:
                    null_y_local = torch.zeros_like(y)
                    null_y_mask_local = torch.zeros_like(y_mask)
                bs = img.shape[0]
                drop_mask = (torch.rand((bs,), device=y.device) < p_drop).view(bs, 1, 1, 1)
                y = torch.where(drop_mask, null_y_local.expand_as(y), y)
                y_mask = torch.where(drop_mask, null_y_mask_local.expand_as(y_mask), y_mask)

            if isinstance(y, torch.Tensor):
                y = y.to(dtype=img.dtype)

            cycle_step_counter += 1
            with accelerator.accumulate(model):
                loss_term = _compute_fm_and_cycle_loss(
                    model=model,
                    clean_images=img,
                    y=y,
                    y_mask=y_mask,
                    data_info=data_info,
                    depth_batch=depth_batch,
                    seg_batch=seg_batch,
                    cycle_loss_module=cycle_loss_module,
                    control_cfg=config.control,
                    scheduler_cfg=config.scheduler,
                    step_counter=cycle_step_counter,
                    device=accelerator.device,
                )
                loss = loss_term["loss"]
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    _mask_inactive_control_grads(model, loss_term["control_mode"])
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if config.train.ema_update and model_ema is not None:
                        rate = config.train.ema_rate
                        for p_dest, p_src in zip(model_ema.parameters(), model.parameters()):
                            p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimized_step += 1

            logs = {
                config.loss_report_name: accelerator.gather(loss.detach()).mean().item(),
                "fm_loss": accelerator.gather(loss_term["fm_loss"]).mean().item(),
                "cycle_loss": accelerator.gather(loss_term["cycle_loss"]).mean().item(),
                "cycle_active": accelerator.gather(loss_term["cycle_active"]).mean().item(),
                "opt_step": int(accelerator.sync_gradients),
            }
            if grad_norm is not None:
                logs["grad_norm"] = accelerator.gather(grad_norm).mean().item()
            log_buffer.update(logs)

            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - time_start) / max(1, step + 1)
                log_buffer.average()
                info = (
                    f"Epoch: {epoch} | MicroStep: {global_step} | "
                    f"OptimizedStep: {optimized_step} | "
                    f"LocalStep: {skipped_batches_this_epoch + step + 1}/{train_dataloader_len} | "
                    f"time/step:{t:.3f}s lr:{lr_scheduler.get_last_lr()[0]:.3e} "
                    f"mode:{loss_term['control_mode']} | "
                    + ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items() if isinstance(v, float)])
                )
                if accelerator.is_main_process:
                    logger.info(info)
                log_buffer.clear()

            gate_every = int(getattr(config.control, "gate_log_every", 0) or 0)
            if (
                accelerator.sync_gradients
                and gate_every > 0
                and optimized_step > 0
                and optimized_step % gate_every == 0
                and accelerator.is_main_process
            ):
                unwrapped_for_gate = accelerator.unwrap_model(model)
                gw = getattr(unwrapped_for_gate, "last_gate_weights", None)
                if gw is not None and gw.ndim == 3:
                    mean_per_layer = gw.mean(dim=1)  # [num_inject, num_controls]
                    overall = mean_per_layer.mean(dim=0).tolist()
                    head = mean_per_layer[0].tolist()
                    tail = mean_per_layer[-1].tolist()
                    logger.info(
                        f"[gate] step={optimized_step} mode={loss_term['control_mode']} "
                        f"overall(d/s/e)=({overall[0]:.3f},{overall[1]:.3f},{overall[2]:.3f}) "
                        f"layer0=({head[0]:.3f},{head[1]:.3f},{head[2]:.3f}) "
                        f"layerN=({tail[0]:.3f},{tail[1]:.3f},{tail[2]:.3f})"
                    )

            if (
                (accelerator.sync_gradients
                 and optimized_step > 0
                 and optimized_step % config.train.save_model_steps == 0)
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=optimized_step,
                        generator=generator,
                        add_symlink=True,
                    )

            global_step += 1

        if (epoch % config.train.save_model_epochs == 0) or (epoch == config.train.num_epochs):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=optimized_step,
                    model=accelerator.unwrap_model(model),
                    model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

    flush()


if __name__ == "__main__":
    main()
