# Minimal Implementation: Independent Multi-Control PixelDiT

This document is the minimal code-level specification for reimplementing the user's method. It intentionally focuses on the smallest set of modules needed to support:

- single-control generation: `depth`, `seg`, `edge`
- multi-control generation: `depth_seg`, `depth_edge`, `seg_edge`, `depth_seg_edge`
- independent modality branches
- layer-wise gated residual fusion
- strict single-condition hard selection

The current repository implementation lives mainly in:

```text
pixdit_core/pixeldit_t2i_control.py
t2i/diffusion/model/control_trainer.py
t2i/diffusion/data/datasets/control_datasets.py
t2i/train_control.py
t2i/diffusion/losses/multi_condition_cycle.py
t2i/diffusion/losses/edge_cycle.py
```

## 1. Core Design

The base model is a PixelDiT text-to-image model. The control extension adds local condition residuals into the transformer blocks.

The key innovation is:

```text
Do not concatenate depth/seg/edge into one shared control encoder.
Use one independent branch per condition, then fuse branch residuals layer-wise.
```

The control order is fixed everywhere:

```python
CONTROL_NAMES = ("depth", "seg", "edge")
```

`control_keep` shape:

```text
[B, 3]
```

`control_keep` meanings:

```text
depth only:       [1, 0, 0]
seg only:         [0, 1, 0]
edge only:        [0, 0, 1]
depth + seg:      [1, 1, 0]
depth + edge:     [1, 0, 1]
seg + edge:       [0, 1, 1]
depth + seg+edge: [1, 1, 1]
```

## 2. Minimal Model Class

Use the original PixelDiT backbone as parent class. Add independent branch modules only when `control_mode == "multi"`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTROL_NAMES = ("depth", "seg", "edge")


class MultiControlPixelDiT(BasePixelDiT):
    def __init__(
        self,
        *,
        use_depth_condition=True,
        control_mode="multi",
        control_names=CONTROL_NAMES,
        depth_channels=1,
        hidden_size=1536,
        depth_base_channels=64,
        depth_max_channels=512,
        num_inject_layers=14,
        init_gate_logits=(0.5, 0.0, -0.5),
        enable_structure_inject=True,
        control_structure_inject=(True, True, False),
        alpha_inject=2.0,
        freeze_backbone=True,
        freeze_control_branches=(),
        pretrained_ckpt=None,
        skip_pretrained_modules=(),
        **backbone_kwargs,
    ):
        super().__init__(**backbone_kwargs)
        self.control_mode = control_mode
        self.control_names = tuple(control_names)
        self.num_controls = len(self.control_names)
        self.depth_channels = depth_channels
        self.num_inject_layers = num_inject_layers
        self.enable_structure_inject = enable_structure_inject
        self.control_structure_inject = tuple(control_structure_inject)
        self.alpha_inject = float(alpha_inject)

        if control_mode == "single":
            # Legacy single-control path. The module names are kept as depth_*
            # for compatibility with old depth-control checkpoints, even when
            # the input condition is seg or edge in single-control baselines.
            self.depth_encoder = DepthEncoder(depth_channels, depth_base_channels, depth_max_channels)
            self.depth_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(hidden_size)
                for _ in range(num_inject_layers)
            ])
            self.seg_encoder = None
            self.seg_adapters = None
            self.edge_encoder = None
            self.edge_adapters = None
            self.control_gate_logits = None

        elif control_mode == "multi":
            # Three independent branches.
            self.depth_encoder = DepthEncoder(depth_channels, depth_base_channels, depth_max_channels)
            self.depth_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(hidden_size)
                for _ in range(num_inject_layers)
            ])
            self.seg_encoder = DepthEncoder(depth_channels, depth_base_channels, depth_max_channels)
            self.seg_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(hidden_size)
                for _ in range(num_inject_layers)
            ])
            self.edge_encoder = DepthEncoder(depth_channels, depth_base_channels, depth_max_channels)
            self.edge_adapters = nn.ModuleList([
                StructureAwareGatedZeroAdapter(hidden_size)
                for _ in range(num_inject_layers)
            ])

            # Layer-wise scalar logits: [L, 3].
            row = torch.tensor(init_gate_logits, dtype=torch.float32)
            self.control_gate_logits = nn.Parameter(
                row.view(1, 3).repeat(num_inject_layers, 1).clone()
            )
        else:
            raise ValueError(f"unsupported control_mode={control_mode}")

        self.last_gate_weights = None

        if pretrained_ckpt is not None:
            self.load_pretrained_backbone(pretrained_ckpt, skip_modules=skip_pretrained_modules)

        if freeze_backbone:
            self.freeze_base_backbone_except_control()

        for branch in freeze_control_branches:
            self.freeze_control_branch(branch)
```

## 3. Structure-Aware Adapter

The adapter must expose `compute_residual()` so the multi-branch model can compute residuals before adding them to the hidden state.

```python
class StructureAwareGatedZeroAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(()))

    def compute_residual(
        self,
        cond_tokens: torch.Tensor,
        structure_map: torch.Tensor | None = None,
        alpha_inject: float = 0.5,
    ) -> torch.Tensor:
        residual = self.gate * self.proj(self.norm(cond_tokens))
        if structure_map is not None and alpha_inject != 0.0:
            residual = residual * (1.0 + float(alpha_inject) * structure_map.to(residual.dtype))
        return residual

    def forward(self, x, cond_tokens, structure_map=None, alpha_inject=0.5):
        return x + self.compute_residual(cond_tokens, structure_map, alpha_inject)
```

## 4. Control Splitting

For multi-control, the model receives `control` with shape `[B, 3, H, W]` or `[B, 3*C, H, W]`. For this project each control is one channel.

```python
def split_controls(control: torch.Tensor, control_keep: torch.Tensor | None):
    # control: [B, 3, H, W]
    assert control.ndim == 4
    b, c, h, w = control.shape
    assert c % 3 == 0
    ch = c // 3
    parts = [
        control[:, 0 * ch:1 * ch],
        control[:, 1 * ch:2 * ch],
        control[:, 2 * ch:3 * ch],
    ]
    if control_keep is None:
        control_keep = control.new_ones((b, 3))
    return parts, control_keep.to(control.dtype)
```

## 5. Gate Weight Function

This is the central single-vs-multi behavior.

Rules:

- If exactly one control is active for a sample, hard select it.
- If more than one control is active, softmax only over active controls.
- Inactive controls receive weight zero.
- Gate logits are used only for multi-condition samples.

```python
def per_layer_per_sample_weights(
    gate_logits: torch.Tensor,  # [L, 3]
    keep: torch.Tensor,         # [B, 3]
) -> torch.Tensor:
    l, n = gate_logits.shape
    b = keep.shape[0]
    assert n == keep.shape[1] == 3

    active_count = keep.sum(dim=1)  # [B]
    weights = gate_logits.new_zeros((l, b, n))

    # Single-condition: hard select. Gate is ignored.
    single = active_count == 1
    if single.any():
        weights[:, single, :] = keep[single].view(1, -1, n)

    # Multi-condition: masked softmax over active controls.
    multi = active_count > 1
    if multi.any():
        logits = gate_logits[:, None, :].expand(l, int(multi.sum()), n)
        mask = keep[multi].bool().view(1, -1, n)
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
        weights[:, multi, :] = torch.softmax(logits, dim=-1)

    return weights  # [L, B, 3]
```

This produces the fusion formula:

```text
R_l = w_l,b,depth * R_l,depth
    + w_l,b,seg   * R_l,seg
    + w_l,b,edge  * R_l,edge
```

## 6. Branch Feature Computation

Each branch is computed independently. For efficiency and strict gradient behavior, a branch can be skipped if no sample in the batch uses it.

```python
def compute_branch_features(self, controls, keep, grid_hw):
    encoders = [self.depth_encoder, self.seg_encoder, self.edge_encoder]
    adapters = [self.depth_adapters, self.seg_adapters, self.edge_adapters]

    branch_tokens = [None, None, None]
    branch_structs = [None, None, None]

    for i, control_i in enumerate(controls):
        if keep[:, i].sum() <= 0:
            continue

        # Encoder returns one token tensor per injection layer.
        feats_i = encoders[i](control_i)  # list length L, each [B, T, D]
        branch_tokens[i] = feats_i

        use_structure = (
            self.enable_structure_inject
            and self.alpha_inject != 0.0
            and self.control_structure_inject[i]
        )
        if use_structure:
            branch_structs[i] = [sobel_structure_map(control_i, grid_hw) for _ in range(self.num_inject_layers)]

    return branch_tokens, branch_structs
```

## 7. Forward Fusion Pseudocode

Inside the PixelDiT block loop, inject the fused residual at each target layer.

```python
def forward(self, x, t, y, *, control=None, control_keep=None, **kwargs):
    if self.control_mode == "single":
        # Legacy path: one control branch only.
        cond_feats = self.depth_encoder(control)
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, t, y)
            if layer_idx in self.inject_layer_indices:
                j = self.inject_layer_indices.index(layer_idx)
                struct = sobel_structure_map(control, grid_hw) if self.enable_structure_inject else None
                x = self.depth_adapters[j](x, cond_feats[j], struct, self.alpha_inject)
        return x

    # Multi-control path.
    controls, keep = split_controls(control, control_keep)
    branch_feats, branch_structs = self.compute_branch_features(controls, keep, grid_hw)
    weights = per_layer_per_sample_weights(self.control_gate_logits.to(x.dtype), keep)
    self.last_gate_weights = weights.detach().float().cpu()

    for layer_idx, block in enumerate(self.blocks):
        x = block(x, t, y)
        if layer_idx not in self.inject_layer_indices:
            continue
        j = self.inject_layer_indices.index(layer_idx)

        fused = 0.0
        for branch_idx in range(3):
            if branch_feats[branch_idx] is None:
                continue
            adapter = [self.depth_adapters, self.seg_adapters, self.edge_adapters][branch_idx][j]
            cond_j = branch_feats[branch_idx][j]
            struct_j = None if branch_structs[branch_idx] is None else branch_structs[branch_idx][j]
            residual = adapter.compute_residual(cond_j, struct_j, self.alpha_inject)
            w = weights[j, :, branch_idx].view(-1, 1, 1)
            fused = fused + w * residual

        x = x + fused

    return x
```

## 8. Training Mode Sampling

The training loop samples one mode per step:

```python
CONTROL_MODES = [
    "depth",
    "seg",
    "edge",
    "depth_seg",
    "depth_edge",
    "seg_edge",
    "depth_seg_edge",
]
```

Final mixed probabilities:

```python
CONTROL_PROBS = [0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19]
```

DDP ranks must use the same mode:

```python
def sample_control_mode(modes, probs, device):
    probs = torch.tensor(probs, dtype=torch.float32, device=device)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(idx, src=0)
    return modes[int(idx.item())]
```

Mode to keep mask:

```python
def mode_to_keep(mode: str):
    tokens = set(mode.split("_"))
    return torch.tensor([
        1.0 if "depth" in tokens else 0.0,
        1.0 if "seg" in tokens else 0.0,
        1.0 if "edge" in tokens else 0.0,
    ])
```

Apply mode:

```python
def apply_multi_control_mode(control, mode):
    # control: [B, 3, H, W]
    keep = mode_to_keep(mode).to(control.device, control.dtype)
    keep_b = keep.view(1, 3, 1, 1)
    control = control * keep_b
    control_keep = keep.view(1, 3).expand(control.shape[0], 3)
    return control, control_keep
```

## 9. Gradient Masking Invariant

For single active modes, only the active branch should update. For multi-condition, only active branches plus the gate update.

```python
def mask_inactive_control_grads(model, control_mode: str):
    tokens = set(control_mode.split("_"))
    active = {
        "depth": "depth" in tokens,
        "seg": "seg" in tokens,
        "edge": "edge" in tokens,
    }
    gate_active = sum(active.values()) > 1

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "control_gate_logits" in name and not gate_active:
            p.grad = None
        elif ("depth_encoder" in name or "depth_adapters" in name) and not active["depth"]:
            p.grad = None
        elif ("seg_encoder" in name or "seg_adapters" in name) and not active["seg"]:
            p.grad = None
        elif ("edge_encoder" in name or "edge_adapters" in name) and not active["edge"]:
            p.grad = None
```

Important exception: for the legacy single-control baseline path, parameters are still named `depth_encoder/depth_adapters` even if the condition is seg or edge. Do not apply this independent-branch gradient mask to the single-control baseline model.

## 10. Dataset Output Contract

For three-control training, each item should include:

```python
data_info = {
    "control": torch.cat([depth, seg, edge], dim=0),  # [3, H, W]
    "control_keep": torch.tensor([1.0, 1.0, 1.0]),
    "control_mode": "depth_seg_edge",
    "depth": depth,
    "seg": seg,
    "edge": edge,
}
```

The training loop samples the active mode and zeroes inactive channels.

For single-control baselines, each item should include:

```python
data_info = {
    "control": control,  # [1, H, W]
    "control_keep": torch.tensor([1.0]),
    "control_mode": "depth" or "seg" or "edge",
}
```

## 11. Cycle Loss Minimal Interface

The multi-condition cycle wrapper receives generated image and condition labels.

```python
class MultiConditionCycleLoss(nn.Module):
    def __init__(self, depth_cycle_loss=None, seg_cycle_loss=None, edge_cycle_loss=None,
                 depth_weight=1.0, seg_weight=1.0, edge_weight=1.0):
        super().__init__()
        self.depth_cycle_loss = depth_cycle_loss
        self.seg_cycle_loss = seg_cycle_loss
        self.edge_cycle_loss = edge_cycle_loss
        self.depth_weight = depth_weight
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight

    def forward(self, gen_image, depth_01=None, seg_01=None, gt_image_m11=None, control_mode="depth_seg"):
        tokens = set(control_mode.split("_"))
        total = gen_image.new_zeros(())
        if "depth" in tokens and self.depth_cycle_loss is not None:
            total = total + self.depth_weight * self.depth_cycle_loss(gen_image, depth_01)
        if "seg" in tokens and self.seg_cycle_loss is not None:
            total = total + self.seg_weight * self.seg_cycle_loss(gen_image, seg_01)
        if "edge" in tokens and self.edge_cycle_loss is not None:
            total = total + self.edge_weight * self.edge_cycle_loss(gen_image, gt_image_m11)
        return total
```

Depth and seg compare generated-image-derived structure to the condition label. Edge uses generated RGB and GT RGB because offline Canny labels are threshold-sensitive.

## 12. SoftCanny Edge Cycle

Minimal idea:

```python
class SoftCannyImagePyramidCycleLoss(nn.Module):
    def forward(self, gen_image_m11, gt_image_m11):
        threshold = uniform(0.2745, 0.5882)
        gen_edge = soft_canny(gen_image_m11, threshold)
        gt_edge = soft_canny(gt_image_m11, threshold).detach()
        return smooth_l1(gen_edge, gt_edge)
```

Current parameters:

```yaml
gaussian_kernel: 11
threshold_min: 0.2745
threshold_max: 0.5882
temperature: 0.03
cycle_scales: [512, 256, 128, 64]
cycle_scale_weights: [0.1, 0.25, 1.0, 0.25]
```

## 13. Minimal Final Config Values

Final mixed-control config:

```yaml
control_names: [depth, seg, edge]
init_gate_logits: [0.5, 0.0, -0.5]
control_structure_inject: [true, true, false]
freeze_backbone: true
pretrained_ckpt: ./universal_pix_t2i_workdirs/exp_pixeldit_threecontrol_v1_mixed_smalllr_from_softcanny2k/checkpoints/epoch_1_step_2000.pth
depth_branch_lr_scale: 0.05
seg_branch_lr_scale: 0.1
edge_branch_lr_scale: 0.1
gate_lr_scale: 0.5
cycle_weight: 0.005
control_probs: [0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19]
```

## 14. Critical Implementation Checks

Before considering an implementation correct:

```text
1. depth-only output ignores gate and uses only depth branch.
2. seg-only output ignores gate and uses only seg branch.
3. edge-only output ignores gate and uses only edge branch.
4. multi-condition output uses masked softmax over active controls only.
5. inactive branches receive no gradients.
6. gate receives gradients only for multi-condition samples.
7. DDP ranks sample the same control_mode each step.
8. edge structure injection is disabled in final mixed training.
9. corrupt edge maps are skipped/replaced, not silently loaded.
10. gate weights are logged for interpretability.
```

