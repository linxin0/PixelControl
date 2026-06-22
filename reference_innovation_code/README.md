# Minimal Code Package For My PixelDiT Three-Control Network

This folder contains extracted useful code from the current project. It is not just a prose document. It is a small Python package that implements the core innovations:

- independent `depth / seg / edge` control branches
- strict single-condition hard selection
- multi-condition layer-wise gated fusion
- DDP-safe mode sampling
- inactive branch gradient masking
- single-control and three-control dataset loading
- multi-condition cycle loss dispatch
- SoftCanny image-cycle edge consistency

## Files

```text
minimal_my_network/
  __init__.py
  independent_gated_control.py
  datasets.py
  losses.py
  README.md
```

## Core Model Code

Use:

```python
from minimal_my_network import IndependentBranchGatedFusion
```

Create fusion module:

```python
fusion = IndependentBranchGatedFusion(
    hidden_size=1536,
    num_layers=14,
    init_gate_logits=(0.5, 0.0, -0.5),
    control_structure_inject=(True, True, False),
    alpha_inject=2.0,
)
```

Inside a PixelDiT block loop, after you compute branch tokens:

```python
x = fusion(
    hidden=x,
    layer_idx=inject_idx,
    branch_tokens=[depth_tokens, seg_tokens, edge_tokens],
    keep_mask=control_keep,  # [B, 3]
    branch_structure_maps=[depth_struct, seg_struct, edge_struct],
)
```

Behavior is exactly:

```text
depth-only: uses depth branch only, gate ignored
seg-only: uses seg branch only, gate ignored
edge-only: uses edge branch only, gate ignored
multi-control: masked softmax gate over active branches only
```

## Training Utilities

```python
from minimal_my_network import (
    apply_multi_control_mode,
    sample_control_mode_ddp,
    mask_inactive_control_grads,
)
```

Sample one mode per step:

```python
mode = sample_control_mode_ddp(
    modes=("depth", "seg", "edge", "depth_seg", "depth_edge", "seg_edge", "depth_seg_edge"),
    probs=(0.15, 0.15, 0.15, 0.12, 0.12, 0.12, 0.19),
    enable_dropout=True,
    device=device,
)
```

Apply sampled mode:

```python
control, control_keep = apply_multi_control_mode(control, mode, num_controls=3)
```

After backward:

```python
mask_inactive_control_grads(model, mode)
```

## Dataset Code

Three-control dataset:

```python
from minimal_my_network.datasets import PixelThreeControlDataset, subdir_range

ds = PixelThreeControlDataset(
    image_root="data/blip/extracted",
    depth_root="data/blip_depth_da3_nested_giant_large_1_1",
    seg_root="data/blip_sam2_large_extracted",
    edge_root="data/blip_edge",
    subdirs=subdir_range(0, 199),
)
```

Single-control dataset:

```python
from minimal_my_network.datasets import PixelSingleControlDataset, subdir_range

seg_ds = PixelSingleControlDataset(
    image_root="data/blip/extracted",
    control_root="data/blip_sam2_large_extracted",
    control_type="seg",
    subdirs=subdir_range(0, 199),
)
```

## Loss Code

```python
from minimal_my_network import MultiConditionCycleLoss, SoftCannyImagePyramidCycleLoss

edge_loss = SoftCannyImagePyramidCycleLoss(
    gaussian_kernel=11,
    threshold_min=0.2745,
    threshold_max=0.5882,
    temperature=0.03,
)

cycle = MultiConditionCycleLoss(
    depth_cycle_loss=depth_loss,
    seg_cycle_loss=seg_loss,
    edge_cycle_loss=edge_loss,
    depth_weight=1.0,
    seg_weight=1.0,
    edge_weight=1.0,
)
```

Call:

```python
loss = cycle(
    gen_image_m11,
    depth_01=depth,
    seg_01=seg,
    gt_image_m11=gt_image_m11,
    control_mode=mode,
)
```

## What Is Not Included

This folder intentionally does not copy the full PixelDiT backbone. You should keep using the original backbone from:

```text
pixdit_core/pixeldit.py
pixdit_core/pixeldit_t2i_control.py
```

This minimal package contains the transferable innovation code that Codex can reuse in another implementation.

