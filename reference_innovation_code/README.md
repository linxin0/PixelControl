# Reference implementation package

This directory contains a compact, framework-independent implementation of
the control-branch fusion and cycle-loss utilities. It does not include the
PixelDiT backbone or external data and model assets.

## Package layout

```text
reference_innovation_code/
├── independent_gated_control.py  # branch adapters, gate, mode utilities
├── datasets.py                   # single- and three-control datasets
├── losses.py                     # cycle-loss dispatch and SoftCanny loss
└── __init__.py
```

## Import and use

```python
from reference_innovation_code import IndependentBranchGatedFusion

fusion = IndependentBranchGatedFusion(
    hidden_size=1536,
    num_layers=14,
    init_gate_logits=(0.5, 0.0, -0.5),
    control_structure_inject=(True, True, False),
    alpha_inject=2.0,
)

residual = fusion.fuse_layer(
    layer_idx=0,
    branch_tokens=[depth_tokens, seg_tokens, edge_tokens],
    keep_mask=control_keep,       # [batch, 3]
    branch_structure_maps=[depth_map, seg_map, edge_map],
)
hidden = hidden + residual
```

`sample_control_mode_ddp`, `apply_multi_control_mode`, and
`mask_inactive_control_grads` are available for the training loop. Dataset
helpers read caller-provided RGB/caption and condition roots and do not
download data.

```python
from reference_innovation_code.datasets import (
    PixelThreeControlDataset,
    PixelSingleControlDataset,
    subdir_range,
)

train_shards = subdir_range(0, 199)
```

For the full trainer, launchers, configs, and evaluation commands, see
[`docs/02_USAGE.md`](../docs/02_USAGE.md).
