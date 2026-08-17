"""Cycle / consistency feedback losses for PixelDiT depth-control and multi-control.

Ported from the PixelGen depth-control / multi-control experiments
(``src/losses/...``). Imports kept self-contained inside ``diffusion.losses``
so PixelDiT can pull them without touching the upstream PixelGen project tree.
"""

from .depth_consistency_da3 import DA3ConsistencyLoss
from .depth_cycle_da3 import DA3DepthCycleLoss
from .depth_cycle_pyramid_da3 import DA3PyramidDepthCycleLoss
from .depth_cycle_coarse_to_fine_da3 import DA3CoarseToFinePyramidDepthCycleLoss
from .sam2_seg_cycle import SAM2SegCycleLoss
from .edge_cycle import EdgePyramidCycleLoss, SoftCannyImagePyramidCycleLoss
from .multi_condition_cycle import MultiConditionCycleLoss

__all__ = [
    "DA3ConsistencyLoss",
    "DA3DepthCycleLoss",
    "DA3PyramidDepthCycleLoss",
    "DA3CoarseToFinePyramidDepthCycleLoss",
    "SAM2SegCycleLoss",
    "EdgePyramidCycleLoss",
    "SoftCannyImagePyramidCycleLoss",
    "MultiConditionCycleLoss",
]
