"""
Multimodal loss computation — identical to v1, re-exported.
The v2 LossComputer uses the same config structure.
"""

from navsafe.modelzoo.navsim.diffusiondrive.modules.multimodal_loss import (
    reduce_loss,
    weight_reduce_loss,
    py_sigmoid_focal_loss,
    LossComputer,
)

__all__ = [
    "reduce_loss",
    "weight_reduce_loss",
    "py_sigmoid_focal_loss",
    "LossComputer",
]
