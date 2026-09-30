"""
Multimodal loss computation for DiffusionDrive.
Ported from BridgeSim — import paths updated to navsafe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from navsafe.modelzoo.navsim.diffusiondrive.transfuser_config import TransfuserConfig


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified."""
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss."""
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                          reduction='mean', avg_factor=None):
    """PyTorch version of Focal Loss."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LossComputer(nn.Module):
    def __init__(self, config: TransfuserConfig):
        self._config = config
        super(LossComputer, self).__init__()
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight

    def forward(self, poses_reg, poses_cls, targets, plan_anchor):
        """
        pred_traj: (bs, 20, 8, 3)
        pred_cls: (bs, 20)
        plan_anchor: (bs, 20, 8, 2)
        targets['trajectory']: (bs, 8, 3)
        """
        bs, num_mode, ts, d = poses_reg.shape
        target_traj = targets["trajectory"]
        dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1)
        dist = dist.mean(dim=-1)
        mode_idx = torch.argmin(dist, dim=-1)
        cls_target = mode_idx
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, ts, d)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)

        target_classes_onehot = torch.zeros(
            [bs, num_mode],
            dtype=poses_cls.dtype,
            layout=poses_cls.layout,
            device=poses_cls.device,
        )
        target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls, target_classes_onehot,
            weight=None, gamma=2.0, alpha=0.25,
            reduction='mean', avg_factor=None,
        )

        reg_loss = self.reg_loss_weight * F.l1_loss(best_reg, target_traj)
        ret_loss = loss_cls + reg_loss
        return ret_loss
