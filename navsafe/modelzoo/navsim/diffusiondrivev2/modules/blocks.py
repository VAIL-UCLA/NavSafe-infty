"""
DiffusionDrive v2 building blocks.
Ported from BridgeSim — extends v1 blocks with GridSampleCrossBEVAttentionScorer
and gen_sineembed_for_position_1d.
"""

# Re-export everything from v1 blocks
from navsafe.modelzoo.navsim.diffusiondrive.modules.blocks import (
    linear_relu_ln,
    gen_sineembed_for_position,
    bias_init_with_prob,
    GridSampleCrossBEVAttention,
)

# v2-specific additions below
import torch
import torch.nn as nn


def gen_sineembed_for_position_1d(theta, hidden_dim):
    dim_t = torch.arange(hidden_dim, device=theta.device).float()
    dim_t = 10000 ** (2 * (dim_t // 2) / hidden_dim)
    emb = theta[..., None] / dim_t
    emb = torch.stack([emb.sin(), emb.cos()], dim=-1)  # (..., num_feats, 2)
    return emb.flatten(-2)


class GridSampleCrossBEVAttentionScorer(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttentionScorer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, embed_dims, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        bs, num_queries, num_points, _ = traj_points.shape

        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x
        normalized_trajectory = normalized_trajectory[..., [1, 0]]

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        sampled_features = torch.nn.functional.grid_sample(
            value, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        out = self.output_proj(out)

        return self.dropout(out) + queries
