"""
DiffusionDrive V2 selection model with scorer heads.
Ported from BridgeSim — import paths updated to navsafe.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from navsafe.modelzoo.navsim.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
from navsafe.modelzoo.navsim.diffusiondrivev2.transfuser_backbone import TransfuserBackbone
from navsafe.modelzoo.navsim.diffusiondrivev2.transfuser_features import BoundingBox2DIndex
from navsafe.modelzoo.common.enums import StateSE2Index
from navsafe.modelzoo.navsim.diffusiondrivev2.modules.conditional_unet1d import ConditionalUnet1D, SinusoidalPosEmb
from navsafe.modelzoo.navsim.diffusiondrivev2.modules.blocks import (
    linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position,
    GridSampleCrossBEVAttention, gen_sineembed_for_position_1d,
    GridSampleCrossBEVAttentionScorer,
)
from navsafe.modelzoo.navsim.diffusiondrivev2.modules.multimodal_loss import LossComputer


class V2TransfuserModel(nn.Module):
    """Torch module for DiffusionDrive v2 Transfuser with scorer."""

    def __init__(self, config: TransfuserConfig):
        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels, config.bev_features_channels,
                kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels, config.num_bev_classes,
                kernel_size=(1, 1), stride=1, padding=0, bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear", align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model, nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn, dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn, d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn, d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path, config=config,
        )

        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )

    def forward(
        self, features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor] = None,
        eta=0.0, metric_cache=None, cal_pdm=True, token=None,
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        pred = self._trajectory_head(
            trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding[:, None], status_feature, camera_feature,
            targets=targets, global_img=None, eta=eta,
            metric_cache=metric_cache, cal_pdm=cal_pdm, token=token,
        )
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def forward_temporal(
        self, features: Dict[str, torch.Tensor],
        current_time: float,
        targets: Dict[str, torch.Tensor] = None,
        eta=0.0, metric_cache=None, cal_pdm=True, token=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with temporal consistency scoring."""
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        pred = self._trajectory_head.forward_test_rl_temporal_scorer(
            trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding[:, None], status_feature, camera_feature,
            targets=targets, global_img=None, metric_cache=metric_cache,
            current_time=current_time, eta=eta, cal_pdm=cal_pdm, token=token,
        )
        output.update(pred)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

    def forward_inference_scaling(
        self, features: Dict[str, torch.Tensor],
        num_groups: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Generate num_groups * 20 trajectory candidates with coarse scores."""
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        traj_output = self._trajectory_head.forward_inference_scaling(
            trajectory_query, agents_query, cross_bev_feature,
            bev_spatial_shape, status_encoding[:, None], status_feature,
            camera_feature, global_img=None, num_groups=num_groups,
        )

        output = {
            "all_candidates": traj_output["all_candidates"],
            "coarse_scores": traj_output["coarse_scores"],
            "confidence_scores": traj_output["confidence_scores"],
            "scorer_context": {
                "bev_feature": cross_bev_feature,
                "bev_spatial_shape": bev_spatial_shape,
                "agents_query": agents_query,
                "ego_query": trajectory_query,
                "status_encoding": status_encoding[:, None],
            },
        }
        return output

    def reset_temporal_history(self):
        """Reset temporal consistency history. Call at start of new scenario."""
        self._trajectory_head.reset_temporal_history()

    def set_temporal_params(self, alpha: float = None, lambda_consist: float = None,
                           max_history: int = None, sigma: float = None,
                           consensus_temperature: float = None):
        """Set temporal consistency parameters."""
        self._trajectory_head.set_temporal_params(alpha, lambda_consist, max_history, sigma, consensus_temperature)

    def get_temporal_history_length(self) -> int:
        """Get current length of temporal history buffer."""
        return len(self._trajectory_head.temporal_history)


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super(AgentHead, self).__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(nn.Linear(self._d_model, 1))

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT()] = agent_states[..., BoundingBox2DIndex.POINT()].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)
        return {"agent_states": agent_states, "agent_labels": agent_labels}


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=8, ego_fut_mode=20, if_zeroinit_reg=True):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode, _ = traj_feature.shape
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
        return plan_reg, plan_cls


class ModulationLayer(nn.Module):
    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(self, traj_feature, time_embed, global_cond=None, global_img=None):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], axis=-1)
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class ScorerTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        tf_d_model: int = 512
        tf_d_ffn: int = 2048
        tf_num_head: int = 16
        tf_dropout: float = 0.1

        self.cross_bev_attention = GridSampleCrossBEVAttentionScorer(
            tf_d_model, tf_num_head, num_points=num_poses, config=config, in_bev_dims=256,
        )
        self.agent_input = nn.Linear(256, tf_d_model)
        self.ego_input = nn.Linear(256, tf_d_model)
        self.cross_agent_attention = nn.MultiheadAttention(
            tf_d_model, tf_num_head, dropout=tf_dropout, batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            tf_d_model, tf_num_head, dropout=tf_dropout, batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            tf_d_model, tf_num_head, dropout=tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(tf_d_model, tf_d_ffn),
            nn.ReLU(),
            nn.Linear(tf_d_ffn, tf_d_model),
        )
        self.norm1 = nn.LayerNorm(tf_d_model)
        self.norm2 = nn.LayerNorm(tf_d_model)
        self.norm3 = nn.LayerNorm(tf_d_model)
        self.norm4 = nn.LayerNorm(tf_d_model)

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        agents_query = self.agent_input(agents_query)
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0])
        traj_feature = self.norm1(traj_feature)

        traj_feature = traj_feature + self.dropout1(
            self.self_attn(traj_feature, traj_feature, traj_feature)[0])
        traj_feature = self.norm2(traj_feature)

        ego_query = self.ego_input(ego_query)
        traj_feature = traj_feature + self.dropout2(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm3(traj_feature)

        traj_feature = self.norm4(self.ffn(traj_feature))
        return traj_feature


class ScorerTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        traj_feature_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            traj_feature = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape,
                              agents_query, ego_query, time_embed, status_encoding, global_img)
            traj_feature_list.append(traj_feature)
        return traj_feature_list


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model, config.tf_num_head, num_points=num_poses,
            config=config, in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model, ego_fut_ts=num_poses, ego_fut_mode=20,
        )

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0])
        traj_feature = self.norm1(traj_feature)

        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)

        # Reshape for multi-group support
        traj_feature = traj_feature.view(traj_feature.shape[0], -1, 20, traj_feature.shape[-1])
        bs, num_groups, _, _ = traj_feature.shape
        traj_feature = traj_feature.view(-1, 20, traj_feature.shape[-1])
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        poses_reg = poses_reg.view(bs, 20 * num_groups, 8, 3)
        poses_cls = poses_cls.view(bs, -1, 20)
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls, traj_feature


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls, traj_feature = mod(
                traj_feature, traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()
        return poses_reg_list, poses_cls_list, traj_feature


class DDIMScheduler_with_logprob(DDIMScheduler):
    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
        eta: float = 1.0, use_clipped_model_output: bool = False,
        generator=None, variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None, return_dict: bool = True,
    ) -> Union[Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range)

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** (0.5)).clamp_(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).clamp_(min=0) ** (0.5) * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if prev_sample_mean is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and prev_sample. Please make sure that either `generator` or `prev_sample` stays `None`."
            )

        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)

        if prev_sample is None:
            variance_noise_horizon = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype
            ) * std_dev_t_mul + 1.0

            variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1, 1, model_output.shape[2], 1)

            variance_noise_x = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            variance_noise_add = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
            variance_noise_add = variance_noise_add.repeat(1, 1, model_output.shape[2], 1)

            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul**2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)


class TrajectoryHead(nn.Module):
    """Trajectory prediction head with coarse and fine scorers."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int,
                 plan_anchor_path: str, config: TransfuserConfig):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000, steps_offset=1,
            beta_schedule="scaled_linear", prediction_type="sample",
        )
        self.diffusionrl_scheduler = DDIMScheduler_with_logprob(
            num_train_timesteps=1000, steps_offset=1,
            beta_schedule="scaled_linear", prediction_type="sample",
        )
        self.num_groups = config.num_groups
        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False,
        )  # 20,8,2
        self.sigmoid = nn.Sigmoid()
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.plan_anchor_scorer_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 2 * 512),
            nn.Linear(d_model, 512),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 1)

        # coarse scorer
        scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        self.scorer_decoder = ScorerTransformerDecoder(scorer_decoder_layer, 1)
        self.NC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.EP_head = nn.Sequential(*linear_relu_ln(512, 2, 2), nn.Linear(512, 1))
        self.DAC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.TTC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.C_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))

        # fine scorer
        fine_scorer_decoder_layer = ScorerTransformerDecoderLayer(
            num_poses=num_poses, d_model=d_model, d_ffn=d_ffn, config=config,
        )
        self.fine_scorer_decoder = ScorerTransformerDecoder(fine_scorer_decoder_layer, 3)
        self.fine_NC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.fine_EP_head = nn.Sequential(*linear_relu_ln(512, 2, 2), nn.Linear(512, 1))
        self.fine_DAC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.fine_TTC_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))
        self.fine_C_head = nn.Sequential(*linear_relu_ln(512, 1, 2), nn.Linear(512, 1))

        self.rank_loss = torch.nn.MarginRankingLoss(margin=0.05)
        self.loss_computer = LossComputer(config)

        self._pdm_pool = None
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_bce_without_reduce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_reg = nn.MSELoss()
        self.diffusion_output = None

        # Temporal consistency parameters
        self.temporal_history = []
        self.temporal_alpha = 1.5
        self.temporal_lambda = 0.3
        self.temporal_max_history = 8
        self.temporal_sigma = 5.0
        self.temporal_traj_dt = 0.5
        self.consensus_temperature = 1.0

    def reset_temporal_history(self):
        """Reset the temporal consistency history buffer."""
        self.temporal_history = []

    def set_temporal_params(self, alpha: float = None, lambda_consist: float = None,
                           max_history: int = None, sigma: float = None,
                           consensus_temperature: float = None):
        if alpha is not None:
            self.temporal_alpha = alpha
        if lambda_consist is not None:
            self.temporal_lambda = lambda_consist
        if max_history is not None:
            self.temporal_max_history = max_history
        if sigma is not None:
            self.temporal_sigma = sigma
        if consensus_temperature is not None:
            self.consensus_temperature = consensus_temperature

    def compute_consensus_trajectory(self, trajectories: torch.Tensor,
                                     pdm_scores: torch.Tensor) -> torch.Tensor:
        """Compute score-weighted consensus trajectory."""
        weights = torch.nn.functional.softmax(
            pdm_scores / self.consensus_temperature, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        consensus = (trajectories[..., :2] * weights).sum(dim=1)
        return consensus

    def compute_temporal_consistency_score(self, candidates: torch.Tensor,
                                          current_time: float) -> torch.Tensor:
        """Compute temporal consistency scores for candidate trajectories."""
        B, G = candidates.shape[:2]
        device = candidates.device

        if len(self.temporal_history) == 0:
            return torch.ones(B, G, device=device)

        cand_first_pos = candidates[:, :, 0, :2]
        score_sum = torch.zeros(B, G, device=device)
        weight_sum = 0.0

        for j, (hist_time, hist_traj) in enumerate(self.temporal_history):
            steps_ago = len(self.temporal_history) - j
            weight = self.temporal_alpha ** steps_ago

            time_diff = current_time - hist_time
            waypoint_idx = int(time_diff / self.temporal_traj_dt) - 1

            if 0 <= waypoint_idx < hist_traj.shape[0]:
                predicted_pos = hist_traj[waypoint_idx, :2]
                if isinstance(predicted_pos, np.ndarray):
                    predicted_pos = torch.from_numpy(predicted_pos).to(device)

                displacement = torch.norm(cand_first_pos - predicted_pos.view(1, 1, 2), dim=-1)
                position_score = torch.exp(-displacement / self.temporal_sigma)
                score_sum += weight * position_score
                weight_sum += weight

        if weight_sum > 0:
            consistency_scores = score_sum / weight_sum
        else:
            consistency_scores = torch.ones(B, G, device=device)

        return consistency_scores

    def update_temporal_history(self, selected_traj: np.ndarray, current_time: float):
        traj_copy = selected_traj.copy() if isinstance(selected_traj, np.ndarray) else selected_traj.cpu().numpy()
        self.temporal_history.append((current_time, traj_copy))
        if len(self.temporal_history) > self.temporal_max_history:
            self.temporal_history.pop(0)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]
        odo_info_fut_x = odo_info_fut_x / 50
        odo_info_fut_y = odo_info_fut_y / 20
        odo_info_fut_head = odo_info_fut_head / 1.57
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]
        odo_info_fut_x = odo_info_fut_x * 50
        odo_info_fut_y = odo_info_fut_y * 20
        odo_info_fut_head = odo_info_fut_head * 1.57
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, status_feature, camera_feature,
                targets=None, global_img=None, eta=0.0, metric_cache=None,
                cal_pdm=True, token=None) -> Dict[str, torch.Tensor]:
        if self.EP_head.training:
            return self.forward_train_rl(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, status_feature, camera_feature, targets,
                global_img, eta, metric_cache, cal_pdm=cal_pdm, token=token)
        else:
            return self.forward_test_rl(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, status_feature, camera_feature, targets,
                global_img, metric_cache, eta, cal_pdm=cal_pdm, token=token)

    def _get_scorer_inputs(self, diffusion_output: torch.Tensor, bs: int, ego_fut_mode: int):
        """Prepare scorer inputs from diffusion output."""
        diffusion_output = self.norm_odo(diffusion_output)
        x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(x_boxes)

        noisy_traj_points_xy = noisy_traj_points[..., :2]
        traj_pos_embed = gen_sineembed_for_position(
            noisy_traj_points_xy, hidden_dim=64).flatten(-2)
        traj_heading_embed = gen_sineembed_for_position_1d(
            noisy_traj_points[..., 2], hidden_dim=32).flatten(-2)

        traj_pos_embed = torch.cat([traj_pos_embed, traj_heading_embed], dim=-1)
        traj_feature = self.plan_anchor_scorer_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        return noisy_traj_points_xy, traj_feature, None

    def _select_topk(self, final_coarse_reward: torch.Tensor, topk: int,
                     traj_feature: torch.Tensor, noisy_traj_points_xy: torch.Tensor,
                     sub_rewards_group):
        """Select top-k candidates by coarse reward."""
        topk_val, topk_idx = torch.topk(
            final_coarse_reward, topk, dim=-1, largest=True, sorted=True)

        idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, traj_feature.size(-1))
        traj_feature_k = torch.gather(traj_feature, 1, idx_feat)

        idx_point = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, noisy_traj_points_xy.size(-2), noisy_traj_points_xy.size(-1))
        noisy_traj_points_k = torch.gather(noisy_traj_points_xy, 1, idx_point)

        if sub_rewards_group is not None:
            sub_rewards_topk = {
                name: torch.gather(val, 1, topk_idx)
                for name, val in sub_rewards_group.items()
            }
        else:
            sub_rewards_topk = None

        return traj_feature_k, noisy_traj_points_k, sub_rewards_topk, topk_idx, topk_val

    def _score_coarse(self, traj_feature: torch.Tensor, sub_rewards_group):
        """Compute coarse scorer loss and rewards."""
        bs = traj_feature.shape[0]
        NC_score = self.NC_head(traj_feature).squeeze(-1)
        EP_score = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score = self.C_head(traj_feature).squeeze(-1)

        gt_nc = sub_rewards_group["no_collision"]
        gt_nc[gt_nc == 0.5] = 0.0

        loss_nc = self.loss_bce(NC_score, gt_nc)
        loss_ep = self.loss_bce(EP_score, sub_rewards_group["progress"])
        loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area"])
        loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["ttc"])
        loss_c = self.loss_bce_without_reduce(C_score, sub_rewards_group["comfort"])
        mask = (sub_rewards_group["comfort"] != -1)
        loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-6)

        gt_ep = sub_rewards_group["progress"]
        B, Gk = EP_score.shape
        idx_i, idx_j = torch.combinations(
            torch.arange(Gk, device=EP_score.device), r=2).unbind(-1)
        pred_i, pred_j = EP_score[:, idx_i], EP_score[:, idx_j]
        gt_i, gt_j = gt_ep[:, idx_i], gt_ep[:, idx_j]
        target = torch.sign(gt_i - gt_j)
        mask = target != 0
        if mask.any():
            loss_rank = self.rank_loss(pred_i[mask], pred_j[mask], target[mask])
        else:
            loss_rank = torch.tensor(0., device=EP_score.device)

        loss_coarse = loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2 * loss_rank

        loss_dict = {
            "coarse_loss_nc": loss_nc, "coarse_loss_ep": loss_ep,
            "coarse_loss_dac": loss_dac, "coarse_loss_ttc": loss_ttc,
            "coarse_loss_c": loss_c, "coarse_loss_rank": loss_rank,
        }

        final_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
        )

        best_idx = torch.argmax(final_coarse_reward, dim=-1)
        coarse_reward = sub_rewards_group['final'][torch.arange(bs), best_idx]
        return loss_coarse, final_coarse_reward, coarse_reward, loss_dict

    def _score_fine_multi(self, traj_feature_list, sub_rewards_group, only_reward=False):
        """Compute fine scorer loss and rewards across multiple layers."""
        loss_fine = 0.0
        loss_dict = {}
        fine_reward_dict = {}
        fine_reward = 0.0
        best_idx_list = []
        bs = traj_feature_list[0].shape[0]

        if sub_rewards_group is not None:
            gt_nc = sub_rewards_group["no_collision"]
            gt_nc[gt_nc == 0.5] = 0.0

        for i, feat in enumerate(traj_feature_list):
            EP_score = self.fine_EP_head(feat).squeeze(-1)
            NC_score = self.fine_NC_head(feat).squeeze(-1)
            DAC_score = self.fine_DAC_head(feat).squeeze(-1)
            TTC_score = self.fine_TTC_head(feat).squeeze(-1)
            C_score = self.fine_C_head(feat).squeeze(-1)

            if not only_reward:
                loss_nc = self.loss_bce(NC_score, gt_nc)
                loss_ep = self.loss_bce(EP_score, sub_rewards_group["progress"])
                loss_dac = self.loss_bce(DAC_score, sub_rewards_group["drivable_area"])
                loss_ttc = self.loss_bce(TTC_score, sub_rewards_group["ttc"])
                loss_c = self.loss_bce_without_reduce(C_score, sub_rewards_group["comfort"])
                mask = (sub_rewards_group["comfort"] != -1)
                loss_c = (loss_c * mask).sum() / (mask.sum() + 1e-6)

                gt_ep = sub_rewards_group["progress"]
                B, Gk = EP_score.shape
                idx_i, idx_j = torch.combinations(
                    torch.arange(Gk, device=EP_score.device), r=2).unbind(-1)
                pred_i, pred_j = EP_score[:, idx_i], EP_score[:, idx_j]
                gt_i, gt_j = gt_ep[:, idx_i], gt_ep[:, idx_j]
                target = torch.sign(gt_i - gt_j)
                mask = target != 0
                if mask.any():
                    loss_rank = self.rank_loss(pred_i[mask], pred_j[mask], target[mask])
                else:
                    loss_rank = torch.tensor(0., device=EP_score.device)

                loss_fine_ = loss_nc + loss_ep + loss_dac + loss_ttc + loss_c + 2 * loss_rank
                loss_dict.update({
                    f"fine_loss_nc_{i}": loss_nc, f"fine_loss_ep_{i}": loss_ep,
                    f"fine_loss_dac_{i}": loss_dac, f"fine_loss_ttc_{i}": loss_ttc,
                    f"fine_loss_c_{i}": loss_c, f"fine_loss_rank_{i}": loss_rank,
                })
                loss_fine = loss_fine + loss_fine_

            final_fine_reward = (
                self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
                (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
            )
            best_idx = torch.argmax(final_fine_reward, dim=-1)
            best_idx_list.append(best_idx)
            if not only_reward:
                fine_reward = sub_rewards_group['final'][torch.arange(bs), best_idx]
                fine_reward_dict.update({f"fine_reward_{i}": fine_reward.mean()})

        loss_fine = loss_fine / len(traj_feature_list)
        return loss_fine, final_fine_reward, fine_reward, loss_dict, fine_reward_dict, best_idx_list

    def add_mul_noise(self, diffusion_output, n_aug=3, std_min=0.1, std_max=0.3):
        diffusion_output_aug_list = [diffusion_output]
        for _ in range(n_aug):
            std_dev_t_mul = torch.empty(1, device=diffusion_output.device).uniform_(std_min, std_max).item()
            variance_noise_horizon = randn_tensor(
                [diffusion_output.shape[0], diffusion_output.shape[1], 1, 1],
                device=diffusion_output.device, dtype=diffusion_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_vert = randn_tensor(
                [diffusion_output.shape[0], diffusion_output.shape[1], 1, 1],
                device=diffusion_output.device, dtype=diffusion_output.dtype
            ) * std_dev_t_mul + 1.0
            variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1, 1, diffusion_output.shape[2], 1)
            diffusion_output_aug_list.append(diffusion_output * variance_noise_mul)
        return torch.cat(diffusion_output_aug_list, dim=1)

    def bezier_xyyaw(self, xy8: torch.Tensor) -> torch.Tensor:
        """Compute yaw from Bezier curve derivatives for (B, G, 8, 2) trajectory points."""
        assert xy8.shape[-2:] == (8, 2), "Input must be (B,G,8,2)"
        B, G, _, _ = xy8.shape
        device, dtype = xy8.device, xy8.dtype

        origin = torch.zeros_like(xy8[..., :1, :])
        ctrl = torch.cat([origin, xy8], dim=-2)  # (B,G,9,2)
        n = ctrl.shape[-2] - 1  # 8

        delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]  # (B,G,8,2)

        binom = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)], device=device, dtype=dtype)

        t = torch.arange(1, n + 1, device=device, dtype=dtype) / n

        t_pow = t.view(-1, 1) ** torch.arange(0, n, device=device, dtype=dtype)
        one_pow = (1 - t).view(-1, 1) ** torch.arange(n - 1, -1, -1, device=device, dtype=dtype)
        basis = binom * t_pow * one_pow

        delta_exp = delta.unsqueeze(2)
        basis_exp = basis.view(1, 1, 8, 8, 1)

        deriv = n * (delta_exp * basis_exp).sum(dim=3)

        dx, dy = deriv[..., 0], deriv[..., 1]
        yaw = torch.atan2(dy, dx).unsqueeze(-1)

        return torch.cat([xy8, yaw], dim=-1)

    def forward_train_rl(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                         status_encoding, status_feature, camera_feature, targets,
                         global_img, eta, metric_cache, cal_pdm, token) -> Dict[str, torch.Tensor]:
        """Training forward pass with RL scoring (requires PDM)."""
        raise NotImplementedError(
            "forward_train_rl requires PDM simulator/scorer which are not ported. "
            "Use inference-only methods: forward_test_rl or forward_inference_scaling."
        )

    def forward_test_rl(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                        status_encoding, status_feature, camera_feature, targets,
                        global_img, metric_cache, eta=1.0, cal_pdm=True,
                        token=None) -> Dict[str, torch.Tensor]:
        """Inference forward pass with coarse + fine scoring."""
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = 10
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])

        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = diffusion_output.shape[1]

        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)

            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            poses_reg_list, poses_cls_list, _ = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            diffusion_output, _, _ = self.diffusionrl_scheduler.step(
                model_output=x_start, timestep=k, sample=diffusion_output, eta=0.0)

        diffusion_output = self.add_mul_noise(diffusion_output)
        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output)

        # Scorer
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(
            diffusion_output, bs, diffusion_output.shape[1])

        # Coarse scorer
        traj_feature_list = self.scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img)
        traj_feature = traj_feature_list[-1]

        NC_score = self.NC_head(traj_feature).squeeze(-1)
        EP_score = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score = self.C_head(traj_feature).squeeze(-1)
        final_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
        )

        best_coarse_flat = torch.argmax(final_coarse_reward, dim=-1)
        coarse_traj = diffusion_output[
            torch.arange(bs, device=device), best_coarse_flat].unsqueeze(1)
        traj_to_score = [coarse_traj]

        topk = 32
        traj_feature, noisy_traj_points_xy, sub_rewards_topk, topk_idx, topk_val = self._select_topk(
            final_coarse_reward=final_coarse_reward, topk=topk,
            traj_feature=traj_feature, noisy_traj_points_xy=noisy_traj_points_xy,
            sub_rewards_group=None)

        fine_traj_feature_list = self.fine_scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img)
        loss_fine, final_fine_reward, fine_reward, fine_sub_loss_dict, fine_reward_dict, fine_best_idx_list = \
            self._score_fine_multi(fine_traj_feature_list, sub_rewards_topk, only_reward=True)

        for best_idx_local in fine_best_idx_list:
            global_best_idx = topk_idx[torch.arange(bs, device=device), best_idx_local]
            fine_traj = diffusion_output[
                torch.arange(bs, device=device), global_best_idx].unsqueeze(1)
            traj_to_score.append(fine_traj)

        traj_to_score = torch.cat(traj_to_score, dim=1)

        if not cal_pdm:
            topk_trajectories = diffusion_output[
                torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk), topk_idx]

            return {
                "trajectory": traj_to_score[:, -1],
                "trajectory_candidates": traj_to_score,
                "trajectory_topk": topk_trajectories,
                "topk_scores": topk_val,
                "trajectory_coarse": diffusion_output,
                "coarse_scores": final_coarse_reward,
            }

        return {"trajectory": traj_to_score[:, -1]}

    def forward_inference_scaling(self, ego_query, agents_query, bev_feature,
                                  bev_spatial_shape, status_encoding, status_feature,
                                  camera_feature, global_img,
                                  num_groups=10) -> Dict[str, torch.Tensor]:
        """Generate num_groups * ego_fut_mode candidates with coarse scores."""
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])

        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = diffusion_output.shape[1]

        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)

            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            poses_reg_list, poses_cls_list, _ = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            diffusion_output, _, _ = self.diffusionrl_scheduler.step(
                model_output=x_start, timestep=k, sample=diffusion_output, eta=0.0)

        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output)

        # Run coarse scorer
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(
            diffusion_output, bs, diffusion_output.shape[1])

        traj_feature_list = self.scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img)
        traj_feature = traj_feature_list[-1]

        NC_score = self.NC_head(traj_feature).squeeze(-1)
        EP_score = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score = self.C_head(traj_feature).squeeze(-1)
        final_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
        )

        return {
            "all_candidates": diffusion_output,
            "coarse_scores": final_coarse_reward,
            "confidence_scores": None,
        }

    def forward_test_rl_temporal_scorer(
        self, ego_query, agents_query, bev_feature, bev_spatial_shape,
        status_encoding, status_feature, camera_feature,
        targets, global_img, metric_cache, current_time: float,
        eta=1.0, cal_pdm=True, token=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with temporal consistency scoring."""
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusionrl_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        num_groups = 10
        plan_anchor = self.plan_anchor.unsqueeze(0).unsqueeze(0).repeat(bs, num_groups, 1, 1, 1)
        plan_anchor = plan_anchor.view(bs, num_groups * self.ego_fut_mode, *plan_anchor.shape[3:])

        diffusion_output = self.norm_odo(plan_anchor)
        noise = torch.randn(diffusion_output.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        diffusion_output = self.diffusion_scheduler.add_noise(
            original_samples=diffusion_output, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = diffusion_output.shape[1]

        for i, k in enumerate(roll_timesteps[:]):
            x_boxes = torch.clamp(diffusion_output, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=diffusion_output.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(diffusion_output.device)

            timesteps = timesteps.expand(diffusion_output.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            poses_reg_list, poses_cls_list, _ = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg = poses_reg_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            diffusion_output, _, _ = self.diffusionrl_scheduler.step(
                model_output=x_start, timestep=k, sample=diffusion_output, eta=0.0)

        diffusion_output = self.add_mul_noise(diffusion_output)
        diffusion_output = self.denorm_odo(diffusion_output)
        diffusion_output = self.bezier_xyyaw(diffusion_output)

        # Scorer with temporal consistency
        noisy_traj_points_xy, traj_feature, time_embed = self._get_scorer_inputs(
            diffusion_output, bs, diffusion_output.shape[1])

        traj_feature_list = self.scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img)
        traj_feature = traj_feature_list[-1]

        NC_score = self.NC_head(traj_feature).squeeze(-1)
        EP_score = self.EP_head(traj_feature).squeeze(-1)
        DAC_score = self.DAC_head(traj_feature).squeeze(-1)
        TTC_score = self.TTC_head(traj_feature).squeeze(-1)
        C_score = self.C_head(traj_feature).squeeze(-1)

        pdm_coarse_reward = (
            self.sigmoid(NC_score) * self.sigmoid(DAC_score) *
            (5 * self.sigmoid(TTC_score) + 5 * self.sigmoid(EP_score) + 2 * self.sigmoid(C_score)) / 12
        )

        temporal_scores = self.compute_temporal_consistency_score(diffusion_output, current_time)

        final_coarse_reward = (
            (1 - self.temporal_lambda) * pdm_coarse_reward +
            self.temporal_lambda * temporal_scores
        )

        best_coarse_flat = torch.argmax(final_coarse_reward, dim=-1)
        coarse_traj = diffusion_output[
            torch.arange(bs, device=device), best_coarse_flat].unsqueeze(1)
        traj_to_score = [coarse_traj]

        topk = 32
        traj_feature, noisy_traj_points_xy, sub_rewards_topk, topk_idx, topk_val = self._select_topk(
            final_coarse_reward=final_coarse_reward, topk=topk,
            traj_feature=traj_feature, noisy_traj_points_xy=noisy_traj_points_xy,
            sub_rewards_group=None)

        fine_traj_feature_list = self.fine_scorer_decoder(
            traj_feature, noisy_traj_points_xy, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img)
        loss_fine, final_fine_reward, fine_reward, fine_sub_loss_dict, fine_reward_dict, fine_best_idx_list = \
            self._score_fine_multi(fine_traj_feature_list, sub_rewards_topk, only_reward=True)

        for best_idx_local in fine_best_idx_list:
            global_best_idx = topk_idx[torch.arange(bs, device=device), best_idx_local]
            fine_traj = diffusion_output[
                torch.arange(bs, device=device), global_best_idx].unsqueeze(1)
            traj_to_score.append(fine_traj)

        traj_to_score = torch.cat(traj_to_score, dim=1)
        selected_traj = traj_to_score[:, -1]

        consensus_traj = self.compute_consensus_trajectory(diffusion_output, pdm_coarse_reward)

        for b in range(bs):
            self.update_temporal_history(consensus_traj[b].cpu().numpy(), current_time)

        if not cal_pdm:
            topk_trajectories = diffusion_output[
                torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk), topk_idx]

            return {
                "trajectory": traj_to_score[:, -1],
                "trajectory_candidates": traj_to_score,
                "trajectory_topk": topk_trajectories,
                "topk_scores": topk_val,
                "trajectory_coarse": diffusion_output,
                "coarse_scores": final_coarse_reward,
                "pdm_coarse_scores": pdm_coarse_reward,
                "temporal_coarse_scores": temporal_scores,
                "temporal_history_length": len(self.temporal_history),
                "consensus_trajectory": consensus_traj,
            }

        return {"trajectory": traj_to_score[:, -1]}
