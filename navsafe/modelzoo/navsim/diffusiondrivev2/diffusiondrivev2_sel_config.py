"""
DiffusionDrive v2 selection model configuration.
Ported from BridgeSim — nuplan enum imports replaced with simple labels.
Extends v1 config with num_groups for the scorer.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from navsafe.modelzoo.common.trajectory_sampling import TrajectorySampling


@dataclass
class TransfuserConfig:
    """Global TransFuser config for DiffusionDrive v2."""

    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(time_horizon=4, interval_length=0.5)
    )

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = ""
    plan_anchor_path: str = "kmeans_navsim_traj_20.npy"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # v2-specific
    num_groups: int = 8
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False

    # BEV mapping — simple integer labels replacing nuplan enums
    bev_semantic_classes = {
        1: ("polygon", ["lane", "intersection"]),
        2: ("polygon", ["walkways"]),
        3: ("linestring", ["lane", "lane_connector"]),
        4: ("box", ["czone_sign", "barrier", "traffic_cone", "generic_object"]),
        5: ("box", ["vehicle"]),
        6: ("box", ["pedestrian"]),
    }

    bev_pixel_width: int = 256
    bev_pixel_height: int = 128
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # optimizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name": {
            "image_encoder": {
                "lr_mult": cfg_lr_mult
            }
        }
    }

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
