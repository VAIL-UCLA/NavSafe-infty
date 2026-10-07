"""Frame transformer (ego-to-world pose) configuration.

Moved from ``navsafe/envs/scenario_replay_env_cfg.py`` to consolidate
sensor configurations under ``navsafe/component/sensors/``.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class EgoFrameTransformerCfg:
    """Configuration for a FrameTransformer tracking ego-to-world pose.

    FrameTransformer provides the SE(3) transform between a source body
    and one or more target frames at every sim step.  Useful for:
    - Extracting ego pose without finite-differencing.
    - Computing relative transforms to other agents for observation.

    Attributes:
        prim_path: Source body prim (ego chassis).
        target_frames: List of target prim paths to track relative to
            the source.  Defaults to world origin.
        update_period: Sensor update period (0.0 = every step).
    """
    prim_path: str = "{ENV_REGEX_NS}/ego_vehicle/chassis"
    target_frames: list | None = None  # defaults to ["/World"] at runtime
    update_period: float = 0.0
    debug_vis: bool = False
