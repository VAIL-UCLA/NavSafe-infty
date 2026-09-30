"""Environment presets for replay and sensor-policy evaluation."""

from navsafe.env.env_cfg import EnvCfg


REPLAY: EnvCfg = EnvCfg(
    obs_kind="state",
    traffic_mode="log_replay",
    scenario_source="py123d",
    render_backend="nurec_grpc",
    dt=0.1,
    max_episode_steps=200,
    num_envs=1,
    reward_weights={
        "progress": 0.0,
        "comfort": 0.0,
        "collision": 0.0,
        "off_road": 0.0,
        "lane_keeping": 0.0,
        "speed": 0.0,
    },
    required_bundles=["scenarios-navhard"],
)


VISION_TRAIN: EnvCfg = EnvCfg(
    obs_kind="sensor",
    traffic_mode="log_replay",
    scenario_source="py123d",
    render_backend="nurec_grpc",
    dt=0.1,
    max_episode_steps=200,
    required_bundles=[],
)
