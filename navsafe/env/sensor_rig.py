"""IsaacLab LiDAR, IMU, contact and frame-transformer sensor support."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SensorRig:
    """Construct + read the ego sensor suite for a :class:`NavSafeEnv`.

    Args:
        env: The owning environment. Used only for shared, non-sensor state
            (``cfg`` and the prim-path / lidar-mesh scene helpers).
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        # Owned sensor handles. Pre-set so the getters and the env's read-only
        # properties can rely on attribute presence even before ``setup()``.
        self.ego_lidar_sensor: Any = None
        self.ego_imu_sensor: Any = None
        self.ego_contact_sensor: Any = None
        self.ego_frame_transformer_sensor: Any = None

    def setup(self) -> None:
        """Instantiate optional IsaacLab sensors declared on ``env.cfg``.

        Stores the resulting sensor objects on ``self`` as:

        * ``self.ego_lidar_sensor``   — :class:`isaaclab.sensors.RayCaster`
        * ``self.ego_imu_sensor``     — :class:`isaaclab.sensors.Imu`
        * ``self.ego_contact_sensor`` — :class:`isaaclab.sensors.ContactSensor`

        Each attribute is set to ``None`` if either the cfg field is
        unset or the corresponding sensor class isn't available in the
        running IsaacLab version.  Failures are logged at WARNING level.

        Ported verbatim from ``ScenarioReplayEnv._setup_extra_sensors``,
        adapted to read the ``EnvCfg`` sensor slots (``cfg.ego_lidar`` /
        ``cfg.ego_imu`` / ``cfg.ego_contact`` /
        ``cfg.ego_frame_transformer``).
        """
        cfg = self._env.cfg
        # Reset owned handles so re-running setup() is idempotent.
        self.ego_lidar_sensor = None
        self.ego_imu_sensor = None
        self.ego_contact_sensor = None
        self.ego_frame_transformer_sensor = None

        lidar_cfg = getattr(cfg, "ego_lidar", None)
        imu_cfg = getattr(cfg, "ego_imu", None)
        contact_cfg = getattr(cfg, "ego_contact", None)
        ft_cfg = getattr(cfg, "ego_frame_transformer", None)
        # Nothing requested → nothing to do. (Previously this only checked
        # lidar/imu/contact and returned early, which silently skipped the
        # camera + frame-transformer sensors when only those were configured.)
        if all(c is None for c in (lidar_cfg, imu_cfg, contact_cfg, ft_cfg)):
            return

        if lidar_cfg is not None or imu_cfg is not None or contact_cfg is not None:
            try:
                from isaaclab.sensors import (
                    ContactSensor,
                    ContactSensorCfg,
                    Imu,
                    ImuCfg,
                    RayCaster,
                    RayCasterCfg,
                    patterns,
                )
            except Exception as exc:  # pragma: no cover — Linux+IsaacSim only
                logger.warning(
                    "[NavSafeEnv] isaaclab.sensors unavailable; "
                    "skipping ego LiDAR/IMU/contact: %s", exc,
                )
                lidar_cfg = imu_cfg = contact_cfg = None

        # ── LiDAR (ray-cast pattern) ──────────────────────────────────
        if lidar_cfg is not None:
            try:
                pattern = patterns.LidarPatternCfg(
                    channels=int(lidar_cfg.vertical_channels),
                    vertical_fov_range=(
                        float(lidar_cfg.vertical_fov_min_deg),
                        float(lidar_cfg.vertical_fov_max_deg),
                    ),
                    horizontal_fov_range=(
                        -float(lidar_cfg.horizontal_fov_deg) / 2.0,
                        float(lidar_cfg.horizontal_fov_deg) / 2.0,
                    ),
                    horizontal_res=float(lidar_cfg.horizontal_res_deg),
                )
                mesh_path = self._env._build_combined_lidar_mesh() or "/World/ground"
                rc_kwargs = dict(
                    prim_path=self._env._resolve_ego_prim_path(lidar_cfg.prim_path),
                    update_period=float(lidar_cfg.update_period),
                    offset=RayCasterCfg.OffsetCfg(pos=tuple(lidar_cfg.offset_pos)),
                    pattern_cfg=pattern,
                    max_distance=float(lidar_cfg.max_distance),
                    debug_vis=bool(lidar_cfg.debug_vis),
                    mesh_prim_paths=[mesh_path],
                )
                # Prefer the new `ray_alignment` API (IsaacLab ≥ 2.1.1);
                # fall back to the deprecated `attach_yaw_only` flag on
                # older installs that don't accept the new kwarg.
                ray_kwargs = (
                    {"ray_alignment": "yaw"} if lidar_cfg.attach_yaw_only
                    else {"ray_alignment": "base"}
                )
                try:
                    rc_cfg = RayCasterCfg(**rc_kwargs, **ray_kwargs)
                except TypeError:
                    rc_cfg = RayCasterCfg(
                        **rc_kwargs, attach_yaw_only=bool(lidar_cfg.attach_yaw_only),
                    )
                self.ego_lidar_sensor = RayCaster(rc_cfg)
                logger.info(
                    "[NavSafeEnv] Ego LiDAR attached (%d channels, %.1f°FoV, %.1fm)",
                    rc_cfg.pattern_cfg.channels,
                    lidar_cfg.horizontal_fov_deg,
                    rc_cfg.max_distance,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("[NavSafeEnv] Ego LiDAR setup failed: %s", exc)

        # ── IMU ───────────────────────────────────────────────────────
        if imu_cfg is not None:
            try:
                imu_prim_path = self._env._resolve_ego_prim_path(imu_cfg.prim_path)
                im_cfg = ImuCfg(
                    prim_path=imu_prim_path,
                    update_period=float(imu_cfg.update_period),
                    debug_vis=bool(imu_cfg.debug_vis),
                    gravity_bias=tuple(imu_cfg.gravity_bias),
                )
                self.ego_imu_sensor = Imu(im_cfg)
                logger.info("[NavSafeEnv] Ego IMU attached at %s", imu_prim_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("[NavSafeEnv] Ego IMU setup failed: %s", exc)

        # ── Contact sensor ────────────────────────────────────────────
        if contact_cfg is not None:
            try:
                # contact_dynamics: the ego's only rigid body (and the only
                # prim with PhysxContactReportAPI) is the physics proxy —
                # the visual ego prim can never report contacts.
                if (getattr(self._env, "_contact_dynamics", False)
                        and self._env._ego_phys_proxy_path):
                    contact_prim_path = self._env._ego_phys_proxy_path
                else:
                    contact_prim_path = self._env._resolve_ego_prim_path(contact_cfg.prim_path)
                cs_cfg = ContactSensorCfg(
                    prim_path=contact_prim_path,
                    update_period=float(contact_cfg.update_period),
                    history_length=int(contact_cfg.history_length),
                    track_pose=bool(contact_cfg.track_pose),
                    track_air_time=bool(contact_cfg.track_air_time),
                    debug_vis=bool(contact_cfg.debug_vis),
                )
                self.ego_contact_sensor = ContactSensor(cs_cfg)
                logger.info(
                    "[NavSafeEnv] Ego ContactSensor attached at %s", contact_prim_path,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("[NavSafeEnv] Ego ContactSensor setup failed: %s", exc)

        # ── FrameTransformer (ego-to-world SE(3) tracking) ────────────
        if ft_cfg is not None:
            try:
                from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
                targets = ft_cfg.target_frames or ["/World"]
                target_cfgs = [
                    FrameTransformerCfg.FrameCfg(prim_path=t) for t in targets
                ]
                ft_prim_path = self._env._resolve_ego_prim_path(ft_cfg.prim_path)
                ftc = FrameTransformerCfg(
                    prim_path=ft_prim_path,
                    target_frames=target_cfgs,
                    update_period=float(ft_cfg.update_period),
                    debug_vis=bool(ft_cfg.debug_vis),
                )
                self.ego_frame_transformer_sensor = FrameTransformer(ftc)
                logger.info(
                    "[NavSafeEnv] Ego FrameTransformer attached (%s → %s)",
                    ft_prim_path, targets,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("[NavSafeEnv] Ego FrameTransformer setup failed: %s", exc)

    def get_lidar_returns(self) -> Optional[np.ndarray]:
        """Return the latest ray-cast LiDAR hit positions or ``None``.

        Output shape ``(num_envs, num_rays, 3)`` (world frame).  Caller is
        responsible for transforming to the body frame if needed.
        """
        sensor = getattr(self, "ego_lidar_sensor", None)
        if sensor is None:
            return None
        try:
            data = getattr(sensor, "data", None)
            if data is None:
                return None
            hits = getattr(data, "ray_hits_w", None)
            if hits is None:
                return None
            return hits.detach().cpu().numpy() if hasattr(hits, "detach") else np.asarray(hits)
        except Exception as exc:  # pragma: no cover
            logger.debug("[NavSafeEnv] LiDAR read failed: %s", exc)
            return None

    def get_imu_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return ``{lin_acc_b, ang_vel_b, quat_w}`` ndarrays or ``None``."""
        sensor = getattr(self, "ego_imu_sensor", None)
        if sensor is None:
            return None
        try:
            data = sensor.data
            def _np(t):
                return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            return {
                "lin_acc_b": _np(data.lin_acc_b),
                "ang_vel_b": _np(data.ang_vel_b),
                "quat_w": _np(data.quat_w),
            }
        except Exception as exc:  # pragma: no cover
            logger.debug("[NavSafeEnv] IMU read failed: %s", exc)
            return None

    def get_contact_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return contact-sensor net forces / pose, or ``None``.

        Keys (when available):
        * ``net_forces_w`` — ``(N, B, 3)`` net contact force in world frame.
        * ``in_contact``   — ``(N, B)`` bool mask, True when force > epsilon.
        * ``last_contact_pos_w`` — last contact point in world frame, when
          ``track_pose=True``.
        """
        sensor = getattr(self, "ego_contact_sensor", None)
        if sensor is None:
            return None
        try:
            data = sensor.data
            def _np(t):
                return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            out: Dict[str, np.ndarray] = {}
            net = getattr(data, "net_forces_w", None)
            if net is not None:
                arr = _np(net)
                out["net_forces_w"] = arr
                out["in_contact"] = (np.linalg.norm(arr, axis=-1) > 1e-3)
            pos = getattr(data, "pos_w", None)
            if pos is not None:
                out["last_contact_pos_w"] = _np(pos)
            return out
        except Exception as exc:  # pragma: no cover
            logger.debug("[NavSafeEnv] Contact read failed: %s", exc)
            return None

    def get_frame_transforms(self) -> Optional[Dict[str, np.ndarray]]:
        """Return ego-to-target SE(3) transforms or ``None``.

        Returns ``{target_prim_path: (num_envs, 7)}`` where each row is
        ``[px, py, pz, qw, qx, qy, qz]``.
        """
        sensor = getattr(self, "ego_frame_transformer_sensor", None)
        if sensor is None:
            return None
        try:
            data = sensor.data
            out: Dict[str, np.ndarray] = {}
            # FrameTransformer stores target_pos_w and target_quat_w
            pos = getattr(data, "target_pos_w", None)
            quat = getattr(data, "target_quat_w", None)
            if pos is not None and quat is not None:
                def _np(t):
                    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
                targets = sensor.cfg.target_frames
                pos_np = _np(pos)  # (num_envs, num_targets, 3)
                quat_np = _np(quat)  # (num_envs, num_targets, 4)
                for i, tf in enumerate(targets):
                    prim = getattr(tf, "prim_path", str(tf))
                    pose = np.concatenate([pos_np[:, i], quat_np[:, i]], axis=-1)
                    out[prim] = pose
            return out if out else None
        except Exception as exc:  # pragma: no cover
            logger.debug("[NavSafeEnv] FrameTransformer read failed: %s", exc)
            return None
