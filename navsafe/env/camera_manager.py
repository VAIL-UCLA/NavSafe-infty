"""NuRec camera dispatch and observer configuration."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)
EGO_AGENT_ID = "__ego__"
PREDICTED_EGO_OBSERVER_ID = "__predicted_ego__"
_OWNED_PREFIXES = ("_cam_",)
_OWNED_EXACT = frozenset({"_logged_cam_scale", "_env"})


class CameraManager:
    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. Owned (but not-yet-created)
        # state must report absent here, never forward to the env; everything
        # else is shared env state/methods and forwards.
        if name in _OWNED_EXACT or name.startswith(_OWNED_PREFIXES):
            raise AttributeError(name)
        return getattr(self._env, name)

    def get_camera_images(self, cam_configs: Dict) -> Dict[str, np.ndarray]:
        """Render camera images at the current ego pose via the active renderer.

        Delegates to the NuRec gRPC renderer.

        Args:
            cam_configs: Dict mapping camera name → config dict with keys
                         x, y, z, yaw, pitch, roll, fov, width, height.

        Returns:
            Dict mapping camera name → (H, W, 3) uint8 BGR numpy array.
        """
        ego_state = self.get_ego_state()
        agent_states = self._collect_agent_states_for_renderer()
        renderer = self.renderer
        # Keep the renderer synchronized with the scenario replay time.
        renderer.set_timestep(self.scenario_timestep)
        return renderer.get_camera_images(
            ego_state=ego_state,
            cam_configs=cam_configs,
            agent_states=agent_states,
        )

    def get_camera_images_at_pose(
        self, observer_pose: Dict, cam_configs: Dict,
    ) -> Dict[str, np.ndarray]:
        """Render the ego rig from a HYPOTHETICAL ego pose; the sim is untouched.

        The scene stays exactly as it is at the current timestep — every other
        agent at its present pose — and only the camera moves. This is the
        observation a trajectory-prediction model needs at a predicted future
        ego state (PPL's ``f(s, a, H)`` under the "other traffic stays put"
        assumption its rule-based predictor makes); nothing here advances
        time, moves the ego prim, or mutates the replay.

        The ego body is excluded from the observer's agent list.

        Args:
            observer_pose: ``get_ego_state``-compatible dict (``position``,
                ``heading``, ``velocity``) in world frame.
            cam_configs: Rig dict keyed by camera name, as for
                :meth:`get_camera_images`.

        Returns:
            ``{cam_name: (H, W, 3) uint8 BGR}``.
        """
        ego_ids = {EGO_AGENT_ID, "ego"}
        manager_ego = getattr(getattr(self, "agent_manager", None), "ego_agent_id", None)
        if manager_ego is not None:
            ego_ids.add(str(manager_ego))
        scenario = getattr(self, "_scenario_data", None)
        if isinstance(scenario, dict):
            sdc = (scenario.get("metadata") or {}).get("sdc_id")
            if sdc is not None:
                ego_ids.add(str(sdc))
        # The ego's real body would otherwise be drawn at the pose the
        # hypothetical camera is looking back over.
        agent_states = [
            state for state in self._collect_agent_states_for_renderer()
            if str(state.get("id", "")) not in ego_ids
        ]
        renderer = self.renderer
        renderer.set_timestep(self.scenario_timestep)
        return renderer.get_agent_camera_images(
            agent_id=PREDICTED_EGO_OBSERVER_ID,
            observer_pose=observer_pose,
            cam_configs=cam_configs,
            agent_states=agent_states,
        )

    def get_agent_camera_images(
        self,
        cam_configs_by_agent: Dict[str, Dict[str, Dict]],
        capture_budget: Optional[int] = None,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """Render camera images for one or more observer agents.

        Args:
            cam_configs_by_agent: ``{agent_id: {cam_name: rig_cfg, ...}, ...}``.
                ``agent_id == EGO_AGENT_ID`` reuses the legacy ego rig and
                prim paths.  Other ids are resolved via
                :meth:`_resolve_agent_world_pose`.
            capture_budget: Optional hard cap on the number of cameras
                captured this call.  When exceeded, agent ids and camera
                names are sorted lexicographically and the first
                ``capture_budget`` pairs are kept; the rest are skipped
                (a single warning is logged).  ``None`` disables the cap.

        Returns:
            ``{(agent_id, cam_name): (H, W, 3) uint8 BGR}``.  Agents that
            cannot be resolved at the current timestep are omitted.
        """
        if not cam_configs_by_agent:
            return {}

        # Expand into a deterministically-ordered list of (agent_id, cam_name, cam_cfg).
        flat: list = []
        for agent_id in sorted(cam_configs_by_agent.keys()):
            rig = cam_configs_by_agent[agent_id] or {}
            for cam_name in sorted(rig.keys()):
                flat.append((agent_id, cam_name, rig[cam_name]))

        if capture_budget is not None and len(flat) > int(capture_budget):
            dropped = [(a, c) for (a, c, _) in flat[int(capture_budget):]]
            logger.warning(
                "[NavSafeEnv] Agent-camera capture budget %d exceeded "
                "(%d requested); dropping %d (first skipped: %s)",
                int(capture_budget), len(flat), len(dropped),
                f"{dropped[0][0]}/{dropped[0][1]}" if dropped else "-",
            )
            flat = flat[:int(capture_budget)]

        # Collect poses for the unique observer ids once per call.
        pose_cache: Dict[str, Dict] = {}
        for agent_id, _, _ in flat:
            if agent_id in pose_cache:
                continue
            pose = self._resolve_agent_world_pose(agent_id)
            if pose is not None:
                pose_cache[agent_id] = pose

        agent_states = self._collect_agent_states_for_renderer()
        images: Dict[Tuple[str, str], np.ndarray] = {}

        # Drive the renderer's replay cursor once for this capture (no-op for
        # stateless backends; time-varying backends like 3dgs use it to pick
        # the frame). Same single source of truth as the ego capture path.
        self.renderer.set_timestep(self.scenario_timestep)

        # Bundle requests by agent so each renderer call sees a single
        # coherent per-observer rig (non-IsaacSim renderers expect this).
        grouped: Dict[str, Dict[str, Dict]] = {}
        for agent_id, cam_name, cam_cfg in flat:
            if agent_id not in pose_cache:
                continue
            grouped.setdefault(agent_id, {})[cam_name] = cam_cfg

        for agent_id, rig in grouped.items():
            pose = pose_cache[agent_id]
            try:
                if hasattr(self.renderer, "get_agent_camera_images"):
                    per_agent = self.renderer.get_agent_camera_images(
                        agent_id=agent_id,
                        observer_pose=pose,
                        cam_configs=rig,
                        agent_states=agent_states,
                    )
                else:
                    # Back-compat: older renderers only know the ego API.
                    per_agent = self.renderer.get_camera_images(
                        ego_state=pose,
                        cam_configs=rig,
                        agent_states=agent_states,
                    )
            except Exception as e:
                logger.warning(
                    "[NavSafeEnv] get_agent_camera_images failed for %s: %s",
                    agent_id, e,
                )
                continue
            for cam_name, img in (per_agent or {}).items():
                images[(agent_id, cam_name)] = img

        return images

    def _scale_cam_configs(self, cam_configs: Dict) -> Dict:
        """Return a copy of ``cam_configs`` with width/height scaled.

        Applies ``cfg.camera_resolution_scale`` (default 1.0 = no change) to
        every camera's ``width`` and ``height`` so render load can be reduced
        without changing the camera rig the policy expects. The aspect ratio
        is preserved; dimensions are clamped to a minimum of 16 px and made
        even. A scale of 1.0 returns the configs unchanged (and logs nothing).
        """
        scale = float(getattr(self.cfg, "camera_resolution_scale", 1.0) or 1.0)
        if scale == 1.0 or not cam_configs:
            return cam_configs

        def _even(v: float) -> int:
            iv = max(16, int(round(v)))
            return iv - (iv % 2)

        scaled: Dict = {}
        for name, cfg in cam_configs.items():
            new_cfg = dict(cfg)
            if "width" in new_cfg:
                new_cfg["width"] = _even(int(new_cfg["width"]) * scale)
            if "height" in new_cfg:
                new_cfg["height"] = _even(int(new_cfg["height"]) * scale)
            scaled[name] = new_cfg
        # Log once per process so the user can confirm the reduction.
        if not getattr(self, "_logged_cam_scale", False):
            self._logged_cam_scale = True
            any_cam = next(iter(scaled.values()))
            print(f"[NavSafeEnv] camera_resolution_scale={scale} -> "
                  f"e.g. {any_cam.get('width')}x{any_cam.get('height')}")
        return scaled

    def get_ego_front_image_bgr(self) -> Optional[np.ndarray]:
        """Request the front observation from NuRec."""
        from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
        return self.get_camera_images({"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}).get("CAM_F0")
