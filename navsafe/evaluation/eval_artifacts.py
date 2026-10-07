"""EvalArtifactWriter - the artifact / visualization output subsystem of Evaluator.

Owns the per-frame and end-of-run output side of evaluation: saving extra-sensor
artifacts (LiDAR BEV, camera extras, npz dumps), rendering the per-frame BEV /
front-cam visualisation, and generating the summary
GIFs. Extracted from the ~1,480-line ``Evaluator`` God class so this output
concern lives behind one narrow object.

Stateless w.r.t. the run: all evaluation state (config, env, model adapter,
frame buffers) is owned by the Evaluator. This writer holds a back reference and
forwards every shared read to it via ``__getattr__``; it writes only files, never
evaluator state (audited). The Evaluator keeps thin delegating wrappers so its
public surface and call sites are unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from navsafe.evaluation import vis_utils
except ImportError:  # pragma: no cover — vis_utils imports cv2 (viz extra)
    vis_utils = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _no_overlay() -> bool:
    """``NAVSAFE_NO_OVERLAY``: write ``cam_f0.jpg`` unannotated.

    Read per frame rather than cached so it can be flipped without rebuilding
    anything; it is one environment lookup against a jpeg encode.
    """
    return os.environ.get("NAVSAFE_NO_OVERLAY", "").lower() not in {
        "", "0", "false", "no", "off"}


class EvalArtifactWriter:
    """Output/visualisation subsystem for an :class:`Evaluator` (composition).

    Forwards all shared run state to the owning evaluator; writes only files.
    """

    def __init__(self, evaluator: Any) -> None:
        self._eval = evaluator

    def __getattr__(self, name: str) -> Any:
        if name == "_eval":
            raise AttributeError(name)
        return getattr(self._eval, name)

    @staticmethod
    def _safe_imwrite(path, img) -> bool:
        """Write an image to disk with error isolation.

        Wraps ``cv2.imwrite`` so a single bad write (e.g. permissions, an
        unexpected dtype, an empty array) doesn't take down the rest of
        the per-frame artifact pipeline.

        Returns True on success, False otherwise.
        """
        import cv2
        try:
            if img is None:
                return False
            return bool(cv2.imwrite(str(path), img))
        except Exception as exc:
            logger.warning("[vis] cv2.imwrite(%s) failed: %s", path, exc)
            return False

    def _save_extra_sensor_artifacts(self, frame_dir) -> None:
        """Persist optional sensor outputs (LiDAR/IMU/contact + cam extras).

        Each block is gated on the corresponding env getter returning
        non-None.  All writes are isolated so a single failure doesn't
        suppress the others.
        """
        env = self.env

        # ── Camera extras (depth / semseg / normals / ...) ────────────
        get_extras = getattr(env, "get_camera_extras", None)
        if callable(get_extras):
            try:
                extras = get_extras() or {}
            except Exception as exc:
                logger.debug("[vis] get_camera_extras failed: %s", exc)
                extras = {}
            for dt, arr in extras.items():
                self._save_camera_extra(frame_dir, dt, arr)

        # ── LiDAR (point cloud) ───────────────────────────────────────
        get_lidar = getattr(env, "get_lidar_returns", None)
        if callable(get_lidar):
            try:
                hits = get_lidar()
            except Exception as exc:
                logger.debug("[vis] get_lidar_returns failed: %s", exc)
                hits = None
            if hits is not None:
                arr = np.asarray(hits)
                try:
                    np.save(str(frame_dir / "lidar.npy"), arr)
                except Exception as exc:
                    logger.warning("[vis] lidar.npy write failed: %s", exc)
                try:
                    self._save_lidar_bev(frame_dir, arr)
                except Exception as exc:
                    logger.debug("[vis] lidar.png render failed: %s", exc)

        # ── IMU ───────────────────────────────────────────────────────
        get_imu = getattr(env, "get_imu_state", None)
        if callable(get_imu):
            try:
                imu = get_imu()
            except Exception as exc:
                logger.debug("[vis] get_imu_state failed: %s", exc)
                imu = None
            if imu is not None:
                try:
                    EvalArtifactWriter._dump_npz(frame_dir / "imu.npz", imu)
                except Exception as exc:
                    logger.warning("[vis] imu.npz write failed: %s", exc)

        # ── Contact sensor ────────────────────────────────────────────
        get_contact = getattr(env, "get_contact_state", None)
        if callable(get_contact):
            try:
                ct = get_contact()
            except Exception as exc:
                logger.debug("[vis] get_contact_state failed: %s", exc)
                ct = None
            if ct is not None:
                try:
                    EvalArtifactWriter._dump_npz(frame_dir / "contact.npz", ct)
                except Exception as exc:
                    logger.warning("[vis] contact.npz write failed: %s", exc)

    def _save_lidar_bev(self, frame_dir, hits) -> None:
        """Render an ego-centred BEV scatter of the LiDAR returns.

        Points arrive in world frame with shape ``(num_envs, num_rays, 3)``
        or ``(num_rays, 3)``.  We pull the first env, transform into the
        ego frame (translate by -ego_xy, rotate by -heading so +x points
        forward / up in the image), drop hits beyond the lidar's max
        range, and rasterise a scatter coloured by height.  Output is
        ``frame_dir/lidar.png`` so the visualisation auto-discovery
        emits ``lidar.gif`` alongside ``topdown.gif`` / ``cam_f0.gif``.
        """
        import cv2  # local import: cv2 is already a hot-path dep above

        pts = np.asarray(hits)
        if pts.ndim == 3:
            pts = pts[0]
        # Drop misses: IsaacLab returns NaN/Inf when a ray doesn't hit
        # anything inside max_distance.  Without this filter the BEV
        # scatter ends up empty (NaN→0 cast collapses every point to
        # the origin) and the cast emits a RuntimeWarning per frame.
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            return

        ego = self.env.get_ego_state()
        ex, ey = float(ego["position"][0]), float(ego["position"][1])
        heading = float(ego.get("heading", 0.0))

        dx = pts[:, 0] - ex
        dy = pts[:, 1] - ey
        c, s = np.cos(-heading), np.sin(-heading)
        fwd = c * dx - s * dy
        left = s * dx + c * dy
        z = pts[:, 2]

        view_range = float(getattr(self.env.cfg.ego_lidar, "max_distance", 100.0))
        size = 512
        scale = size / (2.0 * view_range)
        # Image axes: forward → up, left → left.  Row 0 is top of image.
        u = (size * 0.5 - left * scale).astype(np.int32)
        v = (size * 0.5 - fwd * scale).astype(np.int32)
        valid = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        u, v, z = u[valid], v[valid], z[valid]
        if u.size == 0:
            return

        img = np.zeros((size, size, 3), dtype=np.uint8)
        z_lo, z_hi = float(np.min(z)), float(np.max(z))
        z_norm = (z - z_lo) / max(z_hi - z_lo, 1e-3)
        colors = cv2.applyColorMap(
            (z_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS
        ).reshape(-1, 3)
        img[v, u] = colors

        # Ego marker: small filled triangle pointing up (forward).
        cx, cy = size // 2, size // 2
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=10, thickness=2)
        # Range rings every 25 m for visual reference.
        for r in (25.0, 50.0, 75.0):
            if r < view_range:
                cv2.circle(img, (cx, cy), int(r * scale), (60, 60, 60), 1)

        EvalArtifactWriter._safe_imwrite(frame_dir / "lidar.png", img)

    def _save_camera_extra(self, frame_dir, data_type: str, arr) -> None:
        """Persist one camera-extra annotator output to a sensible format.

        - ``distance_to_image_plane`` / ``distance_to_camera`` / aliases —
          saved as a colorized PNG (turbo colormap, near→far) plus the
          raw float depth as ``<name>.npy``.
        - ``semantic_segmentation`` / ``instance_*_segmentation`` —
          saved as PNG when colorized (uint8x4), as ``.npy`` otherwise.
        - ``normals``, ``motion_vectors`` — saved as ``.npy`` (float).
        - ``bounding_box_*`` — saved as ``.npy`` (structured array).
        - Anything else — saved as ``.npy`` so the data is never lost.
        """
        import cv2
        a = np.asarray(arr)
        name = data_type.lower()
        depth_aliases = ("distance_to_image_plane", "distance_to_camera", "depth")

        try:
            if name in depth_aliases:
                np.save(str(frame_dir / f"{name}.npy"), a)
                if a.ndim == 2 and np.issubdtype(a.dtype, np.floating):
                    finite = np.isfinite(a)
                    if finite.any():
                        d = a.copy()
                        d[~finite] = np.nan
                        lo = float(np.nanmin(d))
                        hi = float(np.nanmax(d))
                        if hi > lo:
                            norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
                            norm[np.isnan(norm)] = 0.0
                            color = cv2.applyColorMap(
                                (norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO,
                            )
                            EvalArtifactWriter._safe_imwrite(frame_dir / f"{name}.png", color)
                return

            if "segmentation" in name:
                if a.ndim == 3 and a.shape[-1] in (3, 4) and a.dtype == np.uint8:
                    img = a[..., :3] if a.shape[-1] == 4 else a
                    EvalArtifactWriter._safe_imwrite(frame_dir / f"{name}.png", img[..., ::-1])
                else:
                    np.save(str(frame_dir / f"{name}.npy"), a)
                return

            if name in ("normals", "motion_vectors") or name.startswith("bounding_box"):
                np.save(str(frame_dir / f"{name}.npy"), a)
                return

            # Fallback: save raw
            np.save(str(frame_dir / f"{name}.npy"), a)
        except Exception as exc:
            logger.warning("[vis] camera-extra '%s' write failed: %s", data_type, exc)

    @staticmethod
    def _dump_npz(path, data: Dict) -> None:
        np.savez(str(path), allow_pickle=True, **{k: np.asarray(v) for k, v in data.items()})

    def _render_frame_vis(
        self,
        ego_state: Dict,
        images: Dict[str, np.ndarray],
        collision: bool,
    ) -> None:
        """Save per-frame BEV and front-camera visualizations."""
        import cv2  # local import — only needed when vis is enabled

        output_dir = self.config.output_dir  # see evaluator._save_results
        frame_dir = output_dir / "frames" / f"{self.frame:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        pos = np.asarray(ego_state["position"], dtype=float)
        heading = float(ego_state["heading"])
        speed_kmh = float(np.linalg.norm(
            np.asarray(ego_state.get("velocity", [0, 0, 0]), dtype=float)[:2]
        )) * 3.6
        model_name = type(self.adapter).__name__.replace("Adapter", "")
        planned_world = self._current_trajectory[:, :2] if self._current_trajectory is not None else None

        # Full scenario dict for rich rendering (map + agents)
        full_scenario = getattr(self.env, "current_scenario", None)

        # ── Compute ego-frame trajectory + driving command ──────────────
        plan_traj_ego = None
        # During the replay prefix _current_trajectory is the observed ego pose
        # (see the evaluator's warm-up branch), so drawing it would show the
        # ribbon only on replan frames. Prefer the held warm-up plan there;
        # outside the prefix this attribute is None and nothing changes.
        overlay_traj = getattr(self, "_warmup_overlay_plan", None)
        if overlay_traj is None:
            overlay_traj = self._current_trajectory
        if overlay_traj is not None:
            world_xy = overlay_traj[:, :2]
            delta = world_xy - pos[:2]
            cos_h = np.cos(-heading)
            sin_h = np.sin(-heading)
            ego_lat = sin_h * delta[:, 0] + cos_h * delta[:, 1]
            ego_fwd = cos_h * delta[:, 0] - sin_h * delta[:, 1]
            plan_traj_ego = np.stack([ego_lat, ego_fwd], axis=1)

        # Route-based command is primary; curvature-based is fallback
        if self._current_command >= 0:
            driving_command = self._current_command
        else:
            driving_command = vis_utils.derive_driving_command(plan_traj_ego)

        # Route waypoint data for visualization
        route_waypoints = self.full_route  # list of (pos, cmd, frame_idx) — never consumed
        target_waypoint = self._current_waypoint  # (pos, cmd, frame_idx) or None

        # ── BEV ─────────────────────────────────────────────────────────
        bev_img = None
        try:
            bev_img = vis_utils.render_bev(
                ego_position=pos,
                ego_heading=heading,
                planned_traj_world=planned_world,
                vehicle_states=self._history["vehicle_states"],
                frame_id=self.frame,
                model_name=model_name,
                speed_kmh=speed_kmh,
                collision=collision,
                scenario_data=full_scenario,
                driving_command=driving_command,
                sim_dt=self.config.sim_dt,
                route_waypoints=route_waypoints,
                target_waypoint=target_waypoint,
                agent_states=getattr(self.env, "agent_states", None),
            )
        except Exception as exc:
            logger.warning("[vis] render_bev failed at frame %d: %s", self.frame, exc)
        if bev_img is not None:
            self._safe_imwrite(frame_dir / "topdown.png", bev_img)

        # ── Front camera ────────────────────────────────────────────────
        cam_vis = None
        cam_cfg = None  # set only on the real-image path; needed for candidate projection
        cam_img = None
        # np.bool_ as well as bool: the assignment below ends in ndarray.any().
        has_real_image: bool | np.bool_ = False
        try:
            cam_configs = self.adapter.get_camera_configs()
            cam_name = "CAM_F0" if "CAM_F0" in cam_configs else next(iter(cam_configs), None)
            cam_img = images.get(cam_name) if cam_name else None

            # Detect zero-filled (fake) camera images
            has_real_image = (cam_img is not None and cam_img.any())

            if has_real_image and cam_name is not None:
                # Real camera image available — overlay trajectory ribbon
                assert cam_img is not None  # implied by has_real_image
                cam_cfg = cam_configs[cam_name]
                cam_vis = vis_utils.render_front_cam(
                    image=cam_img,
                    plan_traj_ego=plan_traj_ego,
                    cam_config=cam_cfg,
                    frame_id=self.frame,
                    model_name=model_name,
                    speed_kmh=speed_kmh,
                    collision=collision,
                    driving_command=driving_command,
                    sim_dt=self.config.sim_dt,
                    route_waypoints=route_waypoints,
                    target_waypoint=target_waypoint,
                    ego_position=pos,
                    ego_heading=heading,
                    scenario_data=full_scenario,
                )
            elif full_scenario is not None:
                # No real image — use perspective-projection fallback from scenario data
                cam_vis = vis_utils.render_front_cam_from_scenario(
                    scenario_data=full_scenario,
                    frame_id=self.frame,
                    ego_position=pos,
                    ego_heading=heading,
                    planned_traj_world=planned_world,
                    model_name=model_name,
                    speed_kmh=speed_kmh,
                    collision=collision,
                    driving_command=driving_command,
                    route_waypoints=route_waypoints,
                    target_waypoint=target_waypoint,
                )
        except Exception as exc:
            logger.warning("[vis] render_front_cam failed at frame %d: %s", self.frame, exc)

        if cam_vis is not None:
            # NAVSAFE_NO_OVERLAY writes the renderer's own frame instead of the
            # annotated one. The HUD, plan ribbon and projected lane lines are a
            # debugging aid; burned into a jpg they cannot be removed again, so
            # anything that consumes the pixels as pixels (video for a paper or
            # a site, a perception model, a human judging render quality) wants
            # them off. The top-down keeps its annotations either way — it is a
            # diagram, not a photograph.
            plain = _no_overlay() and has_real_image and cam_img is not None
            self._safe_imwrite(frame_dir / "cam_f0.jpg",
                               cam_img if plain else cam_vis)

        # ── Vis-only extra cameras (e.g. CAM_B0 rear view) ──────────────
        # Written unannotated: the HUD, plan ribbon and route dots describe
        # where the ego is *going*, and projecting them into a rearward view
        # draws them behind the camera — a wrong picture, not a missing one.
        for extra in getattr(self.config, "vis_extra_cameras", ()) or ():
            extra_img = images.get(extra)
            if extra_img is None or not extra_img.any():
                continue
            self._safe_imwrite(frame_dir / f"{extra.lower()}.jpg", extra_img)

        # ── Online display ──────────────────────────────────────────────
        if self.config.vis_online:
            try:
                vis_utils.show_online(bev_img, cam_vis)
            except Exception as exc:
                logger.debug("[vis] online display failed: %s", exc)

        # ── Candidate trajectory visualization (DiffusionDrive, PDM-Closed) ──
        parsed = self._cached_parsed_output
        if parsed is not None and "trajectory_coarse" in parsed:
            pred_pos = self._cached_prediction_position
            pred_heading = self._cached_prediction_heading
            if pred_pos is not None:
                candidates_ego = parsed["trajectory_coarse"]
                scores = parsed.get("coarse_scores")
                try:
                    cands_img = vis_utils.render_bev_candidates(
                        ego_position=pos,
                        ego_heading=heading,
                        candidates_ego=candidates_ego,
                        scores=scores,
                        planned_traj_world=planned_world,
                        prediction_position=pred_pos,
                        prediction_heading=pred_heading,
                        frame_id=self.frame,
                        model_name=model_name,
                        scenario_data=full_scenario,
                    )
                    self._safe_imwrite(frame_dir / "topdown_candidates.png", cands_img)
                except Exception as exc:
                    logger.warning("[vis] render_bev_candidates failed at frame %d: %s",
                                   self.frame, exc)

                # Same candidate set, projected onto the front camera. Needs the
                # real-image path (cam_cfg carries the intrinsics the fallback
                # perspective renderer doesn't produce).
                if cam_vis is not None and cam_cfg is not None:
                    try:
                        cands_cam = vis_utils.render_front_cam_candidates(
                            image=cam_vis,
                            candidates_ego=vis_utils.reanchor_candidates_to_ego(
                                candidates_ego,
                                prediction_position=pred_pos,
                                prediction_heading=pred_heading,
                                ego_position=pos,
                                ego_heading=heading,
                            ),
                            cam_config=cam_cfg,
                            scores=scores,
                            plan_traj_ego=plan_traj_ego,
                        )
                        self._safe_imwrite(frame_dir / "cam_f0_candidates.png", cands_cam)
                    except Exception as exc:
                        logger.warning("[vis] render_front_cam_candidates failed at frame %d: %s",
                                       self.frame, exc)

        # ── Optional ego sensors (LiDAR / IMU / contact) + cam extras ──
        try:
            self._save_extra_sensor_artifacts(frame_dir)
        except Exception as exc:
            logger.debug("[vis] _save_extra_sensor_artifacts failed: %s", exc)

    def _generate_visualizations(self) -> None:
        """Generate GIFs from saved per-frame images (called from finalize)."""
        output_dir = self.config.output_dir  # see evaluator._save_results
        frame_dir = output_dir / "frames"
        vis_dir = output_dir / "visualization"
        vis_dir.mkdir(parents=True, exist_ok=True)

        frame_ids = list(range(self.frame))

        # Headline GIFs (BridgeSim parity).  Each generate_gif silently
        # skips frames whose source image doesn't exist — so missing
        # extras (e.g. no scorer → no topdown_candidates.png) don't
        # break the rest.
        extra_cams = tuple(getattr(self.config, "vis_extra_cameras", ()) or ())
        gif_specs = [
            ("topdown.gif", "topdown.png"),
            ("cam_f0.gif", "cam_f0.jpg"),
            ("topdown_candidates.gif", "topdown_candidates.png"),
            ("cam_f0_candidates.gif", "cam_f0_candidates.png"),
        ] + [(f"{c.lower()}.gif", f"{c.lower()}.jpg") for c in extra_cams]

        # Auto-discover per-frame camera-extras (depth, semseg, ...)
        # from the first populated frame and add a GIF per type.
        if frame_ids:
            sample_dir = frame_dir / f"{frame_ids[0]:05d}"
            extra_pngs = sorted({
                p.name for p in sample_dir.glob("*.png")
                if p.name not in {"topdown.png", "topdown_candidates.png",
                                  "cam_f0_candidates.png"}
                and p.name != "cam_f0.png"
            })
            for fname in extra_pngs:
                stem = Path(fname).stem
                gif_specs.append((f"{stem}.gif", fname))

        emitted = 0
        for gif_name, src_name in gif_specs:
            try:
                vis_utils.generate_gif(frame_dir, frame_ids, vis_dir / gif_name, src_name)
                emitted += 1
            except Exception as exc:
                logger.warning("[vis] generate_gif(%s) failed: %s", gif_name, exc)

        try:
            vis_utils.generate_combined_gif(
                frame_dir, frame_ids, vis_dir / "combined.gif",
                left_filename="topdown.png", right_filename="cam_f0.jpg",
                extra_filenames=[f"{c.lower()}.jpg" for c in extra_cams],
            )
            emitted += 1
        except Exception as exc:
            logger.warning("[vis] generate_combined_gif failed: %s", exc)

        if self.config.vis_online:
            try:
                vis_utils.close_online()
            except Exception:
                pass

        logger.info("Visualizations saved → %s (%d gifs)", vis_dir, emitted)
