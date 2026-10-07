# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""py123d loader — Apache Arrow log → ``ScenarioDescription``.

Selected when ``EnvCfg.scenario_source == "py123d"``. Reads a converted
py123d arrow scene directly (no ScenarioNet pickle), absorbs it into the
py123d-native ScenarioState
(:class:`~navsafe.scenario.py123d_schema.Py123DScenarioData`), then projects
that state into the universal ScenarioNet ``ScenarioDescription`` dict the base
env / scorers / visualizers consume unchanged.

py123d itself is imported lazily inside :meth:`Py123DLoader.load` so this module
stays import-safe when py123d is absent (e.g. doc builds, schema-only tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
from navsafe.scenario.py123d_scenario_description import py123d_to_scenario_description
from navsafe.scenario.py123d_scenes import enumerate_scenes, scene_id


class Py123DLoader:
    """Load a py123d arrow scene into a runtime ``ScenarioDescription``.

    Implements :class:`~navsafe.env.loaders.ScenarioLoaderProtocol`.
    """

    def load(self, cfg: Any) -> Any:
        """Resolve a py123d arrow data root from ``cfg`` and load one scene.

        Scene selection (in order):

        1. ``cfg.scenario_id`` — exact scene uuid / log name, if set.
        2. ``cfg.start_scenario_index`` — positional index into the filtered
           (optionally shuffled) scene list. Defaults to 0.

        Args:
            cfg: An :class:`~navsafe.env.env_cfg.EnvCfg` (or any object exposing
                ``py123d_data_root`` / ``scenario_path`` / ``data_directory``).

        Returns:
            A ScenarioNet-format ``ScenarioDescription`` dict.
        """
        data_root = self._resolve_data_root(cfg)
        scenes = self._load_scenes(cfg, data_root)
        scene = self._select_scene(cfg, scenes)

        # A py123d scene here IS a whole nuPlan log (650-5320 iterations), while
        # an episode simulates a few hundred frames. Loading the log's every
        # frame of agent tracks made worker RSS scale with the log drawn rather
        # than the work done (measured 34-50 GB, the direct cause of the host
        # OOMs). ``py123d_frame_window`` narrows the load to the iterations the
        # episode can actually reach; absent, behaviour is unchanged.
        frame_window = getattr(cfg, "py123d_frame_window", None)
        scenario = scenario_from_py123d_scene(
            scene,
            Py123DAdapterConfig(
                load_state_payloads=True,
                load_custom_payloads=False,
                load_sensor_payloads=False,
                load_map_objects=True,
                data_root=str(data_root),
                require_map=bool(getattr(cfg, "py123d_require_map", False)),
                frame_window=tuple(frame_window) if frame_window else None,
            ),
        )
        sd = py123d_to_scenario_description(scenario)
        if getattr(cfg, "remove_agents", False):
            # Drop all non-ego tracks so removed objects vanish from the sim
            # STATE (BEV / collision / observation); the nurec_grpc render
            # mirrors agent_states, so the camera drops them too. This is the
            # state-level "remove all objects" (consistent, unlike a
            # render-only hide).
            sdc = sd.get("metadata", {}).get("sdc_id") or sd.get("sdc_id") or "ego"
            tracks = sd.get("tracks", {})
            if sdc in tracks:
                sd["tracks"] = {sdc: tracks[sdc]}
        # Real2sim work dirs are keyed by the source log name (the gs3d clip
        # id), which differs from the scene uuid — try both.
        log_name = ""
        try:
            log_name = str(scene.get_log_metadata().log_name or "")
        except Exception:
            pass
        self._attach_nurec_metadata(cfg, sd, [scene_id(scene), log_name])
        return sd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_data_root(cfg: Any) -> Path:
        """Pick the py123d arrow data root from cfg fields."""
        source = (
            getattr(cfg, "py123d_data_root", None)
            or getattr(cfg, "scenario_path", None)
            or getattr(cfg, "data_directory", None)
        )
        if not source:
            raise ValueError(
                "Py123DLoader requires cfg.py123d_data_root (a directory of "
                "converted py123d arrow logs) — or cfg.scenario_path / "
                "cfg.data_directory as a fallback — to be set."
            )
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"py123d data root does not exist: {source}")
        return path

    @staticmethod
    def _load_scenes(cfg: Any, data_root: Path) -> list:
        """Read the filtered (optionally shuffled) scene list via py123d."""
        # The episode index addresses the explicitly selected UUID list.
        uuids = getattr(cfg, "py123d_scene_uuids", None) or None
        scenes = enumerate_scenes(
            data_root,
            scene_uuids=uuids,
            max_scenes=getattr(cfg, "py123d_max_scenes", None),
            shuffle=bool(getattr(cfg, "shuffle_scenarios", False)),
            seed=int(getattr(cfg, "scenario_seed", 0)),
        )
        if not scenes:
            raise FileNotFoundError(f"No py123d arrow scenes found under: {data_root}")
        return scenes

    @staticmethod
    def _find_recon_origin_sidecar(
        work_dir: "str | None", cands: "list[str]",
    ) -> "Optional[list[float]]":
        """Reconstruction-frame provenance: the re-reference sidecar, if any.

        ``ncore_bridge`` re-references a store to the frame-0 rig xy and writes
        ``nurec_origin_offset.json`` (``offset_xy_utm``; absolute_utm = local +
        offset) next to the clip — "a re-referenced store is self-describing
        via the sidecar". Returns that offset (the reconstruction's own origin,
        ``O_recon``), or ``None`` when no sidecar exists (an absolute-frame
        reconstruction, or one produced outside ``ncore_bridge``).
        """
        if not work_dir:
            return None
        import json
        for cid in cands:
            base = Path(work_dir) / cid
            for p in (base / "clips" / cid / "nurec_origin_offset.json",
                      base / "nurec_origin_offset.json"):
                if p.is_file():
                    try:
                        off = json.loads(p.read_text()).get("offset_xy_utm")
                        if off and len(off) >= 2:
                            return [float(off[0]), float(off[1])]
                    except Exception:  # noqa: BLE001 - malformed sidecar
                        import logging
                        logging.getLogger(__name__).warning(
                            "Py123DLoader: unreadable origin sidecar %s", p)
        return None

    @staticmethod
    def _derive_grpc_anchor(
        sd: dict, meta: dict, clip_ids: "str | list[str]",
        work_dir: "str | None" = None,
    ) -> None:
        """Derive the nurec_grpc render anchor from the Arrow + recon sidecar.

        * ``nurec_grpc_scene_id`` — the reconstruction run id == the clip id.
        * ``gaussian_splat_origin_offset`` — the scenario→reconstruction
          translation. The renderer maps ``nre = (pos - pos0) - offset``, and
          the general transform is ``p_recon = p_scenario + O_s - O_r`` where

          - ``O_s`` = the scenario's own origin shift (``scenario_origin_xy``
            when ``coordinate == "local_frame0"``; zero for ``"world"``);
          - ``O_r`` = the reconstruction's origin shift (the ``ncore_bridge``
            re-reference sidecar via :meth:`_find_recon_origin_sidecar`; zero
            for an absolute-frame reconstruction).

          Solving for the renderer's convention gives
          ``offset = O_r - O_s - pos0`` (``pos0`` = ego frame-0 xy in the
          scenario's own frame). The four frame combinations all fall out of
          this one expression. When the scenario frame cannot be established
          (unknown ``coordinate``, or ``local_frame0`` without a recorded
          ``scenario_origin_xy``) NO offset is derived — the renderer's
          frame-anchor validation then fails closed under strict mode rather
          than rendering from a silently wrong pose.
        * frame timeline — from the Arrow\'s absolute per-frame timestamps
          (``metadata["ts"]``): ``real2sim_start_timestamp_us`` = the first
          stamp, ``nurec_frame_timestamps`` = the rest as relative seconds.

        NB the Arrow ego (imu) timeline runs ~half a frame off the r2s lidar
        timeline the reconstruction was baked on: negligible for a static
        background, a slight temporal offset for dynamic replay.
        """
        import logging
        import numpy as np
        logger = logging.getLogger(__name__)
        # clip_ids is [scene_uuid, log_name]; the reconstruction (and the gRPC
        # scene_id) is keyed by the log_name (== the gs3d clip id), so prefer it.
        cands = [clip_ids] if isinstance(clip_ids, str) else list(clip_ids or [])
        cands = [str(c) for c in cands if c]
        scene_id = next(reversed(cands), None)
        if scene_id:
            meta.setdefault("nurec_grpc_scene_id", scene_id)
        sdc = meta.get("sdc_id") or sd.get("sdc_id") or "ego"
        coord = str(meta.get("coordinate", "")).lower()
        origin_xy = meta.get("scenario_origin_xy")
        o_recon = Py123DLoader._find_recon_origin_sidecar(work_dir, cands)
        try:
            pos = np.asarray(sd["tracks"][sdc]["state"]["position"], dtype=np.float64)
            p0 = pos[0] if pos.ndim == 2 else pos
            if coord == "world":
                o_s = (0.0, 0.0)
            elif coord == "local_frame0" and origin_xy and len(origin_xy) >= 2:
                o_s = (float(origin_xy[0]), float(origin_xy[1]))
            else:
                o_s = None
                logger.warning(
                    "Py123DLoader: scenario frame unknown (coordinate=%r, "
                    "scenario_origin_xy=%r) — not deriving a nurec_grpc "
                    "origin_offset; strict render setup will fail closed.",
                    meta.get("coordinate"), origin_xy)
            if o_s is not None:
                o_r = o_recon if o_recon is not None else (0.0, 0.0)
                meta.setdefault("gaussian_splat_origin_offset", [
                    o_r[0] - o_s[0] - float(p0[0]),
                    o_r[1] - o_s[1] - float(p0[1]),
                    0.0,
                ])
                meta.setdefault(
                    "nurec_recon_frame",
                    "local" if o_recon is not None else "absolute_assumed")
        except Exception as exc:  # pragma: no cover - malformed scenario
            logger.warning("Py123DLoader: could not derive origin_offset (%s)", exc)
        ts = np.asarray(meta.get("ts", []), dtype=np.float64).reshape(-1)
        if ts.size:
            t0 = float(ts[0])
            meta.setdefault("real2sim_start_timestamp_us", int(t0))
            meta.setdefault("nurec_frame_timestamps", ((ts - t0) / 1e6).tolist())
        if not meta.get("nurec_grpc_scene_id"):
            logger.warning("Py123DLoader: nurec_grpc could not resolve a "
                           "scene_id from the clip ids; gRPC may render black.")

    @staticmethod
    def _attach_nurec_metadata(cfg: Any, sd: dict, clip_ids: "str | list[str]") -> None:
        """Derive gRPC scene identity, timeline, and reconstruction alignment."""
        Py123DLoader._derive_grpc_anchor(
            sd, sd.setdefault("metadata", {}), clip_ids,
            work_dir=getattr(cfg, "nurec_work_dir", None))

    @staticmethod
    def _select_scene(cfg: Any, scenes: list) -> Any:
        """Pick a single scene by id, else by index."""
        scenario_id: Optional[str] = getattr(cfg, "scenario_id", None)
        if scenario_id:
            for scene in scenes:
                if scene_id(scene) == str(scenario_id):
                    return scene
            raise FileNotFoundError(
                f"py123d scene '{scenario_id}' not found among {len(scenes)} scenes."
            )

        index = int(getattr(cfg, "start_scenario_index", 0) or 0)
        if not 0 <= index < len(scenes):
            # Fail loudly: a silent fall-back to scene 0 makes every
            # out-of-range index "work" while evaluating the wrong scene.
            raise IndexError(
                f"start_scenario_index={index} out of range for {len(scenes)} "
                f"py123d scene(s) (check py123d_max_scenes / the data root)")
        return scenes[index]


__all__ = ["Py123DLoader"]
