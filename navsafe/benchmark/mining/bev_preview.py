# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""BEV previews of reconstructed clips, for eyeballing what a leaf can use.

Mining predicates say *pass* or *fail*; they do not say *what the road looks
like*. This renders one top-down picture per clip — nuPlan map under the ego's
own logged (expert) trajectory — so a human can sort 200-odd reconstructed
scenes into "ramp / weave", "junction", "straight urban" by looking.

Input is the reconstruction tree, not a converted Arrow root: a clip lives as
four consecutive 5 s NCore segments ``<clip>s1..s4``, each with its own
``clips/<seg>/pai_<seg>.ncore4.zarr.itar`` (rig poses + cuboids) and its own
``output_5cam/<seg>/artifacts/last.usdz`` once 3DGS training finished. Only
clips whose four segments have all finished are drawn, so a preview always
corresponds to a full 20 s scene that can actually be simulated.

Three frames are in play and the conversion between them is the whole trick:

* **NCore local** — what the zarr stores. Each segment is re-referenced to
  *its own* frame-0 rig xy (``NCORE_REREF_FRAME0`` in ``ncore_bridge``), so
  s1..s4 do **not** share an origin and cannot be concatenated raw.
* **UTM** — ``local + offset_xy_utm`` from the segment's
  ``nurec_origin_offset.json`` sidecar. This is the common frame; stitching
  happens here.
* **nuPlan map** — the gpkg geometry is EPSG:4326 lon/lat, not UTM, so the map
  is reprojected into the clip's UTM zone before anything is drawn.

The map location is not recorded anywhere per clip, so it is *derived*: the
four nuPlan cities sit in different UTM zones and their map bounds do not
overlap, so exactly one (zone, city) pair puts the ego inside a map.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from navsafe.benchmark import config as cfg

logger = logging.getLogger(__name__)

# ── deployment layout ───────────────────────────────────────────────────────

DEFAULT_ROOT = cfg.CORPUS
DEFAULT_OUT = cfg.MINING_BEV
DEFAULT_MAPS = cfg.NUPLAN_MAPS_DEVKIT

SEGMENTS = ("s1", "s2", "s3", "s4")

# nuPlan ships one map per city; each city sits in one UTM zone, and ego poses
# are in that zone's metres while the gpkg is lon/lat.
CITY_UTM_EPSG: Dict[str, int] = {
    "us-ma-boston": 32619,
    "us-nv-las-vegas-strip": 32611,
    "us-pa-pittsburgh-hazelwood": 32617,
    "sg-one-north": 32648,
}

# ── map layers, drawn back to front ─────────────────────────────────────────
#
# Style is deliberately flat and low-contrast: the map is context, the ego
# trajectory is the subject. Every layer is optional — a city's gpkg missing
# one is a thinner picture, not a failure.

@dataclass(frozen=True)
class LayerStyle:
    layer: str
    kind: str  # "poly" | "line"
    facecolor: Optional[str] = None
    edgecolor: Optional[str] = None
    linewidth: float = 0.5
    linestyle: str = "-"
    alpha: float = 1.0
    zorder: float = 1.0


MAP_LAYERS: Tuple[LayerStyle, ...] = (
    LayerStyle("road_segments", "poly", "#EDEFF2", "none", zorder=1.0),
    LayerStyle("generic_drivable_areas", "poly", "#EDEFF2", "none", zorder=1.1),
    LayerStyle("carpark_areas", "poly", "#F2EFEA", "#E2DCD2", 0.4, zorder=1.2),
    LayerStyle("intersections", "poly", "#E6EAEF", "none", zorder=1.3),
    LayerStyle("lanes_polygons", "poly", "#E0E5EB", "#C6CDD6", 0.45, zorder=1.5),
    LayerStyle("walkways", "poly", "#F0EEEA", "#E0DCD5", 0.35, zorder=1.6),
    LayerStyle("crosswalks", "poly", "#D8DEE6", "#BFC7D1", 0.4, alpha=0.8, zorder=1.7),
    LayerStyle("baseline_paths", "line", None, "#9AA5B2", 0.7, (0, (5, 3)), 0.9, 2.0),
    LayerStyle("lane_connectors", "line", None, "#AEB8C4", 0.6, (0, (2, 3)), 0.85, 2.1),
)

# One colour per 5 s segment, so the picture also reads as "which quarter of
# the 20 s is where" — that is what tells a reviewer whether the interesting
# geometry is reachable inside a scored window.
SEGMENT_COLORS = ("#1B6FB8", "#178A6E", "#D98218", "#C2373C")

EGO_HALO = "#FFFFFF"
AGENT_COLOR = "#8E98A4"


# ── clip discovery ──────────────────────────────────────────────────────────


def segment_dir(root: Path, seg_id: str) -> Path:
    return cfg.clips_dir(seg_id, root)


def is_trained(root: Path, seg_id: str) -> bool:
    """3DGS training for one 5 s segment finished (its usdz was written)."""
    return cfg.recon_usdz(seg_id, root) is not None


def find_complete_clips(root: Path) -> List[str]:
    """Clip ids whose four 5 s segments have all finished training.

    Four stats per clip over a few thousand segment directories is a minute-plus
    on a network filesystem, so the result is worth caching (``--clips-file``)
    when re-rendering the same sweep.
    """
    trained = {p.split("/")[-1] for p in glob.glob(str(root / "*"))}
    complete: List[str] = []
    for name in sorted(trained):
        if not name.endswith(SEGMENTS):
            continue
        clip = name[:-2]
        if clip in complete:
            continue
        if all(is_trained(root, f"{clip}{s}") for s in SEGMENTS):
            complete.append(clip)
    return complete


# ── worker ──────────────────────────────────────────────────────────────────
#
# One MapCache per process: the layers are small, and sharing them across
# processes would cost more in pickling than the reads themselves.

_WORKER: Dict[str, object] = {}


def _worker_init(root: Path, maps_root: Path, out: Path, opts: dict) -> None:
    _WORKER.update(root=root, out=out, opts=opts, cache=MapCache(maps_root))


def _render_one(clip_id: str) -> Tuple[str, Optional[dict], Optional[str]]:
    """Render one clip. Returns ``(clip_id, summary, error)`` — never raises."""
    opts = dict(_WORKER["opts"])  # type: ignore[arg-type]
    root: Path = _WORKER["root"]  # type: ignore[assignment]
    out: Path = _WORKER["out"]  # type: ignore[assignment]
    cache: MapCache = _WORKER["cache"]  # type: ignore[assignment]
    try:
        clip = read_clip(root, clip_id, with_agents=opts["draw_agents"])
        city = infer_city(clip.xy, cache)
        row = render_clip(clip, city, cache, out / f"{clip_id}.png", **opts)
        return clip_id, row, None
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the sweep
        return clip_id, None, f"{type(exc).__name__}: {exc}"


# ── the log side: rig poses and cuboids out of NCore ────────────────────────


@dataclass
class Segment:
    """One 5 s piece of a clip, already lifted into UTM."""

    seg_id: str
    xy_utm: np.ndarray  # (T, 2)
    heading: np.ndarray  # (T,) rad, UTM frame
    t_us: np.ndarray  # (T,) uint64
    agents: Dict[str, np.ndarray] = field(default_factory=dict)  # track -> (K, 2) UTM

    @property
    def duration_s(self) -> float:
        return float(self.t_us[-1] - self.t_us[0]) / 1e6 if len(self.t_us) > 1 else 0.0


def read_segment(root: Path, seg_id: str, *, with_agents: bool = True) -> Segment:
    """Rig poses (and optionally cuboid tracks) of one segment, in UTM."""
    from ncore.data.v4 import (  # imported late: only this path needs ncore
        CuboidsComponent,
        PosesComponent,
        SequenceComponentGroupsReader,
    )

    clip_dir = segment_dir(root, seg_id)
    store = clip_dir / f"pai_{seg_id}.ncore4.zarr.itar"
    if not store.is_file():
        raise FileNotFoundError(f"{seg_id}: no NCore base shard at {store}")

    sidecar = clip_dir / "nurec_origin_offset.json"
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"{seg_id}: no nurec_origin_offset.json — the segment's local frame "
            f"cannot be lifted to UTM, so it cannot be stitched to its neighbours"
        )
    offset = np.asarray(
        json.loads(sidecar.read_text())["offset_xy_utm"], dtype=np.float64
    )

    reader = SequenceComponentGroupsReader([store])
    poses_reader = reader.open_component_readers(PosesComponent.Reader)["default"]
    rig: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for (source, target), (poses, timestamps) in poses_reader.get_dynamic_poses():
        if str(source) == "rig" and str(target) == "world":
            rig = (np.asarray(poses, np.float64), np.asarray(timestamps))
            break
    if rig is None:
        raise ValueError(f"{seg_id}: no dynamic rig->world pose in the store")
    poses, timestamps = rig
    order = np.argsort(timestamps)
    poses, timestamps = poses[order], timestamps[order]

    xy = poses[:, :2, 3] + offset
    heading = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])

    agents: Dict[str, List[Tuple[int, float, float]]] = {}
    if with_agents:
        try:
            cub_reader = reader.open_component_readers(CuboidsComponent.Reader)["default"]
            for obs in cub_reader.get_observations():
                cx, cy = obs.bbox3.centroid[0], obs.bbox3.centroid[1]
                agents.setdefault(str(obs.track_id), []).append(
                    (int(obs.timestamp_us), float(cx), float(cy))
                )
        except Exception as exc:  # noqa: BLE001 - agents are decoration, not the subject
            logger.debug("%s: no cuboids (%s)", seg_id, exc)

    tracks: Dict[str, np.ndarray] = {}
    for track_id, rows in agents.items():
        rows.sort()
        arr = np.asarray([[x, y] for _, x, y in rows], np.float64) + offset
        tracks[track_id] = arr

    return Segment(seg_id=seg_id, xy_utm=xy, heading=heading, t_us=timestamps, agents=tracks)


@dataclass
class Clip:
    """A 20 s scene: four stitched segments in one UTM frame."""

    clip_id: str
    segments: List[Segment]

    @property
    def xy(self) -> np.ndarray:
        return np.concatenate([s.xy_utm for s in self.segments], axis=0)

    @property
    def t_us(self) -> np.ndarray:
        return np.concatenate([s.t_us for s in self.segments], axis=0)

    @property
    def duration_s(self) -> float:
        t = self.t_us
        return float(t[-1] - t[0]) / 1e6

    @property
    def path_length_m(self) -> float:
        d = np.diff(self.xy, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    def speeds(self) -> np.ndarray:
        """Per-step speed in m/s, computed across the stitched trajectory."""
        t = self.t_us.astype(np.float64) / 1e6
        d = np.diff(self.xy, axis=0)
        dt = np.diff(t)
        good = dt > 1e-6
        return np.hypot(d[good, 0], d[good, 1]) / dt[good]

    def merged_agents(self) -> Dict[str, np.ndarray]:
        """Agent tracks concatenated across segments, in UTM."""
        merged: Dict[str, List[np.ndarray]] = {}
        for seg in self.segments:
            for track_id, arr in seg.agents.items():
                merged.setdefault(track_id, []).append(arr)
        return {k: np.concatenate(v, axis=0) for k, v in merged.items()}


def read_clip(root: Path, clip_id: str, *, with_agents: bool = True) -> Clip:
    segments = [read_segment(root, f"{clip_id}{s}", with_agents=with_agents) for s in SEGMENTS]
    return Clip(clip_id=clip_id, segments=segments)


# ── the map side ────────────────────────────────────────────────────────────


class MapCache:
    """nuPlan gpkg layers, read once per city and reprojected into its UTM zone.

    The layers are small (a few thousand features for the largest city), so
    whole-layer reads beat per-clip bbox queries once more than a handful of
    clips share a city.
    """

    def __init__(self, maps_root: Path) -> None:
        self.maps_root = Path(maps_root)
        self._gpkg: Dict[str, Path] = {}
        self._layers: Dict[Tuple[str, str], object] = {}
        self._bounds_lonlat: Dict[str, Tuple[float, float, float, float]] = {}

    def gpkg(self, city: str) -> Optional[Path]:
        if city not in self._gpkg:
            hits = sorted(self.maps_root.glob(f"{city}/*/map.gpkg"))
            if not hits:
                logger.warning("no map.gpkg for %s under %s", city, self.maps_root)
                return None
            self._gpkg[city] = hits[-1]  # highest map version
        return self._gpkg.get(city)

    def bounds_lonlat(self, city: str) -> Optional[Tuple[float, float, float, float]]:
        """The city's lane extent in lon/lat — the basis of city inference."""
        if city not in self._bounds_lonlat:
            import pyogrio

            path = self.gpkg(city)
            if path is None:
                return None
            info = pyogrio.read_info(str(path), layer="lanes_polygons")
            self._bounds_lonlat[city] = tuple(float(v) for v in info["total_bounds"])
        return self._bounds_lonlat.get(city)

    def layer(self, city: str, layer: str):
        """One layer as a GeoDataFrame in the city's UTM metres, or None."""
        key = (city, layer)
        if key not in self._layers:
            import pyogrio

            path = self.gpkg(city)
            if path is None:
                self._layers[key] = None
            else:
                try:
                    gdf = pyogrio.read_dataframe(str(path), layer=layer)
                    self._layers[key] = gdf.to_crs(epsg=CITY_UTM_EPSG[city])
                except Exception as exc:  # noqa: BLE001 - a missing layer is fine
                    logger.debug("%s: layer %s unavailable (%s)", city, layer, exc)
                    self._layers[key] = None
        return self._layers[key]


def infer_city(xy_utm: np.ndarray, cache: MapCache) -> Optional[str]:
    """Which nuPlan city these UTM metres belong to.

    The four cities are in four different UTM zones, so a point read in the
    wrong zone lands thousands of kilometres outside that city's map. Exactly
    one (zone, city) pair puts the ego inside a map — that is the answer.
    """
    from pyproj import Transformer

    point = xy_utm[len(xy_utm) // 2]
    for city, epsg in CITY_UTM_EPSG.items():
        bounds = cache.bounds_lonlat(city)
        if bounds is None:
            continue
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
            float(point[0]), float(point[1])
        )
        west, south, east, north = bounds
        pad = 0.02  # ~2 km: a clip may run just off the annotated lane extent
        if west - pad <= lon <= east + pad and south - pad <= lat <= north + pad:
            return city
    return None


# ── the picture ─────────────────────────────────────────────────────────────


def _draw_map(ax, cache: MapCache, city: str, bbox: Tuple[float, float, float, float]) -> int:
    """Clip every styled layer to ``bbox`` and draw it. Returns features drawn."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import box as shapely_box

    west, south, east, north = bbox
    window = shapely_box(west, south, east, north)
    drawn = 0

    for style in MAP_LAYERS:
        gdf = cache.layer(city, style.layer)
        if gdf is None or len(gdf) == 0:
            continue
        subset = gdf.iloc[list(gdf.sindex.query(window, predicate="intersects"))]
        if len(subset) == 0:
            continue

        if style.kind == "poly":
            patches = []
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                for part in getattr(geom, "geoms", [geom]):
                    if part.geom_type != "Polygon":
                        continue
                    patches.append(MplPolygon(np.asarray(part.exterior.coords), closed=True))
            if not patches:
                continue
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor=style.facecolor or "none",
                    edgecolor=style.edgecolor or "none",
                    linewidth=style.linewidth,
                    alpha=style.alpha,
                    zorder=style.zorder,
                )
            )
            drawn += len(patches)
        else:
            lines = []
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                for part in getattr(geom, "geoms", [geom]):
                    if part.geom_type != "LineString":
                        continue
                    lines.append(np.asarray(part.coords)[:, :2])
            if not lines:
                continue
            ax.add_collection(
                LineCollection(
                    lines,
                    colors=style.edgecolor,
                    linewidths=style.linewidth,
                    linestyles=style.linestyle,
                    alpha=style.alpha,
                    zorder=style.zorder,
                )
            )
            drawn += len(lines)
    return drawn


def _scale_bar(ax, origin_x: float, origin_y: float, span: float) -> None:
    """A metric scale bar sized to a round fraction of the view."""
    for candidate in (100.0, 50.0, 25.0, 20.0, 10.0):
        if candidate <= span * 0.28:
            length = candidate
            break
    else:
        length = 10.0
    x0 = origin_x + span * 0.05
    y0 = origin_y + span * 0.05
    ax.plot([x0, x0 + length], [y0, y0], color="#333A44", lw=2.0, zorder=9,
            solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + span * 0.012, f"{length:.0f} m", ha="center",
            va="bottom", fontsize=8, color="#333A44", zorder=9)


def render_clip(
    clip: Clip,
    city: Optional[str],
    cache: MapCache,
    out_path: Path,
    *,
    margin_m: float = 45.0,
    min_span_m: float = 150.0,
    draw_agents: bool = True,
    dpi: int = 150,
) -> dict:
    """Draw one clip's BEV and write it to ``out_path``. Returns its summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xy = clip.xy
    # Square window around the trajectory: an aspect-distorted BEV makes a
    # merge angle unreadable, which is the one thing these are drawn for.
    cx, cy = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
    extent = float(
        max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min())
    )
    span = max(extent + 2 * margin_m, min_span_m)
    half = span / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)

    fig, ax = plt.subplots(figsize=(9.0, 9.6))
    ax.set_facecolor("#FBFCFD")

    features = _draw_map(ax, cache, city, bbox) if city else 0

    if draw_agents:
        agent_lines = 0
        for track in clip.merged_agents().values():
            if len(track) < 2:
                continue
            if float(np.linalg.norm(track[-1] - track[0])) < 2.0:
                continue  # parked: a static dot adds clutter, not information
            ax.plot(track[:, 0], track[:, 1], color=AGENT_COLOR, lw=0.9, alpha=0.45,
                    zorder=3, solid_capstyle="round")
            agent_lines += 1
    else:
        agent_lines = 0

    # Ego expert trajectory: white halo first so it stays readable over lanes.
    for seg in clip.segments:
        ax.plot(seg.xy_utm[:, 0], seg.xy_utm[:, 1], color=EGO_HALO, lw=5.4,
                solid_capstyle="round", zorder=4)
    for seg, color in zip(clip.segments, SEGMENT_COLORS):
        ax.plot(seg.xy_utm[:, 0], seg.xy_utm[:, 1], color=color, lw=2.8,
                solid_capstyle="round", zorder=5)

    # Heading arrows roughly every 2 s, so direction of travel is unambiguous.
    for seg, color in zip(clip.segments, SEGMENT_COLORS):
        step = max(1, len(seg.xy_utm) // 3)
        for i in range(step // 2, len(seg.xy_utm), step):
            x, y = seg.xy_utm[i]
            theta = float(seg.heading[i])
            ax.annotate(
                "", xy=(x + 3.2 * math.cos(theta), y + 3.2 * math.sin(theta)),
                xytext=(x, y), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                shrinkA=0, shrinkB=0),
            )

    ax.scatter([xy[0, 0]], [xy[0, 1]], s=95, marker="o", facecolor="#FFFFFF",
               edgecolor=SEGMENT_COLORS[0], linewidth=2.2, zorder=7)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=95, marker="s", facecolor="#FFFFFF",
               edgecolor=SEGMENT_COLORS[-1], linewidth=2.2, zorder=7)
    ax.annotate("start", xy=(xy[0, 0], xy[0, 1]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="#333A44", zorder=7)
    ax.annotate("end", xy=(xy[-1, 0], xy[-1, 1]), xytext=(6, 6),
                textcoords="offset points", fontsize=8, color="#333A44", zorder=7)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D5DAE1")

    _scale_bar(ax, bbox[0], bbox[1], span)
    north_x = bbox[2] - span * 0.05
    ax.annotate("", xy=(north_x, bbox[3] - span * 0.035),
                xytext=(north_x, bbox[3] - span * 0.085),
                arrowprops=dict(arrowstyle="-|>", color="#333A44", lw=1.4), zorder=9)
    ax.annotate("N", xy=(north_x, bbox[3] - span * 0.115), ha="center", va="center",
                fontsize=9, color="#333A44", zorder=9)

    speeds = clip.speeds()
    v_mean = float(np.mean(speeds)) if len(speeds) else 0.0
    v_max = float(np.max(speeds)) if len(speeds) else 0.0
    length_m = clip.path_length_m

    ax.set_title(
        f"{clip.clip_id}   ·   {city or 'unknown map'}",
        fontsize=13, fontweight="bold", color="#1B2129", pad=14, loc="left",
    )
    ax.annotate(
        f"{clip.duration_s:.1f} s  ·  {length_m:.0f} m travelled  ·  "
        f"mean {v_mean:.1f} m/s ({v_mean * 3.6:.0f} km/h)  ·  peak {v_max:.1f} m/s"
        + (f"  ·  {agent_lines} moving agents" if draw_agents else ""),
        xy=(0, 1.005), xycoords="axes fraction", fontsize=9, color="#5A6472",
    )

    handles = [
        Line2D([], [], color=color, lw=2.8, label=f"ego expert · s{i + 1} "
               f"({i * 5}–{(i + 1) * 5} s)")
        for i, color in enumerate(SEGMENT_COLORS)
    ]
    if draw_agents and agent_lines:
        handles.append(Line2D([], [], color=AGENT_COLOR, lw=1.2, alpha=0.6,
                              label="other agents (logged)"))
    if features:
        handles.append(Line2D([], [], color="#9AA5B2", lw=0.9, linestyle=(0, (5, 3)),
                              label="lane centrelines (nuPlan map)"))
    # Below the axes, not inside it: a legend box in a corner hides exactly the
    # kind of side road these previews exist to reveal.
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.012),
              ncol=3, fontsize=8, frameon=False, borderpad=0.4,
              columnspacing=1.6, handlelength=2.4)

    if not city:
        ax.annotate(
            "no nuPlan map matched these UTM coordinates — trajectory only",
            xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=9,
            color="#C2373C",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "clip": clip.clip_id,
        "map": city or "",
        "png": out_path.name,
        "duration_s": round(clip.duration_s, 2),
        "path_length_m": round(length_m, 1),
        "speed_mean_mps": round(v_mean, 2),
        "speed_max_mps": round(v_max, 2),
        "net_displacement_m": round(float(np.linalg.norm(xy[-1] - xy[0])), 1),
        "utm_start": [round(float(xy[0, 0]), 1), round(float(xy[0, 1]), 1)],
        "n_frames": int(len(xy)),
        "moving_agents": int(agent_lines),
        "map_features_drawn": int(features),
    }


# ── driver ──────────────────────────────────────────────────────────────────


def render_previews(
    clips: List[str],
    out: Path,
    *,
    root: Path = DEFAULT_ROOT,
    maps_root: Path = DEFAULT_MAPS,
    workers: int = 1,
    draw_agents: bool = True,
    dpi: int = 150,
    min_span_m: float = 150.0,
    overwrite: bool = False,
    progress: bool = True,
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """Render one BEV PNG per clip. Returns ``(rows, failures)``.

    This is the body `navsafe mine --bev-dir` calls, so a sweep can hand back
    pictures of everything it found rather than a list of tokens to go and look
    up. `main()` is the same thing behind an argparse.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    if not overwrite:
        pending = [c for c in clips if not (out / f"{c}.png").exists()]
        skipped = len(clips) - len(pending)
        if skipped and progress:
            print(f"{skipped} clip(s) already rendered, skipping")
        clips = pending

    rows: List[dict] = []
    failures: List[Tuple[str, str]] = []
    opts = {"draw_agents": draw_agents, "dpi": dpi, "min_span_m": min_span_m}
    if progress:
        print(f"{len(clips)} clip(s) to render -> {out}")

    def _record(n: int, clip_id: str, row: Optional[dict], error: Optional[str]) -> None:
        if error is not None:
            failures.append((clip_id, error))
            if progress:
                print(f"[{n}/{len(clips)}] {clip_id}  FAILED  {error}", flush=True)
            return
        assert row is not None
        rows.append(row)
        if progress:
            print(f"[{n}/{len(clips)}] {clip_id}  {row['map'] or 'NO MAP':<26} "
                  f"{row['path_length_m']:6.0f} m  "
                  f"{row['speed_mean_mps'] * 3.6:5.1f} km/h", flush=True)

    if workers > 1 and clips:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(root, maps_root, out, opts)) as pool:
            for n, (clip_id, row, error) in enumerate(
                pool.map(_render_one, clips, chunksize=1), 1
            ):
                _record(n, clip_id, row, error)
    elif clips:
        _worker_init(root, maps_root, out, opts)
        for n, clip_id in enumerate(clips, 1):
            _record(n, *_render_one(clip_id))

    if rows:
        existing = index_path.read_text().splitlines() if index_path.exists() else []
        keep = [line for line in existing
                if line.strip() and json.loads(line)["clip"] not in
                {r["clip"] for r in rows}]
        index_path.write_text("\n".join(keep + [json.dumps(r) for r in rows]) + "\n")
        if progress:
            print(f"\nwrote {len(rows)} PNG(s); index at {index_path}")
    return rows, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BEV preview per fully-reconstructed 20 s clip, with the ego "
                    "expert trajectory over the nuPlan map.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="reconstruction tree of 5 s segments")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="directory the PNGs and the index are written to")
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS,
                        help="nuPlan maps root (<city>/<version>/map.gpkg)")
    parser.add_argument("--clip", action="append", default=None,
                        help="render only this clip id (repeatable)")
    parser.add_argument("--clips-file", type=Path, default=None,
                        help="read clip ids from this file (one per line) instead "
                             "of stat-ing the whole tree; --write-clips-file makes one")
    parser.add_argument("--write-clips-file", type=Path, default=None,
                        help="save the discovered clip ids here for reuse")
    parser.add_argument("--workers", type=int, default=1,
                        help="render this many clips in parallel")
    parser.add_argument("--limit", type=int, default=0, help="stop after N clips")
    parser.add_argument("--no-agents", action="store_true",
                        help="draw the ego trajectory only, no logged agents")
    parser.add_argument("--overwrite", action="store_true",
                        help="redraw clips whose PNG already exists")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--min-span-m", type=float, default=150.0,
                        help="smallest square window drawn, in metres")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.clip:
        clips = list(args.clip)
    elif args.clips_file:
        clips = [line.strip() for line in args.clips_file.read_text().splitlines()
                 if line.strip()]
    else:
        clips = find_complete_clips(args.root)
        if args.write_clips_file:
            args.write_clips_file.parent.mkdir(parents=True, exist_ok=True)
            args.write_clips_file.write_text("\n".join(clips) + "\n")
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print(f"no clip under {args.root} has all four segments trained")
        return 1

    rows, failures = render_previews(
        clips, args.out, root=args.root, maps_root=args.maps_root,
        workers=args.workers, draw_agents=not args.no_agents, dpi=args.dpi,
        min_span_m=args.min_span_m, overwrite=args.overwrite,
    )
    if failures:
        print(f"\n{len(failures)} clip(s) failed:")
        for clip_id, reason in failures:
            print(f"  {clip_id}  {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
