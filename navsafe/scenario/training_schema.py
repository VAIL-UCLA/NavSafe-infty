"""NexusSim common scenario state for training and simulation.

These dataclasses are the stable semantic boundary between source-specific
readers such as py123d and task-specific consumers such as BC datasets, RL envs,
state observations, or future sensor observations.  They intentionally do not
expose py123d classes on the consumer path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class NexusEgoState:
    timestamp_us: Optional[int]
    x: float
    y: float
    z: float
    heading: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    yaw_rate: float
    steering_angle: float
    length: float
    width: float
    height: float

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.x,
                self.y,
                self.z,
                self.heading,
                self.vx,
                self.vy,
                self.vz,
                self.speed,
                self.ax,
                self.ay,
                self.az,
                self.yaw_rate,
                self.steering_angle,
                self.length,
                self.width,
                self.height,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class NexusActorState:
    track_token: str
    label: str
    x: float
    y: float
    z: float
    heading: float
    vx: float
    vy: float
    vz: float
    length: float
    width: float
    height: float
    num_lidar_points: Optional[int] = None
    # Finite supplied planar values, not a guarantee of measurement accuracy.
    velocity_valid: bool = False

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass(frozen=True)
class NexusTrafficLightState:
    lane_id: str
    status: str


@dataclass(frozen=True)
class NexusLaneState:
    lane_id: str
    lane_type: str
    lane_group_id: Optional[str]
    left_lane_id: Optional[str]
    right_lane_id: Optional[str]
    predecessor_ids: tuple[str, ...]
    successor_ids: tuple[str, ...]
    speed_limit_mps: Optional[float]
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    polygon: np.ndarray


@dataclass(frozen=True)
class NexusLaneGroupState:
    lane_group_id: str
    lane_ids: tuple[str, ...]
    intersection_id: Optional[str]
    predecessor_ids: tuple[str, ...]
    successor_ids: tuple[str, ...]
    polygon: np.ndarray


@dataclass(frozen=True)
class NexusIntersectionState:
    intersection_id: str
    intersection_type: str
    lane_group_ids: tuple[str, ...]
    polygon: np.ndarray


@dataclass(frozen=True)
class NexusSurfaceState:
    object_id: str
    layer: str
    polygon: np.ndarray
    semantic_type: Optional[str] = None
    lane_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NexusLineState:
    object_id: str
    layer: str
    polyline: np.ndarray
    semantic_type: Optional[str] = None


@dataclass(frozen=True)
class NexusSensorReference:
    key: str
    modality_type: str
    modality_id: Optional[str]
    timestamp_us: Optional[int] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NexusRouteState:
    """Canonical route / mission view for one frame.

    Source-specific route payloads, for example py123d nuPlan
    ``custom.scenario.route_roadblock_ids``, stay in ``custom_modalities``.
    This class exposes only the normalized route semantics that NexusSim
    consumers should read.
    """

    roadblock_ids: tuple[str, ...] = ()
    lane_ids: tuple[str, ...] = ()
    lane_group_ids: tuple[str, ...] = ()
    source_key: Optional[str] = None
    support: str = "unsupported"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.support == "supported" and bool(
            self.roadblock_ids or self.lane_ids or self.lane_group_ids
        )


@dataclass
class NexusMapState:
    location: Optional[str]
    lanes: dict[str, NexusLaneState] = field(default_factory=dict)
    lane_groups: dict[str, NexusLaneGroupState] = field(default_factory=dict)
    intersections: dict[str, NexusIntersectionState] = field(default_factory=dict)
    crosswalks: dict[str, NexusSurfaceState] = field(default_factory=dict)
    walkways: dict[str, NexusSurfaceState] = field(default_factory=dict)
    carparks: dict[str, NexusSurfaceState] = field(default_factory=dict)
    generic_drivable: dict[str, NexusSurfaceState] = field(default_factory=dict)
    stop_zones: dict[str, NexusSurfaceState] = field(default_factory=dict)
    speed_bumps: dict[str, NexusSurfaceState] = field(default_factory=dict)
    road_edges: dict[str, NexusLineState] = field(default_factory=dict)
    road_lines: dict[str, NexusLineState] = field(default_factory=dict)
    custom_layers: dict[str, Any] = field(default_factory=dict)
    lane_point_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), dtype=np.float32))

    @property
    def available_layers(self) -> list[str]:
        layers: list[str] = []
        for name, values in (
            ("lane", self.lanes),
            ("lane_group", self.lane_groups),
            ("intersection", self.intersections),
            ("crosswalk", self.crosswalks),
            ("walkway", self.walkways),
            ("carpark", self.carparks),
            ("generic_drivable", self.generic_drivable),
            ("stop_zone", self.stop_zones),
            ("speed_bump", self.speed_bumps),
            ("road_edge", self.road_edges),
            ("road_line", self.road_lines),
        ):
            if values:
                layers.append(name)
        return layers

    def nearest_lane_points(self, x: float, y: float, k: int, radius_m: Optional[float] = None) -> np.ndarray:
        if self.lane_point_features.size == 0 or k <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        points = self.lane_point_features
        d2 = np.square(points[:, 0] - x) + np.square(points[:, 1] - y)
        if radius_m is not None:
            mask = d2 <= radius_m * radius_m
            points = points[mask]
            d2 = d2[mask]
            if points.size == 0:
                return np.zeros((0, 5), dtype=np.float32)
        order = np.argsort(d2)[:k]
        return points[order].astype(np.float32, copy=False)


@dataclass(frozen=True)
class NexusFrameState:
    scenario_id: str
    iteration: int
    timestamp_us: Optional[int]
    ego: NexusEgoState
    actors: tuple[NexusActorState, ...]
    traffic_lights: tuple[NexusTrafficLightState, ...]
    route: NexusRouteState = field(default_factory=NexusRouteState)
    sensor_references: dict[str, NexusSensorReference] = field(default_factory=dict)
    custom_modalities: dict[str, Any] = field(default_factory=dict)

    @property
    def route_supported(self) -> bool:
        return self.route.supported


@dataclass(frozen=True)
class ApproxActionLabel:
    accel_mps2: float
    steering_angle_rad: float
    source: str = "finite_difference_bicycle_model"
    approximate: bool = True


@dataclass
class NexusScenarioLog:
    scenario_id: str
    dataset: Optional[str]
    split: Optional[str]
    location: Optional[str]
    log_name: Optional[str]
    timestamps_us: list[int]
    map_state: NexusMapState
    frames: list[NexusFrameState]
    sensor_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def number_of_iterations(self) -> int:
        return len(self.frames)

    @property
    def route_supported(self) -> bool:
        return any(frame.route_supported for frame in self.frames)

    @property
    def route_empty_frame_count(self) -> int:
        return sum(1 for frame in self.frames if not frame.route_supported)

    def frame(self, iteration: int) -> NexusFrameState:
        if iteration < 0 or iteration >= len(self.frames):
            raise IndexError(f"iteration {iteration} outside scenario range [0, {len(self.frames)})")
        return self.frames[iteration]

    def future_ego_trajectory(self, iteration: int, horizon: int, relative: bool = True) -> np.ndarray:
        if horizon <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        start = self.frame(iteration).ego
        values: list[list[float]] = []
        for idx in range(iteration + 1, min(iteration + 1 + horizon, len(self.frames))):
            ego = self.frames[idx].ego
            if relative:
                values.append([ego.x - start.x, ego.y - start.y, _wrap_angle(ego.heading - start.heading)])
            else:
                values.append([ego.x, ego.y, ego.heading])
        return np.asarray(values, dtype=np.float32)

    def approximate_action(self, iteration: int, wheelbase_m: float = 2.9) -> Optional[ApproxActionLabel]:
        if iteration < 0 or iteration + 1 >= len(self.frames):
            return None
        curr = self.frames[iteration].ego
        nxt = self.frames[iteration + 1].ego
        if curr.timestamp_us is None or nxt.timestamp_us is None:
            return None
        dt = (nxt.timestamp_us - curr.timestamp_us) / 1_000_000.0
        if dt <= 0:
            return None
        accel = (nxt.speed - curr.speed) / dt
        yaw_rate = _wrap_angle(nxt.heading - curr.heading) / dt
        speed = max(curr.speed, 1e-3)
        steer = float(np.arctan(yaw_rate * wheelbase_m / speed))
        return ApproxActionLabel(accel_mps2=float(accel), steering_angle_rad=steer)

    def validate(self) -> None:
        if len(self.timestamps_us) != len(self.frames):
            raise ValueError("timestamps_us length must match number of frames")
        if not self.map_state.lanes:
            raise ValueError("NexusScenarioLog requires lane map objects")
        for expected_iteration, frame in enumerate(self.frames):
            if frame.iteration != expected_iteration:
                raise ValueError(f"frame iteration mismatch: expected {expected_iteration}, got {frame.iteration}")
            if frame.ego.length <= 0 or frame.ego.width <= 0:
                raise ValueError(f"invalid ego dimensions at frame {frame.iteration}")



def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


# Backward-compatible aliases.  New code should use the Nexus* names above.
TrainingEgoState = NexusEgoState
TrainingActorState = NexusActorState
TrainingTrafficLightState = NexusTrafficLightState
TrainingLaneState = NexusLaneState
TrainingLaneGroupState = NexusLaneGroupState
TrainingIntersectionState = NexusIntersectionState
TrainingSurfaceState = NexusSurfaceState
TrainingLineState = NexusLineState
TrainingMapState = NexusMapState
TrainingRouteState = NexusRouteState
TrainingFrameState = NexusFrameState
TrainingScenarioLog = NexusScenarioLog
