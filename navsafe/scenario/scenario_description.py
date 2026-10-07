"""
ScenarioDescription — ported from MetaDrive's metadrive.scenario.scenario_description.

A unified dict-based data format for replaying driving scenarios.
No MetaDrive package dependency.
"""

import logging
import math
import os
from collections import defaultdict
from typing import Optional

import numpy as np

from navsafe.scenario.type import MetaDriveType

_logger = logging.getLogger(__name__)


def _norm(x, y):
    return math.sqrt(x * x + y * y)


class ScenarioDescription(dict):
    """
    Scenario description schema. Stores the string keys of the scenario data dict.
    """
    TRACKS = "tracks"
    VERSION = "version"
    ID = "id"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    LENGTH = "length"
    METADATA = "metadata"
    FIRST_LEVEL_KEYS = {TRACKS, VERSION, ID, DYNAMIC_MAP_STATES, MAP_FEATURES, LENGTH, METADATA}

    # lane keys
    POLYLINE = "polyline"
    POLYGON = "polygon"
    LEFT_BOUNDARIES = "left_boundaries"
    RIGHT_BOUNDARIES = "right_boundaries"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry_lanes"
    EXIT = "exit_lanes"

    # object
    TYPE = "type"
    STATE = "state"
    OBJECT_ID = "object_id"
    STATE_DICT_KEYS = {TYPE, STATE, METADATA}
    ORIGINAL_ID_TO_OBJ_ID = "original_id_to_obj_id"
    OBJ_ID_TO_ORIGINAL_ID = "obj_id_to_original_id"
    TRAFFIC_LIGHT_POSITION = "stop_point"
    TRAFFIC_LIGHT_STATUS = "object_state"
    TRAFFIC_LIGHT_LANE = "lane"
    POSITION = "position"
    HEADING = "heading"

    METADRIVE_PROCESSED = "metadrive_processed"
    TIMESTEP = "ts"
    COORDINATE = "coordinate"
    SDC_ID = "sdc_id"
    METADATA_KEYS = {METADRIVE_PROCESSED, COORDINATE, TIMESTEP}
    OLD_ORIGIN_IN_CURRENT_COORDINATE = "old_origin_in_current_coordinate"

    ALLOW_TYPES = (int, float, str, np.ndarray, dict, list, tuple, type(None), set)

    class SUMMARY:
        OBJECT_SUMMARY = "object_summary"
        NUMBER_SUMMARY = "number_summary"

        TYPE = "type"
        OBJECT_ID = "object_id"
        TRACK_LENGTH = "track_length"
        MOVING_DIST = "moving_distance"
        VALID_LENGTH = "valid_length"
        CONTINUOUS_VALID_LENGTH = "continuous_valid_length"

        OBJECT_TYPES = "object_types"
        NUM_OBJECTS = "num_objects"
        NUM_MOVING_OBJECTS = "num_moving_objects"
        NUM_OBJECTS_EACH_TYPE = "num_objects_each_type"
        NUM_MOVING_OBJECTS_EACH_TYPE = "num_moving_objects_each_type"

        NUM_TRAFFIC_LIGHTS = "num_traffic_lights"
        NUM_TRAFFIC_LIGHT_TYPES = "num_traffic_light_types"
        NUM_TRAFFIC_LIGHTS_EACH_STEP = "num_traffic_light_each_step"

        NUM_MAP_FEATURES = "num_map_features"
        MAP_HEIGHT_DIFF = "map_height_diff"

    class DATASET:
        SUMMARY_FILE = "dataset_summary.pkl"
        MAPPING_FILE = "dataset_mapping.pkl"

    def to_dict(self):
        return dict(self)

    def get_sdc_track(self):
        assert self.SDC_ID in self[self.METADATA]
        sdc_id = str(self[self.METADATA][self.SDC_ID])
        return self[self.TRACKS][sdc_id]

    @staticmethod
    def get_object_summary(object_dict, object_id: str):
        object_type = object_dict["type"]
        state_dict = object_dict["state"]
        track = state_dict["position"]
        valid_track = track[np.where(state_dict["valid"].astype(int))][..., :2]
        distance = float(
            sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1))
        )
        valid_length = int(sum(state_dict["valid"]))
        continuous_valid_length = 0
        for v in state_dict["valid"]:
            if v:
                continuous_valid_length += 1
            if continuous_valid_length > 0 and not v:
                break
        return {
            ScenarioDescription.SUMMARY.TYPE: object_type,
            ScenarioDescription.SUMMARY.OBJECT_ID: str(object_id),
            ScenarioDescription.SUMMARY.TRACK_LENGTH: int(len(track)),
            ScenarioDescription.SUMMARY.MOVING_DIST: float(distance),
            ScenarioDescription.SUMMARY.VALID_LENGTH: int(valid_length),
            ScenarioDescription.SUMMARY.CONTINUOUS_VALID_LENGTH: int(continuous_valid_length),
        }

    @staticmethod
    def get_export_file_name(dataset: str, dataset_version: str, scenario_name: str):
        return "sd_{}_{}_{}.pkl".format(dataset, dataset_version, scenario_name)

    @staticmethod
    def is_scenario_file(file_name: str):
        file_name = os.path.basename(file_name)
        if not file_name.endswith(".pkl"):
            return False
        file_name = file_name.replace(".pkl", "")
        return os.path.basename(file_name)[:3] == "sd_" or all(char.isdigit() for char in file_name)

    @staticmethod
    def get_number_summary(scenario):
        SD = ScenarioDescription
        number_summary_dict = {}
        number_summary_dict[SD.SUMMARY.NUM_OBJECTS] = len(scenario[SD.TRACKS])
        number_summary_dict[SD.SUMMARY.OBJECT_TYPES] = set(v["type"] for v in scenario[SD.TRACKS].values())
        object_types_counter = defaultdict(int)
        for v in scenario[SD.TRACKS].values():
            object_types_counter[v["type"]] += 1
        number_summary_dict[SD.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

        object_summaries = {}
        for track_id, track in scenario[SD.TRACKS].items():
            object_summaries[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)
        scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = object_summaries

        number_summary_dict.update(SD._calculate_num_moving_objects(scenario))

        dynamic_object_states_types = set()
        dynamic_object_states_counter = defaultdict(int)
        for v in scenario[SD.DYNAMIC_MAP_STATES].values():
            for step_state in v["state"]["object_state"]:
                if step_state is None:
                    continue
                dynamic_object_states_types.add(step_state)
                dynamic_object_states_counter[step_state] += 1
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHTS] = len(scenario[SD.DYNAMIC_MAP_STATES])
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = dynamic_object_states_types
        number_summary_dict[SD.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP] = dict(dynamic_object_states_counter)

        number_summary_dict[SD.SUMMARY.NUM_MAP_FEATURES] = len(scenario[SD.MAP_FEATURES])
        number_summary_dict[SD.SUMMARY.MAP_HEIGHT_DIFF] = SD.map_height_diff(scenario[SD.MAP_FEATURES])
        return number_summary_dict

    @staticmethod
    def _calculate_num_moving_objects(scenario):
        SD = ScenarioDescription
        number_summary_dict = {
            SD.SUMMARY.NUM_MOVING_OBJECTS: 0,
            SD.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE: defaultdict(int),
        }
        for v in scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY].values():
            if SD.SUMMARY.MOVING_DIST not in v:
                v[SD.SUMMARY.MOVING_DIST] = v.get("distance", 0)
            if v[SD.SUMMARY.MOVING_DIST] > 1:
                number_summary_dict[SD.SUMMARY.NUM_MOVING_OBJECTS] += 1
                number_summary_dict[SD.SUMMARY.NUM_MOVING_OBJECTS_EACH_TYPE][v["type"]] += 1
        return number_summary_dict

    @staticmethod
    def update_summaries(scenario):
        SD = ScenarioDescription
        summary_dict = {}
        for track_id, track in scenario[SD.TRACKS].items():
            summary_dict[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)
        scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict
        scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(scenario)
        return scenario

    @staticmethod
    def map_height_diff(map_features, target=10):
        max_z = -math.inf
        min_z = math.inf
        for feature in map_features.values():
            if not MetaDriveType.is_road_line(feature[ScenarioDescription.TYPE]):
                continue
            polyline = feature[ScenarioDescription.POLYLINE]
            if len(polyline[0]) == 3:
                z = np.asarray(polyline)[..., -1]
                z_max = np.max(z)
                if z_max > max_z:
                    max_z = z_max
                z_min = np.min(z)
                if z_min < min_z:
                    min_z = z_min
            if max_z - min_z > target:
                break
        return float(max_z - min_z)

    @staticmethod
    def centralize_to_ego_car_initial_position(scenario):
        sdc_id = scenario[ScenarioDescription.METADATA][ScenarioDescription.SDC_ID]
        initial_pos = np.array(
            scenario[ScenarioDescription.TRACKS][sdc_id]["state"]["position"][0], copy=True
        )[:2]
        if abs(np.sum(initial_pos)) < 1e-3:
            return scenario
        return ScenarioDescription.offset_scenario_with_new_origin(scenario, initial_pos)

    @staticmethod
    def offset_scenario_with_new_origin(scenario, new_origin):
        new_origin = np.copy(np.asarray(new_origin))
        for track in scenario[ScenarioDescription.TRACKS].values():
            track["state"]["position"] = np.asarray(track["state"]["position"])
            track["state"]["position"][..., :2] -= new_origin

        for map_feature in scenario[ScenarioDescription.MAP_FEATURES].values():
            if "polyline" in map_feature:
                map_feature["polyline"] = np.asarray(map_feature["polyline"])
                map_feature["polyline"][..., :2] -= new_origin
            if "polygon" in map_feature:
                map_feature["polygon"] = np.asarray(map_feature["polygon"])
                map_feature["polygon"][..., :2] -= new_origin

        for light in scenario[ScenarioDescription.DYNAMIC_MAP_STATES].values():
            if ScenarioDescription.TRAFFIC_LIGHT_POSITION in light:
                light["stop_point"] = np.asarray(light["stop_point"])
                light[ScenarioDescription.TRAFFIC_LIGHT_POSITION][..., :2] -= new_origin

        scenario["metadata"]["old_origin_in_current_coordinate"] = -new_origin
        return scenario

    @staticmethod
    def get_num_objects(scenario, object_type: Optional[str] = None):
        SD = ScenarioDescription
        metadata = scenario[SD.METADATA]
        if SD.SUMMARY.NUMBER_SUMMARY not in metadata:
            scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(scenario)
        if object_type is None:
            return metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS]
        return metadata[SD.SUMMARY.NUMBER_SUMMARY][SD.SUMMARY.NUM_OBJECTS_EACH_TYPE].get(object_type, 0)

    @staticmethod
    def sdc_moving_dist(scenario):
        scenario = ScenarioDescription(scenario)
        SD = ScenarioDescription
        metadata = scenario[SD.METADATA]
        sdc_id = metadata[SD.SDC_ID]
        sdc_info = metadata[SD.SUMMARY.OBJECT_SUMMARY][sdc_id]
        if SD.SUMMARY.MOVING_DIST not in sdc_info:
            sdc_info = SD.get_object_summary(object_dict=scenario.get_sdc_track(), object_id=sdc_id)
        return sdc_info[SD.SUMMARY.MOVING_DIST]


# ---------------------------------------------------------------------------
# Scenario timestep parsing
# ---------------------------------------------------------------------------
#
# ``SD.TIMESTEP`` ("ts") is overloaded: three producers in this repo write
# three incompatible things into it, and every consumer used to assume the
# first one:
#
#   1. scalar dt in seconds          -- procgen/pg_map_to_scenario.py,
#                                       scenario/from_scenario_state.py
#   2. array of ABSOLUTE timestamps  -- scenario/py123d_scenario_description.py
#      in MICROSECONDS                 (``log.timestamps_us``, e.g. 3.16e14)
#   3. array of RELATIVE times in s
#                                       (``[0, dt, 2dt, ...]``)
#
# Reading ``ts[0]`` as a dt is correct for (1), accidentally survivable for (3)
# (``ts[0]`` is 0.0, so a ``> 0`` guard falls back to the 0.1 default), and
# catastrophic for (2) -- the dataset we actually run on. It yielded
# ``scenario_dt = 3.16e14``, hence ``gt_stride = round(0.5 / 3.16e14) = 0``, so
# the EPDMS scorer advanced agents by ZERO frames per scored pose: every
# candidate was judged against a world frozen at the current frame (nc/ttc
# meaningless), and ``route_horizon_s=8.0`` silently became 0.2 s.
#
# The failure was silent and produced plausible numbers, which is why it
# survived. Parse it in ONE place, derive dt from DIFFERENCES (never ``ts[0]``),
# normalise units, and warn rather than silently defaulting.

# A frame interval expressed in SECONDS plausibly lies here. Deliberately
# generous at the top: 2 s/frame (0.5 Hz) is unusual but legitimate, and an
# earlier version of this parser silently rescaled any dt > 1.0 s by 1000x
# (2.0 s -> 0.002 s) because it assumed out-of-window meant wrong-units. That
# reintroduced exactly the silent-plausible-corruption this function exists to
# kill.
_DT_MIN_S = 1e-3
_DT_MAX_S = 10.0
# Only a value too large to BE seconds is a candidate for unit conversion.
# Real sub-second units land far above this (py123d: ~100197 us), so there is
# no overlap with a genuine seconds-valued dt.
_DT_UNIT_THRESHOLD = 10.0
# Sub-second unit scales, coarsest first: milliseconds, microseconds, nanoseconds.
_DT_UNIT_SCALES = (1e-3, 1e-6, 1e-9)
# Warn when median-spacing and total-span/(n-1) disagree by more than this.
_DT_CONSISTENCY_TOL = 0.05


def scenario_dt_seconds(metadata: Optional[dict], *, default: float = 0.1) -> float:
    """Seconds between consecutive scenario frames, from scenario metadata.

    Accepts every ``SD.TIMESTEP`` convention in this repo (scalar dt, absolute
    microsecond timestamps, relative second timestamps) and returns seconds.

    Args:
        metadata: the scenario's ``metadata`` dict (``'timestep'`` wins over
            ``'ts'`` when both are present, matching prior consumer behaviour).
        default: returned when nothing usable is present.

    Returns:
        A positive dt in seconds. Falls back to ``default`` WITH A WARNING --
        a wrong dt silently corrupts agent timing, so it must never pass
        unnoticed.
    """
    if not metadata:
        _logger.warning(
            "scenario_dt: no metadata; using %.3fs. A wrong dt silently "
            "freezes agents during scoring.", default)
        return default
    key = "timestep" if "timestep" in metadata else ScenarioDescription.TIMESTEP
    raw = metadata.get(key)
    if raw is None:
        _logger.warning(
            "scenario_dt: metadata has no %r; using %.3fs.", key, default)
        return default

    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        _logger.warning(
            "scenario_dt: metadata[%r] has no finite entries; using %.3fs",
            key, default)
        return default

    if arr.size == 1:
        # Convention (1): a scalar dt, already in seconds. A lone absolute
        # timestamp is indistinguishable from a huge dt, so it warns out.
        return _normalise_dt(float(arr[0]), key, default, allow_units=False)

    # Conventions (2) and (3): a timestamp series. The SPACING is the dt --
    # ts[0] is an ORIGIN, not an interval. Reading it as one is the bug this
    # replaces.
    diffs = np.diff(arr)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        _logger.warning(
            "scenario_dt: metadata[%r] has %d entries but no positive spacing "
            "(stationary clock?); using %.3fs", key, arr.size, default)
        return default

    # Median over POSITIVE diffs resists dropped frames. But gt_stride indexes
    # FRAMES, and a held/duplicate stamp still consumes a frame index, so the
    # clock-update interval can overstate the per-frame interval. Cross-check
    # against the total span and complain when they disagree rather than
    # silently picking one.
    raw_dt = float(np.median(positive))
    span_dt = float(arr[-1] - arr[0]) / max(1, arr.size - 1)
    if span_dt > 0 and abs(raw_dt - span_dt) > _DT_CONSISTENCY_TOL * max(raw_dt, span_dt):
        _logger.warning(
            "scenario_dt: metadata[%r] spacing is irregular -- median diff "
            "%.6g vs span/(n-1) %.6g over %d entries (%d non-advancing). "
            "Using the median; if frames are duplicated the true per-frame dt "
            "is closer to the span value and agent timing will be off.",
            key, raw_dt, span_dt, arr.size, int((diffs <= 0).sum()))
    return _normalise_dt(raw_dt, key, default, allow_units=True)


def _normalise_dt(raw_dt: float, key: str, default: float,
                  *, allow_units: bool) -> float:
    """Resolve a raw spacing to seconds, converting units only when needed."""
    if _DT_MIN_S <= raw_dt <= _DT_MAX_S:
        # Already a plausible seconds value -- never rescale it. Rescaling
        # here is how a legitimate 2 s/frame clock became 0.002 s.
        return raw_dt
    if allow_units and raw_dt > _DT_UNIT_THRESHOLD:
        # Too large to be seconds: the series is in ms / us (py123d) / ns.
        for scale in _DT_UNIT_SCALES:
            dt = raw_dt * scale
            if _DT_MIN_S <= dt <= _DT_MAX_S:
                return dt
    return _warn_default(_logger, key, raw_dt, default)


def _warn_default(logger, key, value, default: float) -> float:
    logger.warning(
        "scenario_dt: metadata[%r] gave an implausible dt (%.6g s, outside "
        "[%.3g, %.3g]); using %.3fs. A wrong dt silently freezes agents "
        "during scoring -- check the producer's units.",
        key, value, _DT_MIN_S, _DT_MAX_S, default)
    return default
