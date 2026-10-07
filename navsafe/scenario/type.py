"""
Scenario type constants — ported from MetaDrive's metadrive.type module.

Provides MetaDriveType string constants and classification helpers used
throughout UrbanSim for ScenarioDescription-format scenario data.
"""

import logging

logger = logging.getLogger(__name__)


class MetaDriveType:
    """
    String-constant type system for scenario objects and map features.
    Follows Waymo-style naming conventions.
    """

    # ===== Lane, Road =====
    LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
    LANE_SURFACE_UNSTRUCTURE = "LANE_SURFACE_UNSTRUCTURE"
    LANE_UNKNOWN = "LANE_UNKNOWN"
    LANE_FREEWAY = "LANE_FREEWAY"
    LANE_BIKE_LANE = "LANE_BIKE_LANE"

    # ===== Lane Line =====
    LINE_UNKNOWN = "UNKNOWN_LINE"
    LINE_BROKEN_SINGLE_WHITE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    LINE_SOLID_SINGLE_WHITE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    LINE_SOLID_DOUBLE_WHITE = "ROAD_LINE_SOLID_DOUBLE_WHITE"
    LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
    LINE_BROKEN_DOUBLE_YELLOW = "ROAD_LINE_BROKEN_DOUBLE_YELLOW"
    LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
    LINE_PASSING_DOUBLE_YELLOW = "ROAD_LINE_PASSING_DOUBLE_YELLOW"

    # ===== Edge/Boundary/SideWalk/Region =====
    BOUNDARY_UNKNOWN = "UNKNOWN"
    BOUNDARY_LINE = "ROAD_EDGE_BOUNDARY"
    BOUNDARY_MEDIAN = "ROAD_EDGE_MEDIAN"
    BOUNDARY_SIDEWALK = "ROAD_EDGE_SIDEWALK"
    STOP_SIGN = "STOP_SIGN"
    CROSSWALK = "CROSSWALK"
    SPEED_BUMP = "SPEED_BUMP"
    DRIVEWAY = "DRIVEWAY"
    GUARDRAIL = "GUARDRAIL"
    # Drivable surfaces that are not lanes, and junction footprints. nuPlan's
    # ``CARPARK_AREA`` layer is part of PDM-Closed's drivable-area map and its
    # ``INTERSECTION`` layer is the TTC fault-cone predicate; both are emitted
    # as polygon-only features (``polyline`` = the closed ring).
    CARPARK_AREA = "CARPARK_AREA"
    INTERSECTION = "INTERSECTION"

    # ===== Traffic Light =====
    LANE_STATE_UNKNOWN = "LANE_STATE_UNKNOWN"
    LANE_STATE_ARROW_STOP = "LANE_STATE_ARROW_STOP"
    LANE_STATE_ARROW_CAUTION = "LANE_STATE_ARROW_CAUTION"
    LANE_STATE_ARROW_GO = "LANE_STATE_ARROW_GO"
    LANE_STATE_STOP = "LANE_STATE_STOP"
    LANE_STATE_CAUTION = "LANE_STATE_CAUTION"
    LANE_STATE_GO = "LANE_STATE_GO"
    LANE_STATE_FLASHING_STOP = "LANE_STATE_FLASHING_STOP"
    LANE_STATE_FLASHING_CAUTION = "LANE_STATE_FLASHING_CAUTION"

    LIGHT_GREEN = "TRAFFIC_LIGHT_GREEN"
    LIGHT_RED = "TRAFFIC_LIGHT_RED"
    LIGHT_YELLOW = "TRAFFIC_LIGHT_YELLOW"
    LIGHT_UNKNOWN = "TRAFFIC_LIGHT_UNKNOWN"

    # ===== Agent type =====
    UNSET = "UNSET"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    OTHER = "OTHER"

    # ===== Object type =====
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    TRAFFIC_BARRIER = "TRAFFIC_BARRIER"
    TRAFFIC_CONE = "TRAFFIC_CONE"
    TRAFFIC_STOP_SIGN = "TRAFFIC_STOP_SIGN"
    TRAFFIC_OBJECT = "TRAFFIC_OBJECT"
    GROUND = "GROUND"
    INVISIBLE_WALL = "INVISIBLE_WALL"
    BUILDING = "BUILDING"

    # ===== Coordinate system =====
    COORDINATE_METADRIVE = "metadrive"
    COORDINATE_WAYMO = "waymo"

    @classmethod
    def is_traffic_object(cls, type):
        return type in [cls.TRAFFIC_CONE, cls.TRAFFIC_BARRIER, cls.TRAFFIC_STOP_SIGN, cls.TRAFFIC_OBJECT]

    @classmethod
    def has_type(cls, type_string: str):
        return type_string in cls.__dict__

    @classmethod
    def is_lane(cls, type):
        return type in [
            cls.LANE_SURFACE_STREET, cls.LANE_SURFACE_UNSTRUCTURE, cls.LANE_UNKNOWN,
            cls.LANE_BIKE_LANE, cls.LANE_FREEWAY,
        ]

    @classmethod
    def is_road_line(cls, line):
        return line in [
            cls.LINE_UNKNOWN, cls.LINE_BROKEN_SINGLE_WHITE, cls.LINE_SOLID_SINGLE_WHITE,
            cls.LINE_SOLID_DOUBLE_WHITE, cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_DOUBLE_YELLOW,
            cls.LINE_SOLID_SINGLE_YELLOW, cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW,
        ]

    @classmethod
    def is_yellow_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW,
        ]

    @classmethod
    def is_white_line(cls, line):
        return cls.is_road_line(line) and not cls.is_yellow_line(line)

    @classmethod
    def is_broken_line(cls, line):
        return line in [cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_SINGLE_WHITE]

    @classmethod
    def is_solid_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_WHITE, cls.LINE_SOLID_DOUBLE_YELLOW,
            cls.LINE_SOLID_SINGLE_YELLOW, cls.LINE_SOLID_SINGLE_WHITE,
        ]

    @classmethod
    def is_road_boundary_line(cls, edge):
        return edge in [cls.BOUNDARY_UNKNOWN, cls.BOUNDARY_LINE, cls.BOUNDARY_MEDIAN]

    @classmethod
    def is_sidewalk(cls, edge):
        return edge == cls.BOUNDARY_SIDEWALK

    @classmethod
    def is_stop_sign(cls, type):
        return type == cls.STOP_SIGN

    @classmethod
    def is_speed_bump(cls, type):
        return type == cls.SPEED_BUMP

    @classmethod
    def is_driveway(cls, type):
        return type == cls.DRIVEWAY

    @classmethod
    def is_crosswalk(cls, type):
        return type == cls.CROSSWALK

    @classmethod
    def is_vehicle(cls, type):
        return type == cls.VEHICLE

    @classmethod
    def is_pedestrian(cls, type):
        return type == cls.PEDESTRIAN

    @classmethod
    def is_cyclist(cls, type):
        return type == cls.CYCLIST

    @classmethod
    def is_participant(cls, type):
        return type in (cls.CYCLIST, cls.PEDESTRIAN, cls.VEHICLE, cls.UNSET, cls.OTHER)

    @classmethod
    def simplify_light_status(cls, status: str):
        if status in [cls.LANE_STATE_UNKNOWN, cls.LANE_STATE_FLASHING_STOP, cls.LIGHT_UNKNOWN, None]:
            return cls.LIGHT_UNKNOWN
        elif status in [cls.LANE_STATE_ARROW_STOP, cls.LANE_STATE_STOP, cls.LIGHT_RED]:
            return cls.LIGHT_RED
        elif status in [cls.LANE_STATE_ARROW_CAUTION, cls.LANE_STATE_CAUTION,
                        cls.LANE_STATE_FLASHING_CAUTION, cls.LIGHT_YELLOW]:
            return cls.LIGHT_YELLOW
        elif status in [cls.LANE_STATE_ARROW_GO, cls.LANE_STATE_GO, cls.LIGHT_GREEN]:
            return cls.LIGHT_GREEN
        else:
            logger.warning("TrafficLightStatus: {} is not a known MetaDriveType".format(status))
            return cls.LIGHT_UNKNOWN

    @classmethod
    def parse_light_status(cls, status: str, simplifying=True):
        if simplifying:
            return cls.simplify_light_status(status)
        return status

    def __init__(self, type=None):
        self.metadrive_type = MetaDriveType.UNSET
        if type is not None:
            self.set_metadrive_type(type)

    def set_metadrive_type(self, type):
        if type in MetaDriveType.__dict__.values():
            self.metadrive_type = type
        else:
            raise ValueError(f"'{type}' is not a valid MetaDriveType.")
