"""
Scenario constants — ported from MetaDrive's metadrive.constants.
"""


class TerminationState:
    SUCCESS = "arrive_dest"
    OUT_OF_ROAD = "out_of_road"
    MAX_STEP = "max_step"
    CRASH = "crash"
    CRASH_VEHICLE = "crash_vehicle"
    CRASH_HUMAN = "crash_human"
    CRASH_OBJECT = "crash_object"
    CRASH_BUILDING = "crash_building"
    CRASH_SIDEWALK = "crash_sidewalk"
    CURRENT_BLOCK = "current_block"
    ENV_SEED = "env_seed"
    IDLE = "idle"
