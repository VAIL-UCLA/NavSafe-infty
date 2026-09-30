"""
DiffusionDrive TransFuser feature/target builders.
Ported from BridgeSim — nuplan imports replaced with navsafe equivalents.

NOTE: TransfuserFeatureBuilder and TransfuserTargetBuilder are omitted because
they depend on nuplan map APIs, scenario builders, and abstract builders that
are not part of the core model architecture. Only BoundingBox2DIndex (used by
the model) is ported here.
"""

from enum import IntEnum


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4

    @classmethod
    def size(cls):
        return 5

    @classmethod
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls.X, cls.Y + 1)

    @classmethod
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls.X, cls.HEADING + 1)
