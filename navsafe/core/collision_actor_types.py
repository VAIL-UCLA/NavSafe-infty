"""Canonical taxonomy for physical actors that can collide with ego.

Scenario adapters cannot always map a source label to a specialized
MetaDrive type.  Those actors are deliberately preserved as ``OTHER`` rather
than guessed to be vehicles.  ``OTHER`` must still participate in geometry:
an unknown semantic class does not make its rendered collision body vanish.
"""

from __future__ import annotations


COLLIDABLE_ACTOR_TYPES = frozenset({
    "VEHICLE",
    "CYCLIST",
    "PEDESTRIAN",
    "TRAFFIC_CONE",
    "TRAFFIC_BARRIER",
    "TRAFFIC_STOP_SIGN",
    "TRAFFIC_OBJECT",
    # py123d maps unmapped physical labels (including ``generic_object``) to
    # OTHER.  Some direct scenario sources retain the original generic name.
    "OTHER",
    "GENERIC_OBJECT",
})

# Prediction APIs historically expose an ordered tuple as their override
# contract.  Keep that shape while deriving it from the same source of truth.
COLLIDABLE_ACTOR_TYPES_TUPLE = tuple(sorted(COLLIDABLE_ACTOR_TYPES))


def is_collidable_actor_type(actor_type: object) -> bool:
    """Return whether an actor label denotes a physical collision body."""
    return str(actor_type or "").upper() in COLLIDABLE_ACTOR_TYPES
