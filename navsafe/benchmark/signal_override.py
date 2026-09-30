# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Author a traffic-light state the log never recorded.

V-1 Red-Light is scored as a hold: the ego must stay behind the line while the
signal is red. The published corpus cannot pose that question. Measured
2026-08-31 on the two V-1 bundles on disk, the logged states are
``LANE_STATE_GO`` and ``LANE_STATE_UNKNOWN`` -- ``00c1e4eb4a045f20``'s single
signalled lane reads GO for 42 frames then UNKNOWN for 159, and
``_red_lane_ids`` treats UNKNOWN as not-red, which is why the evaluator's ``TL``
subscore is 1.0 on every scored frame of every run of that token and why no
``red_light`` infraction has ever fired in the 41 runs under
``output/navsafe_sweep``. So the red is not something to hold; it has to be
written.

The override is declared per bundle in ``manifest.json`` so it travels with the
scenario and diffs with it:

.. code-block:: json

    "signal_override": {"lane": "52791", "state": "LANE_STATE_STOP",
                        "frames": "all"}

``frames`` is ``"all"`` or ``[start, stop)``. Only ``"all"`` is implemented; the
range is in the schema from the start because adding a green phase later should
be a manifest edit, and re-baking every V-1 bundle to widen a schema is the
expensive half of that change.

It is applied to ``dynamic_map_states`` at the point the ScenarioDescription is
built, so every consumer -- the evaluator's red-light rule, the PDM planner's
own lookup, the trace's ``signal_state`` -- reads one state. Applying it any
later would let a privileged expert see green while the scorer sees red.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

#: Set by a driver that has already read the manifest, for the paths that build
#: a ScenarioDescription without knowing which bundle it came from (the env's
#: scenario manager is several layers from the file). JSON, same shape as the
#: manifest key.
ENV_VAR = "NAVSAFE_SIGNAL_OVERRIDE"

_STATES = ("LANE_STATE_STOP", "LANE_STATE_GO", "LANE_STATE_CAUTION",
           "LANE_STATE_UNKNOWN")


def validate(override: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one override, raising on anything the schema cannot express.

    ``lane`` may be one id or several. Several is usually what a junction needs:
    the evaluator's red-light rule exempts a crossing when a NON-red lane still
    offers a way through on the ego's heading (``_green_way_through``), so
    reddening one connector of a multi-lane approach leaves its parallel
    siblings green and the crossing is never convicted. Measured on
    ``05d0a1a763fc5334``: with only lane 52246 red, SimWAM drove through and the
    ``TL`` column stayed 1.0 on all 599 scored frames.
    """
    raw = override.get("lane")
    lanes = [raw] if isinstance(raw, (str, int)) else list(raw or [])
    lanes = [str(x).strip() for x in lanes if str(x).strip()]
    if not lanes:
        raise ValueError("signal_override needs a 'lane' id, or a list of them")
    state = str(override.get("state") or "LANE_STATE_STOP").strip().upper()
    if state not in _STATES:
        raise ValueError(f"signal_override state {state!r} is not one of {_STATES}")
    frames = override.get("frames", "all")
    if frames != "all":
        pair = list(frames)
        if len(pair) != 2 or int(pair[0]) < 0 or int(pair[1]) <= int(pair[0]):
            raise ValueError("signal_override frames must be 'all' or [start, stop)")
        raise NotImplementedError(
            "only frames='all' is implemented; a partial range needs the "
            "success rule to change meaning at the switch (see the V-1 plan)")
    return {"lane": lanes[0] if len(lanes) == 1 else lanes,
            "lanes": lanes, "state": state, "frames": "all"}


def from_manifest(data_root: str | Path) -> dict[str, Any] | None:
    """The override declared by the bundle a py123d data root belongs to."""
    root = Path(str(data_root))
    man = (root.parent if root.name == "arrow" else root) / "manifest.json"
    if not man.is_file():
        return None
    try:
        raw = json.loads(man.read_text()).get("signal_override")
    except Exception:                                        # noqa: BLE001
        return None
    return validate(raw) if raw else None


def from_env() -> dict[str, Any] | None:
    """The override this process was started with, if any."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    return validate(json.loads(raw))


def apply(sd: Any, override: Mapping[str, Any] | None) -> int:
    """Force one lane's light state for the whole scenario.

    Creates the entry when the log never reported that lane -- which is the
    normal case, since a lane with no detection gets no ``dynamic_map_states``
    row at all and an absent row is exactly what reads as "not red".

    Returns:
        Frames written, so a caller can log that the override took.
    """
    if not override:
        return 0
    override = validate(override)
    states = sd.get("dynamic_map_states")
    if states is None:
        states = {}
        sd["dynamic_map_states"] = states
    length = int(sd.get("length") or 0)
    if not length:
        for entry in states.values():
            seq = (entry.get("state") or {}).get("object_state") or []
            length = max(length, len(seq))
    if not length:
        tracks = sd.get("tracks") or {}
        sdc = (sd.get("metadata") or {}).get("sdc_id")
        track = tracks.get(sdc) or next(iter(tracks.values()), None)
        if track:
            length = len(((track.get("state") or {}).get("position") or []))
    if not length:
        return 0

    written = 0
    for lane in override["lanes"]:
        entry = states.get(lane)
        if entry is None:
            entry = {
                "type": "TRAFFIC_LIGHT",
                "state": {"object_state": ["LANE_STATE_UNKNOWN"] * length},
                "lane": lane,
                "metadata": {"track_length": length, "object_id": lane},
            }
            states[lane] = entry
        seq = entry.setdefault("state", {}).get("object_state")
        if not seq:
            seq = ["LANE_STATE_UNKNOWN"] * length
            entry["state"]["object_state"] = seq
        for i in range(len(seq)):
            seq[i] = override["state"]
        written += len(seq)
    return written


__all__ = ["ENV_VAR", "apply", "from_env", "from_manifest", "validate"]
