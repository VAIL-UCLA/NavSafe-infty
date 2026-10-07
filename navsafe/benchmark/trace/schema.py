"""The canonical episode trace -- the only expensive artifact in the benchmark.

Everything downstream (rubric pass/fail, outcome labels, episode score,
aggregation) is a pure function of this file.  That is the whole point: the
paper leaves the rubric thresholds, the score weights and the envelope bounds
unfixed, so scoring has to be re-runnable without re-simulating.  A trace is
written once per episode and read many times.

Design rules, in order of importance:

1. **Record facts, not verdicts.**  Store the distance to the lead vehicle, not
   "headway ok".  A verdict baked into the trace freezes a threshold that the
   benchmark still intends to tune.
2. **Record what the simulator knows, not what the policy saw.**  The trace is
   ground truth for scoring; policy inputs belong in its own logs.
3. **Mark, never drop.**  A frame that cannot be judged on some axis is kept and
   flagged rather than omitted, because "this was never checked" must stay
   auditable instead of becoming a silent gap in the record.
4. **Frames carry their phase.**  Warm-up frames are replayed ground truth and
   must never be scored; keeping them makes the hand-over visible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import pyarrow as pa

TRACE_VERSION = "0.2.0"  # 0.2.0: + signal_hold

# Phase of each frame.  The policy takes over exactly at t_trig (doc II-B).
PHASE_WARMUP = "warmup"    # ego replays the logged action; not scored
PHASE_SCORED = "scored"    # policy in control; this is the episode

# --- per-frame record ------------------------------------------------------
# One row per simulator step.  Nested lists (agents, contacts) are stored as
# Arrow list<struct>, so the whole trace stays one flat parquet file.

FRAME_SCHEMA = pa.schema([
    ("frame", pa.int32()),
    ("t_sim_s", pa.float64()),            # sim clock, 0 at episode start
    ("t_log_us", pa.int64()),             # the source log timestamp it maps to
    ("phase", pa.string()),

    # --- ego ground truth (float64: absolute UTM would lose 0.5 m in float32)
    ("ego_x", pa.float64()),
    ("ego_y", pa.float64()),
    ("ego_z", pa.float64()),
    ("ego_yaw", pa.float64()),
    ("ego_speed", pa.float64()),
    ("ego_accel", pa.float64()),
    ("ego_jerk", pa.float64()),
    ("ego_lat_accel", pa.float64()),
    ("ego_steer", pa.float64()),

    # --- map context, as measurements rather than booleans where possible
    ("on_drivable", pa.bool_()),          # a hard gate, so a fact by definition
    # DDC == 0.0: metres driven against the local traffic direction passed the
    # full-violation threshold. True whenever there was no DDC to judge, so a
    # scenario with no lane graph reads "not checked", never "wrong way".
    ("driving_direction_ok", pa.bool_()),
    ("lane_id", pa.string()),
    ("lane_heading", pa.float64()),
    ("lateral_offset_m", pa.float64()),   # signed, from lane centre
    ("dist_to_stopline_m", pa.float64()), # negative once past it
    ("in_intersection", pa.bool_()),
    ("in_conflict_zone", pa.bool_()),

    # --- signals
    ("signal_id", pa.string()),
    ("signal_state", pa.string()),        # "red"|"yellow"|"green"|"unknown"
    # A red signal is ahead on the ego's lane (or was just crossed): the
    # evaluator's state fact, from the logged light states. Exempts the frame
    # from the deadlock hold (termination.py) -- waiting at a red is not a
    # freeze. False means "no red ahead" AND "not checked"; both keep the
    # plain rule, so an older writer cannot exempt anything by omission.
    # A declared exception to rule 1: like ``on_drivable`` it is a
    # thresholded verdict (30 m ahead / 2 m lateral / heading-aligned,
    # EPDMSLiveScorer._signal_hold_live), stored because re-deriving it needs
    # the map and the light log, which a trace reader does not have.
    ("signal_hold", pa.bool_()),

    # --- other road users
    ("agents", pa.list_(pa.struct([
        ("id", pa.string()),
        ("cls", pa.string()),             # vehicle|pedestrian|bicycle|...
        ("policy", pa.string()),          # replay|idm|adversarial|scripted
        ("x", pa.float64()), ("y", pa.float64()), ("yaw", pa.float64()),
        ("speed", pa.float64()),
        ("length", pa.float64()), ("width", pa.float64()),
        ("dist_to_ego", pa.float64()),      # centre to centre
        # Body-to-body gap: centre distance less both half-extents along the
        # line of sight. This is what a near-miss floor must be measured
        # against -- a 0.5 m CENTRE distance would mean the boxes overlap.
        ("clearance_m", pa.float64()),
        ("is_lead", pa.bool_()),          # same lane, ahead
        ("in_conflict_zone", pa.bool_()),
        ("ttc_s", pa.float64()),          # inf when not closing
        # Who is closing the distance. Under log-replay a non-reactive
        # follower can drive into a correctly-behaving ego; gating the ego on
        # that would score the replay's physics, not the policy. The writer
        # decides this from the relative velocity, so predicates never have to
        # re-derive geometry.
        ("ego_is_closing", pa.bool_()),
    ]))),

    # --- contacts.  Fault is recorded per contact, because the fault rule is
    # regime-dependent (doc Table VI) and re-deciding it later must be possible.
    ("contacts", pa.list_(pa.struct([
        ("agent_id", pa.string()),
        ("at_fault", pa.bool_()),
        ("kind", pa.string()),            # rear_end|angle|sideswipe|vru|single
        ("rel_speed", pa.float64()),
    ]))),
    ("min_clearance_m", pa.float64()),    # to any agent, this frame

    # --- deviation from the logged ego pose, as facts. These fed a
    # render-validity envelope that once ended episodes (`envelope_exit`);
    # the bound fired on ordinary closed-loop driving and excluded
    # those runs from every denominator, so it was removed. The measurements
    # stay: they are the raw material for certifying a bound properly, and
    # keeping them means that can be done from stored traces.
    ("ego_dev_lat_m", pa.float64()),
    ("ego_dev_lon_m", pa.float64()),
    ("ego_dev_yaw_deg", pa.float64()),
])


@dataclass
class EpisodeMeta:
    """Written alongside the frames; identifies exactly what produced them."""

    trace_version: str = TRACE_VERSION
    episode_id: str = ""
    seed_id: str = ""
    family: str = ""
    regime: str = ""                       # log_replay|reactive|safety_critical
    policy: str = ""
    policy_checkpoint: str = ""
    sim_dt: float = 0.1
    warmup_frames: int = 0
    t_max_s: float | None = None
    # The frozen sample that generated this episode, so it replays exactly.
    program_sample: dict[str, Any] = field(default_factory=dict)
    rng_seed: int = 0
    # Provenance of the world it ran in.
    world_version: str = ""                # recon checkpoint hash / run id
    scenario_origin_xy: tuple[float, float] | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def empty_frame() -> dict:
    """A row with every column present, so partial writers cannot shift schema."""
    row: dict[str, Any] = {}
    for f in FRAME_SCHEMA:
        t = f.type
        if pa.types.is_list(t):
            row[f.name] = []
        elif pa.types.is_boolean(t):
            row[f.name] = False
        elif pa.types.is_string(t):
            row[f.name] = ""
        elif pa.types.is_integer(t):
            row[f.name] = 0
        else:
            row[f.name] = float("nan")
    # Compliance flags are the exception to "booleans default False". They are
    # termination inputs — `on_drivable: False` ends the episode `off_drivable`
    # and `driving_direction_ok: False` ends it `wrong_way` — so a writer that
    # simply has nothing to say about an axis must not thereby fail the ego.
    # Absence means "not checked", which is True here and False nowhere.
    row["on_drivable"] = True
    row["driving_direction_ok"] = True
    return row
