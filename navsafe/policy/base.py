"""Common abstract parent for all NexusSim policy adapters.

This module is the canonical home of :class:`BasePolicyAdapter`, the
abstract base class every first-party and third-party policy adapter
inherits from. Phase 3 task 3.5 of the NexusSim Package Reorg spec
stands it up; it backs Requirements 3.1 (the package exposes
``BasePolicyAdapter`` as the common parent) and 10.5 (the reorg must
not modify the ``prepare_input`` / ``run_inference`` / ``parse_output``
semantics of any first-party adapter).

Per the §12.6-RESOLVED no-shim policy, the pre-reorg base class
``navsafe.evaluation.models.base_adapter:BaseModelAdapter`` is
deleted outright. The contract carried over verbatim into this class,
but callers must update imports to use :class:`BasePolicyAdapter`
from :mod:`navsafe.policy` directly — the old import path is gone.

Sub-classes :class:`navsafe.policy.state_policy.StatePolicy` and
:class:`navsafe.policy.sensor_policy.SensorPolicy` then partition the
adapter space along the modality axis defined by Requirement 3.7
(base-class choice is determined by what ``prepare_input`` consumes).
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict

import numpy as np

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-finite plan telemetry
# ---------------------------------------------------------------------------
#
# The scorer stack already guards non-finite candidates downstream; what it
# cannot do is say WHERE the NaN came from. ``parse_output`` is the single
# choke point where model-output trajectories enter the pipeline, so every
# concrete adapter's ``parse_output`` is wrapped (via ``__init_subclass__``)
# with a finiteness check on the returned ``"trajectory"``. A non-finite plan
# is COUNTED per adapter class and warned once per adapter — it is never
# raised or altered here: the verifier's fail-worst semantics own that
# decision, and the existing downstream guards take over unchanged.

_nonfinite_plan_counts: Counter[str] = Counter()
_nonfinite_plan_warned: set[str] = set()


def nonfinite_plan_counts() -> dict[str, int]:
    """Per-adapter-class counts of non-finite plan trajectories seen so far."""
    return dict(_nonfinite_plan_counts)


def reset_nonfinite_plan_state() -> None:
    """Reset warn-once markers and counts (test isolation)."""
    _nonfinite_plan_counts.clear()
    _nonfinite_plan_warned.clear()


class BasePolicyAdapter(ABC):
    """
    Abstract base class for policy adapters.

    Each policy adapter handles:
    1. Model loading and initialization
    2. Input preprocessing (camera images, ego state, etc.)
    3. Model inference
    4. Output parsing and trajectory extraction
    """

    #: Reentrancy flag for the ``parse_output`` telemetry wrapper (see
    #: ``__init_subclass__``); never touched by adapters.
    _in_validated_parse_output: bool = False

    def __init__(self, checkpoint_path: str, config_path: str | None = None, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model: Any = None
        self.device = "cuda"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap each concrete ``parse_output`` with non-finite plan telemetry.

        Wrapping happens once per class that defines its own ``parse_output``
        (inherited implementations are already wrapped on the parent); a
        reentrancy flag keeps ``super().parse_output()`` delegation from
        double-counting a single plan.
        """
        super().__init_subclass__(**kwargs)
        parse = cls.__dict__.get("parse_output")
        if (
            parse is None
            or not callable(parse)
            # staticmethod/classmethod parse_output would be mis-called as
            # ``parse(self, ...)`` by the wrapper — leave such adapters
            # unwrapped (telemetry gap) rather than break their calls.
            or isinstance(parse, (staticmethod, classmethod))
            or getattr(parse, "__isabstractmethod__", False)
            or getattr(parse, "__navsafe_plan_validated__", False)
        ):
            return

        @functools.wraps(parse)
        def wrapped(self: "BasePolicyAdapter", *args: Any, **inner_kwargs: Any) -> Any:
            if getattr(self, "_in_validated_parse_output", False):
                return parse(self, *args, **inner_kwargs)
            self._in_validated_parse_output = True
            try:
                result = parse(self, *args, **inner_kwargs)
            finally:
                self._in_validated_parse_output = False
            try:
                self._record_nonfinite_plan(result)
            except Exception:  # pragma: no cover - defensive: telemetry must never break inference
                _logger.debug("non-finite plan telemetry failed", exc_info=True)
            return result

        wrapped.__navsafe_plan_validated__ = True  # type: ignore[attr-defined]
        cls.parse_output = wrapped  # type: ignore[method-assign]

    def _record_nonfinite_plan(self, parsed: Any) -> None:
        """Count + warn-once when a parsed plan trajectory is non-finite.

        Telemetry only: the parsed output is returned to the caller unaltered
        regardless, so the existing downstream non-finite guards keep their
        behavior — this exists so NaN sources are attributable to an adapter.
        """
        if not isinstance(parsed, dict):
            return
        trajectory = parsed.get("trajectory")
        if trajectory is None:
            return
        try:
            arr = np.asarray(trajectory)
        except Exception:
            return
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return
        if bool(np.isfinite(arr).all()):
            return
        name = type(self).__name__
        _nonfinite_plan_counts[name] += 1
        if name not in _nonfinite_plan_warned:
            _nonfinite_plan_warned.add(name)
            _logger.warning(
                "policy adapter %s (checkpoint=%s) emitted a non-finite plan trajectory; "
                "passing through unchanged for downstream guards. Warning once per adapter; "
                "occurrences counted in nonfinite_plan_counts().",
                name,
                # getattr: a third-party adapter that skips super().__init__ must
                # not have telemetry raise AttributeError out of parse_output.
                getattr(self, "checkpoint_path", None),
            )

    @abstractmethod
    def load_model(self):
        """Load the model from checkpoint. Must set self.model."""
        pass

    @abstractmethod
    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare model input from raw sensor data and ego state."""
        pass

    @abstractmethod
    def run_inference(self, model_input: Any) -> Any:
        """Run model inference."""
        pass

    @abstractmethod
    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse model output and extract planning trajectory."""
        pass

    def get_waypoint_dt(self) -> float:
        """Return time interval between predicted waypoints in seconds."""
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        """Return the total time horizon of predicted trajectory in seconds."""
        return 4.0

    def perceive(self, env, frame_id: int):
        """
        Optional custom perception hook. Return a dict of camera images if the
        adapter wants to drive image capture itself, or return None to let the
        base evaluator handle perception normally.
        """
        return None

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Return camera configuration for this model."""
        return {
            'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
            'CAM_BACK_LEFT': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK_RIGHT': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_THIRD_PERSON': {'x': -8.0, 'y': 0.0, 'z': 3.0, 'yaw': 0.0, 'pitch': -15.0, 'roll': 0.0, 'fov': 70, 'width': 900, 'height': 900},
        }

    def save_intermediate_outputs(self, model_output: Any, output_path: str):
        """Save intermediate model outputs (optional, for debugging/visualization)."""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(checkpoint={self.checkpoint_path})"


__all__ = ["BasePolicyAdapter", "nonfinite_plan_counts", "reset_nonfinite_plan_state"]
