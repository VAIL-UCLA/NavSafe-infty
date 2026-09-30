"""Structural safety-net for the Evaluator decomposition.

The 1,479-line ``Evaluator`` God class is being broken into focused
collaborators (artifact/visualization writer, route manager, ...). Its runtime
behaviour can only be exercised with the full IsaacSim / torch stack, which is
unavailable in CI. This test is the guardrail that *can* run import-light: it
pins the public surface so a refactor that drops, renames, or fails to re-expose
a method is caught immediately.

``navsafe.evaluation.evaluator`` now imports without the heavy stack because
``torch`` and ``vis_utils`` (which pulls cv2) are guarded — that is what makes
the class body introspectable here. See the module's import guards.

If a member is *intentionally* removed/renamed, update the baseline below in the
same commit and say why.
"""

from __future__ import annotations

import importlib


# Public surface captured immediately before the decomposition began.
EVALUATOR_BASELINE = frozenset({
    '_capture_observer_images', '_dump_npz', '_enrich_ego_state',
    '_generate_visualizations', '_interpolate_trajectory',
    '_invoke_adapter_perceive', '_lift_trajectory', '_render_frame_vis',
    '_reset_state', '_run_inference_and_cache', '_safe_imwrite',
    '_safe_stack', '_save_camera_extra', '_save_extra_sensor_artifacts',
    '_save_lidar_bev', '_save_results', '_step',
    'finalize', 'generate_route', 'get_next_waypoint', 'run', 'setup',
})

EVALUATION_CONFIG_BASELINE = frozenset({
    'controller_type', 'ego_replay_frames', 'enable_vis', 'eval_frames',
    'eval_mode', 'replan_rate', 'save_per_frame', 'sim_dt', 'traffic_mode',
    'vis_online',
    # Added by the execution-mode axis (teleport / controller / physics).
    # (output_dir is intentionally absent: it uses field(default_factory=...)
    # so it is not a class attribute and never appears in dir().)
    'execution_mode',
})


def test_module_imports_without_heavy_stack():
    """The evaluator must import with no torch / cv2 present (guarded)."""
    m = importlib.import_module('navsafe.evaluation.evaluator')
    assert m.Evaluator.__name__ == 'Evaluator'
    assert m.EvaluationConfig.__name__ == 'EvaluationConfig'


def test_evaluator_public_surface_is_preserved():
    """No baseline Evaluator member may vanish during decomposition."""
    from navsafe.evaluation.evaluator import Evaluator

    present = {n for n in dir(Evaluator) if not n.startswith('__')}
    missing = EVALUATOR_BASELINE - present
    assert not missing, (
        f"Evaluator lost {len(missing)} public member(s): {sorted(missing)}. "
        f"If intentional, update EVALUATOR_BASELINE in this file."
    )


def test_evaluation_config_fields_preserved():
    """EvaluationConfig's field set is a stable contract for callers/CLIs."""
    from navsafe.evaluation.evaluator import EvaluationConfig

    present = {n for n in dir(EvaluationConfig) if not n.startswith('__')}
    missing = EVALUATION_CONFIG_BASELINE - present
    assert not missing, f"EvaluationConfig lost fields: {sorted(missing)}"


def test_canonical_import_path_resolves():
    """External code imports these names directly from the module."""
    from navsafe.evaluation.evaluator import Evaluator, EvaluationConfig

    assert Evaluator.__name__ == 'Evaluator'
    assert EvaluationConfig.__name__ == 'EvaluationConfig'
