import numpy as np
import pytest
from navsafe.traffic.timed_cut_in import CutInProfile, TimedCutInDriver
from navsafe.evaluation.perturbation import augmented_warmup_path


def test_cut_in_common_prefix_and_return():
    hazard = CutInProfile(3.5)
    base = CutInProfile(3.5, enabled=False)
    for t in np.linspace(0, 2, 31):
        assert hazard.sample(t) == base.sample(t)
    assert hazard.sample(4) == (3.5, 0.0)
    assert hazard.sample(6) == (3.5, 0.0)
    assert hazard.sample(8) == (0.0, 0.0)
    assert hazard.sample(20) == (0.0, 0.0)


@pytest.mark.parametrize("t", [2, 4, 6, 8])
def test_cut_in_continuous_at_boundaries(t):
    p = CutInProfile(-3.5)
    left, right = np.array(p.sample(t-1e-5)), np.array(p.sample(t+1e-5))
    np.testing.assert_allclose(left, right, atol=1e-7)


def test_driver_heading_tracks_lateral_motion():
    p = dict(path_polyline=[[0,0,0],[200,0,0]],
             profile=dict(speed=10, enabled=False),
             cut_in=dict(lateral_m=3.5))
    d = TimedCutInDriver("car", dict(position=[0,0,0]), policy=p)
    from types import SimpleNamespace
    d.step(SimpleNamespace(t=30), .1)
    pose = d.pose()
    assert pose["position"][0] == pytest.approx(30)
    assert pose["position"][1] == pytest.approx(1.75)
    assert pose["velocity"][1] > 0
    assert pose["heading"] > 0
    d.step(SimpleNamespace(t=70), .1)
    assert d.pose()["heading"] < 0


def test_reference_matches_initial_and_requested_handoff():
    xy=np.column_stack([np.arange(51),np.zeros(51)])
    path=augmented_warmup_path(xy, np.zeros(51), 20, 1, 2, 10)
    np.testing.assert_allclose(path[0], xy[0])
    np.testing.assert_allclose(path[20], [22,1])
    direction=path[21]-path[20]
    assert np.arctan2(direction[1],direction[0]) == pytest.approx(np.deg2rad(10))
    np.testing.assert_allclose(
        augmented_warmup_path(xy,np.zeros(51),20,0,0,0),xy)


@pytest.mark.parametrize("kwargs", [dict(transition_s=0), dict(hold_s=-1), dict(lateral_m=float("nan"))])
def test_invalid_profiles_fail(kwargs):
    with pytest.raises(ValueError):
        CutInProfile(**(dict(lateral_m=3.5) | kwargs))

@pytest.mark.parametrize("lat,lon,steps", [(.5,0,20),(0,.5,8)])
def test_controller_history_uses_integrated_states(tmp_path,lat,lon,steps):
    from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg
    from navsafe.evaluation.evaluator import EvaluationConfig, Evaluator
    class Env:
        def __init__(self):
            self.ego=EgoDynamics(EgoDynamicsCfg(dt=.1))
            self.ego.reset(x=0,y=0,heading=0,speed=5)
            self.overrides=0
            self.logged=np.column_stack([np.arange(101)*.5,np.zeros(101),np.zeros(101)])
        def get_scenario_info(self):
            return {"metadata":{"sdc_id":"ego"},"tracks":{"ego":{"state":{
                "position":self.logged,"heading":np.zeros(101)}}}}
        def get_ego_state(self):
            return self.ego.get_state()
        def clear_ego_override(self):
            pass
        def set_ego_externally_driven(self,active):
            assert active
        def set_ego_override(self,**kwargs):
            self.overrides+=1
            raise AssertionError("history mode must not teleport")
    env=Env()
    cfg=EvaluationConfig(execution_mode="controller",controller_type="lqr",
        ego_replay_frames=steps,ego_perturb_history="controller",ego_perturb_lateral_m=lat,
        ego_perturb_longitudinal_m=lon,
        output_dir=tmp_path)
    ev=Evaluator(env=env,model_adapter=object(),config=cfg)
    positions=[]
    for k in range(steps):
        ev.frame=k
        state=env.get_ego_state()
        positions.append(state["position"].copy())
        action=ev._execute_augmented_warmup(state)
        env.ego.step(float(action[0]),float(action[1]))
    ev.frame=steps
    before=env.get_ego_state()["position"].copy()
    ev._record_controller_perturbation()
    np.testing.assert_array_equal(env.get_ego_state()["position"],before)
    assert env.overrides==0
    assert np.max(np.linalg.norm(np.diff(positions,axis=0),axis=1))<1.
    key="achieved_lateral_m" if lat else "achieved_longitudinal_m"
    assert ev._perturb_record[key]>.01
    assert ev._perturb_record["requested_lateral_m"]==lat
