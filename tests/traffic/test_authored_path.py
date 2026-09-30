"""Physical prefix and event semantics for BEV-authored paired recipes."""
from types import SimpleNamespace
import numpy as np
import pytest
from navsafe.traffic.authored_path import AuthoredMotion, AuthoredPathDriver, smooth_path

def policy(kind,enabled=True):
    return dict(event_kind=kind,enabled=enabled,onset_s=3,speed=6,
                deceleration=3,hold_s=1,recovery_acceleration=2,
                path_polyline=smooth_path([[0,0,0],[10,0,0],[30,3,0],[200,3,0]]))

@pytest.mark.parametrize('kind',['cut_in','lead_brake','vru_cross'])
def test_shared_history_until_onset(kind):
    event=AuthoredMotion(policy(kind)); baseline=AuthoredMotion(policy(kind,False))
    for time in np.linspace(0,3,31):
        for key in ['position','velocity','heading']:
            np.testing.assert_allclose(event.sample(time)[key],baseline.sample(time)[key])

def test_brake_stops_and_recovers():
    m=AuthoredMotion(policy('lead_brake'))
    assert m.sample(4)['speed']==3
    assert m.sample(5)['speed']==0
    assert m.sample(5.5)['speed']==0
    assert m.sample(7)['speed']==2
    assert m.sample(9)['speed']==6

def test_vru_waits_then_crosses_and_stops():
    p=policy('vru_cross'); p['path_polyline']=[[0,0,0],[12,0,0]]
    m=AuthoredMotion(p)
    np.testing.assert_allclose(m.sample(2)['position'],[0,0,0])
    np.testing.assert_allclose(m.sample(4)['position'],[6,0,0])
    assert m.sample(6)['speed']==0
    assert m.sample(6)['exhausted']
    p['enabled']=False
    np.testing.assert_allclose(AuthoredMotion(p).sample(20)['position'],[0,0,0])

def test_short_vehicle_path_fails_in_runtime():
    p=policy('cut_in'); p['path_polyline']=[[0,0,0],[2,0,0]]
    driver=AuthoredPathDriver('test',dict(position=[-18,0,0]),policy=p)
    with pytest.raises(ValueError,match='路径已用尽'):
        driver.step(SimpleNamespace(t=40),.1)

def test_spawn_cannot_disagree_with_history():
    with pytest.raises(ValueError,match='spawn does not match'):
        AuthoredPathDriver('test',dict(position=[0,0,0]),policy=policy('cut_in'))
