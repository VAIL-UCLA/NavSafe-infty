"""NuRec camera dispatch survives removal of local capture implementations."""
from types import SimpleNamespace
import numpy as np
from navsafe.env.camera_manager import CameraManager


class Renderer:
    def set_timestep(self, value):
        self.timestep = value

    def get_camera_images(self, **kwargs):
        self.request = kwargs
        return {name: np.zeros((2, 3, 3), dtype=np.uint8) for name in kwargs['cam_configs']}


def test_ego_images_forward_state_and_timestep():
    renderer = Renderer()
    state = {'position': [1, 2], 'heading': 0.5}
    actors = [{'id': 'other'}]
    env = SimpleNamespace(renderer=renderer, scenario_timestep=17,
                          get_ego_state=lambda: state,
                          _collect_agent_states_for_renderer=lambda: actors)
    cameras = CameraManager(env)
    images = cameras.get_camera_images({'CAM_F0': {'width': 3, 'height': 2}})
    assert renderer.timestep == 17
    assert renderer.request['ego_state'] is state
    assert renderer.request['agent_states'] is actors
    assert images['CAM_F0'].shape == (2, 3, 3)
    assert cameras.get_ego_front_image_bgr().shape == (2, 3, 3)
