"""Route-completion projection must retain progress already attained."""

import numpy as np

from navsafe.evaluation.evaluator import _monotone_projected_route_arc


def test_passing_endpoint_cannot_regress_to_penultimate_waypoint():
    route = np.column_stack((np.arange(5.0), np.zeros(5)))
    # Reach x=4, then travel far beyond it. A final-only nearest projection can
    # pick an earlier sample on curved real routes; attained completion cannot.
    ego = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0],
                    [20.0, 10.0], [20.0, 20.0]])

    attained = _monotone_projected_route_arc(route, ego)

    assert np.all(np.diff(attained) >= 0.0)
    assert attained[-1] == 4.0


def test_moving_backwards_does_not_erase_route_completion():
    route = np.column_stack((np.arange(6.0), np.zeros(6)))
    ego = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])

    assert _monotone_projected_route_arc(route, ego).tolist() == [0.0, 3.0, 3.0]
