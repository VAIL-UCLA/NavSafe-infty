"""
TrajectorySampling dataclass — replaces nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling.
"""

from dataclasses import dataclass


@dataclass
class TrajectorySampling:
    time_horizon: float = 4.0
    interval_length: float = 0.5

    @property
    def num_poses(self) -> int:
        return int(self.time_horizon / self.interval_length)
