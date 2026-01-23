from typing import Tuple
from torch import Tensor
from dataclasses import dataclass, MISSING

import torch


@dataclass
class FlightTrajectory:
    takeoff_pos: Tensor
    landing_pos: Tensor
    takeoff_angel: Tensor
    gravity: float = -3.71 # Mars gravity

    def __post_init__(self):
        self._check_iputs()
        self._takeoff_vel = self._calculate_takeoff_vel()
        self._jump_duration = self._claculate_jump_duration()
        land_pos = self.get_base_pos(self.jump_duration)
        diff = land_pos - self.landing_pos

    @property
    def jump_duration(self) -> Tensor:
        return self._jump_duration

    def get_base_pos(self, t: Tensor) -> Tensor:
        pos = self.takeoff_pos + self._takeoff_vel * t.unsqueeze(1)
        pos[:, 2] += 0.5 * self.gravity * t**2
        return pos

    def get_base_vel(self, t: Tensor) -> Tensor:
        vel = self._takeoff_vel.clone()
        vel[:, 2] += self.gravity * t
        return vel

    def _check_iputs(self):
        assert torch.all(self.takeoff_angel >= 20) and torch.all(
            self.takeoff_angel < 80
        ), "Takeoff angle must be between 30 and 80 degrees"
        assert torch.all(
            self.takeoff_pos[:, 0] < self.landing_pos[:, 0]
        ), "Takeoff position must be in front of landing position"

    def _calculate_takeoff_vel(self) -> Tensor:
        diff = self.landing_pos - self.takeoff_pos
        jump_length = (diff[:, :2]).norm(dim=1)
        jump_heading = torch.atan2(diff[:, 1], diff[:, 0])

        denom = (
            2
            * (torch.cos(self.takeoff_angel.deg2rad()).square())
            * (diff[:, 2] - jump_length * torch.tan(self.takeoff_angel.deg2rad()))
        )
        v_abs_sqr = self.gravity * jump_length**2 / denom

        v_abs = v_abs_sqr.sqrt()

        v_z = v_abs * torch.sin(self.takeoff_angel.deg2rad())
        v_xy = v_abs * torch.cos(self.takeoff_angel.deg2rad())
        v_x = v_xy * torch.cos(jump_heading)
        v_y = v_xy * torch.sin(jump_heading)

        return torch.stack((v_x, v_y, v_z), dim=1)

    def _claculate_jump_duration(self) -> Tensor:
        diff = self.landing_pos - self.takeoff_pos
        t_x = diff[:, 0] / self._takeoff_vel[:, 0]
        return t_x
