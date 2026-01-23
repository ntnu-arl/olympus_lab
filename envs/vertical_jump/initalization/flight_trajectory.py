from typing import Tuple
from torch import Tensor
from dataclasses import dataclass, MISSING

import torch


@dataclass
class FlightTrajectory:
    takeoff_pos: Tensor
    landing_pos: Tensor
    jump_height: Tensor
    gravity: float = -9.81

    def __post_init__(self):
        self._check_iputs()
        self._takeoff_vel = self._calculate_takeoff_vel()
        land_pos = self.get_base_pos(self.jump_duration)
        diff = land_pos - self.landing_pos
        # Calculate the apex height error
        apex_positions = self.get_base_pos(self._t_up)[:, 2]
        apex_height_error = torch.abs(apex_positions - self.jump_height)
        

        assert torch.all(diff.norm(dim=1) < 1e-4), "Landing position does not match the expected landing position"
        assert torch.all((self.get_base_pos(self._t_up)[:, 2] - self.jump_height).norm(dim=-1) < 1e-4)

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
        assert torch.all(self.jump_height >= 0)
        assert torch.all(
            self.takeoff_pos[:, 0] <= self.landing_pos[:, 0]
        ), "Takeoff position must be in front of landing position"

    def _calculate_takeoff_vel(self) -> Tensor:
        diff = self.landing_pos - self.takeoff_pos
        jump_length = (diff[:, :2]).norm(dim=1)
        h_up = self.jump_height - self.takeoff_pos[:, 2]
        h_down = self.jump_height - self.landing_pos[:, 2]
        v_z = torch.sqrt(-2 * self.gravity * h_up)
        t_up = v_z / -self.gravity
        t_down = torch.sqrt(2 * h_down / -self.gravity)
        v_xy = jump_length / (t_up + t_down)
        self._jump_duration = t_up + t_down
        self._t_up = t_up

        jump_heading = torch.atan2(diff[:, 1], diff[:, 0])

        v_x = v_xy * torch.cos(jump_heading)
        v_y = v_xy * torch.sin(jump_heading)

        return torch.stack((v_x, v_y, v_z), dim=1)
