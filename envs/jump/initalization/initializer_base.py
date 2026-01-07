from typing import Tuple, Literal, List
from torch import Tensor

from abc import abstractmethod, ABC
from typing_extensions import override


import torch
from torch.nn.functional import normalize
import time

from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_apply,
    random_orientation,
    quat_from_angle_axis,
)

from kinematics import OlympusKinematics
from utilities import uniform_sample, Stack
from .configs import FlightTrajectoryCfg, CommandCfg
from .flight_trajectory import FlightTrajectory


class JumpInitializerBase(ABC):
    def __init__(
        self,
        olympus: Articulation,
        kinematics: OlympusKinematics,
        flight_trajectory_cfg: FlightTrajectoryCfg,
        command_cfg: CommandCfg,
    ) -> None:
        self._num_envs = olympus.num_instances
        self._device = olympus.device
        self._kinematics = kinematics
        self._default_base_state = olympus.data.default_root_state[0]
        self._default_joint_pos = olympus.data.default_joint_pos[0]
        self._default_joint_vel = olympus.data.default_joint_vel[0]
        self._num_states = 13 + 2 * olympus.num_joints
        self._stack: Stack = None
        self._flight_trajectory_cfg = flight_trajectory_cfg
        self._command_cfg = command_cfg
        self._command_dim = 3
        self._parse_command_cfg(command_cfg)
        self._parse_flight_trajectory_cfg(flight_trajectory_cfg)

    @abstractmethod
    def _make_stack(self) -> Stack:
        pass

    def draw(self, num_envs: int) -> Tuple[Tensor, Tensor]:
        if self._stack is None:
            self._stack = self._make_stack()
        return torch.split(self._stack.pop(num_envs), [self._command_dim, self._num_states], dim=1)

    def _tuple_to_tensor(self, t: Tuple[List[float] | float]) -> Tuple[Tensor, Tensor]:
        return tuple(torch.tensor(l, device=self._device) for l in t)

    def _parse_command_cfg(self, command_cfg: CommandCfg) -> None:
        self._landing_pos_limits = self._tuple_to_tensor(command_cfg.landing_pos_limits)
        assert self._landing_pos_limits[0].shape[0] == self._command_dim
        assert self._landing_pos_limits[1].shape[0] == self._command_dim

    def _parse_flight_trajectory_cfg(self, flight_trajectory_cfg: FlightTrajectoryCfg) -> None:
        if flight_trajectory_cfg is None:
            return
        self._ft_takeoff_pos_limits = self._tuple_to_tensor(flight_trajectory_cfg.takeoff_pos_limits)
        self._ft_takeoff_angle_limits = self._tuple_to_tensor(flight_trajectory_cfg.takeoff_angle_limits)
        self._ft_jump_length_limits = self._tuple_to_tensor(flight_trajectory_cfg.jump_length_limits)

    def _sample_command(self, num_envs: int | None = None) -> Tensor:
        if num_envs is None:
            num_envs = self._num_envs
        return uniform_sample(*self._landing_pos_limits, num_envs)

    def _sample_flight_trajectory(
        self,
        command: Tensor,
        clip_mode: Literal["landing_pos", "takeoff_pos"] = "landing_pos",
    ) -> FlightTrajectory:
        if self._flight_trajectory_cfg is None:
            raise ValueError("Flight trajectory config is not provided")

        land_pos = command.clone()
        # sample takeoff pos and angle
        takeoff_pos = uniform_sample(*self._ft_takeoff_pos_limits, self._num_envs)
        takeoff_angle = uniform_sample(*self._ft_takeoff_angle_limits, self._num_envs)

        # clip the jump length to the limits
        commanded_jump_length = command - takeoff_pos
        too_long = (commanded_jump_length[:, :2] - self._ft_jump_length_limits[1].unsqueeze(0)).clip(min=0)
        too_short = (commanded_jump_length[:, :2] - self._ft_jump_length_limits[0].unsqueeze(0)).clip(max=0)

        if clip_mode == "takeoff_pos":
            takeoff_pos[:, :2] += too_long
            takeoff_pos[:, :2] += too_short

        elif clip_mode == "landing_pos":
            land_pos[:, :2] -= too_long
            land_pos[:, :2] -= too_short
        else:
            raise ValueError("clip_mode must be either 'landing_pos' or 'takeoff_pos'")

        return FlightTrajectory(takeoff_pos, land_pos, takeoff_angle)
