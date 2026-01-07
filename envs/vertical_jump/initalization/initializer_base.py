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
    quat_rotate,
    random_orientation,
    quat_from_angle_axis,
)

from kinematics import OlympusKinematics
from utilities import uniform_sample, Stack, rotations as rotations_utils
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
        self._command_dim = 1
        self._parse_command_cfg(command_cfg)
        self._parse_flight_trajectory_cfg(flight_trajectory_cfg)

    def draw(self, num_envs: int) -> Tuple[Tensor, Tensor]:
        if self._stack is None:
            self._stack = self._make_stack()
        return torch.split(self._stack.pop(num_envs), [self._command_dim, self._num_states], dim=1)

    def _tuple_to_tensor(self, t: Tuple[List[float] | float]) -> Tuple[Tensor, Tensor]:
        return tuple(torch.tensor(l, device=self._device) for l in t)

    def _parse_command_cfg(self, command_cfg: CommandCfg) -> None:
        self._jump_height_limits = self._tuple_to_tensor(command_cfg.jump_height_limits)

    def _parse_flight_trajectory_cfg(self, flight_trajectory_cfg: FlightTrajectoryCfg) -> None:
        if flight_trajectory_cfg is None:
            return
        self._ft_takeoff_pos_limits = self._tuple_to_tensor(flight_trajectory_cfg.takeoff_pos_limits)

    def _sample_command(self) -> Tensor:
        return uniform_sample(*self._jump_height_limits, self._num_envs).view(self._num_envs, -1)

    def _sample_flight_trajectory(self, command: Tensor) -> FlightTrajectory:
        if self._flight_trajectory_cfg is None:
            raise ValueError("Flight trajectory config is not provided")

        jump_height = command.clone()
        # sample takeoff pos and angle
        takeoff_pos = uniform_sample(*self._ft_takeoff_pos_limits, self._num_envs)
        landpos_pos = uniform_sample(*self._ft_takeoff_pos_limits, self._num_envs)
        return FlightTrajectory(takeoff_pos, landpos_pos, jump_height)
