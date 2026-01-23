from torch import Tensor
from typing import Tuple, Dict
from typing_extensions import override

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.utils.math import quat_apply

from kinematics import OlympusKinematics
from utilities import Stack, uniform_sample
from utilities import rotations as rotations_utils
from .initializer_base import JumpInitializerBase
from .configs import InflightInitializerCfg, CommandCfg, FlightTrajectoryCfg


class InflightInitializer(JumpInitializerBase):

    def __init__(
        self,
        cfg: InflightInitializerCfg,
        olympus: Articulation,  # shoud not take in olympus, class method would be better
        kinematics: OlympusKinematics,
        flight_trajectory_cfg: FlightTrajectoryCfg,
        command_cfg: CommandCfg,
    ):
        super().__init__(olympus, kinematics, flight_trajectory_cfg, command_cfg)
        self._resolve_indices(olympus)
        self._parse_cfg(cfg)

    @override
    def _make_stack(self) -> Stack:

        def generator():
            command = self._sample_command()

            flight_traj = self._sample_flight_trajectory(command, clip_mode="landing_pos")
            flight_time = (
                uniform_sample(
                    *self._normalized_time_limits,
                    self._num_envs,
                )
                * flight_traj.jump_duration
            )

            base_pos = flight_traj.get_base_pos(flight_time)
            base_lin_vel_w = flight_traj.get_base_vel(flight_time)

            base_pose = torch.cat(
                (
                    base_pos,
                    rotations_utils.random_orientation_euler(
                        self._num_envs,
                        euler_limits=self._base_euler_limits,
                    ),
                ),
                dim=1,
            )

            base_pose[:, 0] -= torch.rand(self._num_envs, device=self._device) * 0.15
            base_ang_vel_w = quat_apply(
                base_pose[:, 3:],
                uniform_sample(*self._base_ang_vel_limits, self._num_envs),
            )

            q = torch.zeros(self._num_envs, self._kinematics.num_joints, device=self._device)
            q[:, self._lateral_indices] = uniform_sample(
                *self._lateral_joints_limits,
                self._num_envs,
            )
            q[:, self._transversal_indices] = uniform_sample(
                *self._transversal_joints_limits,
                self._num_envs,
            )

            q = q.deg2rad()

            q = self._kinematics.get_consitent_knee_positions(q)
            return torch.cat(
                (
                    command,
                    base_pose,
                    q,
                    base_lin_vel_w,
                    base_ang_vel_w,
                    torch.zeros_like(q),
                ),
                dim=1,
            )

        return Stack(generator)

    def _parse_cfg(self, cfg: InflightInitializerCfg) -> None:
        self._base_ang_vel_limits = self._tuple_to_tensor(cfg.base_ang_vel_limits)
        self._lateral_joints_limits = tuple(
            l.unsqueeze(0).expand(len(self._lateral_indices)) for l in self._tuple_to_tensor(cfg.lateral_joints_limits)
        )
        self._transversal_joints_limits = tuple(
            l.unsqueeze(0).expand(len(self._transversal_indices))
            for l in self._tuple_to_tensor(cfg.transversal_joints_limits)
        )
        self._base_euler_limits = self._tuple_to_tensor(cfg.base_euler_limits)
        self._normalized_time_limits = self._tuple_to_tensor(cfg.normalized_time_limits)

    def _resolve_indices(self, olympus: Articulation):
        self._lateral_indices = olympus.find_joints("LateralMotor.*")[0]
        self._transversal_indices = olympus.find_joints(".*TransversalMotor.*")[0]
