from torch import Tensor
from typing import Tuple, Dict

from typing_extensions import override

import torch
from torch.nn.functional import normalize

from dataclasses import dataclass, MISSING
from isaaclab.assets.articulation import Articulation
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply


from kinematics import OlympusKinematics
from utilities import Stack, uniform_sample, rotations as rotations_utils

from .configs import TouchdownInitializerCfg, FlightTrajectoryCfg, CommandCfg
from .initializer_base import JumpInitializerBase


class TouchdownInitializer(JumpInitializerBase):

    def __init__(
        self,
        cfg: TouchdownInitializerCfg,
        olympus: Articulation,
        kinematics: OlympusKinematics,
        flight_trajectory_cfg: FlightTrajectoryCfg,
        command_cfg: CommandCfg,
    ):
        super().__init__(olympus, kinematics, flight_trajectory_cfg, command_cfg)
        self._resolve_indicies(olympus)
        self._parse_cfg(cfg)

    @override
    def _make_stack(self) -> Stack:
        def generator():
            command = self._sample_command()

            flight_traj = self._sample_flight_trajectory(command, clip_mode="landing_pos")

            flight_time = (
                flight_traj.jump_duration
                - uniform_sample(
                    *self._time_to_land_limits,
                    self._num_envs,
                )
            ).clamp(min=0)

            base_pos = flight_traj.get_base_pos(flight_time)
            base_lin_vel_w = flight_traj.get_base_vel(flight_time)

            base_pose = torch.cat(
                (
                    base_pos,
                    rotations_utils.random_orientation_euler(
                        self._num_envs,
                        self._base_euler_limits,
                    ),
                ),
                dim=1,
            )

            base_ang_vel_w = quat_apply(
                base_pose[:, 3:],
                uniform_sample(*self._base_ang_vel_limits, self._num_envs),
            )

            q = torch.zeros(self._num_envs, self._kinematics.num_joints, device=self._device)
            q[:, self._lateral_indices] = uniform_sample(
                *self._lateral_joints_limits,
                self._num_envs,
            )
            q[:, self._front_transversal_indices] = (
                uniform_sample(
                    *self._front_transversal_joints_limits,
                    self._num_envs,
                )
                .unsqueeze(1)
                .expand(-1, len(self._front_transversal_indices))
            )
            q[:, self._back_transversal_indices] = (
                uniform_sample(
                    *self._back_transversal_joints_limits,
                    self._num_envs,
                )
                .unsqueeze(1)
                .expand(-1, len(self._back_transversal_indices))
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

    def _resolve_indicies(self, olympus: Articulation):
        self._lateral_indices = olympus.find_joints("LateralMotor.*")[0]

        self._front_transversal_indices = [
            olympus.find_joints(jn)[0][0]
            for jn in [
                "OuterTransversalMotor_FL",
                "OuterTransversalMotor_FR",
                "InnerTransversalMotor_BL",
                "InnerTransversalMotor_BR",
            ]
        ]

        self._back_transversal_indices = [
            olympus.find_joints(jn)[0][0]
            for jn in [
                "InnerTransversalMotor_FL",
                "InnerTransversalMotor_FR",
                "OuterTransversalMotor_BL",
                "OuterTransversalMotor_BR",
            ]
        ]

    def _parse_cfg(self, cfg: TouchdownInitializerCfg):
        self._time_to_land_limits = self._tuple_to_tensor(cfg.time_to_land_limits)
        self._base_ang_vel_limits = self._tuple_to_tensor(cfg.base_ang_vel_limits)
        self._lateral_joints_limits = tuple(
            l.unsqueeze(0).expand(len(self._lateral_indices)) for l in self._tuple_to_tensor(cfg.lateral_joints_limits)
        )
        self._front_transversal_joints_limits = self._tuple_to_tensor(cfg.front_transversal_joints_limits)
        self._back_transversal_joints_limits = self._tuple_to_tensor(cfg.back_transversal_joints_limits)
        self._base_euler_limits = self._tuple_to_tensor(cfg.base_euler_limits)
