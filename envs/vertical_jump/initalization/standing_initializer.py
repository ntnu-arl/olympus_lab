from torch import Tensor
from typing import Tuple, Dict

from typing_extensions import override

import torch

from isaaclab.assets.articulation import Articulation

from kinematics import OlympusKinematics
from utilities import Stack, uniform_sample, rotations as rotations_utils

from .configs import StandingInitializerCfg, CommandCfg
from .initializer_base import JumpInitializerBase


class StandingInitializer(JumpInitializerBase):

    def __init__(
        self,
        cfg: StandingInitializerCfg,
        olympus: Articulation,
        kinematics: OlympusKinematics,
        command_cfg: CommandCfg,
    ):
        super().__init__(olympus, kinematics, None, command_cfg)
        self._parse_cfg(cfg)

    @override
    def _make_stack(self) -> Stack:
        def generator():
            command = self._sample_command()

            base_pos = uniform_sample(*self._base_pos_limits, self._num_envs)

            base_rot = rotations_utils.random_orientation_euler(self._num_envs, self._base_euler_limits)
            base_pose = torch.cat((base_pos, base_rot), dim=1)
            base_vel = torch.zeros(self._num_envs, 6, device=self._device)

            paw_pos = {paw: uniform_sample(*limits, self._num_envs) for paw, limits in self._paw_pos_limits.items()}

            base_pose[:, :3], base_pose[:, 3:], q, q_dot = self._kinematics.get_consistent_joint_state(
                base_pose, base_vel[:, :3], base_vel[:, 3:], paw_pos
            )

            return torch.cat(
                (
                    command,
                    base_pose,
                    q,
                    base_vel,
                    q_dot,
                ),
                dim=1,
            )

        return Stack(generator)

    def _parse_cfg(self, cfg: StandingInitializerCfg):
        self._base_pos_limits = self._tuple_to_tensor(cfg.base_pos_limits)
        self._base_euler_limits = self._tuple_to_tensor(cfg.base_euler_limits)
        self._paw_pos_limits = {paw: self._tuple_to_tensor(limits) for paw, limits in cfg.paw_pos_limits.items()}
