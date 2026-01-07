from torch import Tensor
from typing import Tuple, Dict

from typing_extensions import override

import torch

from isaaclab.assets.articulation import Articulation

from kinematics import OlympusKinematics
from utilities import Stack

from .configs import CommandCfg
from .initializer_base import JumpInitializerBase


class DefaultInitializer(JumpInitializerBase):

    def __init__(
        self,
        olympus: Articulation,
        kinematics: OlympusKinematics,
        command_cfg: CommandCfg,
    ):
        super().__init__(olympus, kinematics, None, command_cfg)

    @override
    def _make_stack(self) -> Stack:
        def generator():
            command = self._sample_command()

            return torch.cat(
                (
                    command,
                    self._default_base_state[:7]
                    .unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_joint_pos.unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_base_state[7:]
                    .unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_joint_vel.unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                ),
                dim=1,
            )

        return Stack(generator)
