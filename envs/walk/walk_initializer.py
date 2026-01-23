from typing import Dict, List, Tuple
from dataclasses import MISSING

import torch
from torch import Tensor
from isaaclab.assets.articulation import Articulation
from isaaclab.utils import configclass

from kinematics import OlympusKinematics
from utilities import Stack, uniform_sample, rotations as rotations_utils


_DEFAULT_PAW_POS_LIMITS = {
    "Paw_BL": (
        [-0.22, 0.20, 0.0253],
        [-0.20, 0.22, 0.0253],
    ),
    "Paw_BR": (
        [-0.22, -0.22, 0.0253],
        [-0.20, -0.20, 0.0253],
    ),
    "Paw_FL": (
        [0.20, 0.15, 0.0253],
        [0.22, 0.17, 0.0253],
    ),
    "Paw_FR": (
        [0.20, -0.17, 0.0253],
        [0.22, -0.15, 0.0253],
    ),
}


@configclass
class WalkInitializerCfg:
    base_pos_limits: Tuple[List[float], List[float]] = MISSING
    base_euler_limits: Tuple[List[float], List[float]] = MISSING
    paw_pos_limits: Dict[str, Tuple[List[float], List[float]]] = _DEFAULT_PAW_POS_LIMITS


class WalkInitializer:

    def __init__(
        self,
        cfg: WalkInitializerCfg,
        olympus: Articulation,
        kinematics: OlympusKinematics,
    ):
        self._num_envs = olympus.num_instances
        self._device = olympus.device
        self._kinematics = kinematics
        self._default_base_state = olympus.data.default_root_state[0]
        self._default_joint_pos = olympus.data.default_joint_pos[0]
        self._default_joint_vel = olympus.data.default_joint_vel[0]
        self._num_states = 13 + 2 * olympus.num_joints
        self._stack: Stack = None
        self._parse_cfg(cfg)

    def draw(self, num_envs: int) -> Tensor:
        if self._stack is None:
            self._stack = self._make_stack()
        return self._stack.pop(num_envs)

    def _make_stack(self) -> Stack:
        def generator():

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
                    base_pose,
                    q,
                    base_vel,
                    q_dot,
                ),
                dim=1,
            )

        return Stack(generator)

    def _parse_cfg(self, cfg: WalkInitializerCfg):
        self._base_pos_limits = self._tuple_to_tensor(cfg.base_pos_limits)
        self._base_euler_limits = self._tuple_to_tensor(cfg.base_euler_limits)
        self._paw_pos_limits = {paw: self._tuple_to_tensor(limits) for paw, limits in cfg.paw_pos_limits.items()}

    def _tuple_to_tensor(self, t: Tuple[List[float] | float]) -> Tuple[Tensor, Tensor]:
        return tuple(torch.tensor(l, device=self._device) for l in t)
