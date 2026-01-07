from torch import Tensor
from typing import Tuple
from dataclasses import MISSING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.utils.configclass import configclass

from kinematics import OlympusKinematics
from utilities import Stack, uniform_sample, rotations as rotations_utils
from torch.nn.functional import normalize
import time

from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_rotate,
    random_orientation,
    quat_from_angle_axis,
)

def random_orientation_axis( 
    num_envs: int, device: str, euler_limits: Tuple[Tensor, Tensor] | None = None
) -> Tensor:

    if euler_limits is not None:
        euler = torch.split(
            uniform_sample(*euler_limits, num_envs).deg2rad(), 1, dim=1
        )
        return normalize(quat_from_euler_xyz(*euler).squeeze(1))

    # random axis
    phi = torch.rand(num_envs, device=device) * torch.pi
    theta = torch.rand(num_envs, device=device) * 2 * torch.pi
    s_phi = torch.sin(phi)
    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)

    axis = torch.stack((s_phi * c_theta, s_phi * s_theta, torch.cos(phi)), dim=1)
    # random angle
    angle = torch.rand(num_envs, device=device) * 2 * torch.pi - torch.pi

    return quat_from_angle_axis(angle, axis)

def random_orientation_quat( 
        num_envs: int, device: str, euler_limits: Tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        
    if euler_limits is not None:
        euler = torch.split(
            uniform_sample(*euler_limits, num_envs).deg2rad(), 1, dim=1
        )
        return normalize(quat_from_euler_xyz(*euler).squeeze(1))
    
    i, j, k = torch.rand(num_envs, 3, device=device).unbind(dim=-1)
    x = torch.sqrt(1 - i) * torch.sin(2 * torch.pi * j)
    y = torch.sqrt(1 - i) * torch.cos(2 * torch.pi * j)
    z = torch.sqrt(i) * torch.sin(2 * torch.pi * k)
    w = torch.sqrt(i) * torch.cos(2 * torch.pi * k)
        
    return torch.stack((w, x, y, z), dim=-1)

def predefined_orientation(
    num_envs: int, device: str, euler_limits: Tuple[Tensor, Tensor] | None = None
) -> Tensor:

    if euler_limits is not None:
        euler = torch.split(
            uniform_sample(*euler_limits, num_envs).deg2rad(), 1, dim=1
        )
        return normalize(quat_from_euler_xyz(*euler).squeeze(1))

    # defined orientation
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)
    yaw = torch.zeros(num_envs, device=device)
    # add a random angle from -150 to 150 degrees
    random_added_angle_deg = (2 * torch.rand(num_envs, device=device) - 1) * 179

    roll = roll - 0.0*torch.pi/180
    pitch = pitch + 0.0 *torch.pi/180
    yaw = yaw + random_added_angle_deg*torch.pi/180

    return quat_from_euler_xyz(roll, pitch, yaw)

def random_single_axis_orientation(
    num_envs: int, 
    device: str, 
    angle_limit_deg: float = 179.0,
    axis: str = "random"  # can be "roll", "pitch", "yaw", or "random"
) -> Tensor:
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)
    yaw = torch.zeros(num_envs, device=device)
    
    random_angles = (2 * torch.rand(num_envs, device=device) - 1) * angle_limit_deg
    random_angles = random_angles * torch.pi / 180.0  
    
    if axis == "random":
        axis = torch.randint(1, 4, (1,)).item()  
    if axis == "roll" or axis == 1:
        roll = random_angles
    elif axis == "pitch" or axis == 2:
        pitch = random_angles
    elif axis == "yaw" or axis == 3:
        yaw = random_angles
    else:
        raise ValueError("Invalid axis specified. Must be 'roll', 'pitch', 'yaw', or 'random'")
    return quat_from_euler_xyz(roll, pitch, yaw)

@configclass
class AttitudeControlInitializerCfg:
    """
    Config for attitude control task initializer
    """
    latteral_joints_limits: Tuple[float, float] = MISSING
    transversal_joints_limits: Tuple[float, float] = MISSING


class AttitudeControlInitializer:

    def __init__(
        self,
        cfg: AttitudeControlInitializerCfg,
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
        self._resolve_indices(olympus)
        self._parse_cfg(cfg)

    def draw(self, num_envs: int) -> Tensor:
        if self._stack is None:
            self._stack = self._make_stack()
        return self._stack.pop(num_envs)

    def _make_stack(self) -> Stack:
        def generator():
            base_pos = self._default_base_state[:3].expand(self._num_envs, 3).clone()
            base_pos[:, 2] = 2.0

            base_rot = random_orientation_quat(self._num_envs, device=self._device)  


            # test with rotation velocity
            initial_ang_vel = torch.zeros(self._num_envs, 3, device=self._device)
            initial_ang_vel[:, 2] = 0.0

            with torch.device(self._device):

                q = torch.zeros(self._num_envs, self._kinematics.num_joints)
                lateral = uniform_sample(
                    *self._latteral_limits,
                    self._num_envs,
                )
                # lateral = torch.zeros(self._num_envs, len(self._lateral_indices), device=self._device) # for debugging

                left_lateral = lateral[:, self._left_lateral_indices]
                right_lateral = lateral[:, self._right_lateral_indices]
                lateral_sum = left_lateral + right_lateral
                lateral_clamp_mask = lateral_sum < 0
                left_lateral[lateral_clamp_mask] = -lateral_sum[lateral_clamp_mask] / 2

                right_lateral[lateral_clamp_mask] = -lateral_sum[lateral_clamp_mask] / 2

                lateral[:, self._left_lateral_indices] = left_lateral
                lateral[:, self._right_lateral_indices] = right_lateral

                outer_transversal, inner_transversal = uniform_sample(
                    *self._transversal_limits,
                    self._num_envs,
                ).chunk(2, dim=1)
                # outer_transversal = torch.zeros(self._num_envs, len(self._outer_transversal_indices), device=self._device) +45 # for debugging
                # inner_transversal = torch.zeros(self._num_envs, len(self._inner_transversal_indices), device=self._device)+45
                outer_transversal_sum = inner_transversal + outer_transversal
                clamp_mask = outer_transversal_sum > 220
                inner_transversal[clamp_mask] -= (outer_transversal_sum[clamp_mask] - 220) / 2
                outer_transversal[clamp_mask] -= (outer_transversal_sum[clamp_mask] - 220) / 2
                clamp_mask = outer_transversal_sum < 0
                inner_transversal[clamp_mask] -= outer_transversal_sum[clamp_mask] / 2
                outer_transversal[clamp_mask] -= outer_transversal_sum[clamp_mask] / 2

                large_lateral = lateral >= 70
                inner_transversal[large_lateral] = inner_transversal[large_lateral].clamp(-180, 85)

                large_lateral = lateral >= 88
                outer_transversal[large_lateral] = outer_transversal[large_lateral].clamp(-180, 120)
                large_lateral = lateral >= 90
                outer_transversal[large_lateral] = outer_transversal[large_lateral].clamp(-180, 20)
                inner_transversal[large_lateral] = inner_transversal[large_lateral].clamp(-180, 20)

                q[:, self._lateral_indices] = lateral
                q[:, self._inner_transversal_indices] = inner_transversal
                q[:, self._outer_transversal_indices] = outer_transversal

                back_inner_transversal = q[:, self._back_inner_transversal_indices]
                front_inner_transversal = q[:, self._front_inner_transversal_indices]
                back_lateral = q[:, self._back_lateral_indices]
                front_lateral = q[:, self._front_lateral_indices]
                lateral_diff = (back_lateral - front_lateral).abs()
                inner_transversal_sum = back_inner_transversal + front_inner_transversal
                clamp_mask = inner_transversal_sum > 10
                back_inner_transversal[clamp_mask] = 10.0
                front_inner_transversal[clamp_mask] = 10.0

                q[:, self._back_inner_transversal_indices] = back_inner_transversal
                q[:, self._front_inner_transversal_indices] = front_inner_transversal

                front_outer_transversal = q[:, self._front_outer_transversal_indices]
                back_outer_transversal = q[:, self._back_outer_transversal_indices]
                front_outer_transversal[clamp_mask] = front_outer_transversal[clamp_mask].clamp(min=-10)
                back_outer_transversal[clamp_mask] = back_outer_transversal[clamp_mask].clamp(min=-10)
                q[:, self._front_outer_transversal_indices] = front_outer_transversal
                q[:, self._back_outer_transversal_indices] = back_outer_transversal

            q = q.deg2rad()
            q = self._kinematics.get_consitent_knee_positions(q)

            return torch.cat(
                (
                    base_pos,
                    base_rot,
                    q,
                    torch.zeros(self._num_envs, 3, device=self._device),  
                    initial_ang_vel,  
                    torch.zeros_like(q),
                ),
                dim=1,
            )

        return Stack(generator)

    def _resolve_indices(self, olympus: Articulation):
        self._lateral_indices = olympus.find_joints("LateralMotor.*")[0]
        self._transversal_indices = olympus.find_joints(".*TransversalMotor.*")[0]
        self._inner_transversal_indices = olympus.find_joints("InnerTransversalMotor.*")[0]
        self._outer_transversal_indices = olympus.find_joints("OuterTransversalMotor.*")[0]
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

        self._front_lateral_indices = [olympus.find_joints(jn)[0][0] for jn in ["LateralMotor_FL", "LateralMotor_FR"]]
        self._back_lateral_indices = [olympus.find_joints(jn)[0][0] for jn in ["LateralMotor_BL", "LateralMotor_BR"]]

        self._left_lateral_indices = [olympus.find_joints(jn)[0][0] for jn in ["LateralMotor_FL", "LateralMotor_BL"]]
        self._right_lateral_indices = [olympus.find_joints(jn)[0][0] for jn in ["LateralMotor_FR", "LateralMotor_BR"]]

        self._back_inner_transversal_indices = [
            olympus.find_joints(jn)[0][0] for jn in ["InnerTransversalMotor_BL", "InnerTransversalMotor_BR"]
        ]

        self._front_inner_transversal_indices = [
            olympus.find_joints(jn)[0][0] for jn in ["InnerTransversalMotor_FL", "InnerTransversalMotor_FR"]
        ]

        self._back_outer_transversal_indices = [
            olympus.find_joints(jn)[0][0] for jn in ["OuterTransversalMotor_BL", "OuterTransversalMotor_BR"]
        ]
        self._front_outer_transversal_indices = [
            olympus.find_joints(jn)[0][0] for jn in ["OuterTransversalMotor_FL", "OuterTransversalMotor_FR"]
        ]

    def _parse_cfg(self, cfg: AttitudeControlInitializerCfg):
        with torch.device(self._device):
            self._latteral_limits = tuple(
                torch.tensor([l]).expand(len(self._lateral_indices)) for l in cfg.latteral_joints_limits
            )
            self._transversal_limits = tuple(
                torch.tensor([l]).expand(len(self._transversal_indices)) for l in cfg.transversal_joints_limits
            )

    def _get_random_orientation(self) -> torch.Tensor:
        probabilities = torch.tensor([0.7, 0.25, 0.025, 0.025], device=self._device)
        choice = torch.multinomial(probabilities, 1).item()

        if choice == 0:
            return random_orientation_quat(self._num_envs, device=self._device)
        elif choice == 1:
            return random_single_axis_orientation(self._num_envs, device=self._device, axis="roll")
        elif choice == 2:
            return random_single_axis_orientation(self._num_envs, device=self._device, axis="pitch")
        else:
            return random_single_axis_orientation(self._num_envs, device=self._device, axis="yaw")