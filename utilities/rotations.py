from typing import Tuple
from torch import Tensor
import torch

import isaaclab.utils.math as math_utils

from . import sampling as sampling_utils


def quat_to_euler_zyx(q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert quaternion to euler angles in XYZ order
    """
    qw, qx, qy, qz = q.unbind(-1)

    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = -torch.pi / 2 + 2 * torch.atan2(
        torch.sqrt(1 + 2 * (qw * qw - qx * qx)), torch.sqrt(1 - 2 * (qw * qy + qx * qz))
    )
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    return roll, pitch, yaw


def random_orientation_euler(num_envs: int, euler_limits: Tuple[Tensor, Tensor]) -> Tensor:
    euler = torch.split(sampling_utils.uniform_sample(*euler_limits, num_envs).deg2rad(), 1, dim=1)
    return math_utils.normalize(math_utils.quat_from_euler_xyz(*euler).squeeze(1))


def random_orientation_axis(num_envs: int, device: str) -> Tensor:
    # random axis
    phi = torch.rand(num_envs, device=device) * torch.pi
    theta = torch.rand(num_envs, device=device) * 2 * torch.pi
    s_phi = torch.sin(phi)
    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)
    axis = torch.stack((s_phi * c_theta, s_phi * s_theta, torch.cos(phi)), dim=1)
    # random angle
    angle = torch.rand(num_envs, device=device) * 2 * torch.pi - torch.pi
    return math_utils.quat_from_angle_axis(angle, axis)


def random_orientation_quat(num_envs: int, device: str) -> Tensor:
    
    i, j, k = torch.rand(num_envs, 3, device=device).unbind(dim=-1)
    x = torch.sqrt(1 - i) * torch.sin(2 * torch.pi * j)
    y = torch.sqrt(1 - i) * torch.cos(2 * torch.pi * j)
    z = torch.sqrt(i) * torch.sin(2 * torch.pi * k)
    w = torch.sqrt(i) * torch.cos(2 * torch.pi * k)
    return torch.stack((w, x, y, z), dim=-1)


def predefined_orientation(num_envs: int, device: str) -> Tensor:
    
    # defined orientation
    roll = torch.zeros(num_envs, device=device)
    pitch = torch.zeros(num_envs, device=device)
    yaw = torch.zeros(num_envs, device=device)
    # add a random angle from -150 to 150 degrees
    random_added_angle_deg = (2 * torch.rand(num_envs, device=device) - 1) * 179
    roll = roll - 0.0 * torch.pi / 180
    pitch = pitch + 0.0 * torch.pi / 180
    yaw = yaw + random_added_angle_deg * torch.pi / 180
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def random_single_axis_orientation(
    num_envs: int,
    device: str,
    angle_limit_deg: float = 179.0,
    axis: str = "random",  # can be "roll", "pitch", "yaw", or "random"
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
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)
