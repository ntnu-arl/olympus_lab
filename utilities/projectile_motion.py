from torch import Tensor
import torch


def estimate_land_pos_error(base_pos: Tensor, land_pos: Tensor, base_vel: Tensor, g: float = -9.81) -> Tensor:
    d_vec = land_pos - base_pos
    d_z = d_vec[:, 2]
    v_z = base_vel[:, 2]
    under_root = v_z**2 + 2 * g * d_z
    t = torch.where((under_root >= 0).logical_and(v_z > 0), (v_z + under_root.sqrt()) / -g, 0.0)
    land_pos_est = base_pos[:, :2] + base_vel[:, :2] * t.unsqueeze(1)
    return land_pos_est - land_pos[:, :2]


def estimate_jump_height(base_pos: Tensor, base_vel: Tensor, g: float = -9.81) -> Tensor:
    delta_h = torch.where(
        base_vel[:, 2] > 0,
        (base_vel[:, 2] ** 2) / (-2 * g),
        0.0,
    )

    return base_pos[:, 2] + delta_h


if __name__ == "__main__":
    base_pos = torch.tensor([[0.0, 0.0, 0.0]])
    land_pos = torch.tensor([[1.0, 0.0, 0.0]])
    base_vel = torch.tensor([[1.0, 0.0, -10.0]])
    g = -3.72
    land_pos_error = estimate_land_pos_error(base_pos, land_pos, base_vel, g)
    print(land_pos_error)
