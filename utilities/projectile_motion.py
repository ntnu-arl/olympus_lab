from torch import Tensor
import torch


def estimate_land_pos_error(base_pos: Tensor, land_pos: Tensor, base_vel: Tensor, g: float = -9.81) -> Tensor:
    d_z = -base_pos[:, 2]  
    v_z = base_vel[:, 2]
    under_root = v_z**2 + 2 * g * d_z
    
    t = torch.zeros_like(v_z)
    valid = under_root >= 0
    
    sqrt_term = under_root[valid].sqrt()
    t1 = (v_z[valid] + sqrt_term) / -g
    t2 = (v_z[valid] - sqrt_term) / -g
    
    t_both = torch.stack([t1, t2], dim=0)
    positive_mask = t_both > 0
    t_both_masked = torch.where(positive_mask, t_both, torch.tensor(float('inf'), device=t_both.device))
    t[valid] = t_both_masked.min(dim=0)[0]
    
    t[t == float('inf')] = 0.0
    
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
