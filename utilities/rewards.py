from torch import Tensor
import torch


def exp_kernel(x: Tensor, sigma: float | Tensor) -> Tensor:

    if isinstance(sigma, float):
        return torch.exp(x.square().sum(dim=-1) / (2 * sigma**2))

    return torch.exp(-0.5 * (x / (sigma.unsqueeze(0))).square().sum(dim=-1))
