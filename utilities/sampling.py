import torch


def uniform_sample(
    lower: torch.Tensor, upper: torch.Tensor, batch_size: int
) -> torch.Tensor:
    shape = tuple([batch_size]) + lower.shape
    return lower.unsqueeze(0) + torch.rand(shape, device=lower.device) * (
        upper - lower
    ).unsqueeze(0)
