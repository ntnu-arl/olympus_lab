# import jax.dlpack
# import torch.utils.dlpack

import jax
import torch

"""
Utility functions to transfer jax arrays to torch tensors and vice versa without unloading the data from the GPU
"""


def from_jax_to_torch(jax_array: jax.Array) -> torch.Tensor:
    return torch.from_dlpack(jax.dlpack.to_dlpack(jax_array)).float()


def from_torch_to_jax(x_torch: torch.Tensor) -> jax.Array:
    x_torch = x_torch.double()
    return jax.dlpack.from_dlpack(x_torch)
    


def from_jax_to_torch_dict(jax_dict: dict) -> dict:
    torch_dict = {}

    for key, value in jax_dict.items():
        if isinstance(value, dict):
            torch_dict[key] = from_jax_to_torch_dict(value)
        else:
            torch_dict[key] = from_jax_to_torch(value)

    return torch_dict


def from_torch_to_jax_dict(torch_dict: dict) -> dict:
    jax_dict = {}

    for key, value in torch_dict.items():
        if isinstance(value, dict):
            jax_dict[key] = from_torch_to_jax_dict(value)
        else:
            jax_dict[key] = from_torch_to_jax(value)

    return jax_dict


if __name__ == "__main__":
    jax_device = jax.devices()
    jax.config.update("jax_enable_x64", True)
    print(jax_device)

    jax_array = jax.numpy.array([1, 2, 3, 4, 5])
    torch_tensor = torch.tensor([1, 2, 3, 4], device="cuda:0")

    torch_from_jax = from_jax_to_torch(jax_array)
    jax_from_torch = from_torch_to_jax(torch_tensor)

    jax_dict = {"key_1": jax_array, "key_2": {"nested_key": jax_array}}
    torch_dict = {"key_1": torch_tensor, "key_2": {"nested_key": torch_tensor}}

    jax_dict_2 = from_torch_to_jax_dict(torch_dict)
    torch_dict_2 = from_jax_to_torch_dict(jax_dict)
    print(jax_dict_2)
    print(torch_dict_2)

    bs = 4096
    torch_tensor = torch.zeros(bs, 20).cuda()
    import time

    start = time.time()
    for i in range(1000):
        a = from_torch_to_jax(torch_tensor)
        b = from_jax_to_torch(a)
    end = time.time()

    print("Time taken: ", (end - start) / 1000)
