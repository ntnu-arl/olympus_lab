from utilities import from_jax_to_torch, from_torch_to_jax

from torch import Tensor
from jax import Array

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_rotate
import isaacsim.core.utils.prims as prims_utils

import jax
import jax.numpy as jnp
import jaxsim
import jaxsim.api as js

from jaxsim.api.model import JaxSimModel


class JaxIsaacConverter:
    def __init__(
        self,
        jaxsim_model: JaxSimModel,
        isaac_model: Articulation,
    ) -> None:
        self._num_joints = isaac_model.num_joints
        self._jax_id_to_isaac_id = []
        self._isaac_id_to_jax_id = []
        self._joint_to_sgn = {}

        self._T_isaac_jax = torch.zeros(
            self._num_joints, self._num_joints, device="cuda:0"
        )  # Transformation matrix from Jax to Isaac
        self._T_jax_isaac = torch.zeros(
            self._num_joints, self._num_joints, device="cuda:0"
        )  # Transformation matrix from Isaac to Jax

        for id_jax, jn in enumerate(jaxsim_model.joint_names()):
            id_isaac = isaac_model.find_joints(jn)[0][0]
            rot_axis_isaac = get_rot_axis_isaac(jn)
            rot_axis_jax = from_jax_to_torch(get_rot_axis_jax(jn, jaxsim_model))
            dot = torch.dot(rot_axis_isaac, rot_axis_jax)
            assert torch.abs(dot).allclose(
                torch.tensor(1.0)
            ), "Missmacth in rotaiton axis between URDF and USD"
            if torch.sign(dot) == -1:
                print(f"joint name {jn} has different sign")

            if id_isaac != id_jax:
                print(f"joint name {jn} has different id")
            self._T_isaac_jax[id_jax, id_isaac] = torch.sign(dot)

        self._T_jax_isaac[:] = self._T_isaac_jax.T  # inverse of T_isaac_jax

    def isaac_to_jax(self, q_isaac: Tensor) -> Array:
        return from_torch_to_jax(torch.bmm(self._T_jax_isaac.unsqueeze(0), q_isaac))

    def jax_to_isaac(self, q_jax: Array) -> Tensor:
        return torch.bmm(self._T_isaac_jax.unsqueeze(0), from_jax_to_torch(q_jax))


def get_rot_axis_isaac(joint_name: str) -> Tensor:
    prim_path = prims_utils.find_matching_prim_paths(
        f"/World/envs/env_0/*/*/{joint_name}"
    )[0]
    rot0_isaac = torch.tensor(
        prims_utils.get_prim_attribute_value(
            prim_path, "physics:localRot0", fabric=True
        ),
        device="cuda:0",
    )[[1, 2, 3, 0]]

    rot_axis_isaac = torch.zeros(3, device="cuda:0")
    match prims_utils.get_prim_attribute_value(prim_path, "physics:axis", fabric=True):
        case "X":
            rot_axis_isaac[0] = 1
        case "Y":
            rot_axis_isaac[1] = 1
        case "Z":
            rot_axis_isaac[2] = 1
        case _:
            raise ValueError("Invalid axis")
    rot_axis_isaac = quat_rotate(
        rot0_isaac.unsqueeze(0), rot_axis_isaac.unsqueeze(0)
    ).squeeze()

    return rot_axis_isaac


def get_rot_axis_jax(joint_name: str, jax_model: JaxSimModel) -> Array:
    axis = jax_model.kin_dyn_parameters.joint_model.joint_axis[
        jax_model.joint_names().index(joint_name)
    ]
    return jnp.array(axis.axis, dtype=jnp.float32)
