from typing import Dict, Tuple

import functools
import jax
import jax.numpy as jnp
import torch

# from isaaclab.assets import Articulation

import jaxsim
import jaxsim.api as js

from jaxsim.api.model import JaxSimModel
from utilities import (
    from_jax_to_torch,
    from_torch_to_jax,
    from_torch_to_jax_dict,
    from_jax_to_torch_dict,
)
from . import floating_base
from . import ckc


# from .jax_isaac_converter import JaxIsaacConverter


class OlympusKinematics:
    def __init__(self) -> None:
        self._olympus_model = JaxSimModel.build_from_model_description(
            "submodules/olympus_usd/olympus-urdf/olympus.urdf", is_urdf=True
        )

        self._resolve_indexes()
        self._olympus_leg_model = js.model.reduce(
            self._olympus_model,
            [self._olympus_model.joint_names()[idx] for idx in self._leg_idxs],
        )
        self._resolve_leg_indexes()

        # data
        # self._model_data = js.data.JaxSimModelData.build(model=self._olympus_model)
        # self._leg_data = js.data.JaxSimModelData.build(model=self._olympus_leg_model)

    def get_consistent_joint_state(
        self,
        base_pose: torch.Tensor,
        base_lin_vel_w: torch.Tensor,
        base_ang_vel_b: torch.Tensor,
        paw_position: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the consistent joint state for the leg model

        TODO: Remember omega in correct frame
        """

        q_init = torch.zeros(
            base_pose.size(0),
            self._olympus_model.number_of_joints(),
            device=base_pose.device,
        )
        q_init[:, self._inner_transversal_idxs] = 45
        q_init[:, self._outer_transversal_idxs] = 45
        q_init[:, self._knee_idxs] = 45
        q_init = q_init.deg2rad()

        base_pos_jax, base_quat_jax, q_jax, q_dot_jax, converged = floating_base.get_consistent_joint_state(
            self._olympus_model,
            from_torch_to_jax(base_pose),
            from_torch_to_jax(base_lin_vel_w),
            from_torch_to_jax(base_ang_vel_b),
            from_torch_to_jax_dict(paw_position),
            from_torch_to_jax(q_init),
            max_iter=100,
            tol_paws=1e-4,
            tol_ckc=1e-6,
        )

        if not converged.all():
            raise Exception("Floating base IK did not converge")

        return (
            from_jax_to_torch(base_pos_jax),
            from_jax_to_torch(base_quat_jax),
            from_jax_to_torch(q_jax),
            from_jax_to_torch(q_dot_jax),
        )

    def get_consitent_knee_positions(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the consistent knee positions for the leg model

        TODO: Missleading name and should make unified interface
        """
        ##use mean of transversal joints as initial guess for knee joints ##
        mean = 0.5 * (q[:, self._outer_transversal_idxs] + q[:, self._inner_transversal_idxs])

        guess_inner = torch.where(
            mean < 5 * torch.pi / 180,
            2 * mean * torch.sign(q[:, self._inner_transversal_idxs]),
            mean,
        )
        guess_outer = torch.where(
            mean < 5 * torch.pi / 180,
            2 * mean * torch.sign(q[:, self._outer_transversal_idxs]),
            mean,
        )

        q[:, self._outer_knee_idxs] = guess_outer  # .clamp(min=20 * torch.pi / 180)
        q[:, self._inner_knee_idxs] = guess_inner  # .clamp(min=20 * torch.pi / 180)

        q[:,]
        q_jax, converged = ckc.vmapped_make_configuartion_consistent(
            self._olympus_model,
            from_torch_to_jax(q),
            max_iter=100,
            tol=1e-6,
        )

        if not converged.all():
            q = from_jax_to_torch(q_jax)
            converged = from_jax_to_torch(converged)
            not_converged = torch.nonzero(converged == 0, as_tuple=True)[0]
            for idx in not_converged:
                print(f"CKC did not converge for sample {idx}")
                print(q[idx, self._inner_transversal_idxs].rad2deg())
                print(q[idx, self._outer_transversal_idxs].rad2deg())
                print("=====================================")
            raise Exception("CKC IK did not converge")

        return from_jax_to_torch(q_jax)

    def get_consistent_joint_vel(
        self,
        base_pose: torch.Tensor,
        base_lin_vel_w: torch.Tensor,
        base_ang_vel_b: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the consistent joint state for the leg model
        """

        q_dot_jax = floating_base.get_consistent_joint_vel(
            self._olympus_model,
            from_torch_to_jax(base_pose),
            from_torch_to_jax(base_lin_vel_w),
            from_torch_to_jax(base_ang_vel_b),
            from_torch_to_jax(joint_pos),
        )

        return from_jax_to_torch(q_dot_jax)

    def _resolve_indexes(self) -> None:
        self._lateral_motor_idxs = [
            idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "LateralMotor" in jn
        ]
        self._knee_idxs = [idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "Knee" in jn]

        self._leg_idxs = [idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "LateralMotor" not in jn]

        self._outer_transversal_idxs = [
            idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "OuterTransversal" in jn
        ]
        self._inner_transversal_idxs = [
            idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "InnerTransversal" in jn
        ]
        self._outer_knee_idxs = [idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "OuterKnee" in jn]
        self._inner_knee_idxs = [idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "InnerKnee" in jn]

    def _resolve_leg_indexes(self) -> None:
        self._leg_lateral_motor_idxs = [
            idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "LateralMotor" in jn
        ]
        self._leg_knee_idxs = [idx for idx, jn in enumerate(self._olympus_model.joint_names()) if "Knee" in jn]

    @property
    def num_joints(self) -> int:
        return self._olympus_model.number_of_joints()

    @property
    def dim_q(self) -> int:
        return 7 + self.num_joints

    @property
    def dim_v(self) -> int:
        return 6 + self.num_joints


#
# frame_names = leg_model.frame_names()
# print(frame_names)
# print(leg_model.dofs())
# print(leg_model.joint_names())
#
# idx = js.frame.name_to_idx(leg_model, frame_name="InnerAnkle_FR_joint")
# print(idx)
# jac = js.frame.jacobian(
#    leg_model,
#    leg_data,
#    frame_index=idx,
#    output_vel_repr=jaxsim.VelRepr.Inertial,
# )
# print(jac)
# print(model.floating_base())


if __name__ == "__main__":
    kinematics = OlympusKinematics()
