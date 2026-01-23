import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js

from jaxsim.math import Adjoint, Transform

from jaxsim.api.model import JaxSimModel
from jaxsim.api.data import JaxSimModelData

from . import forward_kinematics


def get_link_jacobian(
    model: JaxSimModel,
    data: JaxSimModelData,
    B_J_full_WX_B: jax.Array,
    B_H_Li: jax.Array,
    *,
    link_index: int,
) -> jax.Array:
    """
    get the link jacobian for a given link index using the full jacobian
    """

    # Compute the actual doubly-left free-floating jacobian of the link.
    κb = model.kin_dyn_parameters.support_body_array_bool[link_index]
    B_J_WL_B = jnp.hstack([jnp.ones(5), κb]) * B_J_full_WX_B

    # Adjust the input representation such that `J_WL_I @ I_ν`.

    W_R_B = data.base_orientation(dcm=True)
    BW_H_B = jnp.eye(4).at[0:3, 0:3].set(W_R_B)
    B_X_BW = Adjoint.from_transform(transform=BW_H_B, inverse=True)
    B_J_WL_I = B_J_WL_B @ jax.scipy.linalg.block_diag(  # noqa: F841
        B_X_BW, jnp.eye(model.dofs())
    )

    B_H_L = B_H_Li[link_index]

    W_H_B = data.base_transform()
    W_H_L = W_H_B @ B_H_L
    LW_H_L = W_H_L.at[0:3, 3].set(jnp.zeros(3))
    LW_H_B = LW_H_L @ jaxsim.math.Transform.inverse(B_H_L)
    LW_X_B = Adjoint.from_transform(transform=LW_H_B)
    LW_J_WL_I = LW_X_B @ B_J_WL_I

    return LW_J_WL_I


def _get_body_link_jacobian(
    model: JaxSimModel,
    data: JaxSimModelData,
    B_J_full_WX_B: jax.Array,
    B_H_Li: jax.Array,
    *,
    link_index: int,
) -> jax.Array:
    pass
    # Compute the actual doubly-left free-floating jacobian of the link.
    κb = model.kin_dyn_parameters.support_body_array_bool[link_index]
    B_J_WL_I = jnp.hstack([jnp.ones(5), κb]) * B_J_full_WX_B

    B_H_L = B_H_Li[link_index]
    L_X_B = Adjoint.from_transform(transform=B_H_L, inverse=True)
    return L_X_B @ B_J_WL_I


def get_frame_jacobian(
    model: JaxSimModel,
    data: JaxSimModelData,
    B_J_full_WX_B: jax.Array,
    B_H_Li: jax.Array,
    link_transforms: jax.Array,
    *,
    frame_index: int,
) -> jax.Array:

    # Get the index of the parent link.
    L = js.frame.idx_of_parent_link(model=model, frame_index=frame_index)

    # Compute the Jacobian of the parent link using body-fixed output representation.
    L_J_WL = _get_body_link_jacobian(model, data, B_J_full_WX_B, B_H_Li, link_index=L)

    # Adjust the output representation.

    W_H_L = link_transforms[L]
    W_H_F = forward_kinematics.get_frame_transform(
        model, link_transforms, frame_index=frame_index
    )
    F_H_L = Transform.inverse(W_H_F) @ W_H_L
    FW_H_F = W_H_F.at[0:3, 3].set(jnp.zeros(3))
    FW_H_L = FW_H_F @ F_H_L
    FW_X_L = Adjoint.from_transform(transform=FW_H_L)
    return FW_X_L @ L_J_WL
