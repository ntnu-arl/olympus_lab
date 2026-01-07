from typing import List, Tuple

import functools

from jaxsim import exceptions

import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js


from jaxsim.api.model import JaxSimModel
from jaxsim.api.data import JaxSimModelData
from .jaxsim_wrapper.jacobians import get_frame_jacobian, get_link_jacobian
from .jaxsim_wrapper.forward_kinematics import get_frame_transform

NUM_CKC = 4
DIM_CKC = 2
IDX_CKC = [0, 2]


@jax.jit
def vmapped_make_configuartion_consistent(
    model: JaxSimModel,
    q: jax.Array,
    *,
    max_iter: int,
    tol: float,
) -> Tuple[jax.Array, jax.Array]:

    quads = [
        ln.split("_")[-1] for ln in model.link_names() if "MotorHousing_" in ln
    ]  # get correct order of quads

    motor_housing_link_idxs = jnp.array(
        [js.link.name_to_idx(model, link_name=f"MotorHousing_{q}") for q in quads]
    )
    inner_frame_idxs = jnp.array(
        [js.frame.name_to_idx(model, frame_name=f"InnerAnkle_{q}") for q in quads]
    )
    outer_frame_idxs = jnp.array(
        [js.frame.name_to_idx(model, frame_name=f"OuterAnkle_{q}") for q in quads]
    )

    ckc_joint_indexes = jnp.array(
        [
            [
                idx
                for idx, jn in enumerate(model.joint_names())
                if jn[-2:] == q and "Lateral" not in jn
            ]
            for q in quads
        ]
    )

    return jax.vmap(
        functools.partial(
            make_configuartion_consistent,
            ckcs_joint_indexes=ckc_joint_indexes,
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
            max_iter=max_iter,
            tol=tol,
        ),
        in_axes=(None, 0),
    )(model, q)


def make_configuartion_consistent(
    model: JaxSimModel,
    q: jax.Array,
    *,
    ckcs_joint_indexes: jax.Array,
    motor_housing_link_idxs: jax.Array,
    inner_frame_idxs: jax.Array,
    outer_frame_idxs: jax.Array,
    max_iter: int,
    tol: float,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute the consistent knee positions for the leg model
    """
    alpha = 1.0

    data = js.data.JaxSimModelData.build(
        model=model,
        joint_positions=q,
        velocity_representation=jaxsim.VelRepr.Mixed,
    )

    def f(data: JaxSimModelData) -> Tuple[jax.Array, jax.Array]:

        link_transforms = js.model.forward_kinematics(model, data)
        J_full, H_Li = jaxsim.rbda.jacobian_full_doubly_left(
            model, joint_positions=data.joint_positions()
        )

        return eval_ckcs(
            model,
            data,
            J_full,
            H_Li,
            link_transforms,
            ckc_joint_indexes=ckcs_joint_indexes,
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
        )

    def step(i, data):
        C, J = f(data)
        q_dot = jax.vmap(damped_lstsq, in_axes=(0, 0, None))(
            J[:, :, 2:], -C, 1e-6
        ).flatten()  # knee joints are the last two joints of the ckc joint indexes
        q_i = data.joint_positions()
        return data.reset_joint_positions(q_i.at[12:].add(alpha * q_dot))

    data = jax.lax.fori_loop(0, 6, step, data)

    c, _ = f(data)
    converged = jnp.linalg.norm(c) < tol
    return data.joint_positions(), converged


def eval_ckcs(
    model: JaxSimModel,
    data: JaxSimModelData,
    J_full: jax.Array,
    H_Li: jax.Array,
    link_transforms: jax.Array,
    *,
    ckc_joint_indexes: jax.Array,
    motor_housing_link_idxs: jax.Array,
    inner_frame_idxs: jax.Array,
    outer_frame_idxs: jax.Array,
) -> Tuple[jax.Array, jax.Array]:

    # c = jnp.zeros((4, 2))
    # J = jnp.zeros((4, 2, 4))
    #
    # for i in range(4):
    #    c_i, J_i = eval_ckc(
    #        model,
    #        data,
    #        J_full,
    #        H_Li,
    #        link_transforms,
    #        ckc_joint_indexes[i],
    #        motor_housing_link_idxs[i],
    #        inner_frame_idxs[i],
    #        outer_frame_idxs[i],
    #    )
    #    c = c.at[i].set(c_i)
    #    J = J.at[i].set(J_i)
    # return c, J

    return jax.vmap(eval_ckc, in_axes=(None, None, None, None, None, 0, 0, 0, 0))(
        model,
        data,
        J_full,
        H_Li,
        link_transforms,
        ckc_joint_indexes,
        motor_housing_link_idxs,
        inner_frame_idxs,
        outer_frame_idxs,
    )


def eval_ckc(
    model: JaxSimModel,
    data: JaxSimModelData,
    J_full: jax.Array,
    H_Li: jax.Array,
    link_transforms: jax.Array,
    ckc_joint_indexes: jax.Array,
    motor_housing_link_idx: int,
    inner_frame_idx: int,
    outer_frame_idx: int,
) -> Tuple[jax.Array, jax.Array]:

    R_bw = jnp.transpose(link_transforms[motor_housing_link_idx][:3, :3])

    C = jnp.zeros(DIM_CKC)
    J = jnp.zeros((DIM_CKC, 4))
    sgn = 1
    for idx in [inner_frame_idx, outer_frame_idx]:
        pos_w = get_frame_transform(model, link_transforms, frame_index=idx)[:3, 3]
        pos_b = R_bw @ pos_w
        C = C.at[:].add(sgn * pos_b[jnp.array(IDX_CKC)])
        J_w = get_frame_jacobian(
            model, data, J_full, H_Li, link_transforms, frame_index=idx
        )[:3, 6 + ckc_joint_indexes]
        J_b = R_bw @ J_w
        J = J.at[:].add(sgn * J_b[jnp.array(IDX_CKC)])
        sgn *= -1
    return C, J


def damped_lstsq(A: jax.Array, b: jax.Array, damping: float) -> jax.Array:
    return (
        A.T @ (jnp.linalg.lstsq(A @ A.T - damping * jnp.eye(A.shape[1]), b)[0])
    ).flatten()
