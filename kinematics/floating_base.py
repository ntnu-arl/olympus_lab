from typing import List, Tuple, Dict
import functools

import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.api as js

from jaxsim.api.model import JaxSimModel
from jaxsim.api.data import JaxSimModelData

from jaxlie import SO3
from .ckc import eval_ckcs
from . import jaxsim_wrapper


@jax.jit
def get_consistent_joint_state(
    model: JaxSimModel,
    base_pose: jax.Array,
    base_lin_vel_w: jax.Array,
    base_ang_vel_b: jax.Array,
    paw_pos: Dict[str, jax.Array],
    q_init: jax.Array,
    *,
    max_iter: int,
    tol_paws: float,
    tol_ckc: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    caution: had to comment out line 32 to 37 in jaxsim/rbda/forward_kinematics.py to make this work
    The utils.process_inputs function makes compilation and execution very slow
    """
    quads = [ln.split("_")[-1] for ln in model.link_names() if "MotorHousing_" in ln]  # get correct order of quads

    motor_housing_link_idxs = jnp.array([js.link.name_to_idx(model, link_name=f"MotorHousing_{q}") for q in quads])
    inner_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"InnerAnkle_{q}") for q in quads])
    outer_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"OuterAnkle_{q}") for q in quads])

    paw_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"Paw_{q}") for q in quads])

    leg_joint_indexes = jnp.array([[idx for idx, jn in enumerate(model.joint_names()) if jn[-2:] == q] for q in quads])

    paw_pos_array = jnp.stack([paw_pos[f"Paw_{q}"] for q in quads], axis=1)

    return jax.vmap(
        functools.partial(
            _get_consistent_joint_state,
            paw_frame_idxs=paw_frame_idxs,
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
            leg_joint_indexes=leg_joint_indexes,
            max_iter=max_iter,
            tol_paws=tol_paws,
            tol_ckc=tol_ckc,
        ),
        in_axes=(None, 0, 0, 0, 0, 0),
    )(model, base_pose, base_lin_vel_w, base_ang_vel_b, paw_pos_array, q_init)


@jax.jit
def get_consistent_joint_vel(
    model: JaxSimModel,
    base_pose: jax.Array,
    base_lin_vel_w: jax.Array,
    base_ang_vel_b: jax.Array,
    joint_pos: jax.Array,
) -> Tuple[jax.Array]:
    """
    caution: had to comment out line 32 to 37 in jaxsim/rbda/forward_kinematics.py to make this work
    The utils.process_inputs function makes compilation and execution very slow
    """
    quads = [ln.split("_")[-1] for ln in model.link_names() if "MotorHousing_" in ln]  # get correct order of quads

    motor_housing_link_idxs = jnp.array([js.link.name_to_idx(model, link_name=f"MotorHousing_{q}") for q in quads])
    inner_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"InnerAnkle_{q}") for q in quads])
    outer_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"OuterAnkle_{q}") for q in quads])

    paw_frame_idxs = jnp.array([js.frame.name_to_idx(model, frame_name=f"Paw_{q}") for q in quads])

    leg_joint_indexes = jnp.array([[idx for idx, jn in enumerate(model.joint_names()) if jn[-2:] == q] for q in quads])

    return jax.vmap(
        functools.partial(
            _get_consistent_joint_vel,
            paw_frame_idxs=paw_frame_idxs,
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
            leg_joint_indexes=leg_joint_indexes,
        ),
        in_axes=(None, 0, 0, 0, 0),
    )(model, base_pose, base_lin_vel_w, base_ang_vel_b, joint_pos)


def _get_consistent_joint_state(
    model: JaxSimModel,
    base_pose: jax.Array,
    base_lin_vel_w: jax.Array,
    base_ang_vel_b: jax.Array,
    paw_pos_array: jax.Array,
    q_init: jax.Array,
    *,
    paw_frame_idxs: jax.Array,
    motor_housing_link_idxs: jax.Array,
    inner_frame_idxs: jax.Array,
    outer_frame_idxs: jax.Array,
    leg_joint_indexes: jax.Array,
    max_iter: int,
    tol_paws: float,
    tol_ckc: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:

    data = js.data.JaxSimModelData.build(
        model=model,
        base_position=base_pose[:3],
        base_quaternion=base_pose[3:],
        joint_positions=q_init,
        base_linear_velocity=base_lin_vel_w,
        base_angular_velocity=base_ang_vel_b,
        velocity_representation=jaxsim.VelRepr.Mixed,
    )

    alpha = 1.0
    dq_max = 20 * jnp.pi / 180
    w_diag = jnp.ones(6 + model.number_of_joints())
    w_diag = w_diag.at[0].set(5e3)
    w_diag = w_diag.at[1].set(1e2)
    w_diag = w_diag.at[2].set(1e6)
    w_diag = w_diag.at[3:6].set(1e2)
    w_diag = w_diag.at[4].set(5e2)
    w_diag = w_diag.at[6 : 6 + 4].set(1e2)
    base_idxs = jnp.arange(6)

    def f(data: JaxSimModelData) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        link_transforms = jaxsim_wrapper.forward_kinematics_model(
            model,
            base_position=data.base_position(),
            base_quaternion=data.base_orientation(dcm=False),
            joint_positions=data.joint_positions(),
        )
        J_full, H_Li = jaxsim.rbda.jacobian_full_doubly_left(model, joint_positions=data.joint_positions())

        c_contact, J_contact = eval_contacts(
            model,
            data,
            paw_pos_array,
            J_full,
            H_Li,
            link_transforms,
            paw_frame_indexes=paw_frame_idxs,
            leg_joint_indexes=leg_joint_indexes,
        )

        c_ckc, J_ckc = eval_ckcs(
            model,
            data,
            J_full,
            H_Li,
            link_transforms,
            ckc_joint_indexes=leg_joint_indexes[:, 1:],
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
        )
        # pad a zero wor motor housing joint
        J_ckc_pad = (
            jnp.zeros((J_ckc.shape[0], J_ckc.shape[1], J_contact.shape[2])).at[:, :, -J_ckc.shape[-1] :].set(J_ckc)
        )
        return c_contact, J_contact, c_ckc, J_ckc_pad

    def step(i, data: JaxSimModelData) -> JaxSimModelData:
        c_contact, J_contact, c_ckc, J_ckc = f(data)
        J = jnp.zeros((20, 26))
        C = jnp.zeros(20)
        row_idx_start = 0
        dim_ckc = c_ckc.shape[1]
        dim_contact = c_contact.shape[1]
        for i in range(c_contact.shape[0]):
            idxs = jnp.concat([base_idxs, 6 + leg_joint_indexes[i]])
            row_idx_end = row_idx_start + dim_contact
            J = J.at[row_idx_start:row_idx_end, idxs].set(J_contact[i])
            C = C.at[row_idx_start:row_idx_end].set(c_contact[i])
            row_idx_start = row_idx_end
            row_idx_end = row_idx_start + dim_ckc
            J = J.at[row_idx_start:row_idx_end, idxs].set(J_ckc[i])
            C = C.at[row_idx_start:row_idx_end].set(c_ckc[i])
            row_idx_start = row_idx_end

        dq = alpha * weighted_lstsq(J, -C, w_diag)
        # dq = alpha * jnp.linalg.lstsq(J[:, 6:], -C)[0]
        dq = jnp.clip(dq, -dq_max, dq_max)
        data = data.reset_base_position(data.base_position() + dq[:3])
        data = data.reset_base_quaternion(SO3(data.base_orientation()).multiply(SO3.exp(dq[3:6])).wxyz)

        return data.reset_joint_positions(data.joint_positions() + dq[-model.number_of_joints() :])

    data: JaxSimModelData = jax.lax.fori_loop(0, 12, step, data)
    q_dot = jnp.zeros_like(data.joint_positions())

    c_contact, J_contact, c_ckc, J_ckc = f(data)
    J_stack = jnp.concatenate([J_contact, J_ckc], axis=1)

    q_dot_stack = jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0))(
        J_stack[:, :, 6:], -J_stack[:, :, :6] @ data.base_velocity()
    )[0]
    q_dot = jnp.zeros_like(data.joint_positions())
    for i in range(q_dot_stack.shape[0]):
        q_dot = q_dot.at[leg_joint_indexes[i]].set(q_dot_stack[i])

    converged = jnp.logical_and(
        jnp.linalg.norm(c_contact, ord=jnp.inf) < tol_paws,
        jnp.linalg.norm(c_ckc, ord=jnp.inf) < tol_ckc,
    )

    return (
        data.base_position(),
        data.base_orientation(),
        data.joint_positions(),
        q_dot,
        converged,
    )


def _get_consistent_joint_vel(
    model: JaxSimModel,
    base_pose: jax.Array,
    base_lin_vel_w: jax.Array,
    base_ang_vel_b: jax.Array,
    joint_pos: jax.Array,
    *,
    paw_frame_idxs: jax.Array,
    motor_housing_link_idxs: jax.Array,
    inner_frame_idxs: jax.Array,
    outer_frame_idxs: jax.Array,
    leg_joint_indexes: jax.Array,
) -> jax.Array:

    data = js.data.JaxSimModelData.build(
        model=model,
        base_position=base_pose[:3],
        base_quaternion=base_pose[3:],
        joint_positions=joint_pos,
        base_linear_velocity=base_lin_vel_w,
        base_angular_velocity=base_ang_vel_b,
        velocity_representation=jaxsim.VelRepr.Mixed,
    )

    def f(data: JaxSimModelData) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        link_transforms = jaxsim_wrapper.forward_kinematics_model(
            model,
            base_position=data.base_position(),
            base_quaternion=data.base_orientation(dcm=False),
            joint_positions=data.joint_positions(),
        )
        J_full, H_Li = jaxsim.rbda.jacobian_full_doubly_left(model, joint_positions=data.joint_positions())

        c_contact, J_contact = eval_contacts(
            model,
            data,
            jnp.zeros((4, 3)),
            J_full,
            H_Li,
            link_transforms,
            paw_frame_indexes=paw_frame_idxs,
            leg_joint_indexes=leg_joint_indexes,
        )

        c_ckc, J_ckc = eval_ckcs(
            model,
            data,
            J_full,
            H_Li,
            link_transforms,
            ckc_joint_indexes=leg_joint_indexes[:, 1:],
            motor_housing_link_idxs=motor_housing_link_idxs,
            inner_frame_idxs=inner_frame_idxs,
            outer_frame_idxs=outer_frame_idxs,
        )
        # pad a zero wor motor housing joint
        J_ckc_pad = (
            jnp.zeros((J_ckc.shape[0], J_ckc.shape[1], J_contact.shape[2])).at[:, :, -J_ckc.shape[-1] :].set(J_ckc)
        )
        return c_contact, J_contact, c_ckc, J_ckc_pad

    c_contact, J_contact, c_ckc, J_ckc = f(data)
    J_stack = jnp.concatenate([J_contact, J_ckc], axis=1)

    q_dot_stack = jax.vmap(jnp.linalg.lstsq, in_axes=(0, 0))(
        J_stack[:, :, 6:], -J_stack[:, :, :6] @ data.base_velocity()
    )[0]
    q_dot = jnp.zeros_like(data.joint_positions())
    for i in range(q_dot_stack.shape[0]):
        q_dot = q_dot.at[leg_joint_indexes[i]].set(q_dot_stack[i])

    return q_dot


def eval_contacts(
    model: JaxSimModel,
    data: JaxSimModelData,
    paw_pos: jax.Array,
    J_full: jax.Array,
    H_Li: jax.Array,
    link_transforms: jax.Array,
    *,
    paw_frame_indexes: jax.Array,
    leg_joint_indexes: jax.Array,
) -> Tuple[jax.Array, jax.Array]:

    return jax.vmap(eval_contact, in_axes=(None, None, None, None, None, 0, 0, 0))(
        model,
        data,
        J_full,
        H_Li,
        link_transforms,
        paw_pos,
        paw_frame_indexes,
        leg_joint_indexes,
    )


def eval_contact(
    model: JaxSimModel,
    data: JaxSimModelData,
    J_full: jax.Array,
    H_Li: jax.Array,
    link_transforms: jax.Array,
    paw_pos: jax.Array,
    paw_frame_index: int,
    leg_joint_indicies: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    c = jaxsim_wrapper.get_frame_transform(model, link_transforms, frame_index=paw_frame_index)[:3, 3]
    c = c.at[:].add(-paw_pos)
    J = jaxsim_wrapper.get_frame_jacobian(model, data, J_full, H_Li, link_transforms, frame_index=paw_frame_index)[
        :3, :
    ]
    J = jnp.concatenate([J[:, :6], J[:, 6 + leg_joint_indicies]], axis=1)
    return c, J


def weighted_lstsq(A: jax.Array, b: jax.Array, w_diag: jax.Array) -> jax.Array:
    W = jnp.diag(w_diag)
    W_inv = jnp.diag(1 / w_diag)

    return (W_inv @ A.T @ (jnp.linalg.lstsq(A @ W_inv @ A.T, b)[0])).flatten()
