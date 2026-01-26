# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Jørgen Anker Olsen, NTNU Autonomous Robots Lab, 2026

from __future__ import annotations
from email.mime import base
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
)
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObject, Articulation

import isaaclab.utils.math as math_utils

from robot import OlympusConfig, cube_mars
from .walk_initializer import WalkInitializerCfg

from control import MotorCommandFilterCfg
from robot import cube_mars


DEG2RAD = torch.pi / 180.0

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.utils.noise as noise

if TYPE_CHECKING:
    from .walk_env import WalkEnv


def sample_command(env: WalkEnv, env_ids: torch.Tesnor) -> None:
    env._sample_commands(env_ids)


def apply_external_force_torque(
    env: WalkEnv,
    env_ids: torch.Tensor,
    probability: float,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    should_apply = (torch.rand(len(env_ids), 1, 1, device=asset.device) < probability).float()

    forces = (
        math_utils.sample_uniform(
            *(torch.tensor(r, device=asset.device).view(1, 1, 3) for r in force_range), size, asset.device
        )
        * should_apply
    )

    torques = (
        math_utils.sample_uniform(
            *(torch.tensor(r, device=asset.device).view(1, 1, 3) for r in torque_range), size, asset.device
        )
        * should_apply
    )
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`

    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.95),
            "dynamic_friction_range": (0.7, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*MotorHousing.*", ".*Thigh.*", ".*Shank.*", ".*Paw.*"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    body_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "com_range": {"x": (-0.02, 0.08), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )


    motor_housing_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*MotorHousing.*"]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    link_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*Thigh.*", ".*Shank.*"]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.02, 0.02)},
        },
    )

    paw_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*Paw.*"]),
            "com_range": {"x": (-0.0025, 0.0025), "y": (-0.0025, 0.0025), "z": (-0.0025, 0.0025)},
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    motor_model = EventTerm(
        func=cube_mars.randomize_cubemars_model,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "no_load_speed_distribution_params": (0.6, 1.2),
            "cutoff_speed_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "friction_distribution_params": (0.01, 0.04),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.05, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "probability": 0.05,
            "force_range": ([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
            "torque_range": ([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]),
        },
    )

    # resample command interval
    sample_commands = EventTerm(
        func=sample_command,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={},
    )

# configuration for the Olympus robot
_OLYMPUS_CONFIG = OlympusConfig(
    actuators={
        "lateral_motors": cube_mars.get_AK809_cfg(
            joint_names_expr=["LateralMotor.*"],
            kp=17.0,
            kd=0.9 * 2.3,
            safe_effort_limit=10.0, # Nm
            safe_velocity=2500.0, # deg/s
            min_delay=0,
            max_delay=0,
        ),
        "transversal_motors": cube_mars.get_AK7010_cfg(
            joint_names_expr=[".*TransversalMotor.*"],
            kp=17.0,
            kd=0.4 * 2.,
            safe_effort_limit=10.0, # Nm
            safe_velocity=2500.0, # deg/s
            min_delay=0,
            max_delay=0,
        ),
        "knees_damping": ImplicitActuatorCfg(
            joint_names_expr=[".*Knee.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.1,
            friction=0.0,
        ),
    },
    motor_command_filter=MotorCommandFilterCfg(
        # filter parameters - degrees
        lateral_motor_joint_limits=(-50.0, 180),
        transversal_motor_joint_limits=(-30.0, 140),
        transversal_joint_sum_limits=(0, 220),
        filter_threshold=6 / 60,
        adaptive_filter=False,
        # angle threshold parameters
        upper_angle_threshold=15.0,
        lower_angle_threshold=10.0,
        upper_sum_threshold=30.0,
        lower_sum_threshold=30.0,
        velocity_threshold=60.0,
    ),
)


@configclass
class WalkEnvCfg(DirectRLEnvCfg):
    ''' configuration class for the Walk environment. '''
    @configclass
    class ObservationNoiseCfg:
        body_lin_vel_noise = 0.1 # m/s
        body_ang_vel_noise = 0.1 # rad/s
        projected_gravity_noise = 0.05
        joint_pos_noise_deg = 3 # degrees
        joint_vel_noise_deg = 10 # degrees/s
        joint_pos_bias_deg = 2.0 # degrees

    @configclass
    class LatencyCfg:
        """Configuration for the latency in the environment."""
        min_observation_time_lag: int = 0
        max_observation_time_lag: int = 1
        min_action_time_lag: int = 0
        max_action_time_lag: int = 1 # in number of simulation steps

    # env
    episode_length_s = 20.0
    decimation = 8
    lateral_action_scale = 60 * DEG2RAD
    transversal_action_scale = 60 * DEG2RAD
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation,
        gravity=(0.0, 0.0, -3.71),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    # robot
    robot: OlympusConfig = _OLYMPUS_CONFIG

    # initializer
    initializer: WalkInitializerCfg = WalkInitializerCfg(
        base_pos_limits=([-0.1, -0.05, 0.30], [0.1, 0.05, 0.5]), base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Olympus/.*", history_length=decimation, track_air_time=True
    )
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    latency: LatencyCfg = LatencyCfg()

    # reward scales walk
    lin_vel_reward_scale = 2.5
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.03
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -100.0
    flat_orientation_reward_scale = -50.0
    standing_joint_pos_reward_scale = 1.5
    lateral_motor_reward_scale = 0.18
    transverse_motor_reward_scale = 0.08
    rapid_stepping_penalty_scale = -0.2

    # experimental reward
    paw_drag_reward_scale = 0  # -0.5
    lateral_symmetry_reward_scale = 0.0
    base_height_reward_scale = 0.0

    # regularization rewards
    action_clip_reward_scale =  -1e-3
    jerk_reward_scale = -0.005
    joint_torque_reward_scale = -1.5e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.06

    paw_forces_reward_scale = -0.000
    contact_change_reward_scale = -1000.0
    