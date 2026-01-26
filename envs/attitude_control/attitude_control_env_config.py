# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Jørgen Anker Olsen, NTNU Autonomous Robots Lab, 2026

from __future__ import annotations

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


from robot import OlympusConfig, cube_mars
from .attitude_control_initializer import AttitudeControlInitializerCfg

from control import MotorCommandFilterCfg
from robot import cube_mars

DEG2RAD = torch.pi / 180.0

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class EventCfg:
    """Configuration for randomization."""
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*MotorHousing.*", ".*Thigh.*", ".*Shank.*", ".*Paw.*"]),
            "mass_distribution_params": (0.8, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
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
            "no_load_speed_distribution_params": (0.8, 1.2),
            "cutoff_speed_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

_OLYMPUS_CONFIG = OlympusConfig(
    actuators={
        "lateral_motors": cube_mars.get_AK809_cfg(
            joint_names_expr=["LateralMotor.*"], kp=17.0, kd=0.9*1.4, safe_effort_limit=8.0, safe_velocity=800.0
        ),
        "transversal_motors": cube_mars.get_AK7010_cfg(
            joint_names_expr=[".*TransversalMotor.*"], kp=17.0, kd=0.4*1.3, safe_effort_limit=8.0, safe_velocity=800.0
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
        # motor command filter settings
        lateral_motor_joint_limits=(-20.0, 90.0),
        transversal_motor_joint_limits=(-30.0, 180),
        transversal_joint_sum_limits=(0, 220),
        filter_threshold=5 / 60,
        adaptive_filter=False,
        # angle threshold parameters
        upper_angle_threshold = 15.0,
        lower_angle_threshold = 10.0,
        upper_sum_threshold = 20.0,
        lower_sum_threshold = 20.0,
        velocity_threshold = 60.0,
    ),
)


@configclass
class AttitudeControlEnvCfg(DirectRLEnvCfg):
    """Configuration for the Attitude Control Environment."""
    @configclass
    class ObservationNoiseCfg:
        """Configuration for observation noise."""
        orientation_quat_noise_deg = 2
        body_ang_vel_noise_deg = 4
        joint_pos_noise_deg = 2
        joint_vel_noise_deg = 10
        joint_pos_bias_deg = 1.0

    @configclass
    class LatencyCfg:
        """Configuration for the latency in the environment."""
        min_observation_time_lag: int = 0
        max_observation_time_lag: int = 1
        min_action_time_lag: int = 0
        max_action_time_lag: int = 1


    # env
    episode_length_s = 5.0
    decimation = 8
    action_scale = 90 * DEG2RAD
    action_space = 12
    observation_space = 43
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 1/ 480, # need smaller dt for stability with closed kinemati chain
        render_interval=decimation,
        gravity=(0.0, 0.0, 0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # physx=PhysxCfg(solver_type=1), # 0: PGS, 1: TGS is default
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: OlympusConfig = _OLYMPUS_CONFIG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Olympus/.*",
        history_length=decimation*2 + 1,
        track_air_time=True,
    )

    # initializer - set initialisation ranges 
    attitude_control_initializer: AttitudeControlInitializerCfg = (
        AttitudeControlInitializerCfg(
            latteral_joints_limits=(-20,90), transversal_joints_limits=(-20, 90)
        )
    )

    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    latency: LatencyCfg = LatencyCfg()

    # reward scales
    attitude_error_small_reward_scale = 2.5
    attitude_error_large_reward_scale = 1.5
    angvel_reward_scale = 0.5
    terminate_on_collision_reward_scale = -100
    landing_pose_lateral_error_reward_scale = 1.25
    landing_pose_transversal_error_reward_scale =  1.0
    stability_reward_scale = 2.0
    symetry_reward_sides_scale = 1.0
    symetry_reward_transversal_scale = 0.25

    # regularization rewards
    action_clip_reward_scale    =  -0.5e-1
    joint_torque_reward_scale   = -3e-4
    joint_accel_reward_scale    = -2.5e-8
    action_rate_reward_scale    = -0.001
    jerk_reward_scale           = -1e-2