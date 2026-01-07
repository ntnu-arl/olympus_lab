from __future__ import annotations
from typing import Dict, TYPE_CHECKING, Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from robot import OlympusConfig, cube_mars
from control import MotorCommandFilterCfg

from . import curriculum
from .initalization import InitializationScheme
from .jump_state_machine import JumpState

if TYPE_CHECKING:
    from .vertical_jump_env import VerticalJumpEnv

from gym import spaces

DEG2RAD = torch.pi / 180.0


_OLYMPUS_CONFIG = OlympusConfig(
    actuators={
        "lateral_motors": cube_mars.get_AK809_cfg(
            joint_names_expr=["LateralMotor.*"],
            kp=17.0,
            kd=0.9 * 2.0,
            safe_effort_limit=10.0, # torque saturation in Nm
            safe_velocity=2000.0,
            min_delay=0,
            max_delay=0,
        ),
        "transversal_motors": cube_mars.get_AK7010_cfg(
            joint_names_expr=[".*TransversalMotor.*"],
            kp=17.0,
            kd=0.4 * 2.0,
            safe_effort_limit=18.0, # torque saturation in Nm
            safe_velocity=2000.0,
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
        # angle threshold parameters are in degrees
        lateral_motor_joint_limits=(-15, 15),
        transversal_motor_joint_limits=(-30.0, 140),
        transversal_joint_sum_limits=(0, 240),
        filter_threshold=8 / 60,
        adaptive_filter=True,
        # angle threshold parameters
        upper_angle_threshold=15.0,
        lower_angle_threshold=10.0,
        upper_sum_threshold=45.0,
        lower_sum_threshold=35.0,
        velocity_threshold=60.0,
    ),
)



def push_robots_on_ground(
    env: VerticalJumpEnv,
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
    should_apply = (
        (torch.rand(len(env_ids), 1, 1, device=asset.device) < probability)
        * (env._jump_state_machine.states[env_ids] != JumpState.IN_FLIGHT).view(-1, 1, 1)
    ).float()

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


def zero_forces_in_flight(
    env: VerticalJumpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        in_air_env_ids = torch.nonzero(env_ids._jump_state_machine.states == JumpState.IN_FLIGHT, as_tuple=False)

    else:
        in_air_env_ids = env_ids[(env._jump_state_machine.states == JumpState.IN_FLIGHT)[env_ids]]

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(in_air_env_ids), num_bodies, 3)

    zeros = torch.zeros(size, device=asset.device)

    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(zeros, zeros, env_ids=in_air_env_ids, body_ids=asset_cfg.body_ids)
    asset.has_external_wrench = True  # ensure that the asset has external wrenches applied


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.9, 1.2),
            "dynamic_friction_range": (0.8, 0.9),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*MotorHousing.*", ".*Thigh.*", ".*Shank.*"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    
    # body_center_of_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
    #     },
    # )
    # motor_housing_center_of_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*MotorHousing.*"]),
    #         "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
    #     },
    # )

    # link_center_of_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*Thigh.*", ".*Shank.*"]),
    #         "com_range": {"x": (-0.001, 0.001), "y": (-0.001, 0.001), "z": (-0.02, 0.02)},
    #     },
    # )

    # paw_center_of_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*Paw.*"]),
    #         "com_range": {"x": (-0.0025, 0.0025), "y": (-0.0025, 0.0025), "z": (-0.0025, 0.0025)},
    #     }
    # )

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
            "friction_distribution_params": (0.005, 0.04),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=push_robots_on_ground,
        mode="interval",
        interval_range_s=(0.2, 0.4),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "probability": 0.25,
            "force_range": ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
            "torque_range": ([-3.5, -3.0, -1.0], [3.5, 3.0, 1.0]),
        },
    )

    zero_forces_in_flight = EventTerm(
        func=zero_forces_in_flight,
        mode="interval",
        interval_range_s=(0.0001, 0.0001),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
        },
    )


@configclass
class VerticalJumpEnvCfg(DirectRLEnvCfg):

    @configclass
    class TerminationConfig:
        """Termination configuration."""
        max_impact_acc: float = 5.0 * 9.81 # m/s^2
        max_impact_acc_stance: float = 2.5 * 9.81 # m/s^2
        min_touchdown_height: float = 0.2 # m
        max_touchdown_jump_height_error: float = 0.1 # m
        max_touchdown_land_pos_error: float = 0.2 # m
        walking_distance: float = 0.20 # m
        min_root_height: float = 0.1 # m
        max_takeoff_velocity_xy: float = 0.75 # m/s
        max_attitude_error_deg: float = 25.0 # degrees
        

    @configclass
    class ObservationNoiseCfg:
        """Observation noise configuration."""
        body_lin_vel_noise = 10 * 1e-3  # m/s
        body_ang_vel_noise = 20 * 1e-3  # rad
        body_pos_noise = 5 * 1e-3 # m
        rot_noise_deg = 3 # degrees
        joint_pos_noise_deg = 3 # degrees
        joint_vel_noise_deg = 10 # degrees
        joint_pos_bias_deg = 2.0 # degrees
        projected_gravity_noise = 0.05 # percent of gravity magnitude

    @configclass
    class LatencyCfg:
        """Latency configuration in time steps."""
        min_observation_time_lag: int = 0
        max_observation_time_lag: int = 0
        min_action_time_lag: int = 0
        max_action_time_lag: int = 1 # introduce max 1 step of action latency

    # env parameters 
    episode_length_s = 20.0
    decimation = 8
    lateral_action_scale = 15 * DEG2RAD
    transversal_action_scale = 90 * DEG2RAD
    action_space = 12
    observation_space = 48
    state_space = 0
    randomize_z_orientation: bool = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
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

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: OlympusConfig = _OLYMPUS_CONFIG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Olympus/.*",
        history_length=decimation,
        update_period=0.005,
        track_air_time=True,
    )

    # initializer
    scheme_fraqs: Dict[str, float] = {
        InitializationScheme.STANDING.name: 0.6,
        InitializationScheme.DEFAULT.name: 0.15,
        InitializationScheme.INFLIGHT.name: 0.2,
        InitializationScheme.TOUCHDOWN.name: 0.0,
        InitializationScheme.LANDED.name: 0.05,
    }

    # curriculum
    num_scheme_curriculums: Dict[str, int] = {
        InitializationScheme.STANDING.name: 1,
        InitializationScheme.DEFAULT.name: 1,
        InitializationScheme.INFLIGHT.name: 1,
        InitializationScheme.TOUCHDOWN.name: 1,
        InitializationScheme.LANDED.name: 1,
    }

    command_curriculum_limits: Dict[str, tuple[int, int]] = {
        InitializationScheme.STANDING.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.DEFAULT.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.INFLIGHT.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.TOUCHDOWN.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.LANDED.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
    }

    # configurations
    termination: TerminationConfig = TerminationConfig()
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    latency: LatencyCfg = LatencyCfg()

    # curriculum parameters
    curriculum_threshold: float = 0.10  # meters - threshold for success
    num_games_per_level: float = 5

    # reward scales
    jump_height_reward_scale = 20.0
    est_jump_height_reward_scale = 20.0
    xy_vel_reward_scale = 10.0
    land_pos_error_reward_scale = 0.0
    est_land_pos_error_reward_scale = 0.0
    angvel_reward_scale = 2.0
    stance_reward_scale = 10.0  # 10 ssem slike smooth landingh
    soft_impact_reward_scale = 5.0
    stance_impact_reward_scale = 100
    retract_feet_in_air_reward_scale = 0.0
    catch_landing_reward_scale = 3.0
    paw_slip_reward_scale = 0.0
    damp_landing_with_legs_reward_scale = 10.0
    attitude_error_reward_scale = 5

    multi_foot_contact_reward_scale = 0.0
    default_position_reward_scale = 30.0  # 1.0
    maintain_stance_height_reward_scale = 0.0
    air_default_position_reward_scale = 80.0  # 1.0


    # regularization rewards
    action_clip_reward_scale = -1.0e0
    joint_torque_reward_scale = -5e-4
    joint_accel_reward_scale = -1.5e-7
    action_rate_reward_scale = -0.1
    paw_forces_reward_scale = -0.002
    jerk_reward_scale = 0  # -1e-1
    symmetry_reward_scale = 35.0  # 2 seems like smooth landing
