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

from envs.jump.initalization import InitializationScheme
from . import curriculum
from .jump_state_machine import JumpState

if TYPE_CHECKING:
    from .jump_env import JumpEnv


from gym import spaces

DEG2RAD = torch.pi / 180.0


_OLYMPUS_CONFIG = OlympusConfig(

# Configuration for the Olympus robot in the Jump environment.

    actuators={
        "lateral_motors": cube_mars.get_AK809_cfg(
            joint_names_expr=["LateralMotor.*"],
            kp=17.0,
            kd=0.9 * 2.0,
            safe_effort_limit=16.0,
            safe_velocity=2500.0,
            min_delay=0,
            max_delay=0,
        ),
        "transversal_motors": cube_mars.get_AK7010_cfg(
            joint_names_expr=[".*TransversalMotor.*"],
            kp=20.0,
            kd=0.4 * 2.0,
            safe_effort_limit=24.8,
            safe_velocity=2500.0,
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
        lateral_motor_joint_limits=(-15, 15),
        transversal_motor_joint_limits=(-30.0, 140),
        transversal_joint_sum_limits=(0, 220),
        filter_threshold=8 / 60,
        adaptive_filter=True,
        # angle threshold parameters
        upper_angle_threshold=15.0,
        lower_angle_threshold=10.0,
        upper_sum_threshold=30.0,
        lower_sum_threshold=30.0,
        velocity_threshold=60.0,
    ),
)



def push_robots_on_ground(
    env: JumpEnv,
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
    env: JumpEnv,
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
            "restitution_range": (0.0, 0.2),
            "make_consistent": True,
            "num_buckets": 64,
        },
    )
    # body maass
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
    # link mass
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
    # body center of mass
    body_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    body_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
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
    # actuator gains
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # motor model
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
    # joint armature
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
    # joint friction
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
    # push robot
    push_robot = EventTerm(
        func=push_robots_on_ground,
        mode="interval",
        interval_range_s=(0.2, 0.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
            "probability": 0.25,
            "force_range": ([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
            "torque_range": ([-3.0, -3.0, -1.0], [3.0, 3.0, 1.0]),
        },
    )
    # zero forces in flight
    zero_forces_in_flight = EventTerm(
        func=zero_forces_in_flight,
        mode="interval",
        interval_range_s=(0.0001, 0.0001),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Body"]),
        },
    )

    def apply_version_settings(self, cfg: 'JumpEnvCfg'):
        # to update friction params for different versions
        self.joint_friction.params["friction_distribution_params"] = cfg._friction_range


@configclass
class JumpEnvCfg(DirectRLEnvCfg):
    """Configuration for the Jump environment."""

    # TOGGLE - change this to switch between versions
    # ============================================================
    use_spring_version: bool = True  # False = original, True = spring version
    # ============================================================

    # spring parameters
    # enable_leg_springs: bool = False # Enable leg springs
    spring_stiffness: float = 1000.0  # N/m
    spring_rest_length: float = 0.17  # meters
    leg_hip_width: float = 0.09  # distance between motors
    leg_upper_link_length: float = 0.18  # meters
    stage2_scale_factor: float = 0.75  # Scale 
    stage2_blend_angle: float = 0.5  # degrees 

    @configclass
    class TerminationConfig:
        ''' Configuration for termination conditions. '''
        max_impact_acc: float = 7.0 * 9.81 # acceleration
        max_impact_acc_stance: float = 5.0 * 9.81 # acceleration
        min_touchdown_height: float = 0.30 # meters
        touchdown_pos_error: float = 1.0 # meters
        touchdown_rot_error: float = 45 # degrees
        walking_distance: float = 0.30 # meters
        min_root_height: float = 0.1 # meters
        close_to_goal_threshold: float = 1.0 # meters

    @configclass
    class ObservationNoiseCfg:
        ''' Configuration for observation noise parameters. '''
        body_lin_vel_noise = 10 * 1e-3
        body_ang_vel_noise = 20 * 1e-3  # rad
        body_pos_noise = 5 * 1e-3
        rot_noise_deg = 3
        joint_pos_noise_deg = 3
        joint_vel_noise_deg = 10
        joint_pos_bias_deg = 2.0
        projected_gravity_noise = 0.05

    @configclass
    class LatencyCfg:
        ''' Configuration for latency. '''
        min_observation_time_lag: int = 0
        max_observation_time_lag: int = 0
        min_action_time_lag: int = 0
        max_action_time_lag: int = 1 # in number of simulation steps

    # env
    episode_length_s = 20.0
    decimation = 8
    lateral_action_scale = 15 * DEG2RAD
    transversal_action_scale = 90 * DEG2RAD
    action_space = 12
    observation_space = 48
    state_space = 0
    touchdown_rejump_prob = 0.1
    landed_rejump_prob = 0.1

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

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: OlympusConfig = _OLYMPUS_CONFIG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Olympus/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # initializer - curriculum - sets the initialization scheme frequencies
    scheme_fraqs: Dict[str, float] = {
        InitializationScheme.STANDING.name: 0.40,
        InitializationScheme.DEFAULT.name: 0.15,
        InitializationScheme.INFLIGHT.name: 0.225,
        InitializationScheme.TOUCHDOWN.name: 0.175,
        InitializationScheme.LANDED.name: 0.05,
    }

    # curriculum - number of scheme curriculums
    num_scheme_curriculums: Dict[str, int] = {
        InitializationScheme.STANDING.name: 2,
        InitializationScheme.DEFAULT.name: 2,
        InitializationScheme.INFLIGHT.name: 2,
        InitializationScheme.TOUCHDOWN.name: 2,
        InitializationScheme.LANDED.name: 2,
    }

    command_curriculum_limits: Dict[str, tuple[int, int]] = {
        InitializationScheme.STANDING.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.DEFAULT.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.INFLIGHT.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.TOUCHDOWN.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
        InitializationScheme.LANDED.name: (0, curriculum.MAX_NUM_COMMAND_CURRICULUMS),
    }
    termination: TerminationConfig = TerminationConfig()
    observation_noise: ObservationNoiseCfg = ObservationNoiseCfg()
    latency: LatencyCfg = LatencyCfg()

    #curriculum parameters
    curriculum_threshold: float = 0.20  # meters
    num_games_per_level: float = 5

    # reward scales
    goal_pos_error_large_reward_scale = 10.0
    goal_pos_error_reward_scale = 50.0
    est_goal_pos_error_reward_scale = 75.0
    angvel_reward_scale = 4.0
    stance_reward_scale = 125  
    soft_impact_reward_scale = 20.0
    retract_feet_in_air_reward_scale = 10.0
    catch_landing_reward_scale = 20.0
    damp_landing_with_legs_reward_scale = 25.0
    break_torque_reward_scale = 2.0
    break_acc_reward_scale = 5.0
    takeoff_paw_pos_reward_scale = 5.0
    attitude_error_reward_scale = 20.0

    # regularization rewards
    action_clip_reward_scale = -2e0
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.03
    paw_forces_reward_scale = -0.002 * 0 
    jerk_reward_scale = 0  # -1e-1
    symmetry_reward_scale = 20.0  

    # experimental rewards
    stance_impact_reward_scale = 0
    paw_slip_reward_scale = 0.0
    distance_bonus_scale = 2.5


    def __post_init__(self):
        """Apply all version-specific changes based on use_spring_version toggle"""
        if self.use_spring_version:
            # SPRING VERSION SETTINGS
            self.enable_leg_springs = True
            
            # termination settings
            self.termination.max_impact_acc = 10.0 * 9.81 # acceleration
            self.termination.max_impact_acc_stance = 10.0 * 9.81 # acceleration
            self.termination.min_touchdown_height = 0.20 # meters
            self.termination.touchdown_pos_error = 1.2 # meters
            
            # curriculum threshold
            self.curriculum_threshold = 0.10 # meters
            
            # Store version-specific parameters for runtime use
            # distance scale parameters
            self._distance_scale_divisor = 1.8 
            self._distance_scale_min = 1.0
            self._distance_scale_max = 5.0
            # estimated error sigmas
            self._est_error_sigma1 = 0.50
            self._est_error_sigma2 = 0.9
            # joint offset adjustment
            self._joint_offset_adjustment = 45.0 # degrees
            # collision threshold
            self._collision_threshold = 2.0 # newtons
            # friction range
            self._friction_range = (0.005, 0.03)
            
        else:
            # ORIGINAL VERSION SETTINGS
            self.enable_leg_springs = False
            
            # termination settings 
            self.termination.max_impact_acc = 7.0 * 9.81 # acceleration
            self.termination.max_impact_acc_stance = 5.0 * 9.81 # acceleration
            self.termination.min_touchdown_height = 0.30 # meters
            self.termination.touchdown_pos_error = 1.0 # meters
            
            # curriculum threshold
            self.curriculum_threshold = 0.20    # meters
            
            # store version-specific parameters
            self._distance_scale_divisor = 2.0
            self._distance_scale_min = 1.0
            self._distance_scale_max = 3.0
            # estimated error sigmas
            self._est_error_sigma1 = 0.30
            self._est_error_sigma2 = 0.5
            # joint offset adjustment
            self._joint_offset_adjustment = 15.0 # degrees
            # collision threshold
            self._collision_threshold = 1.0 # newtons
            # friction range
            self._friction_range = (0.01, 0.04)

        self.events.apply_version_settings(self)
