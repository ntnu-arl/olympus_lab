from __future__ import annotations

from typing import Tuple, List, Dict
from torch import Tensor

import torch
from torch.nn.functional import normalize

import isaaclab.sim as sim_utils
from isaaclab.utils.buffers import TimestampedBuffer
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_error_magnitude,
    quat_conjugate,
    sample_uniform,
    axis_angle_from_quat,
    quat_mul,
    euler_xyz_from_quat,
)
import isaaclab.utils.math as math_utils

from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp

from isaaclab.utils.buffers import DelayBuffer, CircularBuffer

# import springs
from robot.springs.five_bar_spring import FiveBarSpring, FiveBarSpringConfig

from kinematics import OlympusKinematics
from control import MotorCommandFilter
from utilities.projectile_motion import estimate_jump_height, estimate_land_pos_error
from utilities.rotations import quat_to_euler_zyx
from utilities.indicies import make_slice_if_contigious

from .initalization import InitializationScheme, JumpInitializerBase
from .vertical_jump_env_config import VerticalJumpEnvCfg
from .jump_state_machine import JumpStateMachine, JumpState
from .curriculum import make_initializer, get_next_curriculum, TARGET_MAX_JUMP_LENGTH

from .simulation_logger import SimulationLogger

class VerticalJumpEnv(DirectRLEnv):
    '''
    Environment for training the Olympus robot to perform vertical jumps in Martian gravity. 
    Options for training with and without sprrings are available.
    The robot is rewarded for jumping to a commanded height while maintaining
    a stable posture and minimizing impact forces upon landing.
    '''

    cfg: VerticalJumpEnvCfg

    def __init__(self, cfg: VerticalJumpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "jump_height",
                "est_jump_height",
                "xy_vel",
                "orientaion_error",
                "angvel",
                "stance",
                "stance_impact",
                "soft_impact",
                "retract_feet_in_air",
                "catch_landing",
                "paw_slip",
                "damp_landing_with_legs",
                "action_clip",
                "dof_torques",
                "dof_acc_l2",
                "action_rate_l2",
                "paw_forces",
                "jerk",
                "joint_symmetry",
                "multi_foot_contact",
                "default_position",
                "maintain_stance_height",
                "air_default_position",
            ]
        }

        self._init_indices()
        self._init_buffers()
        self._init_initializers()
        self._jump_state_machine = JumpStateMachine(self.num_envs, self.device)
        self._motor_command_filter = MotorCommandFilter(self.cfg.robot.motor_command_filter, self._robot)

       ##################### For jump test #####################
        self.logger = SimulationLogger(enable_logging=False)

        self.window_size = 1 # moving average window size for action filtering
        self._action_history = torch.zeros((self.window_size, self.num_envs, self.cfg.action_space), device=self.device)
        # preprogrammed jump test for open loop jumps
        self.jump_counter = 0
        self.jump_test_enabled = False

    def _init_buffers(self):
        '''Initialize tensors used in the environment.'''
        with torch.device(self.device):
            self._actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._processed_actions = torch.zeros_like(self._actions)
            self._filtered_action = torch.zeros_like(self._actions)
            self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._previous_torque = torch.zeros_like(self._actions)
            self._previous_root_pos = torch.zeros(self.num_envs, 3)
            self._commanded_jump_height = torch.zeros(self.num_envs, 1)
            self._jump_toggle = torch.zeros(self.num_envs, 1)
            self._low_attitude_error_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._terminate_on_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self._episode_end = torch.zeros(self.num_envs, dtype=torch.bool)
            self._has_bin_airborne = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_takeoff = torch.zeros(self.num_envs, dtype=torch.bool)
            self._idle_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._episode_start_step = torch.zeros(self.num_envs, dtype=torch.long)
            self._terminate_touchdown = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_impact = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_takeoff = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_walking = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_landed = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_collision = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_idle = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_nan = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_root_height = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_shank_height = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_attitude = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._terminate_translation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._max_impact_acc = torch.zeros(self.num_envs, dtype=torch.float)
            self._original_jump_height = torch.zeros(self.num_envs, device=self.device)
            self.spring_torques = torch.zeros((self.num_envs, 8), device=self.device)
            self.spring_forces = torch.zeros((self.num_envs, 8), device=self.device)
            self.spring_lengths = torch.zeros((self.num_envs, 8), device=self.device)
            self.knee_distance = torch.zeros((self.num_envs, 4), device=self.device)

            self.spring_system = FiveBarSpring(
                num_envs=self.num_envs,
                device=self.device,
                spring_stiffness=self.cfg.spring_stiffness,
                spring_rest_length=self.cfg.spring_rest_length,
                leg_hip_width=self.cfg.leg_hip_width,
                leg_upper_link_length=self.cfg.leg_upper_link_length,
                stage2_scale_factor=self.cfg.stage2_scale_factor,
                stage2_blend_angle=self.cfg.stage2_blend_angle,
            )   

            self._init_schemes = torch.zeros(self.num_envs, dtype=torch.int)

            self._contact_state = TimestampedBuffer(torch.zeros(self.num_envs, 4, dtype=torch.bool))
            self._collision_state = TimestampedBuffer(torch.zeros(self.num_envs, dtype=torch.bool))

            self._goal_vec = TimestampedBuffer(torch.zeros(self.num_envs, 3, dtype=torch.float))
            self._start_vec = TimestampedBuffer(torch.zeros(self.num_envs, 3, dtype=torch.float))
            self._speed_on_goal = TimestampedBuffer(torch.zeros(self.num_envs, dtype=torch.float))
            self._rot_error_rad = TimestampedBuffer(torch.zeros(self.num_envs, dtype=torch.float))
            self._est_goal_pos_error = TimestampedBuffer(torch.zeros(self.num_envs, 2, dtype=torch.float))

            # measurments
            self._lin_vel_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._ang_vel_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )

            self._pos_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._rot_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._joint_pos_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._joint_vel_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._joint_pos_target_history = DelayBuffer(
                self.cfg.latency.max_action_time_lag, batch_size=self.num_envs, device=self.device
            )

            self._joint_pos_bias = torch.zeros(
                self.num_envs, self._robot.num_joints, dtype=torch.float, device=self.device
            )[:, self._actuated_joint_ids]

            self._projected_gravity_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )

            self._measurement_delay_buffers = [
                self._lin_vel_history,
                self._ang_vel_history,
                self._rot_history,
                self._pos_history,
                self._joint_pos_history,
                self._joint_vel_history,
                self._projected_gravity_history,
            ]

            self._measured_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_position = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_rot = torch.zeros(self.num_envs, 4, dtype=torch.float)
            self._measured_joint_pos = torch.zeros(
                self.num_envs, self._robot.num_joints, dtype=torch.float, device=self.device
            )[:, self._actuated_joint_ids]
            self._measured_joint_vel = torch.zeros(
                self.num_envs, self._robot.num_joints, dtype=torch.float, device=self.device
            )[:, self._actuated_joint_ids]
            self._measured_projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float)

            # curriculums
            self._max_command_curriculum = torch.zeros(self.num_envs, dtype=torch.int)
            self._min_command_curriculum = torch.zeros(self.num_envs, dtype=torch.int)
            self._num_scheme_curriculums = torch.zeros(self.num_envs, dtype=torch.int)
            self._scheme_curiculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._command_curiculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._curriculum_progress = torch.zeros(self.num_envs, dtype=torch.int)
            self._game_won = torch.zeros(self.num_envs, dtype=torch.bool)
            self._scheme_count: Dict[InitializationScheme, int] = {}

    def _init_indices(self):
        # get specific body indices
        self._base_contact_id = self._contact_sensor.find_bodies("Body")[0][0]
        self._feet_contact_ids = make_slice_if_contigious(self._contact_sensor.find_bodies(".*Paw.*")[0])
        self._underisred_contact_body_ids = make_slice_if_contigious(
            self._contact_sensor.find_bodies(["Body", "MotorHousing.*", ".*Thigh.*", ".*Shank.*"])[0]
        )
        self._base_id = self._robot.find_bodies("Body")[0][0]
        self._shank_ids = make_slice_if_contigious(self._robot.find_bodies(".*Shank.*")[0])
        self._feet_ids, self._feet_names = self._robot.find_bodies(".*Paw.*")
        self._feet_ids = make_slice_if_contigious(self._feet_ids)
        self._motor_housing_ids = make_slice_if_contigious(self._robot.find_bodies("MotorHousing.*")[0])
        self._actuated_joint_ids = make_slice_if_contigious(self._robot.find_joints(".*Motor.*")[0])

        self._left_transversal_indices = []
        self._right_transversal_indices = []
        for end in ["F", "B"]:
            for side in ["Inner", "Outer"]:
                self._left_transversal_indices.append(self._robot.find_joints(f"{side}TransversalMotor_{end}L")[0][0])

                self._right_transversal_indices.append(self._robot.find_joints(f"{side}TransversalMotor_{end}R")[0][0])

        self._left_transversal_indices = make_slice_if_contigious(self._left_transversal_indices)
        self._right_transversal_indices = make_slice_if_contigious(self._right_transversal_indices)

    def _init_initializers(self):
        '''Initialize the jump initializers for different curriculum schemes.'''
        self._kinematics = OlympusKinematics()
        self._initializers: Dict[InitializationScheme, List[List[JumpInitializerBase]]] = {}

        for scheme in self.cfg.num_scheme_curriculums.keys():
            self._initializers[InitializationScheme[scheme]] = [
                [
                    make_initializer(
                        InitializationScheme[scheme],
                        sc,
                        cc,
                        self._robot,
                        self._kinematics,
                        use_spring_version=self.cfg.use_spring_version, 
                    )
                    for cc in range(
                        self.cfg.command_curriculum_limits[scheme][0],
                        self.cfg.command_curriculum_limits[scheme][1],
                    )
                ]
                for sc in range(self.cfg.num_scheme_curriculums[scheme])
            ]

        idx_start = 0
        schemes = list(self.cfg.scheme_fraqs.keys())
        for scheme in schemes:
            fraction = self.cfg.scheme_fraqs[scheme]
            num_scheme_curriculums = self.cfg.num_scheme_curriculums[scheme]
            command_curriculum_limits = self.cfg.command_curriculum_limits[scheme]
            idx_end = idx_start + int(fraction * self.num_envs) if scheme != schemes[-1] else self.num_envs
            self._init_schemes[idx_start:idx_end] = InitializationScheme[scheme]
            self._num_scheme_curriculums[idx_start:idx_end] = num_scheme_curriculums
            self._min_command_curriculum[idx_start:idx_end] = command_curriculum_limits[0]
            self._max_command_curriculum[idx_start:idx_end] = command_curriculum_limits[1]
            self._scheme_count[InitializationScheme[scheme]] = idx_end - idx_start
            idx_start = idx_end

        assert idx_start == self.num_envs

        self._command_curiculum_level[:] = self._min_command_curriculum


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def preprogrammed_jump_test(self):
        ''' A preprogrammed jump test that makes the robot squat and jump, for open loop jump tests. '''
        default_pos = self._robot.data.default_joint_pos[:, : self.cfg.action_space]
        self._processed_actions[:] = default_pos

        decimation_factor = 1
                    
        stand_time_len = 50*decimation_factor
        squat_time_len = 25*decimation_factor
        squat_hold_time_len = 50*decimation_factor
        jump_time_len = 70*decimation_factor
        after_jump_time_len = 350*decimation_factor

        t1 = stand_time_len
        t2 = t1 + squat_time_len
        t3 = t2 + squat_hold_time_len
        t4 = t3 + jump_time_len
        t5 = t4 + after_jump_time_len

        if self.jump_counter < t1: # stand phase
            if self.jump_counter == 0:
                print("Jump test: Standing at default position...")

        elif self.jump_counter < t2: # squat phase
            if self.jump_counter == t1:
                print("Jump test: Squatting...")
            progress = (self.jump_counter - t1) / squat_time_len
            self._processed_actions[:, 4:12] = default_pos[:, 4:12] + torch.deg2rad(
                torch.tensor(90.0 * progress, device=self.device)
            )
        
        elif self.jump_counter < t3: # squat hold phase
            if self.jump_counter == t2:
                print("Jump test: Holding squat...")
            self._processed_actions[:, 4:12] = default_pos[:, 4:12] + torch.deg2rad(
                torch.tensor(90.0, device=self.device)
            )

        elif self.jump_counter < t4: # jump phase
            if self.jump_counter == t3:
                print("Jump test: Jumping!")
            self._processed_actions[:, 4:12] = default_pos[:, 4:12] + torch.deg2rad(
                torch.tensor(-90.0, device=self.device)
            )

        elif self.jump_counter < t5:   # after jump phase
            if self.jump_counter == t4:
                print("Jump test: Returning to default position...")
            self._processed_actions[:, 4:12] = default_pos[:, 4:12]

        else: # after jump phase - test complete
            self.jump_test_enabled = False
            print("Jump test complete!")
        
        self.jump_counter += 1

    def _pre_physics_step(self, actions: torch.Tensor):
        '''Process and filter the actions before applying them to the robot.'''
        self._actions[:] = actions
        if self.jump_test_enabled:
            self.preprogrammed_jump_test()
        else:
            self._processed_actions[:, :4] = (
                self.cfg.lateral_action_scale * self._actions[:, :4] + self._robot.data.default_joint_pos[:, :4]
            )
            self._processed_actions[:, 4:] = (
                self.cfg.transversal_action_scale * self._actions[:, 4:] + self._robot.data.default_joint_pos[:, 4:12]
            )

        self._filtered_action[:] = self._joint_pos_target_history.compute(
            self._motor_command_filter.filter(
                joint_commands=self._processed_actions,
                joint_positions=self._robot.data.joint_pos[:, self._actuated_joint_ids],
                joint_velocities=self._robot.data.joint_vel[:, self._actuated_joint_ids],
            )
        )
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=0)
        self._action_history[-1] = self._filtered_action
        mean_filtered_actions = torch.mean(self._action_history, dim=0)
        self._mean_filtered_action = mean_filtered_actions
        self._filtered_action = mean_filtered_actions

    def _apply_action(self):
        '''Apply the filtered joint position targets to the robot, and compute spring forces if enabled.'''
        self._robot.set_joint_position_target(self._filtered_action, self._actuated_joint_ids)
        
        if self.cfg.enable_leg_springs: # add spring forces if enabled
            left_motors = self._robot.data.joint_pos[:, self._left_transversal_indices] # [FL, BL, FR, BR]
            right_motors = self._robot.data.joint_pos[:, self._right_transversal_indices]
            
            theta_inner = torch.stack([
                left_motors[:, 0],   # FL_inner
                left_motors[:, 2],   # BL_inner
                right_motors[:, 0],  # FR_inner
                right_motors[:, 2],  # BR_inner
            ], dim=1)
            
            theta_outer = torch.stack([
                left_motors[:, 1],   # FL_outer
                left_motors[:, 3],   # BL_outer
                right_motors[:, 1],  # FR_outer
                right_motors[:, 3],  # BR_outer
            ], dim=1)
            
            spring_torques = self.spring_system.compute_spring_torques(theta_inner, theta_outer)
            self.spring_torques = spring_torques
            
            self._robot.actuators["transversal_motors"].set_passive_torques(spring_torques)


    def _get_observations(self) -> dict:
        '''Get the current observations for the policy.'''
        self._previous_actions[:] = self._actions
        self._previous_torque[:] = self._robot.data.applied_torque[:, self._actuated_joint_ids]

        self._measured_position[:] = self._pos_history.compute(
            self._robot.data.root_pos_w
            + torch.randn_like(self._robot.data.root_pos_w) * self.cfg.observation_noise.body_pos_noise
        )
        self._measured_rot[:] = self._rot_history.compute(
            math_utils.quat_mul(
                self._robot.data.root_quat_w,
                math_utils.quat_from_angle_axis(
                    torch.randn(self.num_envs, device=self.device)
                    * (self.cfg.observation_noise.rot_noise_deg * (torch.pi / 180.0)),
                    torch.rand(self.num_envs, 3, device=self.device),
                ),
            )
        )

        self._measured_lin_vel[:] = self._lin_vel_history.compute(
            self._robot.data.root_lin_vel_b
            + torch.randn_like(self._robot.data.root_lin_vel_b).clamp(-1, 1)
            * self.cfg.observation_noise.body_lin_vel_noise
        )

        self._measured_ang_vel[:] = self._ang_vel_history.compute(
            self._robot.data.root_ang_vel_b
            + torch.randn_like(self._robot.data.root_ang_vel_b).clamp(-1, 1)
            * self.cfg.observation_noise.body_ang_vel_noise
        )

        self._measured_joint_pos[:] = self._joint_pos_history.compute(
            self._robot.data.joint_pos[:, self._actuated_joint_ids]
            + torch.randn_like(self._measured_joint_pos).clamp(-1, 1)
            * (self.cfg.observation_noise.joint_pos_noise_deg * torch.pi / 180)
            + self._joint_pos_bias[:, self._actuated_joint_ids]
        )

        self._measured_joint_vel[:] = self._joint_vel_history.compute(
            self._robot.data.joint_vel[:, self._actuated_joint_ids]
            + torch.randn_like(self._measured_joint_vel).clamp(-1, 1)
            * (self.cfg.observation_noise.joint_vel_noise_deg * torch.pi / 180)
        )

        self._measured_projected_gravity[:] = quat_apply_inverse(self._measured_rot, self._robot.data.GRAVITY_VEC_W)

        obs = torch.cat(
            [
                self._commanded_jump_height,
                (self._jump_state_machine.states != JumpState.LANDED).float().view(-1, 1),
                self._measured_position[:, [2]],
                self._measured_lin_vel,
                self._measured_projected_gravity,
                self._measured_ang_vel,
                self._measured_joint_pos - self._robot.data.default_joint_pos[:, self._actuated_joint_ids],
                self._measured_joint_vel,
                self._actions,
            ],
            dim=-1,
        )

        nan_mask = torch.isnan(obs).any(dim=-1)
        if nan_mask.any():
            print(f"NaN in observations: {obs[nan_mask]}")
            obs[nan_mask] = 0.0

        observations = {"policy": obs}

        # logging for plotting
        if self.logger.enable_logging:
            self.logger.update_extras(
                robot=self._robot,
                actions=self._actions,
                processed_actions=self._processed_actions,
                filtered_actions=self._filtered_action,
                obs=obs,
                spring_torques=self.spring_torques,
                spring_forces=self.spring_system.spring_forces, 
                spring_lengths=self.spring_system.spring_lengths,
                knee_distance=self.spring_system.knee_distances,
            )
            self.logger.log_step(obs, self.logger.extras)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        '''Compute the reward for the current timestep based on various criteria.'''
        has_landed = self._jump_state_machine.states == JumpState.LANDED

        short_since_landing = (self._jump_state_machine.steps_since_touchdown < 0.3 / self.step_dt) * has_landed

        # height scale
        height_scale = torch.clamp(self._original_jump_height/2.0, min=0.1, max=3.0)
        if self.cfg._height_scale_squared:
            height_scale = torch.where(height_scale > 1.0, height_scale**2, height_scale)
        
        # estimated jump height reward
        est_jump_height = estimate_jump_height(self._robot.data.root_pos_w, self._robot.data.root_lin_vel_w, -3.71)
        
        height_rew_func = lambda x: torch.exp(
            -((x - self._commanded_jump_height.view(-1)) ** 2) / (0.35**2)
        ) + 3 * torch.exp(-((x - self._commanded_jump_height.view(-1)).abs()) / (self.cfg._height_rew_func_sigma))

        est_jump_height_error_reward = torch.where(
            (self._jump_state_machine.states == JumpState.IN_FLIGHT)
            * (~self._jump_state_machine.has_reached_max_height),
            height_rew_func(est_jump_height),
            0.0,
        )
        est_jump_height_error_reward *= height_scale

        if self.cfg._use_linear_jump_height_reward:
            # spring
            jump_heigth_reward = torch.clamp(self._robot.data.root_pos_w[:, 2] / 2.0, min=0.0, max=6.0)
        else:
            jump_heigth_reward = torch.where(
                self._jump_state_machine.just_reached_max_height,
                height_rew_func(self._jump_state_machine.max_height),
                0.0,
            )
            jump_heigth_reward *= height_scale
        # lateral velocity penalty
        xy_vel = self._robot.data.root_lin_vel_w[:, :2].square().sum(dim=1).sqrt()
        xy_vel_reward = torch.exp(-xy_vel / (0.2**2))

        xy_vel_reward[self._jump_state_machine.states != JumpState.STANCE] *= 2
        # orientation penalty
        oreientation_reward = torch.exp(
            -self._robot.data.projected_gravity_b[:, :2].square().sum(dim=1) / (self.cfg._orientation_sigma**2)
        )
        # angular velocity penalty
        ang_vel = self._robot.data.root_ang_vel_w.square().sum(dim=1)
        ang_vel_reward = torch.exp(-ang_vel / (2.0**2))

        # stance when landed reward
        joint_offset = (
            self._robot.data.joint_pos[:, self._actuated_joint_ids]
            - self._robot.data.default_joint_pos[:, self._actuated_joint_ids]
        )

        joint_offset[:, 4:12] -= torch.pi / 180 * self.cfg._joint_offset_adjustment
        #  stance posture reward
        stance_reward = torch.where(
            short_since_landing,
            torch.exp(-joint_offset.square().mean(dim=1) / ((15 * torch.pi / 180) ** 2)),
            0,
        )
        # feet retraction in air reward
        feet_retract_reward = torch.where(
            (self._jump_state_machine.states == JumpState.IN_FLIGHT),
            torch.exp(-joint_offset.square().mean(dim=1) / ((20 * torch.pi / 180) ** 2)),
            0.0,
        )

        impact_acc = torch.sum(
            self._robot.data.body_acc_w[:, self._base_id, :3]
            * normalize(self._robot.data.body_vel_w[:, self._base_id, :3]),
            dim=-1,
        ).clamp(max=0.0)
        # soft landing impact reward
        rew_impact = torch.where(
            short_since_landing, (1 - impact_acc / (self.cfg.termination.max_impact_acc)).clamp(min=0.0), 0
        )

        rew_stance_impact = torch.where(
            self._jump_state_machine.states == JumpState.STANCE,
            (impact_acc / (self.cfg.termination.max_impact_acc_stance)).clamp(-1, 0),
            0,
        )
        # catch landing reward
        catch_landing = torch.where(short_since_landing, (-self._robot.data.root_lin_vel_w[:, 2]).clamp(0, 1), 0.0)
        # paw slip reward
        paw_vel = self._robot.data.body_vel_w[:, self._feet_ids, :3].norm(dim=-1)
        rew_paw_slip = torch.sum(
            short_since_landing.unsqueeze(1) * self.contact_state.int() * torch.exp(-paw_vel.square() / 0.1**2), dim=1
        )

        rew_paw_slip[:] /= torch.maximum(
            torch.ones(1, device=self.device, dtype=torch.int), self.contact_state.int().sum(dim=1)
        ).float()
        # damp landing with legs reward
        rew_damp_landing_with_legs = torch.where(
            short_since_landing, self._robot.data.joint_vel[:, 4:12].clamp(0, 1).mean(dim=-1), 0
        )

        current_contact_state = torch.any(
            torch.abs(self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_contact_ids]) > 0.1,
            dim=1,
        ).float()
        # multi foot contact reward
        paw_height = self._robot.data.body_pos_w[:, self._feet_ids, 2]
        multi_foot_contact_reward = torch.exp(-paw_height.square().mean(dim=1) / (0.1**2))

        joint_offset_default = (
            self._robot.data.joint_pos[:, self._actuated_joint_ids]
            - self._robot.data.default_joint_pos[:, self._actuated_joint_ids]
        ).clone()

        joint_offset_landed = joint_offset_default.clone()
        joint_offset_stance = joint_offset_default.clone()

        joint_offset_landed[:, self._left_transversal_indices] -= torch.pi / 180 * self.cfg._landed_joint_offset
        joint_offset_landed[:, self._right_transversal_indices] -= torch.pi / 180 * self.cfg._landed_joint_offset
        
        joint_offset_stance[:, self._left_transversal_indices] -= torch.pi / 180 * self.cfg._stance_joint_offset
        joint_offset_stance[:, self._right_transversal_indices] -= torch.pi / 180 * self.cfg._stance_joint_offset
    
        variance = (10 * torch.pi / 180) ** 2

        default_position_reward = torch.where(
            self._jump_state_machine.states == JumpState.LANDED,
            torch.exp(-joint_offset_landed.square().mean(dim=1) / variance),
            torch.where(
                self._jump_state_machine.states == JumpState.STANCE,
                torch.exp(-joint_offset_stance.square().mean(dim=1) / variance),
                0.0
            )
        )

        target_stance_height = self.cfg._target_stance_height
        current_height = self._robot.data.root_pos_w[:, 2]

        height_error = torch.abs(current_height - target_stance_height)
        stance_height_reward = torch.exp(-height_error / 0.1)

        # maintain stance height reward
        maintain_height_reward = torch.where(
            self._jump_state_machine.states == JumpState.LANDED, stance_height_reward, 0.0
        )
        # air default position reward
        joint_offset_air = (
                self._robot.data.joint_pos[:, self._actuated_joint_ids]
                - self._robot.data.default_joint_pos[:, self._actuated_joint_ids]
            )
        if self.cfg._air_joint_offset_bias > 0:
            joint_offset_air += torch.pi / 180 * self.cfg._air_joint_offset_bias

        air_default_position_reward = torch.where(
            self._jump_state_machine.states == JumpState.IN_FLIGHT,
            torch.exp(-joint_offset_air.square().mean(dim=1) / ((15 * torch.pi / 180) ** 2)),
            0.0,
        )
        air_default_position_reward = torch.where(
            self._robot.data.root_pos_w[:, 2] < 0.9, 0.0, air_default_position_reward
        )


        rewards = {
            "jump_height": jump_heigth_reward * self.cfg.jump_height_reward_scale,
            "est_jump_height": est_jump_height_error_reward * self.cfg.est_jump_height_reward_scale * self.step_dt,
            "xy_vel": xy_vel_reward * self.cfg.xy_vel_reward_scale * self.step_dt,
            "orientaion_error": oreientation_reward * self.cfg.attitude_error_reward_scale * self.step_dt,
            "angvel": ang_vel_reward * self.cfg.angvel_reward_scale * self.step_dt,
            "stance": stance_reward * self.cfg.stance_reward_scale * self.step_dt,
            "soft_impact": rew_impact * self.cfg.soft_impact_reward_scale * self.step_dt,
            "stance_impact": rew_stance_impact * self.cfg.stance_impact_reward_scale * self.step_dt,
            "retract_feet_in_air": feet_retract_reward * self.cfg.retract_feet_in_air_reward_scale * self.step_dt,
            "catch_landing": catch_landing * self.cfg.catch_landing_reward_scale * self.step_dt,
            "paw_slip": rew_paw_slip * self.cfg.paw_slip_reward_scale * self.step_dt,
            "damp_landing_with_legs": rew_damp_landing_with_legs
            * self.cfg.damp_landing_with_legs_reward_scale
            * self.step_dt,
            "multi_foot_contact": multi_foot_contact_reward * self.cfg.multi_foot_contact_reward_scale * self.step_dt,
            "default_position": default_position_reward * self.cfg.default_position_reward_scale * self.step_dt,
            "maintain_stance_height": maintain_height_reward
            * self.cfg.maintain_stance_height_reward_scale
            * self.step_dt,
            "air_default_position": air_default_position_reward
            * self.cfg.air_default_position_reward_scale
            * self.step_dt,
        }

        rewards.update(self._calculate_regularization_rewards())

        for key, value in rewards.items():
            inf_mask = torch.isinf(value)
            nan_mask = torch.isnan(value)
            value[inf_mask] = 0.0
            value[nan_mask] = 0.0
            if inf_mask.any():
                print(f"Inf in reward {key}")
            if nan_mask.any():
                print(f"NaN in reward {key}")


        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        metrics = dict()
        height_error = (self._jump_state_machine.max_height - self._commanded_jump_height.view(-1)).abs()
        land_pos_error = (self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]).norm(dim=1)
        if self.reset_time_outs.any():
            metrics["terminal_land_pos_error"] = land_pos_error[self.reset_time_outs].mean().item()
            metrics["terminal_jump_height_error"] = height_error[self.reset_time_outs].mean().item()
        metrics["num_airborn"] = (self._jump_state_machine.states == JumpState.IN_FLIGHT).count_nonzero().item()
        metrics["num_airborn_standing"] = (
            (
                (self._jump_state_machine.states == JumpState.IN_FLIGHT)
                * (self._init_schemes == InitializationScheme.STANDING)
            )
            .count_nonzero()
            .item()
        )

        if metrics["num_airborn"] > 0:
            metrics["airborn_attitude_error"] = (
                self.rot_error_rad[self._jump_state_machine.states == JumpState.IN_FLIGHT].mean().item()
            )

        if self._jump_state_machine.touchdown.any():

            metrics["touchdown_jump_height_error"] = (
                height_error[self._jump_state_machine.touchdown].abs().mean().item()
            )
            metrics["touchdown_land_pos_error"] = land_pos_error[self._jump_state_machine.touchdown].mean().item()

            standing = (self._init_schemes == InitializationScheme.STANDING) * self._jump_state_machine.touchdown

            assert (self._has_bin_airborne[standing]).all()

            if standing.any():

                metrics["standing_touchdown_jump_height_error"] = height_error[standing].mean().item()
                metrics["standing_touchdown_land_pos_error"] = land_pos_error[standing].mean().item()

        stance = (self._jump_state_machine.states == JumpState.STANCE).nonzero(as_tuple=False)
        if len(stance) > 0:
            metrics["stance_paw_heigth"] = self._robot.data.body_pos_w[stance, self._feet_ids, 2].mean().item()

        if self._jump_state_machine.takeoff.any():
            metrics["takeoff_min_height"] = (
                self._jump_state_machine.min_height[self._jump_state_machine.takeoff].mean().item()
            )

        # add metric for highest jump achieved
        metrics["max_jump_height"] = self._jump_state_machine.max_height.max().item()
        metrics["avg_jump_height"] = self._jump_state_machine.max_height.mean().item()
        metrics["min_jump_height"] = self._jump_state_machine.max_height.min().item()

        self.extras.update({f"metrics/{key}": value for key, value in metrics.items()})

        # bookkeeping

        return reward.clamp(min=-0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Compute the termination conditions for the vertical jump task. '''
        self._jump_state_machine.update(
            root_pos_w=self._robot.data.root_pos_w,
            root_vel_w=self._robot.data.root_vel_w,
            contact_state=self.contact_state,
        )

        self._has_bin_airborne[self._jump_state_machine.states == JumpState.IN_FLIGHT] = True

        is_idle = self._robot.data.root_lin_vel_w.norm(dim=1) < 0.1
        self._idle_count[self._jump_state_machine.states == JumpState.STANCE] += is_idle[
            self._jump_state_machine.states == JumpState.STANCE
        ].int()
        self._idle_count[self._jump_state_machine.states != JumpState.STANCE] = 0

        self._episode_end[:] = self.episode_length_buf >= self.max_episode_length - 1
        self._terminate_on_goal[:] = self._jump_state_machine.steps_since_touchdown >= 1.0 / self.step_dt

        time_out = self._episode_end | self._terminate_on_goal

        self._terminate_collision[:] = self.collision_state

        self._terminate_touchdown[:] = self._jump_state_machine.touchdown * (
            (
                (self._jump_state_machine.max_height - self._commanded_jump_height.view(-1)).abs()
                > self.cfg.termination.max_touchdown_jump_height_error
            )
            | (self._robot.data.root_pos_w[:, 2] < self.cfg.termination.min_touchdown_height)
        )
        self._max_impact_acc[self._jump_state_machine.states != JumpState.IN_FLIGHT] = (
            self.cfg.termination.max_impact_acc_stance
        )
        self._max_impact_acc[
            (self._jump_state_machine.states == JumpState.IN_FLIGHT) | self._jump_state_machine.touchdown
        ] = self.cfg.termination.max_impact_acc
        self._terminate_impact[:] = (
            torch.sum(
                self._robot.data.body_acc_w[:, self._base_id, :3]
                * normalize(self._robot.data.body_vel_w[:, self._base_id, :3]),
                dim=-1,
            )
            < -self._max_impact_acc
        )

        self._terminate_nan[:] = (
            torch.isnan(self._robot.data.body_state_w).any(dim=(1, 2)) 
            | torch.isnan(self._robot.data.joint_pos).any(dim=-1)
            | torch.isnan(self._robot.data.joint_vel).any(dim=-1)
            | torch.isnan(self._robot.data.joint_acc).any(dim=-1)
            | torch.isnan(self._robot.data.body_acc_w).any(dim=(1, 2))  
            | torch.isnan(self._robot.data.applied_torque).any(dim=-1)
        )

        self._terminate_landed[:] = self._jump_state_machine.takeoff * (
            self._jump_state_machine.states == JumpState.LANDED
        )

        self._terminate_takeoff[:] = self._jump_state_machine.takeoff * (
            self._robot.data.root_vel_w[:, :2].norm(dim=-1) > self.cfg.termination.max_takeoff_velocity_xy
        )
        # self._terminate_takeoff[:] = False

        self._terminate_walking[:] = (self._jump_state_machine.states == JumpState.STANCE) * (
            self.walk_vec[:, :2].norm(dim=-1) > self.cfg.termination.walking_distance
        )

        self._terminate_idle[:] = (self._jump_state_machine.states == JumpState.STANCE) * (
            ((self.episode_length_buf - self._episode_start_step) > 3.0 / self.step_dt)
            | (self._idle_count > 1.0 / self.step_dt)
        )

        shank_height = self._robot.data.body_pos_w[:, self._shank_ids, 2]
        self._terminate_shank_height[:] = (shank_height < 0.025).any(dim=1)

        self._terminate_root_height[:] = self._robot.data.root_pos_w[:, 2] < self.cfg.termination.min_root_height

        self._terminate_dropping_during_stance = (self._jump_state_machine.states == JumpState.STANCE) & (
            self._robot.data.root_pos_w[:, 2] < 0.3
        )  

        roll_raw, pitch_raw, yaw_raw = euler_xyz_from_quat(self._robot.data.root_quat_w)

        roll = torch.atan2(torch.sin(roll_raw), torch.cos(roll_raw))
        pitch = torch.atan2(torch.sin(pitch_raw), torch.cos(pitch_raw))
        yaw = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))

        max_attitude_limits = self.cfg.termination.max_attitude_error_deg * (torch.pi / 180.0)

        roll_limit = max_attitude_limits
        pitch_limit = max_attitude_limits
        yaw_limit = max_attitude_limits

        self._terminate_attitude[:] = (
            (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit) | (torch.abs(yaw) > yaw_limit)
        )

        x_translation_limit = self.cfg.termination.max_touchdown_land_pos_error
        y_translation_limit = self.cfg.termination.max_touchdown_land_pos_error

        x_distance = torch.abs(self._robot.data.root_pos_w[:, 0] - self._terrain.env_origins[:, 0])
        y_distance = torch.abs(self._robot.data.root_pos_w[:, 1] - self._terrain.env_origins[:, 1])

        self._terminate_translation[:] = (x_distance > x_translation_limit) | (y_distance > y_translation_limit)
        # termination conditions
        died = (
            self._terminate_collision
            | (self._terminate_walking)
            | self._terminate_touchdown
            | self._terminate_takeoff
            | self._terminate_landed
            | self._terminate_root_height
            | self._terminate_shank_height
            | self._terminate_idle
            | self._terminate_impact
            | self._terminate_nan
            | self._terminate_attitude
            | self._terminate_translation
        )
        # curriculum update
        curriculum_thresh = (
            0.05 if self.cfg._curriculum_threshold_override == 0.05
            else self.cfg.curriculum_threshold
        )
        # update curriculum progress
        within_curriculum_thresh = (
            (self._commanded_jump_height.view(-1) - self._jump_state_machine.max_height).abs() < curriculum_thresh
        ) * (self._jump_state_machine.states == JumpState.LANDED)
        self._curriculum_progress[within_curriculum_thresh * time_out] += 1
        self._curriculum_progress[(~within_curriculum_thresh) * time_out] = 0
        self._curriculum_progress[died] = 0

        (
            self._scheme_curiculum_level[time_out],
            self._command_curiculum_level[time_out],
            self._curriculum_progress[time_out],
            new_game_won,
        ) = get_next_curriculum(
            self._scheme_curiculum_level[time_out],
            self._command_curiculum_level[time_out],
            self._curriculum_progress[time_out],
            self._num_scheme_curriculums[time_out],
            self._max_command_curriculum[time_out],
            self.cfg.num_games_per_level,
        )

        timeout_idx = time_out.nonzero(as_tuple=True)[0]

        self._game_won[timeout_idx] |= new_game_won

        random_curriculum_mask = time_out * self._game_won
        num_random_curriculum = int(random_curriculum_mask.count_nonzero().item())
        if num_random_curriculum > 0:
            self._command_curiculum_level[random_curriculum_mask] = sample_uniform(
                self._min_command_curriculum[random_curriculum_mask],
                self._max_command_curriculum[random_curriculum_mask],
                num_random_curriculum,
                self.device,
            ).int()
            self._scheme_curiculum_level[random_curriculum_mask] = sample_uniform(
                0,
                self._num_scheme_curriculums[random_curriculum_mask],
                num_random_curriculum,
                self.device,
            ).int()

        # logging
        curriculums = dict()
        for scheme in list(self.cfg.scheme_fraqs.keys()):
            mask = self._init_schemes == InitializationScheme[scheme]
            if mask.sum() == 0:
                continue
            curriculums[f"{scheme}/mean_scheme_curriclum"] = self._scheme_curiculum_level[mask].float().mean().item()
            curriculums[f"{scheme}/mean_command_curriclum"] = self._command_curiculum_level[mask].float().mean().item()
            curriculums[f"{scheme}/game_won_fraq"] = self._game_won[mask].float().mean().item()

        self.extras.update(curriculums)

        if self.jump_test_enabled:
            died = False

        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        '''Reset the environments.'''
        num_resets = len(env_ids)
        if env_ids is None or num_resets == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        # if self.jump_test_enabled:
            # self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(1))


        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._idle_count[env_ids] = 0
        self._episode_start_step[env_ids] = self.episode_length_buf[env_ids]

        # reset robot state and command
        state = torch.zeros(num_resets, 13 + 2 * self._robot.num_joints, device=self.device)
        commanded_jump_height = torch.zeros(num_resets, 1, device=self.device)
        air_born = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        jump_state = torch.zeros(num_resets, dtype=torch.int, device=self.device)
        for scheme, initializers in self._initializers.items():
            scheme_mask = self._init_schemes[env_ids] == scheme
            for sc in range(self.cfg.num_scheme_curriculums[scheme.name]):
                sc_mask = self._scheme_curiculum_level[env_ids] == sc
                for cc in range(
                    self.cfg.command_curriculum_limits[scheme.name][1]
                    - self.cfg.command_curriculum_limits[scheme.name][0]
                ):
                    cc_mask = self._command_curiculum_level[env_ids] == (
                        cc + self.cfg.command_curriculum_limits[scheme.name][0]
                    )
                    mask = scheme_mask * sc_mask * cc_mask
                    count = torch.count_nonzero(mask).item()
                    if count > 0:
                        commanded_jump_height[mask], state[mask] = initializers[sc][cc].draw(count)

            if scheme in [
                InitializationScheme.TOUCHDOWN,
                InitializationScheme.INFLIGHT,
            ]:
                air_born[scheme_mask] = True

            match scheme:
                case InitializationScheme.STANDING:
                    jump_state[scheme_mask] = JumpState.STANCE
                case InitializationScheme.DEFAULT:
                    jump_state[scheme_mask] = JumpState.STANCE
                case InitializationScheme.INFLIGHT:
                    jump_state[scheme_mask] = JumpState.IN_FLIGHT
                case InitializationScheme.TOUCHDOWN:
                    jump_state[scheme_mask] = JumpState.IN_FLIGHT
                case InitializationScheme.LANDED:
                    jump_state[scheme_mask] = JumpState.LANDED
                case _:
                    raise ValueError(f"Unknown initialization scheme: {scheme}")

        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :2] = 0.0
        root_pose[:, :3] += self._terrain.env_origins[env_ids]

        if self.cfg.randomize_z_orientation:
            z_angle_randomization = (torch.rand(env_ids.shape[0], device=self.device) * 2 - 1) * 3.14
            z_quat = torch.stack(
                [
                    torch.cos(z_angle_randomization / 2),
                    torch.zeros_like(z_angle_randomization),
                    torch.zeros_like(z_angle_randomization),
                    torch.sin(z_angle_randomization / 2),
                ],
                dim=1,
            )

            root_pose[:, 3:7] = quat_mul(z_quat, root_pose[:, 3:7])

        # reset state machine
        takeoff_pos = root_pose[:, :3].clone()
        takeoff_pos[~air_born, 0] -= 2 * TARGET_MAX_JUMP_LENGTH
        takeoff_pos[air_born] = self._terrain.env_origins[env_ids[air_born]] + torch.tensor(
            [[0, 0, 0.45]], device=self.device
        )
        landing_pos = root_pose[:, :3].clone()
        landing_pos[air_born] = self._terrain.env_origins[env_ids[air_born]] + torch.tensor(
            [[0, 0, 0.45]], device=self.device
        )

        max_height = torch.where(
            ((jump_state == JumpState.IN_FLIGHT) * (root_vel[:, 2] < 0) | (jump_state == JumpState.LANDED)),
            commanded_jump_height.view(-1),
            root_pose[:, 2],
        )

        jump_signal_delay = torch.randint(
            0, int(0.5 / self.step_dt), (num_resets,), device=self.device, dtype=torch.int
        )

        self._jump_state_machine.reset(
            env_ids,
            states=jump_state,
            takeoff_pos=takeoff_pos,
            land_pos=landing_pos,
            max_height=max_height,
            stance_delay=jump_signal_delay,
        )

        self._has_bin_airborne[env_ids] = jump_state != JumpState.STANCE

        # update buffers
        self._joint_pos_bias[env_ids] = sample_uniform(
            -torch.pi / 180 * self.cfg.observation_noise.joint_pos_bias_deg,
            torch.pi / 180 * self.cfg.observation_noise.joint_pos_bias_deg,
            (num_resets, 12),
            device=self.device,
        )

        for buffer in self._measurement_delay_buffers:
            buffer.reset(env_ids)
            timelags = torch.randint(
                self.cfg.latency.min_observation_time_lag,
                self.cfg.latency.max_observation_time_lag + 1,
                (num_resets,),
                device=self.device,
                dtype=torch.int,
            )
            buffer.set_time_lag(timelags, env_ids)

        self._joint_pos_target_history.reset(env_ids)
        timelags = torch.randint(
            self.cfg.latency.min_action_time_lag,
            self.cfg.latency.max_action_time_lag + 1,
            (num_resets,),
            device=self.device,
            dtype=torch.int,
        )
        self._joint_pos_target_history.set_time_lag(timelags, env_ids)

        self._goal_vec.data[env_ids] = commanded_jump_height - root_pose[:, :3]
        self._start_vec.data[env_ids] = self._terrain.env_origins[env_ids] - root_pose[:, :3]
        self._previous_root_pos[env_ids] = root_pose[:, :3]
        self._commanded_jump_height[env_ids] = commanded_jump_height
        self._original_jump_height[env_ids] = commanded_jump_height.norm(dim=1)
        self._filtered_action[env_ids] = joint_pos[:, self._actuated_joint_ids]
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/collision"] = torch.count_nonzero(self._terminate_collision[env_ids]).item()
        extras["Episode_Termination/episode_end"] = torch.count_nonzero(self._episode_end[env_ids]).item()
        extras["Episode_Termination/on_goal"] = torch.count_nonzero(self._terminate_on_goal[env_ids]).item()
        extras["Episode_Termination/walking"] = torch.count_nonzero(self._terminate_walking[env_ids]).item()
        extras["Episode_Termination/touchdown"] = torch.count_nonzero(self._terminate_touchdown[env_ids]).item()
        extras["Episode_Termination/idle"] = torch.count_nonzero(self._terminate_idle[env_ids]).item()
        extras["Episode_Termination/takeoff"] = torch.count_nonzero(self._terminate_takeoff[env_ids]).item()
        extras["Episode_Termination/landed"] = torch.count_nonzero(self._terminate_landed[env_ids]).item()
        extras["Episode_Termination/impact"] = torch.count_nonzero(self._terminate_impact[env_ids]).item()
        extras["Episode_Termination/nan"] = torch.count_nonzero(self._terminate_nan[env_ids]).item()
        extras["Episode_Termination/root_height"] = torch.count_nonzero(self._terminate_root_height[env_ids]).item()
        extras["Episode_Termination/shank_height"] = torch.count_nonzero(self._terminate_shank_height[env_ids]).item()
        self.extras.update(extras)

        # logging for plotter
        if self.logger.enable_logging:
            self.logger.save_and_clear()

    def _calculate_regularization_rewards(self) -> dict[str, Tensor]:
        # joint torques
        joint_torques = self._robot.data.applied_torque[:, self._actuated_joint_ids].square().sum(dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._actuated_joint_ids]), dim=1)
        joint_accel[self._jump_state_machine.states != JumpState.STANCE] *= 3
        
        # Use version-specific clamping
        if self.cfg._clamp_joint_accel:
            joint_accel = torch.clamp(joint_accel, max=100000000.0)
        # jerk
        jerk = (
            torch.isclose(
                self._robot.data.applied_torque[:, self._actuated_joint_ids].sgn() * self._previous_torque.sgn(),
                torch.tensor(-1.0, device=self.device).view(1, 1).expand_as(self._previous_torque),
            )
            .float()
            .sum(dim=1)
        )
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        action_rate[self._jump_state_machine.states != JumpState.STANCE] *= 3

        # action clip
        action_clip = (self._processed_actions - self._filtered_action).square().sum(dim=1)

        # contact state

        contact_state = torch.any(
            torch.abs(self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_contact_ids]) > 0.1,
            dim=1,
        ).float()

        prev_contact_state = torch.any(
            torch.abs(self._contact_sensor.data.net_forces_w_history[:, 1, self._feet_contact_ids]) > 0.1,
            dim=1,
        ).float()

        contact_change = (contact_state - prev_contact_state).abs().sum(dim=1)

        paw_forces = (
            self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_ids]
            .square()
            .sum(dim=-1)
            .mean(dim=-1)
            .mean(dim=-1)
        )

        joint_symmetry = torch.var(self._robot.data.joint_pos[:, 4:12], dim=1) + self._robot.data.joint_pos[
            :, :4
        ].square().mean(dim=1)
        rew_joint_symmetry = torch.exp(-joint_symmetry / 0.2**2)

        # joint vel symetry
        joint_vel_symmetry = torch.var(self._robot.data.joint_vel[:, 4:12], dim=1) + self._robot.data.joint_vel[
            :, :4
        ].square().mean(dim=1)
        rew_joint_vel_symmetry = torch.exp(-joint_vel_symmetry / 0.2**2)

        rewards = {
            "action_clip": action_clip * self.cfg.action_clip_reward_scale * self.step_dt,
            "dof_torques": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "paw_forces": paw_forces * self.cfg.paw_forces_reward_scale * self.step_dt,
            "jerk": jerk * self.cfg.jerk_reward_scale * self.step_dt,
            "joint_symmetry": rew_joint_symmetry * self.cfg.symmetry_reward_scale * self.step_dt,
        }

        return rewards

    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return torch.split(state, [7, self._robot.num_joints, 6, self._robot.num_joints], dim=1)

    @property
    def goal_vec(self) -> torch.Tensor:
        if self._goal_vec.timestamp < self.common_step_counter:
            self._goal_vec.data[:] = self._commands - self._robot.data.root_pos_w
            self._goal_vec.timestamp = self.common_step_counter
        return self._goal_vec.data

    @property
    def start_vec(self) -> torch.Tensor:
        if self._start_vec.timestamp < self.common_step_counter:
            self._start_vec.data[:] = self._terrain.env_origins - self._robot.data.root_pos_w
            self._start_vec.timestamp = self.common_step_counter
        return self._start_vec.data

    @property
    def takeoff_vec(self) -> Tensor:
        return self._robot.data.root_pos_w - self._jump_state_machine.takeoff_pos

    @property
    def walk_vec(self) -> Tensor:
        return self._robot.data.root_pos_w - self._jump_state_machine.land_pos

    @property
    def speed_on_goal(self) -> torch.Tensor:
        if self._speed_on_goal.timestamp < self.common_step_counter:
            self._speed_on_goal.data[:] = (
                self._robot.data.root_lin_vel_w[:, :2] * normalize(self.goal_vec[:, :2])
            ).sum(dim=1)
            self._speed_on_goal.timestamp = self.common_step_counter
        return self._speed_on_goal.data

    @property
    def rot_error_rad(self) -> torch.Tensor:
        if self._rot_error_rad.timestamp < self.common_step_counter:
            self._rot_error_rad.data[:] = quat_error_magnitude(
                self._robot.data.root_quat_w,
                self._robot.data.default_root_state[:, 3:7],
            )
            self._rot_error_rad.timestamp = self.common_step_counter
        return self._rot_error_rad.data

    @property
    def contact_state(self) -> torch.Tensor:
        if self._contact_state.timestamp < self.common_step_counter:
            self._contact_state.data[:] = (
                self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_contact_ids].norm(dim=-1) > 0.1
            )
            self._contact_state.timestamp = self.common_step_counter
        return self._contact_state.data

    @property
    def in_contact(self) -> torch.Tensor:
        return torch.any(self.contact_state, dim=1)

    @property
    def collision_state(self) -> torch.Tensor:
        if self._collision_state.timestamp < self.common_step_counter:
            self._collision_state.data[:] = torch.any(
                torch.max(
                    torch.norm(
                        self._contact_sensor.data.net_forces_w_history[:, :, self._underisred_contact_body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > self.cfg._collision_threshold, 
                dim=1,
            )
            self._collision_state.timestamp = self.common_step_counter
        return self._collision_state.data
