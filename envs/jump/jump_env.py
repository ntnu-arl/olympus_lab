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
)
import isaaclab.utils.math as math_utils

from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp

from isaaclab.utils.buffers import DelayBuffer, CircularBuffer

from envs.jump.initalization import InitializationScheme, JumpInitializerBase
from kinematics import OlympusKinematics
from control import MotorCommandFilter
from utilities import estimate_land_pos_error
from utilities.rotations import quat_to_euler_zyx
from utilities.indicies import make_slice_if_contigious


from .jump_env_config import JumpEnvCfg
from .jump_state_machine import JumpStateMachine, JumpState
from .curriculum import make_initializer, get_next_curriculum, TARGET_MAX_JUMP_LENGTH

from .simulation_logger import SimulationLogger

class JumpEnv(DirectRLEnv):

    cfg: JumpEnvCfg

    def __init__(self, cfg: JumpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_pos_error",
                "orientaion_error",
                "est_goal_pos_error",
                "angvel",
                "stance",
                "soft_impact",
                "retract_feet_in_air",
                "catch_landing",
                "damp_landing_with_legs",
                "action_clip",
                "dof_torques",
                "dof_acc_l2",
                "action_rate_l2",
                "contact_change",
                "jerk",
                "joint_symmetry",
            ]
        }

        self._init_buffers()
        self._init_indices()
        self._init_initializers()
        self._jump_state_machine = JumpStateMachine(self.num_envs, self.device)
        self._motor_command_filter = MotorCommandFilter(self.cfg.robot.motor_command_filter, self._robot)

        ################## testing and logging ##################
        self.test = False
        self.logger = SimulationLogger(enable_logging=False)
        self._touchdown_rejump_enabled = False

    def _init_buffers(self):
        with torch.device(self.device):
            self._actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._processed_actions = torch.zeros_like(self._actions)
            self._filtered_action = torch.zeros_like(self._actions)
            self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._previous_torque = torch.zeros_like(self._actions)
            self._previous_root_pos = torch.zeros(self.num_envs, 3)
            # X/Y/Z landing position commands
            self._commands = torch.zeros(self.num_envs, 3)
            self._close_to_goal_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._low_attitude_error_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._terminate_on_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self._episode_end = torch.zeros(self.num_envs, dtype=torch.bool)
            self._should_rejump = torch.zeros(self.num_envs, dtype=torch.bool)
            self._rejump_time = torch.zeros(self.num_envs, dtype=torch.long)

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
            self._max_impact_acc = torch.zeros(self.num_envs, dtype=torch.float)

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
                self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
            )

            self._measurement_delay_buffers = [
                self._lin_vel_history,
                self._ang_vel_history,
                self._rot_history,
                self._pos_history,
                self._joint_pos_history,
                self._joint_vel_history,
            ]

            self._measured_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_position = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_rot = torch.zeros(self.num_envs, 4, dtype=torch.float)
            self._measured_joint_pos = torch.zeros(
                self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
            )
            self._measured_joint_vel = torch.zeros(
                self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
            )

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
        ''' Get specific body indices '''
        self._base_contact_id = self._contact_sensor.find_bodies("Body")[0][0]
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(".*Paw.*")
        self._underisred_contact_body_ids, _ = self._contact_sensor.find_bodies(
            ["Body", "MotorHousing.*", ".*Thigh.*", ".*Shank.*"]
        )
        self._base_id = self._robot.find_bodies("Body")[0][0]
        self._shank_ids, _ = self._robot.find_bodies(".*Shank.*")
        self._feet_ids, self._feet_names = self._robot.find_bodies(".*Paw.*")
        self._motor_housing_ids, _ = self._robot.find_bodies("MotorHousing.*")
        self._actuated_joint_ids = torch.tensor(
            self._robot.find_joints(".*Motor.*")[0], device=self.device, dtype=torch.int
        )

        self._left_transversal_indices = []
        self._right_transversal_indices = []
        for end in ["F", "B"]:
            for side in ["Inner", "Outer"]:
                self._left_transversal_indices.append(self._robot.find_joints(f"{side}TransversalMotor_{end}L")[0][0])

                self._right_transversal_indices.append(self._robot.find_joints(f"{side}TransversalMotor_{end}R")[0][0])

        self._left_transversal_indices = torch.tensor(
            self._left_transversal_indices, device=self.device, dtype=torch.int
        )
        self._right_transversal_indices = torch.tensor(
            self._right_transversal_indices, device=self.device, dtype=torch.int
        )

    def _init_initializers(self):
        ''' Initialize the various initializers for each initialization scheme and curriculum level '''
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
        ''' Setup the simulation scene, including robot, terrain, and sensors '''
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

    def _pre_physics_step(self, actions: torch.Tensor):
        ''' rescale, offset, and filter the actions before applying them to the robot '''
        self._actions[:] = actions
        self._processed_actions[:, :4] = (
            self.cfg.lateral_action_scale * self._actions[:, :4] + self._robot.data.default_joint_pos[:, :4]
        )
        self._processed_actions[:, 4:] = (
            self.cfg.transversal_action_scale * self._actions[:, 4:] + self._robot.data.default_joint_pos[:, 4:12]
        )

        self._filtered_action[:] = self._joint_pos_target_history.compute(
            self._motor_command_filter.filter(
                joint_commands=self._processed_actions,
                joint_positions=self._robot.data.joint_pos[:, : self.cfg.action_space],
                joint_velocities=self._robot.data.joint_vel[:, : self.cfg.action_space],
            )
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._filtered_action, self._actuated_joint_ids)

    def _get_observations(self) -> dict:
        ''' Get observations with noise and delay, and compute observations '''
        self._previous_actions[:] = self._actions
        self._previous_torque[:] = self._robot.data.applied_torque[:, : self.cfg.action_space]

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

        commands = self._commands.clone()
        commands[:, 2] = 0

        v_w = commands - self._measured_position

        v_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(self._measured_rot), v_w)

        obs = torch.cat(
            [
                v_yaw,
                self._measured_lin_vel,
                quat_apply_inverse(self._measured_rot, self._robot.data.GRAVITY_VEC_W),
                self._measured_ang_vel,
                self._measured_joint_pos - self._robot.data.default_joint_pos[:, : self.cfg.action_space],
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
                episode_sums=self._episode_sums,
                commands=commands,
                v_yaw=v_yaw,
            )
            self.logger.log_step(obs, self.logger.extras)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        '''
        Returns a dictionary of reward components
        State-based rewards are only given when in the relevant states

        '''
        has_landed = self._jump_state_machine.states == JumpState.LANDED

        short_since_landing = (self._jump_state_machine.steps_since_touchdown < 0.3 / self.step_dt) * has_landed

        a_while_since_landing = (self._jump_state_machine.steps_since_touchdown > 0.5 / self.step_dt) * has_landed

        # position error
        goal_pos_error_squared = self.goal_vec[:, :2].square().sum(dim=1)

        goal_pos_error_reward = torch.where(
            self._jump_state_machine.steps_since_takeoff > 0,
            torch.exp(
                -(self.goal_vec[:, :2].square() / (torch.tensor([[0.15**2, 0.15**2]], device=self.device))).sum(dim=1)
            ),
            0.0,
        )

        oreientation_reward = torch.where(
            (self.speed_on_goal > 0.1 * (self._jump_state_machine.states == JumpState.IN_FLIGHT)),
            torch.exp(-self.rot_error_rad.square() / ((10 * torch.pi / 180) ** 2)),
            0.0,
        )

        estimated_land_pos_error = estimate_land_pos_error(
            self._robot.data.root_pos_w, self._commands, self._robot.data.root_lin_vel_w, -9.81
        )[:, :2]
        # estimated_land_pos_error 
        est_goal_pos_error = torch.where(
            (self._jump_state_machine.states == JumpState.IN_FLIGHT),
            torch.exp(-estimated_land_pos_error.square().sum(dim=1) / (0.30**2))
            + 0.1 * torch.exp(-estimated_land_pos_error.square().sum(dim=1) / (0.5**2)),
            0,
        )

        # angular velocity
        ang_vel = self._robot.data.root_ang_vel_w.square().sum(dim=1)

        ang_vel_reward = torch.where(
            self._jump_state_machine.states == JumpState.IN_FLIGHT, torch.exp(-ang_vel / (10.0**2)), 0.0
        )

        # stance when landed
        joint_offset = (
            self._robot.data.joint_pos[:, : self.cfg.action_space]
            - self._robot.data.default_joint_pos[:, : self.cfg.action_space]
        )
        # encourage a spesofic angle on the transversal joints
        joint_offset[:, 4:12] -= torch.pi / 180 * 25 

        stance_reward = torch.where(
            a_while_since_landing,
            torch.exp(-joint_offset.square().mean(dim=1) / ((20 * torch.pi / 180) ** 2)),
            0,
        )
        # retract feet when in air
        feet_retract_reward = torch.where(
            (self._jump_state_machine.states == JumpState.IN_FLIGHT),
            torch.exp(-joint_offset.square().mean(dim=1) / ((20 * torch.pi / 180) ** 2)),
            0.0,
        )
        # soft impact
        impact_acc = torch.sum(
            self._robot.data.body_acc_w[:, self._base_id, :3]
            * normalize(self._robot.data.body_vel_w[:, self._base_id, :3]),
            dim=-1,
        ).clamp(max=0.0)

        rew_impact = torch.where(
            short_since_landing, (1 - impact_acc / (self.cfg.termination.max_impact_acc)).clamp(min=0.0), 0
        )
        # catch landing
        catch_landing = torch.where(short_since_landing, (-self._robot.data.root_lin_vel_w[:, 2]).clamp(0, 1), 0.0)
        # damp landing with legs
        rew_damp_landing_with_legs = torch.where(
            short_since_landing, self._robot.data.joint_vel[:, :4:12].clamp(0, 1).mean(dim=-1), 0
        )

        rewards = {
            "goal_pos_error": goal_pos_error_reward
            * self.cfg.goal_pos_error_reward_scale
            * self.step_dt,
            "orientaion_error": oreientation_reward * self.cfg.attitude_error_reward_scale * self.step_dt,
            "est_goal_pos_error": est_goal_pos_error * self.cfg.est_goal_pos_error_reward_scale * self.step_dt,
            "angvel": ang_vel_reward * self.cfg.angvel_reward_scale * self.step_dt,
            "stance": stance_reward * self.cfg.stance_reward_scale * self.step_dt,
            "soft_impact": rew_impact * self.cfg.soft_impact_reward_scale * self.step_dt,
            "retract_feet_in_air": feet_retract_reward * self.cfg.retract_feet_in_air_reward_scale * self.step_dt,
            "catch_landing": catch_landing * self.cfg.catch_landing_reward_scale * self.step_dt,
            "damp_landing_with_legs": rew_damp_landing_with_legs
            * self.cfg.damp_landing_with_legs_reward_scale
            * self.step_dt,
        }

        rewards.update(self._calculate_regularization_rewards())

        for key, value in rewards.items(): # check for nan and inf
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
        if self.reset_time_outs.any():
            metrics["terminal_goal_pos_error"] = goal_pos_error_squared[self.reset_time_outs].sqrt().mean().item()
        metrics["num_airborn"] = (self._jump_state_machine.states == JumpState.IN_FLIGHT).count_nonzero().item()
        metrics["num_airborn_standing"] = (
            (self._jump_state_machine.states == JumpState.IN_FLIGHT)[
                self._init_schemes == InitializationScheme.STANDING
            ]
            .count_nonzero()
            .item()
        )

        if metrics["num_airborn"] > 0:
            metrics["airborn_attitude_error"] = (
                self.rot_error_rad[self._jump_state_machine.states == JumpState.IN_FLIGHT].mean().item()
            )

        if self._jump_state_machine.touchdown.any():
            metrics["touchdown_pos_error"] = (
                self.goal_vec[self._jump_state_machine.touchdown, :2].norm(dim=1).mean().item()
            )
            metrics["touchdown_rot_error"] = (
                self.rot_error_rad[self._jump_state_machine.touchdown].mean().rad2deg().item()
            )
            standing = (self._init_schemes == InitializationScheme.STANDING) * self._jump_state_machine.touchdown
            if standing.any():
                metrics["standing_touchdown_pos_error"] = self.goal_vec[standing, :2].norm(dim=1).mean().item()
                metrics["standing_touchdown_rot_error"] = self.rot_error_rad[standing].mean().rad2deg().item()

        stance = (self._jump_state_machine.states == JumpState.STANCE).nonzero(as_tuple=False)
        if len(stance) > 0:
            metrics["stance_paw_heigth"] = self._robot.data.body_pos_w[stance, self._feet_ids, 2].mean().item()

        metrics["touchdown_rejum_enabled"] = int(self._touchdown_rejump_enabled)

        self.extras.update({f"metrics/{key}": value for key, value in metrics.items()})

        # bookkeeping
        close_to_goal = goal_pos_error_squared < self.cfg.termination.close_to_goal_threshold**2
        self._close_to_goal_count[close_to_goal] += 1
        self._close_to_goal_count[~close_to_goal] = 0

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Update jump state machine and handle rejump logic
        Also handles termination conditions
        '''
        self._jump_state_machine.update(
            root_pos_w=self._robot.data.root_pos_w,
            root_vel_w=self._robot.data.root_vel_w,
            contact_state=self.contact_state,
        )

        ## rejump
        if not self._touchdown_rejump_enabled:

            mask = self._init_schemes == InitializationScheme.STANDING
            num = self._scheme_count[InitializationScheme.STANDING]

            self._touchdown_rejump_enabled = (
                (
                    self._command_curiculum_level[mask]
                    > self.cfg.command_curriculum_limits[InitializationScheme.STANDING.name][0]
                )
                | (self._scheme_curiculum_level[mask] > 0)
                | (self._game_won[mask])
            ).sum() > 0.9 * num

        if self._touchdown_rejump_enabled:

            self._should_rejump[self._jump_state_machine.touchdown] = (
                torch.rand(self._jump_state_machine.touchdown.count_nonzero(), device=self.device)
                < self.cfg.touchdown_rejump_prob
            )
        else:
            self._should_rejump[self._jump_state_machine.touchdown] = False

        self._rejump_time[self._jump_state_machine.touchdown] = torch.randint(
            1, int(0.7 / self.step_dt), (self._jump_state_machine.touchdown.count_nonzero(),), device=self.device
        )

        rejump_mask = (self._jump_state_machine.steps_since_touchdown == self._rejump_time) * self._should_rejump
        rejump_idx = rejump_mask.nonzero(as_tuple=False).squeeze(-1)
        num_rejump = rejump_idx.numel()
        if num_rejump > 0:
            scheme = InitializationScheme.STANDING
            commands = torch.zeros(num_rejump, 3, device=self.device)
            for cc in range(
                self.cfg.command_curriculum_limits[scheme.name][1] - self.cfg.command_curriculum_limits[scheme.name][0]
            ):

                if cc == self.cfg.command_curriculum_limits[scheme.name][1] - 1:
                    cc_mask = self._command_curiculum_level[rejump_mask] >= (
                        cc + self.cfg.command_curriculum_limits[scheme.name][0]
                    )

                else:
                    cc_mask = self._command_curiculum_level[rejump_mask] >= (
                        cc + self.cfg.command_curriculum_limits[scheme.name][0]
                    )

                count = torch.count_nonzero(cc_mask).item()
                if count > 0:
                    commands[cc_mask] = self._initializers[scheme][0][cc]._sample_command(count)
            self._commands[rejump_idx] += commands

            takeoff_pos = self._robot.data.root_pos_w[rejump_idx, :3].clone()
            takeoff_pos[:, 0] -= 2 * TARGET_MAX_JUMP_LENGTH

            landing_pos = self._robot.data.root_pos_w[rejump_idx, :3].clone()

            self._jump_state_machine.reset(
                rejump_idx,
                JumpState.STANCE,
                takeoff_pos=takeoff_pos,
                land_pos=landing_pos,
            )

            self._idle_count[rejump_idx] = 0
            self._episode_start_step[rejump_idx] = self.episode_length_buf[rejump_idx]

        is_idle = self._robot.data.root_lin_vel_w.norm(dim=1) < 0.1
        self._idle_count[self._jump_state_machine.states == JumpState.STANCE] += is_idle[
            self._jump_state_machine.states == JumpState.STANCE
        ].int()
        self._idle_count[self._jump_state_machine.states != JumpState.STANCE] = 0

        self._episode_end[:] = (self.episode_length_buf - self._episode_start_step) >= self.max_episode_length - 1
        self._terminate_on_goal[:] = self._jump_state_machine.steps_since_touchdown >= 1.0 / self.step_dt

        time_out = self._episode_end | self._terminate_on_goal

        self._terminate_collision[:] = self.collision_state

        yaw = quat_to_euler_zyx(self._robot.data.root_quat_w)[-1]
        self._terminate_touchdown[:] = self._jump_state_machine.touchdown * (
            ((self.goal_vec[:, :2].norm(dim=1) > self.cfg.termination.touchdown_pos_error))
            | (self._robot.data.root_pos_w[:, 2] < self.cfg.termination.min_touchdown_height)
            | (self.rot_error_rad.rad2deg() > 30)
            | (yaw.abs() > 10 * torch.pi / 180)
        )

        self._max_impact_acc[self._jump_state_machine.states == JumpState.STANCE] = (
            self.cfg.termination.max_impact_acc_stance
        )
        self._max_impact_acc[self._jump_state_machine.states != JumpState.STANCE] = self.cfg.termination.max_impact_acc
        self._terminate_impact[:] = (
            torch.sum(
                self._robot.data.body_acc_w[:, self._base_id, :3]
                * normalize(self._robot.data.body_vel_w[:, self._base_id, :3]),
                dim=-1,
            )
            < -self._max_impact_acc
        )

        self._terminate_nan[:] = (
            torch.isnan(self._robot.data.body_state_w[:, self._base_id]).any(dim=-1)
            | (torch.isnan(self._robot.data.joint_pos).any(dim=-1))
            | (torch.isnan(self._robot.data.joint_vel).any(dim=-1))
            | (torch.isnan(self._robot.data.joint_acc).any(dim=-1))
            | (torch.isnan(self._robot.data.body_acc_w[:, self._base_id]).any(dim=-1))
            | (torch.isnan(self._robot.data.applied_torque).any(dim=-1))
        )

        self._terminate_landed[:] = (self._jump_state_machine.states == JumpState.LANDED) * (
            yaw.abs().rad2deg() > self.cfg.termination.touchdown_rot_error
        )

        self._terminate_landed[:] = False

        self._terminate_takeoff[:] = self._jump_state_machine.takeoff * (
            (self._jump_state_machine.states == JumpState.LANDED)
        )
        estimated_landing_pos = estimate_land_pos_error(
            self._robot.data.root_pos_w, self._commands, self._robot.data.root_vel_w, -9.81
        )

        terminate_bad_takeoff = (estimated_landing_pos.norm(dim=1) > 0.45) * (
            self._jump_state_machine.steps_since_takeoff > 0.1 / self.step_dt
        )
        self._terminate_takeoff.logical_or_(terminate_bad_takeoff)

        self._terminate_walking[:] = (self._jump_state_machine.states == JumpState.STANCE) * (
            self.walk_vec[:, :2].norm(dim=-1) > self.cfg.termination.walking_distance
        )

        self._terminate_idle[:] = (self._jump_state_machine.states == JumpState.STANCE) * (
            ((self.episode_length_buf - self._episode_start_step) > 2.0 / self.step_dt)
        ) | (self._idle_count > 0.3 / self.step_dt)

        shank_height = self._robot.data.body_pos_w[:, self._shank_ids, 2]
        self._terminate_shank_height[:] = (shank_height < 0.025).any(dim=1)
        self._terminate_shank_height[:] = False

        self._terminate_root_height[:] = self._robot.data.root_pos_w[:, 2] < self.cfg.termination.min_root_height
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
        )

        # curriculums
        within_curriculum_thresh = self.goal_vec[:, :2].norm(dim=1) < self.cfg.curriculum_threshold
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

        # Logging
        curriculums = dict()
        for scheme in list(self.cfg.scheme_fraqs.keys()):
            mask = self._init_schemes == InitializationScheme[scheme]
            curriculums[f"{scheme}/mean_scheme_curriclum"] = self._scheme_curiculum_level[mask].float().mean().item()
            curriculums[f"{scheme}/mean_command_curriclum"] = self._command_curiculum_level[mask].float().mean().item()
            curriculums[f"{scheme}/game_won_fraq"] = self._game_won[mask].float().mean().item()

        self.extras.update(curriculums)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        ''' Reset the environments at the given indices '''
        num_resets = len(env_ids)
        if env_ids is None or num_resets == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._close_to_goal_count[env_ids] = 0
        self._idle_count[env_ids] = 0
        self._episode_start_step[env_ids] = self.episode_length_buf[env_ids]

        # reset robot state and command
        state = torch.zeros(num_resets, 13 + 2 * self._robot.num_joints, device=self.device)
        command = torch.zeros(num_resets, 3, device=self.device)
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
                        converged = False
                        while not converged:
                            try:
                                command[mask], state[mask] = initializers[sc][cc].draw(count)
                                converged = True
                            except Exception as e:
                                print(f"Error while drawing command for {scheme.name}. Retrying...")
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

        yaw_angle = torch.rand(num_resets, device=self.device) * 2 * torch.pi * 0  ## this doeas not work yet
        yaw_quat = torch.stack(
            [
                torch.cos(yaw_angle / 2),
                torch.zeros(num_resets, device=self.device),
                torch.zeros(num_resets, device=self.device),
                torch.sin(yaw_angle / 2),
            ],
            dim=-1,
        )

        math_utils.quat_apply_yaw

        command = math_utils.quat_apply(yaw_quat, command)

        command += self._terrain.env_origins[env_ids]
        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :2] += self._terrain.env_origins[env_ids, :2]
        root_pose[:, 3:7] = math_utils.quat_mul(yaw_quat, root_pose[:, 3:7])

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

        self._jump_state_machine.reset(
            env_ids,
            states=jump_state,
            takeoff_pos=takeoff_pos,
            land_pos=landing_pos,
        )

        should_rejump = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        num_landed = torch.count_nonzero(self._init_schemes[env_ids] == InitializationScheme.LANDED)
        should_rejump[self._init_schemes[env_ids] == InitializationScheme.LANDED] = (
            torch.rand(num_landed, device=self.device) < self.cfg.landed_rejump_prob
        )
        num_rejump = torch.count_nonzero(should_rejump)
        rejmup_time = (
            self._jump_state_machine.steps_since_touchdown[env_ids[should_rejump]]
            + torch.randint(0, int(0.7 / self.step_dt), (num_rejump,), device=self.device)
        ).clamp(max=(int(0.7 / self.step_dt) - 1))

        self._should_rejump[env_ids] = should_rejump
        self._rejump_time[env_ids] = 0
        self._rejump_time[env_ids[should_rejump]] = rejmup_time

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

        self._goal_vec.data[env_ids] = command - root_pose[:, :3]
        self._start_vec.data[env_ids] = self._terrain.env_origins[env_ids] - root_pose[:, :3]
        self._previous_root_pos[env_ids] = root_pose[:, :3]
        self._commands[env_ids] = command
        self._filtered_action[env_ids] = joint_pos[:, : self.cfg.action_space]
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
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
        joint_torques = self._robot.data.applied_torque[:, : self.cfg.action_space].square().sum(dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, : self.cfg.action_space]), dim=1)
        # jerk
        jerk = (
            torch.isclose(
                self._robot.data.applied_torque[:, : self.cfg.action_space].sgn() * self._previous_torque.sgn(),
                torch.tensor(-1.0, device=self.device).view(1, 1).expand_as(self._previous_torque),
            )
            .float()
            .sum(dim=1)
        )
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # action clip
        action_clip = (self._processed_actions - self._filtered_action).square().sum(dim=1)
        # contact state
        contact_state = torch.any(
            torch.abs(self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_contact_ids]) > 0.1,
            dim=1,
        ).float()
        # previous contact state
        prev_contact_state = torch.any(
            torch.abs(self._contact_sensor.data.net_forces_w_history[:, 1, self._feet_contact_ids]) > 0.1,
            dim=1,
        ).float()
        # contact change
        contact_change = (contact_state - prev_contact_state).abs().sum(dim=1)
        # symmetry reward
        joint_symmetry = (
            (
                self._robot.data.joint_pos[:, self._left_transversal_indices]
                - self._robot.data.joint_pos[:, self._right_transversal_indices]
            )
            .square()
            .mean(dim=1)
        )
        rew_joint_symmetry = torch.exp(-joint_symmetry / 0.5**2)

        rewards = {
            "action_clip": action_clip * self.cfg.action_clip_reward_scale * self.step_dt,
            "dof_torques": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "contact_change": contact_change * self.cfg.contact_change_reward_scale * self.step_dt,
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
                > 1.0,
                dim=1,
            )
            self._collision_state.timestamp = self.common_step_counter
        return self._collision_state.data
