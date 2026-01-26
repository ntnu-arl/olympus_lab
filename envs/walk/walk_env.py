# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Jørgen Anker Olsen, NTNU Autonomous Robots Lab, 2026

from __future__ import annotations

from typing import Tuple, List
from torch import Tensor
import numpy as np

import torch
from torch.nn.functional import normalize

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_error_magnitude,
    axis_angle_from_quat,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)
from isaaclab.utils.math import quat_mul, sample_uniform

from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp
from isaaclab.utils.buffers import DelayBuffer, CircularBuffer


from kinematics import OlympusKinematics
from .walk_initializer import WalkInitializer
from control import MotorCommandFilter
from utilities.indicies import make_slice_if_contigious

from .walk_env_config import WalkEnvCfg

from .keyboard_control import KeyboardCommands
from .simulation_logger import SimulationLogger


class WalkEnv(DirectRLEnv):
    ''' Walk environment for the robot simulation. '''
    cfg: WalkEnvCfg

    def __init__(self, cfg: WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "stand_joint_pos",
                "flat_orientation_l2",
                "action_clip",
                "contact_change",
                "jerk",
                "lateral_motor_penalty",
                "transversal_motor_penalty",
                "rapid_stepping_penalty",
            ]
        }
        self._init_buffers()
        self._init_indices()
        self._init_initializer()
        self._motor_command_filter = MotorCommandFilter(self.cfg.robot.motor_command_filter, self._robot)
        self.commands_extras = dict()

        ##### Test mode #####
        self.test = False
        self.logger = SimulationLogger(enable_logging=False)  # Set to True to enable logging
        if self.test:
            # self.keyboard_control = KeyboardCommands(num_envs=self.num_envs, device=self.device) # manual keyboard control
            self.keyboard_control = KeyboardCommands( # use preprogrammed command sequence
            num_envs=self.num_envs, 
            device=self.device,
            obs_freq=60,
            auto_start_script=True 
        )

    def _init_buffers(self):
        ''' Initialize buffers and tensors for actions, observations, and delays. '''
        with torch.device(self.device):
            self._actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._processed_actions = torch.zeros_like(self._actions)
            self._filtered_action = torch.zeros_like(self._actions)
            self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._previous_torque = torch.zeros_like(self._actions)
            self._commands = torch.zeros(self.num_envs, 3)
            self._standing_still = torch.zeros(self.num_envs, dtype=torch.bool)
            self._joint_pos_bias = torch.zeros(self.num_envs, self._robot.num_joints)

            self._terminate_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self._lin_vel_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._ang_vel_history = DelayBuffer(
                self.cfg.latency.max_observation_time_lag, batch_size=self.num_envs, device=self.device
            )
            self._projected_gravity_history = DelayBuffer(
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

            self._measurement_delay_buffers = [
                self._lin_vel_history,
                self._ang_vel_history,
                self._projected_gravity_history,
                self._joint_pos_history,
                self._joint_vel_history,
            ]

            self._measured_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float)
            self._measured_joint_pos = torch.zeros(
                self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
            )
            self._measured_joint_vel = torch.zeros(
                self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
            )

    def _init_indices(self):
        ''' Get specific body indices '''
        self._base_id = make_slice_if_contigious(self._robot.find_bodies("Body")[0])
        self._feet_ids = make_slice_if_contigious(self._robot.find_bodies(".*Paw.*")[0])
        self._actuated_joint_ids = make_slice_if_contigious(self._robot.find_joints(".*Motor.*")[0])
        self._feet_contact_ids = make_slice_if_contigious(self._contact_sensor.find_bodies(".*Paw.*")[0])
        self._undesired_contact_ids = make_slice_if_contigious(
            self._contact_sensor.find_bodies(["Body", ".*MotorHousing.*", ".*Thigh.*", ".*Shank.*"])[0]
        )
        assert self._robot.find_bodies(".*Paw.*")[1] == self._contact_sensor.find_bodies(".*Paw.*")[1]

    def _init_initializer(self):
        ''' Initialise kinematics and initializer '''
        self._kinematics = OlympusKinematics()

        self._intializer = WalkInitializer(
            self.cfg.initializer,
            self._robot,
            self._kinematics,
        )

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
        dome_light = sim_utils.DomeLightCfg(
            intensity=500.0,
            color=(0.8, 0.8, 0.8),
        )
        dome_light.func("/World/DomeLight", dome_light)

        dir_light = sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 1.0, 1.0),
            angle=45.0,
        )
        dir_light.func("/World/DirectionalLight", dir_light)

    def _pre_physics_step(self, actions: torch.Tensor):
        ''' Rescale, offset, and filter the actions before applying them to the robot '''  
        self._actions[:] = actions

        self._processed_actions[:, :4] = (
            self.cfg.lateral_action_scale * self._actions[:, :4] + self._robot.data.default_joint_pos[:, :4]
        )
        self._processed_actions[:, 4:12] = (
            self.cfg.transversal_action_scale * self._actions[:, 4:12] + self._robot.data.default_joint_pos[:, 4:12]
        )
        self._previous_processed_actions = self._processed_actions
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
        ''' Get the observations for the robot, including adding noise and handling delays. '''
        self._previous_actions[:] = self._actions.clone()
        self._previous_torque[:] = self._robot.data.applied_torque[:, self._actuated_joint_ids]
        if self.test:
            self.keyboard_control.update_commands()
            self._commands = self.keyboard_control.commands

        add_observation_noise = True
        if add_observation_noise:
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

            self._measured_projected_gravity[:] = self._projected_gravity_history.compute(
                self._robot.data.projected_gravity_b
                + torch.randn_like(self._robot.data.projected_gravity_b).clamp(-1, 1)
                * self.cfg.observation_noise.projected_gravity_noise
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

        else:
            self._measured_lin_vel[:] = self._robot.data.root_lin_vel_b
            self._measured_ang_vel[:] = self._robot.data.root_ang_vel_b
            self._measured_projected_gravity[:] = self._robot.data.projected_gravity_b
            self._measured_joint_pos[:] = self._robot.data.joint_pos[:, self._actuated_joint_ids]
            self._measured_joint_vel[:] = self._robot.data.joint_vel[:, self._actuated_joint_ids]

        obs = torch.cat(
            [
                self._measured_lin_vel,
                self._measured_ang_vel,
                self._measured_projected_gravity,
                self._commands,
                self._measured_joint_pos - self._robot.data.default_joint_pos[:, self._actuated_joint_ids],
                self._measured_joint_vel,
                self._actions,
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        # prototype logging for plotting
        if self.logger.enable_logging:

            self.commands_extras["commands"] = self._commands
            self.commands_extras["lin_vel"] = self._robot.data.root_lin_vel_b[:, :2]
            self.commands_extras["lin_vel_error"] = torch.sum(
                torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),
                dim=1,
            )
            self.commands_extras["yaw_rate"] = self._robot.data.root_ang_vel_b[:, 2]
            self.commands_extras["yaw_rate_error"] = torch.square(
                self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
            )
            self.commands_extras["z_vel"] = self._robot.data.root_lin_vel_b[:, 2]
            self.commands_extras["ang_vel_error"] = torch.sum(
                torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
            )
            self.commands_extras["projected_gravity"] = self._robot.data.projected_gravity_b
            self.commands_extras["quaternion"] = self._robot.data.root_quat_w

            self.commands_extras["paw_height"] = self._robot.data.body_pos_w[:, self._feet_ids, 2]
            self.commands_extras["paw_velocities"] = self._robot.data.body_lin_vel_w[:, self._feet_ids]

            self.logger.update_extras(
                robot=self._robot,
                actions=self._actions,
                processed_actions=self._processed_actions,
                filtered_actions=self._filtered_action,
                mean_filtered_actions=self._filtered_action,
                obs=obs,
                episode_sums=self._episode_sums,
                commands_extras=self.commands_extras,
            )
            self.logger.log_step(obs, self.logger.extras)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        ''' calculate the reward for the current timestep based on various criteria. '''
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        joint_accel[self._standing_still] *= 3  # tripple the reward when standing still
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        air_time = torch.sum((last_air_time - 0.25) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undersired contacts
        contacts = torch.sum(
            torch.max(
                torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # hip position reward
        hip_pos = self._robot.data.joint_pos[:, :4]
        hip_angle_error = torch.abs(hip_pos)
        hip_reward = torch.where(hip_angle_error < 0.2, 1.0 - hip_angle_error / 0.2, -hip_angle_error)
        hip_reward = torch.mean(hip_reward, dim=1)

        # nominal mactuator angle reward
        lateral_motor_angle = self._robot.data.joint_pos[:, :4]
        lateral_motor_error = lateral_motor_angle  # default is 0
        lateral_motor_penalty = torch.exp(-100 * torch.pow(lateral_motor_error, 8)) - 1
        lateral_motor_penalty = torch.sum(lateral_motor_penalty, dim=1)

        # transversal motor angle reward
        transversal_motor_angle = self._robot.data.joint_pos[:, 4:12]
        transversal_motor_angle_added_rad = torch.tensor(10 * np.pi / 180, device=self.device)
        transversal_motor_error = (
            transversal_motor_angle - self._robot.data.default_joint_pos[:, 4:12] - transversal_motor_angle_added_rad
        )
        transversal_motor_penalty = torch.exp(-5000 * torch.pow(transversal_motor_error, 20)) - 1
        transversal_motor_penalty = torch.sum(transversal_motor_penalty, dim=1)

        # standing joint position reward
        stand_joint_pos = torch.where(
            self._standing_still,
            torch.exp(-transversal_motor_error.square().mean(dim=1) / ((10 * torch.pi / 180) ** 2)),
            0,
        )

        # high motor velocity penalty
        motor_velocity = self._robot.data.joint_vel
        velocity_threshold = 8.0
        steepness = 1.0

        motor_velocity_penalty = torch.sigmoid(steepness * (torch.abs(motor_velocity) - velocity_threshold))
        motor_velocity_penalty = torch.sum(motor_velocity_penalty, dim=1)

        motor_velocity_penalty = self._robot.data.joint_vel[:, :12].mean(dim=1)

        # track contact history with a time window
        contact_window = 0.75
        current_contacts = self._contact_sensor.data.current_contact_time[:, self._feet_contact_ids]

        if not hasattr(self, "_contact_history"):
            history_length = int(contact_window / self.step_dt)
            self._contact_history = torch.zeros(
                (self._robot.data.joint_pos.shape[0], len(self._feet_contact_ids), history_length),
                device=self._robot.device,
            )

        self._contact_history = torch.roll(self._contact_history, shifts=-1, dims=-1)
        self._contact_history[..., -1] = (current_contacts > 0.01).float()

        contact_events = torch.zeros_like(current_contacts)
        for i in range(1, self._contact_history.shape[-1]):
            contact_events += (self._contact_history[..., i] - self._contact_history[..., i - 1]).clamp(min=0)

        contact_threshold = 2
        excess_contacts = (contact_events - contact_threshold).clamp(min=0)
        rapid_stepping_penalty = torch.sum(torch.exp(excess_contacts) - 1, dim=-1)
        rapid_stepping_penalty = torch.clamp(rapid_stepping_penalty, max=5.0)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "lateral_motor_penalty": lateral_motor_penalty * self.cfg.lateral_motor_reward_scale * self.step_dt,
            "stand_joint_pos": stand_joint_pos * self.cfg.standing_joint_pos_reward_scale * self.step_dt,
            "transversal_motor_penalty": transversal_motor_penalty
            * self.cfg.transverse_motor_reward_scale
            * self.step_dt,
            "rapid_stepping_penalty": rapid_stepping_penalty * self.cfg.rapid_stepping_penalty_scale * self.step_dt,
        }
        
        # regularization rewards
        rewards.update(self._calculate_regularization_rewards())

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.clamp(reward, 0, 1000)

        # logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Determine if the episode should terminate due to collision or timeout. '''
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._terminate_collision[:] = torch.any(
            torch.max(
                torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_ids], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

        return self._terminate_collision, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        ''' Reset the specified environments to initial states. '''
        num_resets = len(env_ids)
        if env_ids is None or num_resets == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.test:
            if len(env_ids) == self.num_envs:
                # for test mode, use full episode lengths
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=10)
        else:
            if len(env_ids) == self.num_envs:
                # spread out the resets to avoid spikes in training when many environments reset at a similar time
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.air_time_violation_died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ground_time_violation_died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._joint_pos_bias[env_ids] = sample_uniform(
            -torch.pi / 180 * self.cfg.observation_noise.joint_pos_bias_deg,
            torch.pi / 180 * self.cfg.observation_noise.joint_pos_bias_deg,
            (num_resets, self._robot.num_joints),
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
        # reset robot state
        self._sample_commands(env_ids)

        state = self._intializer.draw(num_resets)
        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]

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
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._processed_actions[env_ids] = joint_pos[:, self._actuated_joint_ids]

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
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self.extras["log"].update(extras)

        if self.logger.enable_logging:
            self.logger.save_and_clear()

        if self.test:
            if hasattr(self, "keyboard_control"):
                self.keyboard_control.cleanup()

    def _sample_commands(self, env_ids) -> None:
        ''' Sample new commands for the specified environments. '''
        num_resets = len(env_ids)
        sgns = torch.sgn(torch.rand((num_resets, 3), device=self.device) - 0.5)
        commands = torch.zeros((num_resets, 3), device=self.device).uniform_(0.12, 0.75) * sgns 

        standing = torch.rand((num_resets,), device=self.device) < 0.12 # chance to stand still
        commands[standing] = 0.0

        yaw_only = torch.rand((num_resets,), device=self.device) < 0.05 # chance to do yaw only
        yaw_only = yaw_only & (~standing)

        if yaw_only.any():
            yaw_rates = torch.zeros(yaw_only.sum(), device=self.device).uniform_(0.12, 0.75)
            yaw_signs = torch.sgn(torch.rand(yaw_only.sum(), device=self.device) - 0.5)
            commands[yaw_only, 0] = 0.0
            commands[yaw_only, 1] = 0.0
            commands[yaw_only, 2] = yaw_rates * yaw_signs

        self._standing_still[env_ids] = standing
        self._commands[env_ids] = commands

    def _calculate_regularization_rewards(self) -> dict[str, Tensor]:
        # joint torques
        joint_torques = self._robot.data.applied_torque[:, self._actuated_joint_ids].square().sum(dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._actuated_joint_ids]), dim=1)
        # jerk
        jerk = (
            torch.isclose(
                self._robot.data.applied_torque[:, self._actuated_joint_ids].sgn() * self._previous_torque.sgn(),
                torch.tensor(-1.0, device=self.device).view(1, 1).expand_as(self._previous_torque),
            )
            .float()
            .sum(dim=1)
        )
        jerk = torch.clamp(jerk, -3, 10)
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

        rewards = {
            "action_clip": action_clip * self.cfg.action_clip_reward_scale * self.step_dt,
            "jerk": jerk * self.cfg.jerk_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "contact_change": contact_change * self.cfg.contact_change_reward_scale * self.step_dt,
        }

        return rewards

    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return torch.split(state, [7, self._robot.num_joints, 6, self._robot.num_joints], dim=1)
