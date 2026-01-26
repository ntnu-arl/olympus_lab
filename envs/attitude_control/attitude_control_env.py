# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Jørgen Anker Olsen, NTNU Autonomous Robots Lab, 2026

from __future__ import annotations

from math import pi
from typing import Tuple, List
from torch import Tensor

import torch
from torch.nn.functional import normalize

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_error_magnitude, axis_angle_from_quat, quat_from_angle_axis, euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_mul, sample_uniform

from kinematics import OlympusKinematics
from .attitude_control_initializer import AttitudeControlInitializer
from control import MotorCommandFilter

from .attitude_control_env_config import AttitudeControlEnvCfg

from .simulation_logger import SimulationLogger

class AttitudeControlEnv(DirectRLEnv):
    '''Environment for training attitude control RL policy of the Olympus robot.'''
    cfg: AttitudeControlEnvCfg

    def __init__(self, cfg: AttitudeControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._default_lateral_stiffness = self._robot.actuators["lateral_motors"].stiffness.clone()
        self._default_lateral_damping = self._robot.actuators["lateral_motors"].damping.clone()
        self._default_transversal_stiffness = self._robot.actuators["transversal_motors"].stiffness.clone()
        self._default_transversal_damping = self._robot.actuators["transversal_motors"].damping.clone()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "orientaion_error_small",
                "orientaion_error_large",
                "angvel",
                "lateral_landing_pose_angel",
                "transversal_landing_pose_angel",
                "terminate_on_collision",
                "stability",
                "symetry_reward_sides",
                "symetry_reward_transversal",
                "action_clip",
                "dof_torques",
                "dof_acc_l2",
                "action_rate_l2",
                "jerk",
            ]
        }
        self._init_buffers()
        self._init_indices()
        self._init_initializers()
        self._motor_command_filter = MotorCommandFilter(self.cfg.robot.motor_command_filter, self._robot)

        # testing and logging
        self.test = True
        self.logger = SimulationLogger(enable_logging=False)

        # mean filter
        self.window_size = 1
        self._action_history = torch.zeros((self.window_size, self.num_envs, self.cfg.action_space), device=self.device)

    def _init_buffers(self):
        '''Initialize tensors'''
        with torch.device(self.device):
            # Joint position command (deviation from default joint positions)
            self._actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._processed_actions = torch.zeros_like(self._actions)
            self._filtered_action = torch.zeros_like(self._actions)
            self._mean_filtered_action = torch.zeros_like(self._actions)
            self._filtered_noisy_action = torch.zeros_like(self._actions)
            self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._previous_torque = torch.zeros_like(self._actions)
            self._low_attitude_error_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._terminate_orientation_error = torch.zeros(self.num_envs, dtype=torch.bool)

            self._last_orientation_error = torch.zeros(self.num_envs)
            self._idle_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._terminate_idle = torch.zeros(self.num_envs, dtype=torch.bool)
            self._inside_threshold = torch.zeros(self.num_envs, dtype=torch.bool)

            self._terminate_collision = torch.zeros(self.num_envs, dtype=torch.bool)

            self._step_counter = torch.zeros(self.num_envs, dtype=torch.int)
            self._last_action = None

            self._step_counter = torch.zeros(self.num_envs, device=self.device)
            self._step_threshold = 6000  
            self._angle_change = 35.0  
            self._initial_action = None
            self._stepped_action = None
            self.tempt_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

            self._steps_inside_threshold = torch.zeros(self.num_envs, device=self.device)

            self._default_orientation_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)
            )
            self._orientation_with_noise = None

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

            self._zero_orientation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


    def _init_indices(self):
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("Body")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*Paw.*")
        self._underisred_contact_body_ids, _ = self._contact_sensor.find_bodies(
            ["Body", "MotorHousing.*", ".*Thigh.*", ".*Shank.*"]
        )
        self._actuated_joint_ids = torch.tensor(
            self._robot.find_joints(".*Motor.*")[0], device=self.device, dtype=torch.int
        )

    def _init_initializers(self):
        self._kinematics = OlympusKinematics()

        self._initializer = AttitudeControlInitializer(
            self.cfg.attitude_control_initializer,
            self._robot,
            self._kinematics,
        )


    def _setup_scene(self):
        '''Set up the simulation scene, including robot and terrain.'''
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
        '''Process and apply actions before the physics step.'''
        self._actions[:] = actions
        # scale and offset actions
        self._processed_actions[:] = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:, : self.cfg.action_space]
        )
        # enable for smother actions during small orientation errors
        interpol_coeff = torch.exp(-self._last_orientation_error**2 / 0.03).unsqueeze(-1) * 0.0 # changed to 1.0 for testing

        current_positions = self._robot.data.joint_pos[:, : self.cfg.action_space]

        self._processed_actions[:] = (1 - interpol_coeff) * self._processed_actions + interpol_coeff * current_positions
        # motor command filter
        self._filtered_action[:] = self._motor_command_filter.filter(
            joint_commands=self._processed_actions,
            joint_positions=self._robot.data.joint_pos[:, : self.cfg.action_space],
            joint_velocities=self._robot.data.joint_vel[:, : self.cfg.action_space],
        )

        # mean filter
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=0)
        self._action_history[-1] = self._filtered_action
        mean_filtered_actions = torch.mean(self._action_history, dim=0)
        self._mean_filtered_action = mean_filtered_actions

        # add noise to the action if enabled
        self._filtered_noisy_action = self._mean_filtered_action + torch.randn_like(self._mean_filtered_action) * 0.0


    def _apply_action(self):
        '''Apply the processed actions.'''
        self._robot.set_joint_position_target(self._mean_filtered_action, self._actuated_joint_ids)


    def _get_observations(self) -> dict:
        '''Get the current observations with noise and delay.'''
        self._previous_actions[:] = self._actions
        self._previous_torque[:] = self._robot.data.applied_torque[:, : self.cfg.action_space]

        add_observation_noise = True
        if add_observation_noise:
            orientation_noise_deg = self.cfg.observation_noise.orientation_quat_noise_deg

            axis_angle = axis_angle_from_quat(self._robot.data.root_quat_w)
            angle = torch.norm(axis_angle, dim=-1)
            axis = axis_angle / (angle.unsqueeze(-1) + 1e-8)

            noisy_angle = angle + torch.randn_like(angle) * (orientation_noise_deg * torch.pi / 180)
            self._measured_orientation = quat_from_angle_axis(noisy_angle, axis)

            self._measured_ang_vel[:] = self._ang_vel_history.compute(
                self._robot.data.root_ang_vel_b
                + torch.randn_like(self._robot.data.root_ang_vel_b).clamp(-1, 1)
                * (self.cfg.observation_noise.body_ang_vel_noise_deg * torch.pi / 180)
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
            self._measured_projected_gravity[:] = self._robot.data.root_quat_w
            self._measured_joint_pos[:] = self._robot.data.joint_pos[:, self._actuated_joint_ids]
            self._measured_joint_vel[:] = self._robot.data.joint_vel[:, self._actuated_joint_ids]



        obs = torch.cat(
            [
                self._measured_orientation,
                self._measured_ang_vel,
                self._measured_joint_pos - self._robot.data.default_joint_pos[:, self._actuated_joint_ids],
                self._measured_joint_vel,
                self._actions,
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        # Logging for plotting
        if self.logger.enable_logging:
            self.logger.update_extras(
                robot=self._robot,
                actions=self._actions,
                processed_actions=self._processed_actions,
                filtered_actions=self._filtered_action,
                mean_filtered_actions=self._mean_filtered_action,
                obs=obs,
                episode_sums=self._episode_sums,
            )
            self.logger.log_step(obs, self.logger.extras)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        '''Calculate the rewards.'''
        # orientation
        rot_error_rad = quat_error_magnitude(self._robot.data.root_quat_w, self._robot.data.default_root_state[:, 3:7])

        # orientation error small
        rot_error_mapped_small = torch.exp(-rot_error_rad.square() / ((10 * torch.pi / 180) ** 2))
        # orientation error large
        rot_error_mapped_large = torch.exp(-rot_error_rad.square() / ((90 * torch.pi / 180) ** 2))

        # angular velocity reward
        tau = axis_angle_from_quat(self._robot.data.root_quat_w)
        ang_vel_proj = -(self._robot.data.root_ang_vel_w * normalize(tau)).sum(dim=1)

        ang_vel = self._robot.data.root_ang_vel_w.square().sum(dim=1)

        ang_vel_reward = torch.where(
            rot_error_rad < 170 * torch.pi / 180,
            ang_vel_proj,
            0.1 * ang_vel,
        )
        ang_vel_reward = torch.where(rot_error_rad < 5 * torch.pi / 180, 0.0, ang_vel_reward)

        # Lateral motors landing pose reward
        lateral_motors_pos = self._robot.data.joint_pos[:, 0:4]
        lateral_motors_default = (self._robot.data.default_joint_pos[:, 0:4] + 1) * torch.pi / 180
        lateral_errors = torch.abs(lateral_motors_pos - lateral_motors_default)
        landing_pose_lateral_error = torch.mean(lateral_errors, dim=1)

        # transversal motors landing pose reward
        transversal_motors_pos = self._robot.data.joint_pos[:, 4:12]
        transversal_motors_default = self._robot.data.default_joint_pos[:, 4:12]
        transversal_errors = torch.abs(transversal_motors_pos - transversal_motors_default)
        landing_pose_transversal_error = torch.mean(transversal_errors, dim=1)

        landing_pose_lateral_error_reward = rot_error_mapped_small * torch.exp(-3 * landing_pose_lateral_error**2)
        landing_pose_transversal_error_reward = rot_error_mapped_small * torch.exp(
            -5 * landing_pose_transversal_error**2
        )

        ang_vel_zero_reward = rot_error_mapped_small * torch.exp(-1000000 * ang_vel**4)

        # symetry reward lateral motors
        lateral_motors_left = self._robot.data.joint_pos[:, [0, 2]]
        lateral_motors_right = self._robot.data.joint_pos[:, [1, 3]]

        right_right_diff = torch.abs(lateral_motors_right[:, 0] - lateral_motors_right[:, 1])
        left_left_diff = torch.abs(lateral_motors_left[:, 0] - lateral_motors_left[:, 1])

        symetry_reward_sides = torch.exp(-0.5 * (right_right_diff + left_left_diff))

        # transversal symetry reward
        front_left_transversal = self._robot.data.joint_pos[:, [8, 10]]
        front_right_transversal = self._robot.data.joint_pos[:, [9, 11]]
        back_left_transversal = self._robot.data.joint_pos[:, [4, 6]]
        back_right_transversal = self._robot.data.joint_pos[:, [5, 7]]

        front_left_transversal_diff = torch.abs(front_left_transversal[:, 0] - front_left_transversal[:, 1])
        front_right_transversal_diff = torch.abs(front_right_transversal[:, 0] - front_right_transversal[:, 1])
        back_left_transversal_diff = torch.abs(back_left_transversal[:, 0] - back_left_transversal[:, 1])
        back_right_transversal_diff = torch.abs(back_right_transversal[:, 0] - back_right_transversal[:, 1])

        transversal_diff = (
            front_left_transversal_diff
            + front_right_transversal_diff
            + back_left_transversal_diff
            + back_right_transversal_diff
        )
        symetry_reward_transversal = torch.exp(-0.5 * transversal_diff**2)

        rewards = {
            "orientaion_error_small": rot_error_mapped_small
            * self.cfg.attitude_error_small_reward_scale
            * self.step_dt,
            "orientaion_error_large": rot_error_mapped_large
            * self.cfg.attitude_error_large_reward_scale
            * self.step_dt,
            "angvel": ang_vel_reward * self.cfg.angvel_reward_scale * self.step_dt,
            "terminate_on_collision": self._terminate_collision
            * self.cfg.terminate_on_collision_reward_scale
            * self.step_dt,
            "lateral_landing_pose_angel": landing_pose_lateral_error_reward
            * self.cfg.landing_pose_lateral_error_reward_scale
            * self.step_dt,
            "transversal_landing_pose_angel": landing_pose_transversal_error_reward
            * self.cfg.landing_pose_transversal_error_reward_scale
            * self.step_dt,
            "stability": ang_vel_zero_reward * self.cfg.stability_reward_scale * self.step_dt,
            "symetry_reward_sides": symetry_reward_sides * self.cfg.symetry_reward_sides_scale * self.step_dt,
            "symetry_reward_transversal": symetry_reward_transversal
            * self.cfg.symetry_reward_transversal_scale
            * self.step_dt,
        }

        rewards.update(self._calculate_regularization_rewards())
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.clamp(reward, -0.1, 1000.0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        metrics = dict()
        metrics["metrics/attitude_error"] = rot_error_rad.mean().item()

        terminate_attitude_control = self.reset_time_outs

        if terminate_attitude_control.any():
            metrics["metrics/terminal_attitude_error"] = rot_error_rad[self.reset_time_outs].mean().item()

        # Bookkeeping
        low_attitude_error = rot_error_rad < 5 * torch.pi / 180
        self._low_attitude_error_count[low_attitude_error] += 1
        self._low_attitude_error_count[~low_attitude_error] = 0

        idle = (rot_error_rad > 20 * torch.pi / 180).logical_and(
            (self._last_orientation_error - rot_error_rad).abs() < 1 * torch.pi / 180
        )
        self._idle_count[idle] += 1
        self._idle_count[(~idle)] = 0
        self._last_orientation_error[:] = rot_error_rad
        self._inside_threshold[:] = low_attitude_error

        metrics["metrics/inside_threshold"] = low_attitude_error.count_nonzero().item()

        self.extras.update(metrics)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check termination conditions
        episode_end = self.episode_length_buf >= self.max_episode_length - 1
        self._terminate_orientation_error[:] = self._low_attitude_error_count >= 4.0 / self.step_dt
        time_out = episode_end.logical_or(self._terminate_orientation_error)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self._terminate_collision[:] = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._underisred_contact_body_ids], dim=-1),
                dim=1,
            )[0]
            > 0.25,
            dim=1,
        )

        self._terminate_idle[:] = self._idle_count >= 1.5 / self.step_dt
        died = self._terminate_collision.logical_or(self._terminate_idle)
        # died = False # for testing

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        '''Reset the specified environments.'''
        num_resets = len(env_ids)
        if env_ids is None or num_resets == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            if self.test:
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=10)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._low_attitude_error_count[env_ids] = 0
        self._idle_count[env_ids] = 0

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

        # Reset robot state
        state = self._initializer.draw(num_resets)

        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, 2] = 2.0

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
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode_Termination/idle"] = torch.count_nonzero(self._terminate_idle[env_ids]).item()
        self.extras["log"].update(extras)

        # Logging for plotter
        if self.logger.enable_logging:
            self.logger.save_and_clear()

    def _calculate_regularization_rewards(self) -> dict[str, Tensor]:
        '''Calculate regularization rewards.'''
        # joint torques
        joint_torques = self._robot.data.applied_torque[:, : self.cfg.action_space].square().sum(dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, : self.cfg.action_space]), dim=1)
        joint_accel[self._zero_orientation] *= 3
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

        rewards = {
            "action_clip": action_clip * self.cfg.action_clip_reward_scale * self.step_dt,
            "dof_torques": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "jerk": jerk * self.cfg.jerk_reward_scale * self.step_dt,
        }

        return rewards

    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return torch.split(state, [7, self._robot.num_joints, 6, self._robot.num_joints], dim=1)
