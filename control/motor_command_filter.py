from typing import List, Tuple
from torch import Tensor

from dataclasses import dataclass, MISSING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.utils.configclass import configclass


@configclass
class MotorCommandFilterCfg:
    lateral_motor_joint_limits: Tuple[float, float] = MISSING
    transversal_motor_joint_limits: Tuple[float, float] = MISSING
    transversal_joint_sum_limits: Tuple[float, float] = MISSING
    filter_threshold: float = MISSING
    adaptive_filter: bool = MISSING
    upper_angle_threshold: float = MISSING 
    lower_angle_threshold: float = MISSING
    upper_sum_threshold: float = MISSING  
    lower_sum_threshold: float = MISSING  
    velocity_threshold: float = MISSING  


class MotorCommandFilter:
    '''
    Motor command filter that adaptively clamps joint commands based on proximity to joint limits and joint velocities.
    Supports both adaptive filtering and fixed threshold filtering modes.
    '''
    def __init__(self, cfg: MotorCommandFilterCfg, olympus: Articulation):
        self._cfg = cfg
        self._num_envs = olympus.num_instances
        self._device = olympus.device
        self._adaptive_filter = cfg.adaptive_filter

        self._lateral_motor_idx = torch.tensor(
            olympus.find_joints("LateralMotor.*")[0], device=olympus.device, dtype=torch.int
        )
        self._transversal_motor_idx = olympus.find_joints(".*TransversalMotor.*")[0]
        self._inner_transversal_motor_idx = olympus.find_joints("InnerTransversalMotor.*")[0]
        self._outer_transversal_motor_idx = olympus.find_joints("OuterTransversalMotor.*")[0]

        self._upper_motor_joint_limit = torch.zeros(
            self._num_envs,
            len(self._lateral_motor_idx) + len(self._transversal_motor_idx),
            device=olympus.device,
        )
        self._lower_motor_joint_limit = torch.zeros_like(self._upper_motor_joint_limit)
        self._lower_motor_joint_limit[:, self._lateral_motor_idx] = cfg.lateral_motor_joint_limits[0]
        self._lower_motor_joint_limit[:, self._transversal_motor_idx] = cfg.transversal_motor_joint_limits[0]
        self._upper_motor_joint_limit[:, self._lateral_motor_idx] = cfg.lateral_motor_joint_limits[1]
        self._upper_motor_joint_limit[:, self._transversal_motor_idx] = cfg.transversal_motor_joint_limits[1]
        self._lower_motor_joint_limit = self._lower_motor_joint_limit.deg2rad()
        self._upper_motor_joint_limit = self._upper_motor_joint_limit.deg2rad()

        self._lower_transversal_sum_limit = torch.full(
            (self._num_envs, 1), cfg.transversal_joint_sum_limits[0], device=olympus.device, dtype=torch.float
        ).deg2rad()

        self._upper_transversal_sum_limit = torch.full(
            (self._num_envs, 1), cfg.transversal_joint_sum_limits[1], device=olympus.device, dtype=torch.float
        ).deg2rad()

        self._filter_threshold = torch.full(
            (self._num_envs, 1), cfg.filter_threshold, device=olympus.device, dtype=torch.float
        )

    def set_motor_joint_limits(
        self,
        lateral_motor_joint_limits: tuple[Tensor, Tensor] | None,
        transversal_motor_joint_limits: tuple[Tensor, Tensor] | None,
        env_ids: Tensor | slice,
        is_degrees: bool = True,
    ):

        env_ids = env_ids.view(-1, 1)
        if lateral_motor_joint_limits is not None:
            self._lower_motor_joint_limit[env_ids, self._lateral_motor_idx] = lateral_motor_joint_limits[0].view(-1, 1)
            self._upper_motor_joint_limit[env_ids, self._lateral_motor_idx] = lateral_motor_joint_limits[1].view(-1, 1)

        if transversal_motor_joint_limits is not None:
            self._lower_motor_joint_limit[env_ids, self._transversal_motor_idx] = transversal_motor_joint_limits[
                0
            ].view(-1, 1)
            self._upper_motor_joint_limit[env_ids, self._transversal_motor_idx] = transversal_motor_joint_limits[
                1
            ].view(-1, 1)

        if is_degrees:
            self._lower_motor_joint_limit[env_ids] = self._lower_motor_joint_limit[env_ids].deg2rad()
            self._upper_motor_joint_limit[env_ids] = self._upper_motor_joint_limit[env_ids].deg2rad()

    def set_transversal_joint_sum_limits(
        self,
        transversal_joint_sum_limits: tuple[Tensor, Tensor],
        env_ids: Tensor | slice,
        is_degrees: bool = True,
    ):
        self._lower_transversal_sum_limit[env_ids] = transversal_joint_sum_limits[0].view(-1, 1)
        self._upper_transversal_sum_limit[env_ids] = transversal_joint_sum_limits[1].view(-1, 1)

        if is_degrees:
            self._lower_transversal_sum_limit[env_ids] = self._lower_transversal_sum_limit[env_ids].deg2rad()
            self._upper_transversal_sum_limit[env_ids] = self._upper_transversal_sum_limit[env_ids].deg2rad()

    def set_filter_threshold(self, filter_threshold: Tensor, env_ids: Tensor | slice):
        self._filter_threshold[env_ids] = filter_threshold.view(-1, 1)

    def filter(self, joint_commands: Tensor, joint_positions: Tensor, joint_velocities: Tensor) -> Tensor:

        if self._adaptive_filter:
            margin_motor_limits_lower_rad = joint_positions[:, :12] - self._lower_motor_joint_limit
            margin_motor_limits_upper_rad = self._upper_motor_joint_limit - joint_positions[:, :12]

            out_of_joint_limits = torch.logical_or(margin_motor_limits_lower_rad < 0, margin_motor_limits_upper_rad < 0)

            velocity_threshold = self._cfg.velocity_threshold * torch.pi / 180  
            safe_velocities = torch.where(
                torch.abs(joint_velocities[:, :12]) < velocity_threshold,
                torch.sign(joint_velocities[:, :12]) * velocity_threshold,
                joint_velocities[:, :12]
            )

            time_to_upper_limit = torch.where(
                (safe_velocities > 0) & (margin_motor_limits_upper_rad > 0),
                margin_motor_limits_upper_rad / safe_velocities,
                float('inf')
            )
            
            time_to_lower_limit = torch.where(
                (safe_velocities < 0) & (margin_motor_limits_lower_rad > 0),
                margin_motor_limits_lower_rad / (-safe_velocities),
                float('inf')
            )

            min_time_to_limit = torch.minimum(time_to_upper_limit, time_to_lower_limit)
            
            blend_factor_limits = torch.where(
                out_of_joint_limits,
                1.0,
                torch.clamp(1.0 - (min_time_to_limit / self._cfg.filter_threshold), 0.0, 1.0)
            )

            upper_angle_threshold = self._cfg.upper_angle_threshold * torch.pi / 180  
            lower_angle_threshold = self._cfg.lower_angle_threshold * torch.pi / 180  

            near_upper_limit = margin_motor_limits_upper_rad < upper_angle_threshold
            near_lower_limit = margin_motor_limits_lower_rad < lower_angle_threshold

            command_toward_upper = joint_commands[:, :12] > joint_positions[:, :12]
            command_toward_lower = joint_commands[:, :12] < joint_positions[:, :12]

            safe_direction_upper = near_upper_limit & command_toward_lower  
            safe_direction_lower = near_lower_limit & command_toward_upper 

            blend_factor_limits = torch.where(safe_direction_upper | safe_direction_lower, 0.0, blend_factor_limits)

            dangerous_direction_upper = near_upper_limit & command_toward_upper
            dangerous_direction_lower = near_lower_limit & command_toward_lower
            blend_factor_limits = torch.where(dangerous_direction_upper | dangerous_direction_lower, 1.0, blend_factor_limits)

            outer_transversal_joint_pos = joint_positions[:, self._outer_transversal_motor_idx]
            inner_transversal_joint_pos = joint_positions[:, self._inner_transversal_motor_idx]
            outer_transversal_joint_vel = joint_velocities[:, self._outer_transversal_motor_idx]
            inner_transversal_joint_vel = joint_velocities[:, self._inner_transversal_motor_idx]
            sum_transversal_joint_pos = outer_transversal_joint_pos + inner_transversal_joint_pos
            sum_transversal_joint_vel = outer_transversal_joint_vel + inner_transversal_joint_vel

            margin_transversal_joint_sum_lower_rad = sum_transversal_joint_pos - self._lower_transversal_sum_limit
            margin_transversal_joint_sum_upper_rad = self._upper_transversal_sum_limit - sum_transversal_joint_pos

            out_of_transversal_joint_sum_limits = torch.logical_or(
                margin_transversal_joint_sum_lower_rad < 0,
                margin_transversal_joint_sum_upper_rad < 0,
            )

            safe_sum_velocities = torch.where(
                torch.abs(sum_transversal_joint_vel) < velocity_threshold,
                torch.sign(sum_transversal_joint_vel) * velocity_threshold,
                sum_transversal_joint_vel
            )

            time_to_upper_sum_limit = torch.where(
                (safe_sum_velocities > 0) & (margin_transversal_joint_sum_upper_rad > 0),
                margin_transversal_joint_sum_upper_rad / safe_sum_velocities,
                float('inf')
            )
            
            time_to_lower_sum_limit = torch.where(
                (safe_sum_velocities < 0) & (margin_transversal_joint_sum_lower_rad > 0),
                margin_transversal_joint_sum_lower_rad / (-safe_sum_velocities),
                float('inf')
            )

            min_time_to_sum_limit = torch.minimum(time_to_upper_sum_limit, time_to_lower_sum_limit)
            
            blend_factor_sum = torch.where(
                out_of_transversal_joint_sum_limits,
                1.0,
                torch.clamp(1.0 - (min_time_to_sum_limit / self._filter_threshold), 0.0, 1.0)
            )

            upper_sum_threshold = self._cfg.upper_sum_threshold * torch.pi / 180   
            lower_sum_threshold = self._cfg.lower_sum_threshold * torch.pi / 180   

            near_upper_sum_limit = margin_transversal_joint_sum_upper_rad < upper_sum_threshold
            near_lower_sum_limit = margin_transversal_joint_sum_lower_rad < lower_sum_threshold

            sum_commands = joint_commands[:, self._inner_transversal_motor_idx] + joint_commands[:, self._outer_transversal_motor_idx]
            command_toward_upper_sum = sum_commands > sum_transversal_joint_pos
            command_toward_lower_sum = sum_commands < sum_transversal_joint_pos

            safe_direction_upper_sum = near_upper_sum_limit & command_toward_lower_sum
            safe_direction_lower_sum = near_lower_sum_limit & command_toward_upper_sum

            blend_factor_sum = torch.where(safe_direction_upper_sum | safe_direction_lower_sum, 0.0, blend_factor_sum)

            dangerous_direction_upper_sum = near_upper_sum_limit & command_toward_upper_sum
            dangerous_direction_lower_sum = near_lower_sum_limit & command_toward_lower_sum
            blend_factor_sum = torch.where(dangerous_direction_upper_sum | dangerous_direction_lower_sum, 1.0, blend_factor_sum)

            clamped_joint_commands = self._clamp_joint_limits(joint_commands)
            blended_commands = (1.0 - blend_factor_limits) * joint_commands + blend_factor_limits * clamped_joint_commands

            blend_factor_sum_expanded = torch.zeros_like(joint_commands)
            blend_factor_sum_expanded[:, self._inner_transversal_motor_idx] = blend_factor_sum.squeeze(-1)
            blend_factor_sum_expanded[:, self._outer_transversal_motor_idx] = blend_factor_sum.squeeze(-1)

            clamped_sum_commands = self._clamp_transversal_sum(blended_commands)
            final_commands = (1.0 - blend_factor_sum_expanded) * blended_commands + blend_factor_sum_expanded * clamped_sum_commands

            return final_commands

        else:
            margin_motor_limits_lower_rad = joint_positions[:, :12] - self._lower_motor_joint_limit
            margin_motor_limits_upper_rad = joint_positions[:, :12] - self._upper_motor_joint_limit

            out_of_joint_limits = torch.logical_or(margin_motor_limits_lower_rad < 0, margin_motor_limits_upper_rad > 0)

            margin_motor_limits_upper_sek = torch.where(
                joint_velocities[:, :12] == 0,
                2 * self._cfg.filter_threshold,
                margin_motor_limits_upper_rad / joint_velocities[:, :12],
            ).clamp(min=0)

            margin_motor_limits_lower_sek = torch.where(
                joint_positions[:, :12] == 0,
                2 * self._cfg.filter_threshold,
                margin_motor_limits_lower_rad / joint_velocities[:, :12],
            ).clamp(min=0)

            close_to_joint_limits = torch.logical_or(
                margin_motor_limits_lower_sek < self._cfg.filter_threshold,
                margin_motor_limits_upper_sek < self._cfg.filter_threshold,
            )
            ## transversal joint sum limits ##
            outer_transversal_joint_pos = joint_positions[:, self._outer_transversal_motor_idx]
            inner_transversal_joint_pos = joint_positions[:, self._inner_transversal_motor_idx]
            outer_transversal_joint_vel = joint_velocities[:, self._outer_transversal_motor_idx]
            inner_transversal_joint_vel = joint_velocities[:, self._inner_transversal_motor_idx]
            sum_transversal_joint_pos = outer_transversal_joint_pos + inner_transversal_joint_pos
            sum_transversal_joint_vel = outer_transversal_joint_vel + inner_transversal_joint_vel

            margin_transversal_joint_sum_lower_rad = sum_transversal_joint_pos - self._lower_transversal_sum_limit

            margin_transversal_joint_sum_upper_rad = sum_transversal_joint_pos - self._upper_transversal_sum_limit

            out_of_transversal_joint_sum_limits = torch.logical_or(
                margin_transversal_joint_sum_lower_rad < 0,
                margin_transversal_joint_sum_upper_rad > 0,
            )

            margin_transversal_joint_sum_upper_sek = torch.where(
                sum_transversal_joint_vel == 0,
                2 * self._cfg.filter_threshold,
                margin_transversal_joint_sum_upper_rad / sum_transversal_joint_vel,
            ).clamp(min=0)
            margin_transversal_joint_sum_lower_sek = torch.where(
                sum_transversal_joint_vel == 0,
                2 * self._cfg.filter_threshold,
                margin_transversal_joint_sum_lower_rad / sum_transversal_joint_vel,
            ).clamp(min=0)

            close_to_transversal_joint_sum_limits = torch.logical_or(
                margin_transversal_joint_sum_lower_sek < self._filter_threshold,
                margin_transversal_joint_sum_upper_sek < self._filter_threshold,
            )

            # calcualte clamping masks
            should_clamp_mask_limts = torch.logical_or(out_of_joint_limits, close_to_joint_limits)

            should_clamp_mask_sum = torch.zeros_like(joint_commands, dtype=torch.bool)
            should_clamp_mask_sum[:, self._inner_transversal_motor_idx] = torch.logical_or(
                close_to_transversal_joint_sum_limits, out_of_transversal_joint_sum_limits
            )
            should_clamp_mask_sum[:, self._outer_transversal_motor_idx] = should_clamp_mask_sum[
                :, self._inner_transversal_motor_idx
            ]

            clamped_joint_commands = torch.where(
                should_clamp_mask_limts,
                self._clamp_joint_limits(joint_commands),
                joint_commands,
            )

            return torch.where(
                should_clamp_mask_sum,
                self._clamp_transversal_sum(clamped_joint_commands),
                clamped_joint_commands,
            )

    def _clamp_joint_limits(self, joint_commands: Tensor) -> Tensor:
        return joint_commands.clamp(min=self._lower_motor_joint_limit, max=self._upper_motor_joint_limit)

    def _clamp_transversal_sum(self, joint_commands: Tensor) -> Tensor:

        clamped_joint_commands = joint_commands.clamp(
            min=self._lower_motor_joint_limit, max=self._upper_motor_joint_limit
        )

        clamped_joint_commands = joint_commands.clone()

        inner_joint_pos = clamped_joint_commands[:, self._inner_transversal_motor_idx]
        outer_joint_pos = clamped_joint_commands[:, self._outer_transversal_motor_idx]

        sum_joint_pos = inner_joint_pos + outer_joint_pos

        clamp_upper_mask = sum_joint_pos > self._cfg.transversal_joint_sum_limits[1] * torch.pi / 180
        clamp_lower_mask = sum_joint_pos < self._cfg.transversal_joint_sum_limits[0] * torch.pi / 180

        outer_joint_pos[clamp_upper_mask] -= (
            sum_joint_pos[clamp_upper_mask] - self._cfg.transversal_joint_sum_limits[1] * torch.pi / 180
        ) / 2
        inner_joint_pos[clamp_upper_mask] -= (
            sum_joint_pos[clamp_upper_mask] - self._cfg.transversal_joint_sum_limits[1] * torch.pi / 180
        ) / 2
        inner_joint_pos[clamp_lower_mask] -= (
            sum_joint_pos[clamp_lower_mask] - self._cfg.transversal_joint_sum_limits[0] * torch.pi / 180
        ) / 2
        outer_joint_pos[clamp_lower_mask] -= (
            sum_joint_pos[clamp_lower_mask] - self._cfg.transversal_joint_sum_limits[0] * torch.pi / 180
        ) / 2

        clamped_joint_commands[:, self._inner_transversal_motor_idx] = inner_joint_pos
        clamped_joint_commands[:, self._outer_transversal_motor_idx] = outer_joint_pos

        return clamped_joint_commands

    @property
    def device(self) -> str:
        return self._device