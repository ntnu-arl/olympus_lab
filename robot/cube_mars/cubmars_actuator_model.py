from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.actuators import IdealPDActuator
from isaacsim.core.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .cubemars_actuator_config import CubeMarsMotorCfg


class CubeMarsMotor(IdealPDActuator):  # DelayedPDActuator makes stuff slow and has some wierd behavior

    cfg: CubeMarsMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: CubeMarsMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration

        self._cutoff_speed = self._parse_joint_parameter(self.cfg.cutoff_speed, torch.inf)
        self._no_load_speed = self._parse_joint_parameter(self.cfg.no_load_speed, torch.inf)
        self._safe_velocity = self._parse_joint_parameter(self.cfg.safe_velocity, torch.inf)
        self._safe_effort_limit = self._parse_joint_parameter(self.cfg.safe_effort_limit, torch.inf)
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        self._passive_torques = torch.zeros_like(self.computed_effort)
        
        assert (self.cfg.no_load_speed >= self._cutoff_speed).all(), \
            "The no load speed must be greater or equal the cutoff speed."
        assert self.cfg.cutoff_speed >= 0.0, "The cutoff speed must be non-negative."
    
    def set_passive_torques(self, torques: torch.Tensor):
        if torques.shape[1] != self._passive_torques.shape[1]:
            raise ValueError(
                f"Torque shape mismatch: expected {self._passive_torques.shape}, "
                f"got {torques.shape}"
            )
        self._passive_torques[:] = torques
    
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """
        Compute total joint torques = motor torques (clipped) + passive torques (unclipped).
        
        This implements superposition: motor and passive forces are independent.
        Motor torques are subject to motor performance limits.
        Passive torques bypass these limits as they're mechanical.
        """
        self._joint_vel[:] = joint_vel
        
        #  error-based PD + feed-forward from control_action.joint_efforts
        result = super().compute(control_action, joint_pos, joint_vel)
        
        # result.joint_efforts now contains the motor torques
        motor_torques = result.joint_efforts
        
        clipped_motor_torques = self._clip_effort(motor_torques)
        
        total_torques = clipped_motor_torques + self._passive_torques

        result.joint_efforts = total_torques
        
        self.computed_effort = motor_torques  
        self.applied_effort = total_torques   
        
        return result
    
    def set_motor_speed_params(
        self, no_load_speed: torch.Tensor | None, cutoff_speed: torch.Tensor | None, env_ids: torch.Tensor | slice
    ):
        """Set the no load speed for the actuator model."""
        if no_load_speed is not None:
            self._no_load_speed[env_ids] = no_load_speed.clamp(min=0)
        if cutoff_speed is not None:
            self._cutoff_speed[env_ids] = cutoff_speed.clamp(
                min=torch.zeros(1, 1, device=self._device, dtype=torch.float), max=self._no_load_speed[env_ids]
            )

    def set_safety_params(
        self, safe_velocity: torch.Tensor | None, safe_effort_limit: torch.Tensor | None, env_ids: torch.Tensor | slice
    ):
        if safe_effort_limit is not None:
            self._safe_effort_limit[env_ids] = safe_effort_limit
        if safe_velocity is not None:
            self._safe_velocity[env_ids] = safe_velocity.clamp(min=0)

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        effort = self._clip_to_torque_curve(effort)
        effort = self._clip_to_safe_velocity(effort)
        return effort

    def _clip_to_torque_curve(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits
        # -- max limit
        abs_effort_limit = (
            self.effort_limit
            * (1.0 - (self._joint_vel.abs() - self._cutoff_speed) / (self._no_load_speed - self._cutoff_speed))
        ).clip(min=self._zeros_effort, max=self.effort_limit)
        # clip the torques based on the motor limits
        return torch.clip(effort, min=-abs_effort_limit, max=abs_effort_limit)

    def _clip_to_safe_velocity(self, effort: torch.Tensor) -> torch.Tensor:
        safe_velocity = self._safe_velocity
        upper_effort_limit = torch.where(self._joint_vel > safe_velocity, 0.0, self._safe_effort_limit)
        lower_effort_limit = torch.where(self._joint_vel < -safe_velocity, 0.0, -self._safe_effort_limit)

        return torch.clip(effort, min=lower_effort_limit, max=upper_effort_limit)
