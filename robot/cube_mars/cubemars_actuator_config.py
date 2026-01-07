from dataclasses import MISSING
from isaaclab.utils import configclass

import torch

from isaaclab.actuators import DelayedPDActuatorCfg

from . import cubmars_actuator_model


@configclass
class CubeMarsMotorCfg(DelayedPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = cubmars_actuator_model.CubeMarsMotor

    cutoff_speed: float = MISSING
    """The speed at which the motor torque starts to saturate (in rad/s)."""
    no_load_speed: float = MISSING
    """The speed at which the motor torque is zero (in rad/s)."""
    safe_velocity: float | None = None
    """The safe velocity for the motor (in rad/s)."""
    safe_effort_limit: float = MISSING
    """The safe effort limit for the motor (in Nm).NOTE: this is the one we clip to."""
