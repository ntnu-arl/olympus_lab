from torch import pi
from .cubemars_actuator_config import CubeMarsMotorCfg


RPM2RADPS = 2.0 * pi / 60.0
DEG2RAD = pi / 180.0


def get_AK7010_cfg(
    joint_names_expr: list[str],
    kp: float,
    kd: float,
    safe_effort_limit: float,
    safe_velocity: float,
    min_delay: int = 0,
    max_delay: int = 0,
) -> CubeMarsMotorCfg:
    return CubeMarsMotorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=24.8,
        safe_effort_limit=safe_effort_limit,
        no_load_speed=382 * RPM2RADPS,
        cutoff_speed=225 * RPM2RADPS,
        safe_velocity=safe_velocity * DEG2RAD,
        stiffness={".*": kp},
        damping={".*": kd},
        friction={".*": 0.00},  # could be randomized
        armature={".*": 414 * 1e-7},
        min_delay=min_delay,
        max_delay=max_delay,
    )


def get_AK809_cfg(
    joint_names_expr: list[str],
    kp: float,
    kd: float,
    safe_effort_limit: float,
    safe_velocity: float,
    min_delay: int = 0,
    max_delay: int = 0,
) -> CubeMarsMotorCfg:
    return CubeMarsMotorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=20.0,
        safe_effort_limit=safe_effort_limit,
        no_load_speed=470 * RPM2RADPS,
        cutoff_speed=335 * RPM2RADPS,
        safe_velocity=safe_velocity * DEG2RAD,
        stiffness={".*": kp},
        damping={".*": kd},
        friction={".*": 0.00},
        armature={".*": 579 * 1e-7},
        min_delay=min_delay,
        max_delay=max_delay,
    )
