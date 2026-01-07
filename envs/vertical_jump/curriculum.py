from typing import Tuple, Callable, Any
from torch import Tensor

import torch

from isaaclab.assets.articulation import Articulation

from kinematics import OlympusKinematics
from .initalization import (
    DefaultInitializer,
    StandingInitializerCfg,
    InflightInitializerCfg,
    TouchdownInitializerCfg,
    LandedInitializerCfg,
    CommandCfg,
    FlightTrajectoryCfg,
    InitializationScheme,
    JumpInitializerBase,
    StandingInitializer,
    InflightInitializer,
    TouchdownInitializer,
    LandedInitializer,
)

DEFAULT_PAW_POS_LIMITS = {
    "Paw_BL": (
        [-0.22, 0.20, 0.0253],
        [-0.20, 0.22, 0.0253],
    ),
    "Paw_BR": (
        [-0.22, -0.22, 0.0253],
        [-0.20, -0.20, 0.0253],
    ),
    "Paw_FL": (
        [0.20, 0.15, 0.0253],
        [0.22, 0.17, 0.0253],
    ),
    "Paw_FR": (
        [0.20, -0.17, 0.0253],
        [0.22, -0.15, 0.0253],
    ),
}

DEFAULT_PAW_POS_LIMITS_SQUAT = {
    "Paw_BL": (
        [-0.22, 0.20, 0.03],
        [-0.20, 0.22, 0.03],
    ),
    "Paw_BR": (
        [-0.22, -0.22, 0.03],
        [-0.20, -0.20, 0.03],
    ),
    "Paw_FL": (
        [0.20, 0.15, 0.03],
        [0.22, 0.17, 0.03],
    ),
    "Paw_FR": (
        [0.20, -0.17, 0.03],
        [0.22, -0.15, 0.03],
    ),
}


def _num_levels(f: Callable[[int], Any]) -> int:
    num_levels = 0
    while True:
        try:
            _ = f(num_levels)
            num_levels += 1
        except ValueError:
            break
    return num_levels


MAX_NUM_COMMAND_CURRICULUMS = 1
# MAX_NUM_STANDING_CURRICULUMS = property(lambda: _num_levels(_get_standing_cfg))
# MAX_NUM_INFLIGHT_CURRICULUMS = property(lambda: _num_levels(_get_inflight_cfg))
# MAX_NUM_TOUCHDOWN_CURRICULUMS = property(lambda: _num_levels(_get_touchdown_cfg))
# MAX_NUM_LANDED_CURRICULUMS = property(lambda: _num_levels(_get_landed_cfg))
# MAX_NUM_DEFAULT_CURRICULUMS = property(lambda: _num_levels(_get_command_cfg))


TARGET_MAX_JUMP_LENGTH = 0.5


DEFAULT_FLIGHT_TRAJECTORY_CFG = FlightTrajectoryCfg(
    takeoff_pos_limits=(
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.45],
    ),
)


def make_initializer(
    scheme: InitializationScheme,
    scheme_curriculum: int,
    command_curriculum: int,
    olympus: Articulation,
    kinematics: OlympusKinematics,
) -> JumpInitializerBase:
    match scheme:
        case InitializationScheme.STANDING:
            return StandingInitializer(
                cfg=_get_standing_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.INFLIGHT:
            return InflightInitializer(
                cfg=_get_inflight_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.TOUCHDOWN:
            return TouchdownInitializer(
                cfg=_get_touchdown_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.LANDED:
            return LandedInitializer(
                cfg=_get_landed_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.DEFAULT:
            return DefaultInitializer(
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case _:
            raise ValueError(f"NO curriculum defined for initalization scheme: '{scheme.name}'")


def get_next_curriculum(
    current_scheme_curriculum: Tensor,
    current_command_curriculum: Tensor,
    current_progress: Tensor,
    num_scheme_curriculums: Tensor,
    num_command_curriculums: Tensor,
    num_games_per_level: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Get the next curriculum level for the initialization scheme and command gieven the current curriculum levels

    Args:
        current_scheme_curriculum: current curriculum level for the initialization scheme
        current_command_curriculum: current curriculum level for the command
        num_scheme_curriculum: number of curriculum levels for the initialization scheme
        num_command_curriculum: number of curriculum levels for the command
    returns:
        Tuple[Tensor, Tensor, Tensor,Tensor]: next curriculum levels for the initialization scheme and command, the updated progress and a game won flag
    """
    level_up = current_progress >= (num_games_per_level - 1)
    level_up_scheme = ((current_scheme_curriculum + 1) < num_scheme_curriculums) * level_up
    level_up_command = ((current_command_curriculum + 1) < num_command_curriculums) * level_up * (~level_up_scheme)
    level_up_reset = level_up * (~level_up_scheme) * (~level_up_command)

    next_scheme_curriculum = current_scheme_curriculum.clone()
    next_command_curriculum = current_command_curriculum.clone()
    next_progress = current_progress.clone()

    next_scheme_curriculum[level_up_scheme] += 1
    next_command_curriculum[level_up_command] += 1
    next_scheme_curriculum[level_up_command] = 0
    next_scheme_curriculum[level_up_reset] = 0
    next_command_curriculum[level_up_reset] = num_command_curriculums[level_up_reset] - 1
    next_progress[level_up] = 0

    return (
        next_scheme_curriculum,
        next_command_curriculum,
        next_progress,
        level_up_reset,
    )

# curriculum ranges define the min and max values for each curriculum level

def _get_command_cfg(curriculum: int) -> CommandCfg:
    ''' Command configuration for the given curriculum level '''
    match curriculum:
        case 0:
            return CommandCfg(jump_height_limits=(0.6, 1.1))
        case 1:
            return CommandCfg(jump_height_limits=(0.6, 0.95))
        case 2:
            return CommandCfg(jump_height_limits=(0.6, 1.0))
        case 3:
            return CommandCfg(jump_height_limits=(0.65, 1.15))
        case _:
            raise ValueError(f"Invalid curriculum {curriculum}")


def _get_standing_cfg(curriculum: int) -> StandingInitializerCfg:
    ''' Standing configuration for the given curriculum level '''
    match curriculum:
        case 0:
            return StandingInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.3], [-0.0, 0.0, 0.35]),
                base_euler_limits=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 1:
            return StandingInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.25], [-0.05, 0.0, 0.45]),
                base_euler_limits=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 2:
            return StandingInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.25], [-0.05, 0.0, 0.45]),
                base_euler_limits=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 3:
            return StandingInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.25], [-0.05, 0.0, 0.45]),
                base_euler_limits=([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")


def _get_inflight_cfg(curriculum: int) -> InflightInitializerCfg:
    ''' Inflight configuration for the given curriculum level '''
    match curriculum:
        case 0:
            return InflightInitializerCfg(
                lateral_joints_limits=(0, 10),
                transversal_joints_limits=(0, 90),
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.01, 0.95),
            )
        case 1:
            return InflightInitializerCfg(
                lateral_joints_limits=(0, 10),
                transversal_joints_limits=(0, 90),
                base_euler_limits=([-2.0, -5.0, -3.0], [2, 5.0, 3.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.01, 0.95),
            )
        case 2:
            return InflightInitializerCfg(
                lateral_joints_limits=(0, 10),
                transversal_joints_limits=(0, 90),
                base_euler_limits=([-3.0, -7.0, -4.0], [3.0, 7.0, 4.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.01, 0.95),
            )
        case 3:
            return InflightInitializerCfg(
                lateral_joints_limits=(0, 10),
                transversal_joints_limits=(0, 90),
                base_euler_limits=([-3.0, -7.0, -4.0], [3.0, 7.0, 4.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.01, 0.95),
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")


def _get_touchdown_cfg(curriculum: int) -> TouchdownInitializerCfg:
    ''' Touchdown configuration for the given curriculum level '''
    match curriculum:
        case 0:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(90, 30),
                time_to_land_limits=(0.2, 0.5),
            )
        case 1:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(90, 30),
                time_to_land_limits=(0.3, 0.6),
            )
        case 2:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(90, 30),
                time_to_land_limits=(0.3, 0.6),
            )
        case 3:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(90, 30),
                time_to_land_limits=(0.2, 0.6),
            )


def _get_landed_cfg(curriculum: int) -> LandedInitializerCfg:
    ''' Get the landed configuration for the given curriculum level '''
    match curriculum:
        case 0:
            return LandedInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.4], [0.0, 0.0, 0.5]),
                base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )

        case 1:
            return LandedInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.4], [0.0, 0.0, 0.5]),
                base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 2:
            return LandedInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.4], [0.0, 0.0, 0.5]),
                base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 3:
            return LandedInitializerCfg(
                base_pos_limits=([-0.0, 0.0, 0.35], [0.0, 0.0, 0.45]),
                base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")
