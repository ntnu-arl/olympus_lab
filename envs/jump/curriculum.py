from typing import Tuple, Callable, Any
import copy
from unittest import case
from webbrowser import get

from torch import Tensor

from isaaclab.assets.articulation import Articulation

from kinematics import OlympusKinematics
from envs.jump.initalization import (
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

# set max number of curriculums for each initialization scheme
MAX_NUM_COMMAND_CURRICULUMS = 4
# MAX_NUM_STANDING_CURRICULUMS = property(lambda: _num_levels(_get_standing_cfg))
# MAX_NUM_INFLIGHT_CURRICULUMS = property(lambda: _num_levels(_get_inflight_cfg))
# MAX_NUM_TOUCHDOWN_CURRICULUMS = property(lambda: _num_levels(_get_touchdown_cfg))
# MAX_NUM_LANDED_CURRICULUMS = property(lambda: _num_levels(_get_landed_cfg))
# MAX_NUM_DEFAULT_CURRICULUMS = property(lambda: _num_levels(_get_command_cfg))


TARGET_MAX_JUMP_LENGTH = 4.5


DEFAULT_FLIGHT_TRAJECTORY_CFG = FlightTrajectoryCfg(
    takeoff_pos_limits=(
        [0.0, 0.0, 0.4],
        [0.1, 0.0, 0.45],
    ),
    takeoff_angle_limits=(50.0, 75.0),
    jump_length_limits=(
        [0.05, -1.0],
        [TARGET_MAX_JUMP_LENGTH, 1.0],
    ),
)


def make_initializer(
    scheme: InitializationScheme,
    scheme_curriculum: int,
    command_curriculum: int,
    olympus: Articulation,
    kinematics: OlympusKinematics,
    use_spring_version: bool = False,  
) -> JumpInitializerBase:
    ''' get initializer instance based on the given scheme and curriculum levels '''
    match scheme:
        case InitializationScheme.STANDING:
            return StandingInitializer(
                cfg=_get_standing_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum, use_spring_version),
            )
        case InitializationScheme.INFLIGHT:
            return InflightInitializer(
                cfg=_get_inflight_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum, use_spring_version),
            )
        case InitializationScheme.TOUCHDOWN:
            fligt_traj_cfg = copy.deepcopy(DEFAULT_FLIGHT_TRAJECTORY_CFG)
            return TouchdownInitializer(
                cfg=_get_touchdown_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum, use_spring_version),
            )
        case InitializationScheme.LANDED:
            return LandedInitializer(
                cfg=_get_landed_cfg(scheme_curriculum),
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum, use_spring_version),
            )
        case InitializationScheme.DEFAULT:
            return DefaultInitializer(
                olympus=olympus,
                kinematics=kinematics,
                command_cfg=_get_command_cfg(command_curriculum, use_spring_version),
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


def _get_command_cfg(curriculum: int, use_spring_version: bool = False) -> CommandCfg:
    ''' Get the command configuration for the given curriculum level - Units in meters '''
    if use_spring_version:
        # SPRING VERSION CURRICULUM
        match curriculum:
            case 0:
                return CommandCfg(
                    landing_pos_limits=([2.0, 0.0, 0.3], [4.85, 0.0, 0.3]),
                )
            case 1:
                return CommandCfg(
                    landing_pos_limits=([1.6, 0.0, 0.3], [2.25, 0.0, 0.3]),
                )
            case 2:
                return CommandCfg(
                    landing_pos_limits=([2.0, -0.5, 0.3], [3.75, 0.5, 0.3]),
                )
            case 3:
                return CommandCfg(
                    landing_pos_limits=([1.5, -0.5, 0.3], [4.5, 0.5, 0.3]),
                )
            case _:
                raise ValueError(f"Invalid curriculum level: {curriculum}")
    else:
        # ORIGINAL VERSION CURRICULUM
        match curriculum:
            case 0:
                return CommandCfg(
                    landing_pos_limits=([1.4, 0.0, 0.3], [1.85, 0.0, 0.3]),
                )
            case 1:
                return CommandCfg(
                    landing_pos_limits=([1.6, 0.0, 0.3], [2.25, 0.0, 0.3]),
                )
            case 2:
                return CommandCfg(
                    landing_pos_limits=([2.0, 0.0, 0.3], [3.75, 0.0, 0.3]),
                )
            case 3:
                return CommandCfg(
                    landing_pos_limits=([3.0, 0.0, 0.3], [4.75, 0.0, 0.3]),
                )
            case _:
                raise ValueError(f"Invalid curriculum level: {curriculum}")
   


def _get_standing_cfg(curriculum: int) -> StandingInitializerCfg:
    ''' Get the standing configuration for the given curriculum level - Units in meters and degrees '''
    match curriculum:
        case 0:
            return StandingInitializerCfg(
                base_pos_limits=([-0.05, -0.05, 0.22], [0.05, 0.05, 0.35]),
                base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 1:
            return StandingInitializerCfg(
                base_pos_limits=([-0.05, -0.05, 0.22], [0.05, 0.05, 0.45]),
                base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 2:
            return StandingInitializerCfg(
                base_pos_limits=([-0.05, 0.0, 0.22], [0.05, 0.0, 0.45]),
                base_euler_limits=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case 3:
            return StandingInitializerCfg(
                base_pos_limits=([-0.05, -0.05, 0.22], [0.05, 0.05, 0.5]),
                base_euler_limits=([-10.0, -1.0, -1.0], [10.0, 1.0, 1.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")


def _get_inflight_cfg(curriculum: int) -> InflightInitializerCfg:
    ''' Get the inflight configuration for the given curriculum level - Units in degrees and seconds '''
    match curriculum:
        case 0:
            return InflightInitializerCfg(
                lateral_joints_limits=(-10, 10),
                transversal_joints_limits=(40, 80),
                base_euler_limits=([-3.0, -5.0, -2.0], [2, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.02, 0.1),
            )
        case 1:
            return InflightInitializerCfg(
                lateral_joints_limits=(-15, 10),
                transversal_joints_limits=(0, 120),
                base_euler_limits=([-20, -40, -20], [20, 40, 20.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                normalized_time_limits=(0.02, 0.1),
            )
        case 2:
            return InflightInitializerCfg(
                lateral_joints_limits=(-15, 15),
                transversal_joints_limits=(0, 120),
                base_euler_limits=([-40, -40, -40], [40, 40, 40.0]),
                base_ang_vel_limits=([-0.5, -0.5, -0.5], [0.5, 1.0, 0.5]),
                normalized_time_limits=(0.02, 0.1),
            )
        case 3:
            return InflightInitializerCfg(
                lateral_joints_limits=(-15, 15),
                transversal_joints_limits=(0, 120),
                base_euler_limits=([-60.0, -60.0, -60.0], [60.0, 60.0, 60.0]),
                base_ang_vel_limits=([-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]),
                normalized_time_limits=(0.01, 0.05),
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")


def _get_touchdown_cfg(curriculum: int) -> TouchdownInitializerCfg:
    ''' Get the touchdown configuration for the given curriculum level - Units in degrees and seconds '''
    match curriculum:
        case 0:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
                base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(50, 90),
                back_transversal_joints_limits=(10, 20),
                time_to_land_limits=(0.03, 0.15),
            )
        case 1:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -10, -2.0], [1, 10.0, 2.0]),
                base_ang_vel_limits=([0.0, -0.5, 0.0], [0.0, 0.5, 0.0]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(-20, -30),
                time_to_land_limits=(0.2, 0.5),
            )

        case 2:
            return TouchdownInitializerCfg(
                base_euler_limits=([-1.0, -10.0, -2.0], [1, 10.0, 2.0]),
                base_ang_vel_limits=([-0.5, -1.0, -0.5], [0.5, 1.0, 0.5]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(-20, -30),
                time_to_land_limits=(0.3, 0.5),
            )

        case 3:
            return TouchdownInitializerCfg(
                base_euler_limits=([-10.0, -20.0, -10.0], [10, 20.0, 10.0]),
                base_ang_vel_limits=([-0.5, -1.0, -0.5], [0.5, 1.0, 0.5]),
                lateral_joints_limits=(-5, 5),
                front_transversal_joints_limits=(90, 30),
                back_transversal_joints_limits=(-20, -30),
                time_to_land_limits=(0.3, 0.5),
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")


def _get_landed_cfg(curriculum: int) -> LandedInitializerCfg:
    ''' Get the landed configuration for the given curriculum level - Units in meters and degrees '''
    match curriculum:
        case 0:
            return LandedInitializerCfg(
                base_pos_limits=([-0.05, -0.05, 0.3], [0.05, 0.05, 0.5]),
                base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )

        case 1:
            return LandedInitializerCfg(
                base_pos_limits=([-0.15, -0.05, 0.4], [0.10, 0.05, 0.5]),
                base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
                paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")
