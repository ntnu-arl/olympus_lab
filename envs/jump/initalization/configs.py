from torch import Tensor
from typing import Tuple, Dict, List

from dataclasses import dataclass, MISSING

import torch

from isaaclab.utils.configclass import configclass


@configclass
class CommandCfg:
    landing_pos_limits: Tuple[List[float], List[float]] = MISSING


@configclass
class FlightTrajectoryCfg:
    takeoff_pos_limits: Tuple[List[float], List[float]] = MISSING
    takeoff_angle_limits: Tuple[float, float] = MISSING
    jump_length_limits: Tuple[List[float], List[float]] = MISSING


@configclass
class TouchdownInitializerCfg:
    base_euler_limits: Tuple[List[float], List[float]] = MISSING
    base_ang_vel_limits: Tuple[List[float], List[float]] = MISSING
    lateral_joints_limits: Tuple[float, float] = MISSING
    front_transversal_joints_limits: Tuple[float, float] = MISSING
    back_transversal_joints_limits: Tuple[float, float] = MISSING
    time_to_land_limits: Tuple[float, float] = MISSING


@configclass
class StandingInitializerCfg:
    base_pos_limits: Tuple[List[float], List[float]] = MISSING
    base_euler_limits: Tuple[List[float], List[float]] = MISSING
    paw_pos_limits: Dict[str, Tuple[List[float], List[float]]] = MISSING


@configclass
class LandedInitializerCfg(StandingInitializerCfg):
    pass


@configclass
class InflightInitializerCfg:
    lateral_joints_limits: Tuple[float, float] = MISSING
    transversal_joints_limits: Tuple[float, float] = MISSING
    base_euler_limits: Tuple[List[float], List[float]] = MISSING
    base_ang_vel_limits: Tuple[List[float], List[float]] = MISSING
    normalized_time_limits: Tuple[float, float] = MISSING

