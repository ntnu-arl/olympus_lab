from dataclasses import MISSING
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from control import MotorCommandFilterCfg
from .cube_mars import get_AK7010_cfg, get_AK809_cfg

DEG2RAD = torch.pi / 180.0

##
# Configuration - Actuators.
##

"""Configuration for CubeMars motor with DC actuator model."""

##
# Configuration - Articulation.
##

@configclass
class OlympusConfig(ArticulationCfg):
    """Configuration of olympus robot using cube mars motor model."""

    # default articulation settings
    prim_path = "/World/envs/env_.*/Olympus"

    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{os.getcwd()}/submodules/olympus_usd/olympus-knee-damping.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,  # anymal uses 0
        ),
        
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4086),
        joint_pos={
            "LateralMotor.*": 0 * DEG2RAD,
            ".*TransversalMotor.*": 45 * DEG2RAD,
            "InnerKnee.*": 70.4116 * DEG2RAD,
            "OuterKnee.*": 70.8849 * DEG2RAD,
        },
    )

    actuators = {
        "lateral_motors": get_AK809_cfg(
            joint_names_expr=["LateralMotor.*"], kp=17.0, kd=0.9 * 1.2, safe_effort_limit=10.0, safe_velocity=2500.0
        ),
        "transversal_motors": get_AK7010_cfg(
            joint_names_expr=[".*TransversalMotor.*"],
            kp=17.0,
            kd=0.4 * 1.2,
            safe_effort_limit=10.0,
            safe_velocity=2500.0,
        ),
        "knees_damping": ImplicitActuatorCfg(
            joint_names_expr=[".*Knee.*"],
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.1,
            friction=0.0,
        ),
    }
    soft_joint_pos_limit_factor = 1.0

    ## specific olympus settings

    motor_command_filter: MotorCommandFilterCfg = MISSING



