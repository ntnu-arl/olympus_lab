from typing import Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from .cubemars_actuator_config import CubeMarsMotorCfg
from .cubmars_actuator_model import CubeMarsMotor


def randomize_cubemars_model(
    env: DirectRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    no_load_speed_distribution_params: tuple[float, float] | None = None,
    cutoff_speed_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = slice(None)
        num_envs = asset.num_instances
    else:
        num_envs = len(env_ids)

    # Loop through actuators and randomize gains
    for name, actuator in asset.actuators.items():
        if not isinstance(actuator, CubeMarsMotor):
            continue

        act_cfg: CubeMarsMotorCfg = asset.cfg.actuators[name]

        randomized_cutoff_speed = None
        randomized_no_load_speed = None

        if cutoff_speed_distribution_params is not None:
            randomized_cutoff_speed = torch.full(
                (num_envs, actuator.num_joints),
                act_cfg.cutoff_speed,
                device=asset.device,
                dtype=torch.float,
            )
            _randomize_prop_by_op(
                randomized_cutoff_speed,
                cutoff_speed_distribution_params,
                None,
                slice(0, actuator.num_joints),
                operation,
                distribution,
            )

        if no_load_speed_distribution_params is not None:
            randomized_no_load_speed = torch.full(
                (num_envs, actuator.num_joints),
                act_cfg.no_load_speed,
                device=asset.device,
                dtype=torch.float,
            )
            _randomize_prop_by_op(
                randomized_no_load_speed,
                no_load_speed_distribution_params,
                None,
                slice(0, actuator.num_joints),
                operation,
                distribution,
            )

        actuator.set_motor_speed_params(
            no_load_speed=randomized_no_load_speed,
            cutoff_speed=randomized_cutoff_speed,
            env_ids=env_ids,
        )


