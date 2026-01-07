# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents
from .attitude_control_env import AttitudeControlEnv
from .attitude_control_env_config import AttitudeControlEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Olympus-Attitude-Control",
    entry_point="envs.attitude_control:AttitudeControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AttitudeControlEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
