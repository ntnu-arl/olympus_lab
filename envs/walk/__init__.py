# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents
from .walk_env import WalkEnv
from .walk_env_config import WalkEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Olympus-Walk",
    entry_point="envs.walk:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": WalkEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": None,
    },
)
