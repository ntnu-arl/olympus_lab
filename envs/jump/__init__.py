# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents
from .jump_env import JumpEnv
from .jump_env_config import JumpEnvCfg
from .jump_play_env import JumpPlayEnv, JumpEnvPlayCfg

##
# Register Gym environments.
##

gym.register(
    id="Olympus-Jump",
    entry_point="envs.jump:JumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JumpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Olympus-Jump-Play",
    entry_point="envs.jump:JumpPlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JumpEnvPlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
