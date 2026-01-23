# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents
from .vertical_jump_env import VerticalJumpEnv
from .vertical_jump_env_config import VerticalJumpEnvCfg
from .vertical_jump_play_env import VerticalJumpPlayEnv, JumpEnvPlayCfg

##
# Register Gym environments.
##

gym.register(
    id="Olympus-Vertical-Jump-Mars",
    entry_point="envs.vertical_jump:VerticalJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VerticalJumpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Olympus-Vertical-Jump-Mars-Play",
    entry_point="envs.vertical_jump:VerticalJumpPlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JumpEnvPlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": None,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
