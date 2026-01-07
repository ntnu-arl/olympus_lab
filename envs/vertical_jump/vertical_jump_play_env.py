from typing_extensions import override
from typing import Dict
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from .initalization import InitializationScheme
from logger.logger import OlympusLogger
from .vertical_jump_env import VerticalJumpEnv
from .vertical_jump_env_config import VerticalJumpEnvCfg


@configclass
class JumpEnvPlayCfg(VerticalJumpEnvCfg):
    scheme_fraqs: Dict[str, float] = {
        InitializationScheme.STANDING.name: 0.0,
        InitializationScheme.DEFAULT.name: 1.0,
        InitializationScheme.INFLIGHT.name: 0.0,
        InitializationScheme.TOUCHDOWN.name: 0.0,
        InitializationScheme.LANDED.name: 0.0,
    }


class VerticalJumpPlayEnv(VerticalJumpEnv):
    cfg: JumpEnvPlayCfg

    def __init__(self, cfg: JumpEnvPlayCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self._logger = OlympusLogger(logdir="logs/logger/jump")
        self._game_won[:] = True  # random initializing
        # self._command_curiculum_level[:] = 3
        # self._scheme_curiculum_level[:] = 0  # self._num_scheme_curriculums - 1

    @override
    def _get_observations(self) -> dict:


        obs_buf = super()._get_observations()["policy"]
        obs_buf[(self.episode_length_buf - self._episode_start_step) < 0.5 / self.step_dt, 1] = ( # commanded to jump after 0.5 s
            0.0  
        )
        return {"policy": obs_buf}

    @override
    def _get_dones(self):
        out = super()._get_dones()
        self._curriculum_progress[:] = 0.0

        terms = {
            "collision": self._terminate_collision,
            "walking": (self._terminate_walking),
            "touchdown": self._terminate_touchdown,
            "takeoff": self._terminate_takeoff,
            "landed": self._terminate_landed,
            "root_height": self._terminate_root_height,
            "idle": self._terminate_idle,
            "impact": self._terminate_impact,
            "nan": self._terminate_nan,
            "shank_height": self._terminate_shank_height,
            "attitude": self._terminate_attitude,
            "translation": self._terminate_translation,
        }
        # print("================== ======================")
        for key, value in terms.items():
            if value[0]:
                print(f"Terminate due to {key}")

        return out
