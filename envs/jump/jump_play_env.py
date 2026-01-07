from typing_extensions import override
from typing import Dict
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from .initalization import InitializationScheme
from logger.logger import OlympusLogger
from .jump_env import JumpEnv
from .jump_env_config import JumpEnvCfg


@configclass
class JumpEnvPlayCfg(JumpEnvCfg):

    scheme_fraqs: Dict[str, float] = {
        InitializationScheme.STANDING.name: 0.0,
        InitializationScheme.DEFAULT.name: 1.0,
        InitializationScheme.INFLIGHT.name: 0.0,
        InitializationScheme.TOUCHDOWN.name: 0.0,
        InitializationScheme.LANDED.name: 0.0,
    }

    landed_rejump_prob = 0.0
    touchdown_rejump_prob = 0.5


class JumpPlayEnv(JumpEnv):
    
    cfg: JumpEnvPlayCfg

    def __init__(self, cfg: JumpEnvPlayCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self._logger = OlympusLogger(logdir="logs/logger/jump")
        self._game_won[:] = True  # random initializing
        self._touchdown_rejump_enabled = True
        # self._command_curiculum_level[:] = 3
        # self._scheme_curiculum_level[:] = 0  # self._num_scheme_curriculums - 1

    @override
    def _get_observations(self) -> dict:
        # print(self.goal_vec.norm(dim=-1).max())
        # self._logger.write_state(self._robot, self.common_step_counter * self.step_dt)
        return super()._get_observations()

    @override
    def _get_dones(self):
        out = super()._get_dones()
        self._curriculum_progress[:] = 0.0

        terms = {
            "on_goal": self._terminate_on_goal,
            "episode_end": self._episode_end,
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
        }
        # print("================== ======================")
        for key, value in terms.items():
            if value[0]:
                print(f"Terminate due to {key}")

        return out
