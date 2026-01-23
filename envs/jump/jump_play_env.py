from typing_extensions import override
from typing import Dict
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from .initalization import InitializationScheme
from logger.logger import OlympusLogger
from .jump_env import JumpEnv
from .jump_env_config import JumpEnvCfg


@configclass
class EventCfg:
    """Configuration for Jump Play Environment. Dont use all randomizations from training."""

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (1, 1),
    #         "dynamic_friction_range": (0.85, 0.85),
    #         "restitution_range": (0.1, 0.1),
    #         "make_consistent": True,
    #         "num_buckets": 64,
    #     },
    # )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Motor.*"),
            "friction_distribution_params": (0.01, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    def apply_version_settings(self, parent_cfg):
        ''''Apply version specific settings to the configuration.'''
        pass


@configclass
class JumpEnvPlayCfg(JumpEnvCfg):
    '''Configuration for Jump Play Environment.'''
    scheme_fraqs: Dict[str, float] = {
        InitializationScheme.STANDING.name: 0.0,
        InitializationScheme.DEFAULT.name: 1.0, # initialize always in default scheme
        InitializationScheme.INFLIGHT.name: 0.0,
        InitializationScheme.TOUCHDOWN.name: 0.0,
        InitializationScheme.LANDED.name: 0.0,
    }

    events = EventCfg()
    landed_rejump_prob = 0.0
    touchdown_rejump_prob = 0.0
    termination = JumpEnvCfg.TerminationConfig(walking_distance=0.5)
    goal_marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/testMarkers",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    start_marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/testMarkers",
        markers={
            "start": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    


class JumpPlayEnv(JumpEnv):
    """Jump Play Environment."""
    cfg: JumpEnvPlayCfg

    def __init__(self, cfg: JumpEnvPlayCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._game_won[:] = True  # random initializing
        self._touchdown_rejump_enabled = True
        self._command_curiculum_level[:] = 1
        self._counter = 0

    def _setup_scene(self):
        super()._setup_scene()
        # self._start_marker = VisualizationMarkers(self.cfg.start_marker)
        self._goal_marker = VisualizationMarkers(self.cfg.goal_marker)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._init_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self._init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

    @override
    def _get_observations(self) -> dict:
        self._update_markers()

        return super()._get_observations()

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        return

    def _update_markers(self):

        trans = self._commands.clone()
        trans[:, 2] = 0.0  # set z to 0
        self._goal_marker.visualize(translations=trans)
        trans = self._init_pos.clone()
        trans[:, 2] = 0.0  # set z to 0

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
        for key, value in terms.items():
            if value[0]:
                print(f"Terminate due to {key}")

        return out
