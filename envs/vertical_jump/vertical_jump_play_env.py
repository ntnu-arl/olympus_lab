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
from .vertical_jump_env import VerticalJumpEnv
from .vertical_jump_env_config import VerticalJumpEnvCfg


@configclass
class EventCfg:
    """Configuration for Vertical Jump Play Environment. Dont use all randomizations from training.
    Recomend slacking termination conditions for play mode directly in the env class if needed for evaluation.
    """

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
            "friction_distribution_params": (0.00, 0.00),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    
    def apply_version_settings(self, parent_cfg):
        ''''Apply version specific settings to the configuration.'''
        pass


@configclass
class JumpEnvPlayCfg(VerticalJumpEnvCfg):
    '''Configuration for Vertical Jump Play Environment.'''
    
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
    
    # marker for the target height - using a sphere at the commanded height
    height_marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/heightMarkers",
        markers={
            "target_height": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    
    # optional: marker for robot starting position
    start_marker = VisualizationMarkersCfg(
        prim_path="/World/Visuals/startMarkers",
        markers={
            "start": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )


class VerticalJumpPlayEnv(VerticalJumpEnv):
    """Vertical Jump Play Environment."""
    cfg: JumpEnvPlayCfg

    def __init__(self, cfg: JumpEnvPlayCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._game_won[:] = True  # random initializing
        self._touchdown_rejump_enabled = True
        self._command_curiculum_level[:] = 1
        self._counter = 0

    def _setup_scene(self):
        super()._setup_scene()
        # iInitialize visualization markers
        self._height_marker = VisualizationMarkers(self.cfg.height_marker)
        # self._start_marker = VisualizationMarkers(self.cfg.start_marker)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._init_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self._init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

    @override
    def _get_observations(self) -> dict:        
        # update the height marker visualization
        self._update_markers()
        
        obs_buf = super()._get_observations()["policy"]
        
        # set jump signal to zero for the first 0.5 seconds - then give the jump command
        obs_buf[(self.episode_length_buf - self._episode_start_step) < 0.5 / self.step_dt, 1] = 0.0
        
        return {"policy": obs_buf}

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        return

    def _update_markers(self):
        robot_pos = self._robot.data.root_pos_w.clone()
        
        marker_translations = torch.zeros_like(robot_pos)
        marker_translations[:, 0] = self.scene.env_origins[:, 0]  
        marker_translations[:, 1] = self.scene.env_origins[:, 1]
                
        if hasattr(self, '_commanded_jump_height'):
            marker_translations[:, 2] = self._commanded_jump_height.squeeze(-1)  
        elif hasattr(self, '_commands'):
            marker_translations[:, 2] = self._commands[:, 0] if self._commands.dim() > 1 else self._commands
        else:
            marker_translations[:, 2] = 1.0  
        
        self._height_marker.visualize(translations=marker_translations)

    @override
    def _get_dones(self):
        out = super()._get_dones()
        self._curriculum_progress[:] = 0.0

        terms = {
            # "episode_end": self._episode_end,  # Comment out if not defined
            "collision": self._terminate_collision,
            "walking": self._terminate_walking,
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
            if value.any() if torch.is_tensor(value) else value[0]:
                print(f"Terminate due to {key}")

        return out