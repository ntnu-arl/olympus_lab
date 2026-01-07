from torch import Tensor
from typing import Tuple, Dict

from typing_extensions import override

from isaaclab.assets.articulation import Articulation

from kinematics import OlympusKinematics

from .configs import LandedInitializerCfg, CommandCfg
from .standing_initializer import StandingInitializer


class LandedInitializer(StandingInitializer):

    def __init__(
        self,
        cfg: LandedInitializerCfg,
        olympus: Articulation,
        kinematics: OlympusKinematics,
        command_cfg: CommandCfg,
    ):
        super().__init__(cfg, olympus, kinematics, command_cfg)

    @override
    def draw(self, num_envs: int) -> Tuple[Tensor]:
        command, state = super().draw(num_envs)
        state[:, :2] += command[:, :2]
        return command, state
