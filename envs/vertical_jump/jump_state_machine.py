from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class JumpState(IntEnum):
    STANCE = 0
    IN_FLIGHT = 1
    LANDED = 2


class JumpStateMachine:
    '''
    State machine to track the jumping states of multiple environments.
    States include STANCE, IN_FLIGHT, and LANDED.
    '''
    def __init__(self, num_envs, device: str):
        self._device = device
        self._global_step = 0
        with torch.device(device):
            self._states = torch.full((num_envs,), JumpState.STANCE, device=device, dtype=torch.int)
            self._takeoff = torch.full((num_envs,), 0.0, device=device, dtype=torch.bool)
            self._touchdown = torch.full((num_envs,), 0.0, device=device, dtype=torch.bool)
            self._time_since_takeoff = torch.full((num_envs,), 0.0, device=device, dtype=torch.int)
            self._time_since_touchdown = torch.full((num_envs,), 0.0, device=device, dtype=torch.int)
            self._takeoff_pos = torch.full((num_envs, 3), 0.0, device=device, dtype=torch.float)
            self._land_pos = torch.full((num_envs, 3), 0.0, device=device, dtype=torch.float)
            self._has_transitioned = torch.full((num_envs,), False, device=device, dtype=torch.bool)
            self._just_reached_max_height = torch.full((num_envs,), False, device=device, dtype=torch.bool)
            self._has_reached_max_height = torch.full((num_envs,), False, device=device, dtype=torch.bool)
            self._max_height = torch.full((num_envs,), 0.0, device=device, dtype=torch.float)
            self._min_height = torch.full((num_envs,), 0.0, device=device, dtype=torch.float)

            # other buffers
            self._next_states = self._states.clone()
            self._in_contact = torch.full((num_envs,), False, device=device, dtype=torch.bool)
            self._initial_state = self._states.clone()
            self._stance_delay = torch.full((num_envs,), 0.0, device=device, dtype=torch.int)
            self._time_since_reset = torch.full((num_envs,), 0.0, device=device, dtype=torch.int)

    def update(self, root_pos_w: Tensor, root_vel_w: Tensor, contact_state: Tensor):

        to_stance_mask = (
            (self._states == JumpState.LANDED)
            & (self._time_since_reset == self._stance_delay)
            & (self._initial_state == JumpState.STANCE)
        )
        self._states[to_stance_mask] = JumpState.STANCE

        self._in_contact[:] = contact_state.any(dim=1)

        self._takeoff[:] = (~self._in_contact) * (self._states != JumpState.IN_FLIGHT) * (root_vel_w[:, 2] > 0.75)
        self._touchdown[:] = (self._in_contact) * (self._states == JumpState.IN_FLIGHT)

        self._next_states[:] = self._states  # .clone()
        self._next_states[self._takeoff * (self._states == JumpState.STANCE)] = JumpState.IN_FLIGHT
        rejumpmask = torch.rand((self._touchdown.count_nonzero(),), device=self._device) < 0.0
        rejump_idx = self._touchdown.nonzero(as_tuple=False)[rejumpmask]

        self._next_states[self._touchdown] = torch.where(rejumpmask, JumpState.STANCE, JumpState.LANDED).int()

        self._time_since_takeoff[rejump_idx] = 0
        self._time_since_touchdown[rejump_idx] = 0
        self._max_height[rejump_idx] = root_pos_w[rejump_idx, 2]
        self._min_height[rejump_idx] = root_pos_w[rejump_idx, 2]
        self._has_reached_max_height[rejump_idx] = False

        self._states[:] = self._next_states  # .clone()

        self._time_since_takeoff[self._states != JumpState.STANCE] += 1
        self._time_since_takeoff[self._takeoff] = 0

        self._time_since_touchdown[self._states == JumpState.LANDED] += 1
        self._time_since_touchdown[self._touchdown] = 0
        self._time_since_reset[:] += 1

        self._takeoff_pos[self._takeoff] = root_pos_w[self._takeoff]
        self._land_pos[self._touchdown] = root_pos_w[self._touchdown]
        self._just_reached_max_height = (
            (self._states == JumpState.IN_FLIGHT) * (root_vel_w[:, 2] < 0) * (~self._has_reached_max_height)
        )
        self._has_reached_max_height[self._just_reached_max_height] = True
        self._max_height[:] = torch.max(self._max_height, root_pos_w[:, 2])
        self._min_height[:] = torch.min(self._min_height, root_pos_w[:, 2])

    def reset(
        self,
        env_ids: Tensor,
        states: Tensor,
        takeoff_pos: Tensor,
        land_pos: Tensor,
        max_height: Tensor,
        stance_delay: Tensor,
    ):

        self._initial_state[env_ids] = states
        states = states.clone()
        states[states == JumpState.IN_FLIGHT] = JumpState.STANCE
        self._states[env_ids] = states
        self._takeoff[env_ids] = 0
        self._touchdown[env_ids] = 0
        self._time_since_takeoff[env_ids] = 0.0
        self._time_since_takeoff[env_ids[states == JumpState.IN_FLIGHT]] = 1
        self._time_since_takeoff[env_ids[states == JumpState.LANDED]] = 100
        self._has_transitioned[env_ids] = False
        self._has_reached_max_height[env_ids] = False
        self._just_reached_max_height[env_ids] = False
        self._time_since_reset[env_ids] = 0.0
        self._stance_delay[env_ids] = stance_delay

        self._time_since_touchdown[env_ids] = 0.0
        self._time_since_touchdown[env_ids[states == JumpState.LANDED]] = torch.randint(
            int(0.2 * 60),
            int(0.35 * 60),
            ((states == JumpState.LANDED).count_nonzero(),),
            device=self._device,
            dtype=torch.int,
        )

        self._max_height[env_ids] = max_height
        self._min_height[env_ids] = torch.inf
        self._takeoff_pos[env_ids, :] = takeoff_pos
        self._land_pos[env_ids, :] = land_pos

    @property
    def states(self) -> Tensor:
        return self._states

    @property
    def takeoff(self) -> Tensor:
        return self._takeoff

    @property
    def touchdown(self) -> Tensor:
        return self._touchdown

    @property
    def steps_since_takeoff(self) -> Tensor:
        return self._time_since_takeoff

    @property
    def steps_since_touchdown(self) -> Tensor:
        return self._time_since_touchdown

    @property
    def takeoff_pos(self) -> Tensor:
        return self._takeoff_pos

    @property
    def land_pos(self) -> Tensor:
        return self._land_pos

    @property
    def initial_state(self) -> Tensor:
        return self._initial_state

    @property
    def has_reached_max_height(self) -> Tensor:
        return self._has_reached_max_height

    @property
    def just_reached_max_height(self) -> Tensor:
        return self._just_reached_max_height

    @property
    def max_height(self) -> Tensor:
        return self._max_height

    @property
    def min_height(self) -> Tensor:
        return self._min_height
