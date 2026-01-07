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
            self._time_since_takeoff = torch.full((num_envs,), 0.0, device=device, dtype=torch.long)
            self._time_since_touchdown = torch.full((num_envs,), 0.0, device=device, dtype=torch.long)
            self._takeoff_pos = torch.full((num_envs, 3), 0.0, device=device, dtype=torch.float)
            self._land_pos = torch.full((num_envs, 3), 0.0, device=device, dtype=torch.float)
            self._has_transitioned = torch.full((num_envs,), False, device=device, dtype=torch.bool)

            # other buffers
            self._next_states = self._states.clone()
            self._in_contact = torch.full((num_envs,), False, device=device, dtype=torch.bool)
            self._initial_state = self._states.clone()

    def update(self, root_pos_w: Tensor, root_vel_w: Tensor, contact_state: Tensor):
        self._in_contact[:] = contact_state.any(dim=1)

        self._takeoff[:] = (~self._in_contact) * (self._states != JumpState.IN_FLIGHT) * (root_vel_w[:, 2] > 0.25)
        self._touchdown[:] = (self._in_contact) * (self._states == JumpState.IN_FLIGHT)

        self._next_states[:] = self._states
        self._next_states[self._takeoff * (self._states == JumpState.STANCE)] = JumpState.IN_FLIGHT
        self._next_states[self._touchdown] = JumpState.LANDED

        self._states[:] = self._next_states

        self._time_since_takeoff[self._states != JumpState.STANCE] += 1
        self._time_since_takeoff[self._takeoff] = 0

        self._time_since_touchdown[self._states == JumpState.LANDED] += 1
        self._time_since_touchdown[self._touchdown] = 0

        self._takeoff_pos[self._takeoff] = root_pos_w[self._takeoff]
        self._land_pos[self._touchdown] = root_pos_w[self._touchdown]

    def reset(self, env_ids: Tensor, states: Tensor, takeoff_pos: Tensor, land_pos: Tensor):
        if not isinstance(states, Tensor):
            states = torch.full((len(env_ids),), states, device=self._device, dtype=torch.int)
        self._initial_state[env_ids] = states
        self._states[env_ids] = states
        self._takeoff[env_ids] = 0
        self._touchdown[env_ids] = 0
        self._time_since_takeoff[env_ids] = 0.0
        self._time_since_takeoff[env_ids[states == JumpState.IN_FLIGHT]] = 1
        self._time_since_takeoff[env_ids[states == JumpState.LANDED]] = 2

        self._time_since_touchdown[env_ids] = 0.0
        num_landed = torch.count_nonzero(states == JumpState.LANDED)
        self._time_since_touchdown[env_ids[states == JumpState.LANDED]] = torch.randint(
            1, int(0.3 * 60), (num_landed,), device=self._device
        )

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
