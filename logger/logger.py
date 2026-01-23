from __future__ import annotations
from typing import Any

import os
import json

from torch import Tensor
from isaaclab.assets import Articulation


class OlympusLogger:
    def __init__(self, logdir: str) -> None:
        self._logdir = logdir
        self._buffers: dict[str, list[Any]] = {}

    @classmethod
    def from_json(cls, json: str) -> OlympusLogger:
        pass

    def write_state(self, olympus: Articulation, time_stamp: float) -> None:
        self._write_field("root_state", time_stamp, olympus.data.root_state_w)
        self._write_field("joint_pos", time_stamp, olympus.data.joint_pos)
        self._write_field("joint_vel", time_stamp, olympus.data.joint_vel)
        self._write_field("joint_torque", time_stamp, olympus.data.applied_torque)
        self._write_field("joint_acc", time_stamp, olympus.data.joint_acc)

    def write_statistics(
        self, stat: str, data: float | Tensor, time_stamp: float
    ) -> None:
        self._write_field(stat, time_stamp, data)

    def _write_field(self, field: str, time_stamp: float, data: Any) -> None:
        if field not in self._buffers.keys():
            self._buffers[field] = []

        self._buffers[field].append({"time": time_stamp, "data": data})

    def to_json(self, aggregate_batch_size: bool = False) -> None:
        json_path = os.path.join(self._logdir, "data.json")
        with open("data.json", "w") as json_file:
            json.dump(convert_tensors_to_lists(self._buffers), json_file, indent=4)


# Recursive function to convert tensors to lists in nested dictionaries
def convert_tensors_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_tensors_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensors_to_lists(v) for v in d]
    elif isinstance(d, Tensor):
        return d.tolist()
    else:
        return d
