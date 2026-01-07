import os
import pickle
import torch
from datetime import datetime

from isaaclab.utils.math import (
    quat_error_magnitude,
    axis_angle_from_quat,
    quat_from_angle_axis,
)


class SimulationLogger:
    def __init__(self, log_dir="logs", enable_logging=False):
        self.log_dir = log_dir
        self.enable_logging = enable_logging

        if self.enable_logging:
            os.makedirs(log_dir, exist_ok=True)

        self.episode_count = 0

        self.logging_dict = {
            "observations": [],
            "joint_names": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_applied_torques": [],
            "policy_actions": [],
            "processed_actions": [],
            "filtered_actions": [],
            "rewards_goal_pos_error_small": [],
            "rewards_orientaion_error": [],
            "rewards_speed_on_goal": [],
            "rewards_angvel": [],
            "rewards_action_clip": [],
            "rewards_dof_torques": [],
            "rewards_dof_acc_l2": [],
            "rewards_action_rate_l2": [],
            "rewards_contact_change": [],
            "rewards_jerk": [],
            "rewards_joint_symmetry": [],
            "root_pos": [],
            "root_quat": [],
            "root_ang_vel_b": [],
            "root_ang_vel_w": [],
            "rot_error_rad": [],
        }

        self.extras = {
            "joint_info": dict(),
            "policy": dict(),
            "rewards_log": dict(),
            "body_info": dict(),
        }

    def set_logging(self, enable: bool):
        self.enable_logging = enable
        if enable and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def update_extras(
        self, robot, actions, processed_actions, filtered_actions, obs,
    ):
        if not self.enable_logging:
            return

        # body info
        self.extras["body_info"]["root_pos"] = robot.data.root_pos_w
        self.extras["body_info"]["root_quat"] = robot.data.root_quat_w
        self.extras["body_info"]["root_ang_vel_b"] = robot.data.root_ang_vel_b
        self.extras["body_info"]["root_ang_vel_w"] = robot.data.root_ang_vel_w
   
        # joint info
        self.extras["joint_info"]["joint_names"] = robot.joint_names
        self.extras["joint_info"]["joint_positions"] = robot.data.joint_pos
        self.extras["joint_info"]["joint_velocities"] = robot.data.joint_vel
        self.extras["joint_info"]["joint_applied_torques"] = robot.data.applied_torque

        # policy info
        self.extras["policy"]["actions"] = actions
        self.extras["policy"]["processed_actions"] = processed_actions
        self.extras["policy"]["filtered_actions"] = filtered_actions
        self.extras["policy"]["observation"] = obs


    def log_step(self, obs, extras):
        if not self.enable_logging:
            return

        self.logging_dict["observations"].append(obs.to("cpu"))
        self.logging_dict["joint_names"].append(extras["joint_info"]["joint_names"])
        self.logging_dict["joint_positions"].append(
            extras["joint_info"]["joint_positions"].to("cpu")
        )
        self.logging_dict["joint_velocities"].append(
            extras["joint_info"]["joint_velocities"].to("cpu")
        )
        self.logging_dict["joint_applied_torques"].append(
            extras["joint_info"]["joint_applied_torques"].to("cpu")
        )
        self.logging_dict["policy_actions"].append(
            extras["policy"]["actions"].to("cpu")
        )
        self.logging_dict["processed_actions"].append(
            extras["policy"]["processed_actions"].to("cpu")
        )
        self.logging_dict["filtered_actions"].append(
            extras["policy"]["filtered_actions"].to("cpu")
        )

        for reward_key in extras["rewards_log"]:
            self.logging_dict[reward_key].append(extras["rewards_log"][reward_key])

        self.logging_dict["root_pos"].append(extras["body_info"]["root_pos"].to("cpu"))
        self.logging_dict["root_quat"].append(
            extras["body_info"]["root_quat"].to("cpu")
        )
        self.logging_dict["root_ang_vel_b"].append(
            extras["body_info"]["root_ang_vel_b"].to("cpu")
        )
        self.logging_dict["root_ang_vel_w"].append(
            extras["body_info"]["root_ang_vel_w"].to("cpu")
        )


    def save_and_clear(self, custom_name=None):
        if not self.enable_logging:
            return

        if custom_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"logs_{timestamp}_episode_{self.episode_count}.pkl"
        else:
            filename = f"{custom_name}.pkl"

        save_path = "logger/simulation_logs/jump/data"
        filepath = os.path.join(save_path, filename)

        os.makedirs(save_path, exist_ok=True)

        print(
           f"Saving logs to: {filepath}, custom_name: {custom_name}, time: {timestamp}"
        )
        with open(filepath, "wb") as f:
            pickle.dump(self.logging_dict, f)

        self.clear()
        self.episode_count += 1

    def clear(self):
        if not self.enable_logging:
            return

        for key in self.logging_dict.keys():
            self.logging_dict[key] = []
