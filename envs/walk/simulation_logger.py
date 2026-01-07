import os
import pickle
import torch
from datetime import datetime

from isaaclab.utils.math import (
    quat_error_magnitude,
    axis_angle_from_quat,
    quat_from_angle_axis
)

class SimulationLogger:    
    def __init__(self, log_dir="logs", enable_logging=False):
        self.log_dir = log_dir
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            os.makedirs(log_dir, exist_ok=True)
            
        self.episode_count = 0
        
        self.logging_dict = {
            # Observations and actions
            "observations": [],
            "joint_names": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_applied_torques": [],
            "policy_actions": [],
            "processed_actions": [],
            "filtered_actions": [],
            "mean_filtered_actions": [],
            "root_pos": [],
            "root_quat": [],
            "root_ang_vel_b": [],
            "root_ang_vel_w": [],
            "rot_error_rad": [],
            "projected_gravity": [],

            #rewards
            "track_lin_vel_xy_exp": [],
            "track_ang_vel_z_exp": [],
            "lin_vel_z_l2": [],
            "ang_vel_xy_l2": [],
            "dof_torques_l2": [],
            "dof_acc_l2": [],
            "action_rate_l2": [],
            "feet_air_time": [],
            "undesired_contacts": [],
            "flat_orientation_l2": [],
            "height_reward": [],
            "hip_reward": [],
            "action_clip": [],
            "dof_torques": [],
            "dof_acc_l2": [],
            "action_rate_l2": [],
            "contact_change": [],
            "jerk": [],

            # Commands
            "commands": [],
            "lin_vel": [],
            "lin_vel_error": [],
            "yaw_rate": [],
            "yaw_rate_error": [],
            "z_vel": [],
            "paw_height": [],
            "paw_velocities": []
            # "current_forces": []
        }
        
        self.extras = {
            "joint_info": dict(),
            "policy": dict(),
            "rewards_log": dict(),
            "body_info": dict(),
            "commands_extras": dict()
        }

    def set_logging(self, enable: bool):
        self.enable_logging = enable
        if enable and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def update_extras(self, robot, actions, processed_actions, filtered_actions, mean_filtered_actions, obs, episode_sums, commands_extras):
        if not self.enable_logging:
            return
            
        # Body info
        self.extras["body_info"]["root_pos"] = robot.data.root_pos_w
        self.extras["body_info"]["root_lin_vel_w"] = robot.data.root_lin_vel_w
        self.extras["body_info"]["root_quat"] = robot.data.root_quat_w
        self.extras["body_info"]["root_ang_vel_b"] = robot.data.root_ang_vel_b
        self.extras["body_info"]["root_ang_vel_w"] = robot.data.root_ang_vel_w
        self.extras["body_info"]["rot_error_rad"] = quat_error_magnitude(
            robot.data.root_quat_w, robot.data.default_root_state[:, 3:7]
        )

        # Joint info
        self.extras["joint_info"]["joint_names"] = robot.joint_names
        self.extras["joint_info"]["joint_positions"] = robot.data.joint_pos
        self.extras["joint_info"]["joint_velocities"] = robot.data.joint_vel
        self.extras["joint_info"]["joint_applied_torques"] = robot.data.applied_torque

        # Policy info
        self.extras["policy"]["actions"] = actions
        self.extras["policy"]["processed_actions"] = processed_actions
        self.extras["policy"]["filtered_actions"] = filtered_actions
        self.extras["policy"]["mean_filtered_actions"] = mean_filtered_actions
        self.extras["policy"]["observation"] = obs

        # Commanded linear and angular velocities
        self.extras["commands_extras"]["commands"] = commands_extras["commands"]
        self.extras["commands_extras"]["lin_vel"] = commands_extras["lin_vel"]
        self.extras["commands_extras"]["lin_vel_error"] = commands_extras["lin_vel_error"]
        self.extras["commands_extras"]["yaw_rate"] = commands_extras["yaw_rate"]
        self.extras["commands_extras"]["yaw_rate_error"] = commands_extras["yaw_rate_error"]
        self.extras["commands_extras"]["z_vel"] = commands_extras["z_vel"]
        self.extras["commands_extras"]["projected_gravity"] = commands_extras["projected_gravity"]
        self.extras["commands_extras"]["paw_height"] = commands_extras["paw_height"]
        self.extras["commands_extras"]["paw_velocities"] = commands_extras["paw_velocities"]
        # self.extras["commands_extras"]["current_forces"] = commands_extras["current_forces"]

        # Rewards
        self.extras["rewards_log"]["track_lin_vel_xy_exp"] = episode_sums["track_lin_vel_xy_exp"]
        self.extras["rewards_log"]["track_ang_vel_z_exp"] = episode_sums["track_ang_vel_z_exp"]
        self.extras["rewards_log"]["lin_vel_z_l2"] = episode_sums["lin_vel_z_l2"]
        self.extras["rewards_log"]["ang_vel_xy_l2"] = episode_sums["ang_vel_xy_l2"]
        self.extras["rewards_log"]["dof_torques_l2"] = episode_sums["dof_torques_l2"]
        self.extras["rewards_log"]["dof_acc_l2"] = episode_sums["dof_acc_l2"]
        self.extras["rewards_log"]["action_rate_l2"] = episode_sums["action_rate_l2"]
        self.extras["rewards_log"]["feet_air_time"] = episode_sums["feet_air_time"]
        self.extras["rewards_log"]["undesired_contacts"] = episode_sums["undesired_contacts"]
        self.extras["rewards_log"]["flat_orientation_l2"] = episode_sums["flat_orientation_l2"]
        # self.extras["rewards_log"]["height_reward"] = episode_sums["height_reward"]
        # self.extras["rewards_log"]["hip_reward"] = episode_sums["hip_reward"]
        self.extras["rewards_log"]["action_clip"] = episode_sums["action_clip"]
        # self.extras["rewards_log"]["dof_torques"] = episode_sums["dof_torques"]
        # self.extras["rewards_log"]["dof_acc_l2"] = episode_sums["dof_acc_l2"]
        # self.extras["rewards_log"]["action_rate_l2"] = episode_sums["action_rate_l2"]
        self.extras["rewards_log"]["contact_change"] = episode_sums["contact_change"]
        self.extras["rewards_log"]["jerk"] = episode_sums["jerk"]


    def log_step(self, obs, extras):
        if not self.enable_logging:
            return
            
        self.logging_dict["observations"].append(obs.to("cpu"))
        self.logging_dict["joint_names"].append(extras["joint_info"]["joint_names"])
        self.logging_dict["joint_positions"].append(extras["joint_info"]["joint_positions"].to("cpu"))
        self.logging_dict["joint_velocities"].append(extras["joint_info"]["joint_velocities"].to("cpu"))
        self.logging_dict["joint_applied_torques"].append(extras["joint_info"]["joint_applied_torques"].to("cpu"))
        self.logging_dict["policy_actions"].append(extras["policy"]["actions"].to("cpu"))
        self.logging_dict["processed_actions"].append(extras["policy"]["processed_actions"].to("cpu"))
        self.logging_dict["filtered_actions"].append(extras["policy"]["filtered_actions"].to("cpu"))
        self.logging_dict["mean_filtered_actions"].append(extras["policy"]["mean_filtered_actions"].to("cpu"))
       # Log all rewards, including landing pose angles
        for reward_key in extras["rewards_log"]:
            self.logging_dict[reward_key].append(extras["rewards_log"][reward_key])

        # Log commands
        self.logging_dict["commands"].append(extras["commands_extras"]["commands"].to("cpu"))
        self.logging_dict["lin_vel"].append(extras["commands_extras"]["lin_vel"].to("cpu"))
        self.logging_dict["lin_vel_error"].append(extras["commands_extras"]["lin_vel_error"].to("cpu"))
        self.logging_dict["yaw_rate"].append(extras["commands_extras"]["yaw_rate"].to("cpu"))
        self.logging_dict["yaw_rate_error"].append(extras["commands_extras"]["yaw_rate_error"].to("cpu"))
        self.logging_dict["z_vel"].append(extras["commands_extras"]["z_vel"].to("cpu"))
        # self.logging_dict["z_vel_error"] = extras["commands_extras"]["z_vel_error"]

        
        # Log body info
        self.logging_dict["root_pos"].append(extras["body_info"]["root_pos"].to("cpu"))
        self.logging_dict["root_quat"].append(extras["body_info"]["root_quat"].to("cpu"))
        self.logging_dict["root_ang_vel_b"].append(extras["body_info"]["root_ang_vel_b"].to("cpu"))
        self.logging_dict["root_ang_vel_w"].append(extras["body_info"]["root_ang_vel_w"].to("cpu"))
        self.logging_dict["rot_error_rad"].append(extras["body_info"]["rot_error_rad"].to("cpu"))

    def save_and_clear(self, custom_name=None):
        print("Saving logs")
        if not self.enable_logging:
            print("Logging is disabled")
            return
            
        if custom_name is None:
            print("Custom name is None")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"logs_{timestamp}_episode_{self.episode_count}.pkl"
        else:
            print("Custom name is   ", custom_name)
            filename = f"{custom_name}.pkl"
            
        # filepath = os.path.join(self.log_dir, filename)
        save_path = "logger/simulation_logs/walk/data"
        filepath = os.path.join(save_path, filename)
        
        print(f"Saving logs to: {filepath}, custom_name: {custom_name}, time: {timestamp}")
        with open(filepath, "wb") as f:
            pickle.dump(self.logging_dict, f)

            
        self.clear()
        self.episode_count += 1
        
    def clear(self):
        if not self.enable_logging:
            return
            
        for key in self.logging_dict.keys():
            self.logging_dict[key] = []