from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class FiveBarSpringConfig:
    enable_leg_springs: bool = True
    spring_stiffness: float = 1000.0  # N/m
    spring_rest_length: float = 0.17  # meters
    leg_hip_width: float = 0.09  # distance between hip motors (m)
    leg_upper_link_length: float = 0.18  # length of upper links (m)
    stage2_scale_factor: float = 0.75  # scale for post-90° torque magnitude
    stage2_blend_angle: float = 0.5  # degrees - smoothing region around 90°


class FiveBarSpring:
    ''' class to compute five-bar leg spring torques '''
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        spring_stiffness: float,
        spring_rest_length: float,
        leg_hip_width: float,
        leg_upper_link_length: float,
        stage2_scale_factor: float = 0.0,
        stage2_blend_angle: float = 0.5,
    ):
        self.num_envs = num_envs
        self.device = device
        
        self.spring_stiffness = spring_stiffness
        self.spring_rest_length = spring_rest_length
        self.leg_hip_width = leg_hip_width
        self.leg_upper_link_length = leg_upper_link_length
        self.stage2_scale_factor = stage2_scale_factor
        self.stage2_blend_angle = stage2_blend_angle
        
        self.spring_forces = torch.zeros((num_envs, 4), device=device)
        self.spring_lengths = torch.zeros((num_envs, 4), device=device)
        self.knee_distances = torch.zeros((num_envs, 4), device=device)
    
    def compute_spring_torques(
        self,
        theta_inner: torch.Tensor,
        theta_outer: torch.Tensor,
    ) -> torch.Tensor:
        ''' compute spring torques for all legs'''

        knee_distance = self._compute_knee_distance(theta_inner, theta_outer)
        
        spring_extension = torch.clamp(knee_distance - self.spring_rest_length, min=0.0)
        spring_force = self.spring_stiffness * spring_extension
        
        self.spring_lengths = spring_extension 
        self.spring_forces = spring_force
        self.knee_distances = knee_distance
        
        torques_inner_stage1, torques_outer_stage1 = self._spring_force_to_joint_torques_stage1(
            spring_force, theta_inner, theta_outer, knee_distance
        )
        
        if self.stage2_scale_factor > 0.0:
            combined_angle = theta_inner + torch.abs(theta_outer)
            stage2_threshold = torch.pi
            
            torques_inner_stage2, torques_outer_stage2 = self._spring_force_to_joint_torques_stage2(
                spring_force, theta_inner, theta_outer, knee_distance, combined_angle
            )
            
            blend_width = torch.deg2rad(torch.tensor(self.stage2_blend_angle, device=self.device))
            blend_start = stage2_threshold - blend_width
            blend_end = stage2_threshold + blend_width
            
            blend_factor = torch.clamp(
                (combined_angle - blend_start) / (2 * blend_width),
                min=0.0,
                max=1.0
            )
            
            torques_inner = (1 - blend_factor) * torques_inner_stage1 + blend_factor * torques_inner_stage2
            torques_outer = (1 - blend_factor) * torques_outer_stage1 + blend_factor * torques_outer_stage2
        else:
            torques_inner = torques_inner_stage1
            torques_outer = torques_outer_stage1
    
        # rearrange from [FL, BL, FR, BR] to actuator order
        # actuator order: [BL_inner, BL_outer, BR_inner, BR_outer, FL_inner, FL_outer, FR_inner, FR_outer]
        torques = torch.cat([
            torques_inner[:, 1:2],  # BL_inner
            torques_outer[:, 1:2],  # BL_outer
            torques_inner[:, 3:4],  # BR_inner
            torques_outer[:, 3:4],  # BR_outer
            torques_inner[:, 0:1],  # FL_inner
            torques_outer[:, 0:1],  # FL_outer
            torques_inner[:, 2:3],  # FR_inner
            torques_outer[:, 2:3],  # FR_outer
        ], dim=1)
        
        max_spring_torque = 30.0  # Nm
        torques = torch.clamp(torques, min=-max_spring_torque, max=max_spring_torque)
        
        return torques
    
    def _compute_knee_distance(
        self,
        theta_inner: torch.Tensor,
        theta_outer: torch.Tensor,
    ) -> torch.Tensor:
        l0 = self.leg_hip_width
        l1 = self.leg_upper_link_length
        l2 = self.leg_upper_link_length
        
        knee_inner_x = -l1 * torch.sin(theta_inner)
        knee_inner_y = l1 * torch.cos(theta_inner)
        
        knee_outer_x = l0 + l2 * torch.sin(theta_outer)
        knee_outer_y = l2 * torch.cos(theta_outer)
        
        dx = knee_outer_x - knee_inner_x
        dy = knee_outer_y - knee_inner_y
        distance = torch.sqrt(dx**2 + dy**2)
        
        return distance
    
    def _spring_force_to_joint_torques_stage1(
        self,
        spring_force: torch.Tensor,
        theta_inner: torch.Tensor,
        theta_outer: torch.Tensor,
        knee_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''stage 1: direct Jacobian transpose method'''

        l0 = self.leg_hip_width
        l1 = self.leg_upper_link_length
        l2 = self.leg_upper_link_length

        knee_inner_x = l1 * torch.sin(theta_inner)
        knee_inner_y = l1 * torch.cos(theta_inner)
        
        theta_outer_physical = -theta_outer
        knee_outer_x = l0 + l2 * torch.sin(theta_outer_physical)
        knee_outer_y = l2 * torch.cos(theta_outer_physical)
        
        dx = knee_outer_x - knee_inner_x
        dy = knee_outer_y - knee_inner_y
        distance_safe = torch.clamp(knee_distance, min=0.05)  # 5 minimum for stability
        
        spring_dir_x = dx / distance_safe
        spring_dir_y = dy / distance_safe
        
        force_inner_x = spring_force * spring_dir_x
        force_inner_y = spring_force * spring_dir_y
        
        force_outer_x = -spring_force * spring_dir_x
        force_outer_y = -spring_force * spring_dir_y
        
        # Jacobian transpose: 
        torque_inner = (
            l1 * torch.cos(theta_inner) * force_inner_x +
            (-l1 * torch.sin(theta_inner)) * force_inner_y
        )

        torque_outer_physical = (
            l2 * torch.cos(theta_outer_physical) * force_outer_x +
            (-l2 * torch.sin(theta_outer_physical)) * force_outer_y
        )
        
        # convert to reported angle space
        torque_outer = -torque_outer_physical
        
        return torque_inner, torque_outer
    
    def _spring_force_to_joint_torques_stage2(
        self,
        spring_force: torch.Tensor,
        theta_inner: torch.Tensor,
        theta_outer: torch.Tensor,
        knee_distance: torch.Tensor,
        combined_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ''' stage 2: simplified model for post-90° behavior '''

        l1 = self.leg_upper_link_length
        
        angle_past_90 = combined_angle - torch.pi
        angle_past_90 = torch.clamp(angle_past_90, min=0.0)
        
        moment_arm = l1 * 0.15 * (1 - torch.cos(angle_past_90))
        
        min_moment_arm = 0.05 * l1  
        moment_arm = torch.clamp(moment_arm, min=min_moment_arm)
        
        torque_magnitude = spring_force * moment_arm * self.stage2_scale_factor
        
        # always negative 
        torque_inner = -torque_magnitude
        torque_outer = -torque_magnitude
        
        return torque_inner, torque_outer