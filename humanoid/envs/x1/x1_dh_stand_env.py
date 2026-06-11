# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from humanoid.utils.math import wrap_to_pi


import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  Terrain

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class X1DHStandEnv(LeggedRobot):
    '''
    X1DHStandEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = self.cfg.rewards.feet_to_ankle_distance
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)      


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)  #angular vel xyz

        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            self.phase_length_buf[stand_command] = 0 # set this as 0 for which env is standing
            # self.gait_start is rand 0 or 0.5
            phase = (self.phase_length_buf * self.dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time + self.gait_start

        # phase continue increase，if want robot stand, set 0
        return phase

    def _get_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Add double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        # stand mask == 1 means stand leg 
        return stance_mask

    def generate_gait_time(self,envs):
        if len(envs) == 0:
            return

        # rand sample 
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                            gait_time_range[1],
                                            (len(envs), 1),device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor,dim=1,keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:,1:] = scaled_tensor[:,:-1].clone()
        scaled_tensor[:,0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor,dim=1).int()
     
    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
            
    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
    
    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self,env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)
        
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        cos_pos = torch.cos(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)

        # ====== V9: 非对称摆动轨迹 — 下降段 sin² 减速，柔和着地 ======
        # 上升段 (抬腿): 保持 |sin| — 快速抬腿
        # 下降段 (落地): 使用 sin² — 脚接近地面时速度 → 0，减少冲击
        #
        # 方向判定: sin 和 cos 同号 → 下降段 (sin 正在趋向 0)
        #           sin 和 cos 异号 → 上升段 (sin 正在远离 0)

        # Left swing (sin_pos_l < 0)
        sin_pos_l[sin_pos_l > 0] = 0
        abs_sin_l = torch.abs(sin_pos_l)
        is_descent_l = cos_pos > 0  # sin 从负趋向 0
        scale_l = torch.where(is_descent_l, abs_sin_l ** 2, abs_sin_l)

        self.ref_dof_pos[:, 0] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[0]
        self.ref_dof_pos[:, 1] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[1]
        self.ref_dof_pos[:, 2] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[2]
        self.ref_dof_pos[:, 3] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[3]
        self.ref_dof_pos[:, 4] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[4]
        self.ref_dof_pos[:, 5] = -scale_l * self.cfg.rewards.final_swing_joint_delta_pos[5]

        # Right swing (sin_pos_r > 0)
        sin_pos_r[sin_pos_r < 0] = 0
        abs_sin_r = torch.abs(sin_pos_r)
        is_descent_r = cos_pos < 0  # sin 从正趋向 0
        scale_r = torch.where(is_descent_r, abs_sin_r ** 2, abs_sin_r)

        self.ref_dof_pos[:, 6] = scale_r *  self.cfg.rewards.final_swing_joint_delta_pos[6]
        self.ref_dof_pos[:, 7] = scale_r *  self.cfg.rewards.final_swing_joint_delta_pos[7]
        self.ref_dof_pos[:, 8] = scale_r *  self.cfg.rewards.final_swing_joint_delta_pos[8]
        self.ref_dof_pos[:, 9] = scale_r *  self.cfg.rewards.final_swing_joint_delta_pos[9]
        self.ref_dof_pos[:, 10] = scale_r * self.cfg.rewards.final_swing_joint_delta_pos[10]
        self.ref_dof_pos[:, 11] = scale_r * self.cfg.rewards.final_swing_joint_delta_pos[11]

        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.

        # if use_ref_actions=True, action += ref_action
        self.ref_action = 2 * self.ref_dof_pos

        # self.ref_dof_pos set ref dof pos for swing leg, ref_dof_pos=0 for stance leg
        self.ref_dof_pos += self.default_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: self.cfg.env.num_commands] = 0.  # commands
        noise_vec[self.cfg.env.num_commands: self.cfg.env.num_commands+self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[self.cfg.env.num_commands+self.num_actions: self.cfg.env.num_commands+2*self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[self.cfg.env.num_commands+2*self.num_actions: self.cfg.env.num_commands+3*self.num_actions] = 0.  # previous actions
        noise_vec[self.cfg.env.num_commands+3*self.num_actions: self.cfg.env.num_commands+3*self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[self.cfg.env.num_commands+3*self.num_actions + 3: self.cfg.env.num_commands+3*self.num_actions + 6] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec



    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        return super().step(actions)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        # critic no lag
        diff = self.dof_pos - self.ref_dof_pos
        # 73
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 10.,  # 1 # sum of all fix link mass
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)
        
        # random add dof_pos and dof_vel same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]  
        # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]
                
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        # imu lag, including rpy and omega
        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]
        
        # obs q and dq
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel  

        # 47
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12
            dq,  # 12
            self.actions,   # 12
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.env.num_single_obs == 48:
            stand_command = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) <= self.cfg.commands.stand_com_threshold)
            obs_buf = torch.cat((obs_buf, stand_command),dim=1)
            
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)
        
        if self.add_noise:  
            # add obs noise
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)
        
        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        
        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)*0.5
        
        #resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)
        
        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait) ,dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5

# ================================================ Rewards ================================================== #
    # ============================================================
    # 精简 Reward 设计 (29→10)
    # ① velocity_tracking — 合并 tracking_lin/ang_vel, vel_mismatch, low_speed, track_vel_hard, stand_still
    # ② stability        — 合并 orientation, base_height, base_acc, default_joint_pos, feet_rotation
    # ③ efficiency        — 合并 torques, dof_vel, dof_acc, action_smoothness, feet_contact_forces
    # ④ ref_joint_pos     — 保留，步态轨迹引导
    # ⑤ feet_contact_number — 保留，步态相位核心
    # ⑥ feet_clearance    — 保留，抬脚高度
    # ⑦ foot_slip         — 保留，脚底打滑
    # ⑧ collision         — 保留，安全硬约束
    # ⑨ safety_limits     — 合并 dof_pos/vel/torque_limits
    # ============================================================

    def _reward_tracking_lin_vel(self):
        """
        ①-a 线速度跟踪 — 独立 reward，保持独立梯度
        v4: 从 weighted average 拆出，exp 系数降低 k=5→2 扩大梯度区域
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        r = torch.where(stand_command,
                        torch.exp(-torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) / self.cfg.rewards.tracking_sigma),
                        r)
        return r

    def _reward_tracking_ang_vel(self):
        """
        ①-b 角速度跟踪 — 独立 reward
        v4: 从 weighted average 拆出，独立梯度方向
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        r = torch.where(stand_command,
                        torch.exp(-torch.square(self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma),
                        r)
        return r

    def _reward_symmetry(self):
        """
        ② 对称性 — 基于相位的左右腿交替步态对称性 reward (V14)
        
        V14 vs V13 关键修正 — 基于真实步态数据分析:
        
        数据发现 (gait_20260608_084521.csv, vx>0.3m/s 区间):
        1. 支撑腿 hip_pitch 不是不动的! 支撑相内 dev 从 +0.10→+0.27→-0.05 (幅度0.31 rad)
           物理原因: 身体重心前移过支撑脚 → 髋从后仰变前倾
        2. 两腿 hip_pitch dev 经常同号! 因为左右髋轴方向相同(URDF: 0,0,1)
           身体前移时两腿同方向旋转, 这是正确的物理行为
        3. 摆动/支撑 |dev| 峰值比 ≈ 1.4x (0.39/0.27), 不是远大于
        
        V13 的错误:
        - anti_phase 维度 (|left_dev + right_dev| ≈ 0) 完全错误
          两腿同号是正确行为, anti_phase 在正确步态时≈0.4, 严重拖累reward
        - phase_agree 陡度 x10 在 1.4x 比值时区分力不足
        
        V14 修正:
        - 移除 anti_phase (两腿同号 ≠ 不对称)
        - phase_agree 陡度 x10 → x15 (1.4x比值 → sigmoid(6)=0.997)
        - 保留 amp_gate (不奖励站立不动)
        
        量级校准 (基于真实数据反向计算, scale=1.0):
        - 正确交替步态: phase_agree≈0.90, amp_gate≈0.82 → reward≈0.76
        - 站立不动: amp_gate≈0 → reward≈0.10 (不奖励)
        - 同相跳跃: phase_agree≈0.50 → reward≈0.35 (惩罚)
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        # 关节偏离默认位置
        lhp_dev = self.dof_pos[:, 0] - self.default_dof_pos[:, 0]   # left hip pitch
        rhp_dev = self.dof_pos[:, 6] - self.default_dof_pos[:, 6]   # right hip pitch
        lkp_dev = self.dof_pos[:, 3] - self.default_dof_pos[:, 3]   # left knee pitch
        rkp_dev = self.dof_pos[:, 9] - self.default_dof_pos[:, 9]   # right knee pitch
        
        # 摆动相判定: sin > 0 → 右腿摆动, sin < 0 → 左腿摆动
        is_right_swing = (sin_pos > 0).float()
        
        # === 1. 相位一致性: 摆动腿的|偏离量| > 支撑腿的|偏离量| ===
        # 绝对值比较, 不依赖关节方向符号
        # V14: sigmoid 陡度 x15 (原 x10), 适配 1.4x 实际比值
        
        # 髋pitch (主, 70%)
        swing_hip = is_right_swing * torch.abs(rhp_dev) + (1 - is_right_swing) * torch.abs(lhp_dev)
        stance_hip = is_right_swing * torch.abs(lhp_dev) + (1 - is_right_swing) * torch.abs(rhp_dev)
        phase_agree_hip = torch.sigmoid((swing_hip - stance_hip) * 15.0)
        
        # 膝pitch (辅助, 30%)
        swing_knee = is_right_swing * torch.abs(rkp_dev) + (1 - is_right_swing) * torch.abs(lkp_dev)
        stance_knee = is_right_swing * torch.abs(lkp_dev) + (1 - is_right_swing) * torch.abs(rkp_dev)
        phase_agree_knee = torch.sigmoid((swing_knee - stance_knee) * 15.0)
        
        phase_agree = 0.7 * phase_agree_hip + 0.3 * phase_agree_knee
        
        # === 2. 幅度门控: 不奖励站立不动 ===
        amplitude = (torch.abs(lhp_dev) + torch.abs(rhp_dev) +
                     0.5 * (torch.abs(lkp_dev) + torch.abs(rkp_dev)))
        amp_gate = 1.0 - torch.exp(-amplitude / 0.2)
        
        # === 综合: phase_agree × amp_gate (移除 anti_phase) ===
        symmetry = phase_agree * amp_gate
        
        # 基线 0.1: 避免完全零梯度
        return 0.1 + 0.9 * symmetry

    def _reward_stability(self):
        """
        ② 稳定性 — V10: 精简为姿态 + 高度
        移除: COM加速度、关节yaw/roll偏差、脚旋转 — 这些是自然步态的结果，不需显式约束
        摔倒 = episode终止，是最终保险；此处只提供保持直立/高度的梯度
        """
        # --- 躯干姿态 ---
        # V11-c: ×20→×10, pitch=3°时 raw从0.35→0.59, 给走路留梯度
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10)

        # --- 质心高度稳定 ---
        # V11-c: ×100→×10, Δh=2cm时 raw从0.14→0.82, 给走路留梯度
        stance_mask = self._get_stance_mask()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        r_height = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10)

        return (orientation + r_height) / 2.

    def _reward_efficiency(self):
        """
        ③ 能效 — 机械功率 |τ·q̇| 惩罚 (V10-c: 方案B)
        
        核心改进: 从 τ² 切换到 |τ·q̇|
        - τ² 对"站立"不公平: 站立需要大 τ 支撑体重(等长收缩), τ² ≈ 100
        - |τ·q̇| 只惩罚机械做功: 站立时 q̇ ≈ 0, 功率 ≈ 0
        - 消除了"站着不动"局部最优的效率优势
        
        量级对比:
        - 旧 τ²: 站立 ≈ 100, 走路 ≈ 1750 (17× 差)
        - 新 |τ·q̇|: 站立 ≈ 0, 走路 ≈ 360 (∞ × 差, 彻底消除站立优势)
        
        Scale 校准:
        - 走路时 |τ·q̇| ≈ 360, scale -1e-3 → 惩罚 ≈ -0.36
        - tracking_lin_vel 走路 ≈ +1.15, 3.2× 余量
        - 站立时 |τ·q̇| ≈ 0, 惩罚 ≈ 0, 但 tracking 也 ≈ 0
        - 探索走路: tracking +0.05, efficiency -0.01 → 净正! 探索被鼓励!
        """
        mechanical_power = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        return mechanical_power

    def _reward_ref_joint_pos(self):
        """
        ④ 步态轨迹引导 — 关节跟踪参考轨迹
        v6: 移除 walk_decay，全程不衰减
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        r[stand_command] = 1.0
        # v6: 不衰减，全程生效
        return r

    def _reward_feet_contact_number(self):
        """
        ⑤ 步态相位-接触对齐
        v6: 移除 walk_decay，全程不衰减
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        stance_mask[stand_command] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        r = torch.mean(reward, dim=1)
        # v6: 不衰减，全程生效
        return r

    def _reward_feet_clearance(self):
        """
        ⑥ 抬脚高度 — 高斯平滑奖励
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_to_ankle_distance
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        swing_mask = 1 - self._get_stance_mask()
        height_error = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height)
        rew = torch.exp(-height_error * 20) * swing_mask
        self.feet_height *= ~contact
        return torch.sum(rew, dim=1)

    def _reward_swing_foot_forward(self):
        """
        ⑥-a 前进动力 — 鼓励摆动脚向前迈
        V7: 加 max=0.5/脚 上限 → scaled max = 2.0 × 1.0 = 2.0
        防止 swing 线性增长碾压 stability (V6 根因)
        """
        swing_mask = 1 - self._get_stance_mask()  # 1=swing, 0=stance
        # 脚在世界坐标系下的 x 线速度
        foot_vel_x = self.rigid_state[:, self.feet_indices, 7]  # shape: [N, 2]
        # 前进命令方向
        cmd_dir = torch.sign(self.commands[:, 0]).unsqueeze(1)
        # 摆动脚速度方向与命令方向一致时奖励，上限 0.5 m/s/脚
        rew = torch.clamp(foot_vel_x * cmd_dir, min=0, max=0.5)
        rew *= swing_mask
        # 仅在有移动命令时生效
        has_cmd = (torch.abs(self.commands[:, 0]) > 0.05).unsqueeze(1)
        rew *= has_cmd
        return torch.sum(rew, dim=1)

    def _reward_landing_impact(self):
        """
        ⑧ 落地冲击惩罚 — V9: 降低阈值到 500N (≈3.4× 体重)
        使用平方惩罚: excess = (cfz - threshold)² / threshold²
        鼓励策略通过膝关节屈曲/全身吸收来降低接触力
        参考: QuietWalk (2026) — GRF 惩罚让策略自主学习吸收策略
        """
        max_f = self.cfg.rewards.max_contact_force
        cfz = self.contact_forces[:, self.feet_indices, 2]  # [N, 2]
        excess = torch.clamp(cfz - max_f, min=0.) ** 2 / (max_f ** 2)
        return torch.sum(excess, dim=1)

    def _reward_landing_compliance(self):
        """
        ⑧-a 落地柔顺奖励 — V9 新增
        在高接触力阶段，奖励整条腿的屈曲吸收。
        
        整腿视角 (用户反馈): 同样的膝关节角度在不同髋关节角度下
        对应不同的有效腿长变化。因此直接用接触力作为"需要吸收"的信号，
        用 hip+knee 的角速度之和作为"正在吸收"的信号。
        
        物理含义: 当脚受到高接触力时，如果同侧髋+膝正在屈曲
        (角速度 > 0)，说明腿在主动缩短以吸收冲击。
        奖励 = 归一化屈曲角速度 × 高力 mask。
        
        信号密度: 在高接触力期间持续激活 (不是单帧脉冲)，
        约占 10-15% 的步数，足以形成有效梯度。
        """
        cfz = self.contact_forces[:, self.feet_indices, 2]  # [N, 2]
        
        # 高接触力阈值: 2× 体重 (~300N)，正常单脚站立约 150N
        high_force = cfz > self.cfg.rewards.compliance_force_threshold  # [N, 2]
        
        # 整腿屈曲角速度: hip_pitch_vel + knee_pitch_vel
        # 左腿: joints [0]=hip_pitch, [3]=knee_pitch
        # 右腿: joints [6]=hip_pitch, [9]=knee_pitch
        left_leg_flex = torch.clamp(self.dof_vel[:, 0] + self.dof_vel[:, 3], min=0, max=5.0)
        right_leg_flex = torch.clamp(self.dof_vel[:, 6] + self.dof_vel[:, 9], min=0, max=5.0)
        
        # 归一化屈曲速度 (0~1)，然后乘以高力 mask
        left_rew = (left_leg_flex / 5.0) * high_force[:, 0].float()
        right_rew = (right_leg_flex / 5.0) * high_force[:, 1].float()
        
        return left_rew + right_rew

    def _reward_foot_slip(self):
        """
        ⑦ 脚底打滑惩罚
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_collision(self):
        """
        ⑧ 碰撞惩罚（安全硬约束）
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        """⑨-a 关节位置限制"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        """⑨-b 关节速度限制"""
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_torque_limits(self):
        """⑨-c 力矩限制"""
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def update_command_curriculum(self, env_ids):
        """
        v4.2: 基于 common_step_counter 的渐进式 curriculum
        目标速度上限 0.6 m/s（稳定行走目标 0.5 m/s，留余量）
          Phase 1: steps 0→20000     (iter 0→833)    cmd 0.3→0.4 低速适应
          Phase 2: steps 20000→60000 (iter 833→2500)  cmd 0.4→0.6 逐步提速
          Phase 3: steps 60000+      (iter 2500+)     cmd 0.6 固定
        """
        steps = self.common_step_counter
        max_cmd = self.cfg.commands.max_curriculum  # 0.6
        start_cmd = 0.3

        if steps < 20000:
            progress = steps / 20000.0
            current_max = start_cmd + progress * (0.4 - start_cmd)
        elif steps < 60000:
            progress = (steps - 20000) / 40000.0
            current_max = 0.4 + progress * (max_cmd - 0.4)
        else:
            current_max = max_cmd

        self.command_ranges["lin_vel_x"][1] = current_max
        self.command_ranges["lin_vel_x"][0] = -current_max / 2

    # ============================================================
    # 以下为保留的辅助 reward（不在 scales 中，仅供 termination 等内部使用）
    # feet_air_time 用于 _reward_feet_clearance 内部状态维护
    # ============================================================
    def _reward_feet_air_time(self):
        """维护 feet_air_time 状态（被 feet_clearance 间接依赖）"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)