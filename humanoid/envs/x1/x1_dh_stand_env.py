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
        # V13: single_foot_contact 历史缓冲 (0.2s / dt 帧 @50Hz = 10 帧)
        grace_frames = int(self.cfg.rewards.single_contact_grace / self.dt)
        self.single_contact_history = torch.zeros((self.num_envs, grace_frames), device=self.device)
        # V13: feet_airtime 需要的接触缓冲 (独立于旧 _reward_feet_air_time)
        self.airtime_contact_prev = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        # === Round-4 (loop iter4) explicit-alternating-contact 状态缓冲 ===
        # alt_contact_prev: 上一步真实接触 (检测落地事件 first_contact)
        # alt_last_foot: 上次落地的脚 (-1=未初始化/0=左/1=右), 交替判定基准
        self.alt_contact_prev = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.alt_last_foot = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # === Round-1redo gait telemetry (行为无关，仅诊断) ===
        # 上轮 reward 指标无法证明是否真前向行走；加入 per-env 累计缓冲，
        # 在 reset_idx 时按 episode 均值上报为 Episode/ 指标。
        # forward vel: base_lin_vel[:,0] (机体系前向)
        # 接触状态: 单脚/双脚/零脚 占比 (区分交替行走 vs 静态站立)
        # 足部离地高度差: max(footz)-min(footz) (交替抬脚 proxy, shuffle/站立≈0)
        n = self.num_envs
        self._diag_steps = torch.zeros(n, device=self.device)
        self._fwd_vel_sum = torch.zeros(n, device=self.device)
        self._fwd_vel_abs_sum = torch.zeros(n, device=self.device)
        self._walk_steps = torch.zeros(n, device=self.device)
        self._walk_fwd_vel_sum = torch.zeros(n, device=self.device)
        self._single_c_sum = torch.zeros(n, device=self.device)
        self._double_c_sum = torch.zeros(n, device=self.device)
        self._zero_c_sum = torch.zeros(n, device=self.device)
        self._foot_diff_sum = torch.zeros(n, device=self.device)


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

        # === Round-1redo gait telemetry accumulation (行为无关) ===
        # 此处 base_lin_vel / contact_forces / rigid_state 均已 refresh (post_physics_step 内)
        self._diag_steps += 1.
        fwd = self.base_lin_vel[:, 0]
        self._fwd_vel_sum += fwd
        self._fwd_vel_abs_sum += torch.abs(fwd)
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        n_contact = contact.float().sum(dim=1)
        self._single_c_sum += (n_contact == 1).float()
        self._double_c_sum += (n_contact == 2).float()
        self._zero_c_sum += (n_contact == 0).float()
        footz = self.rigid_state[:, self.feet_indices, 2]
        self._foot_diff_sum += footz.max(dim=1).values - footz.min(dim=1).values
        # walk-command-only forward velocity (排除站立命令段)
        walk_cmd = (torch.norm(self.commands[:, :3], dim=1) > self.cfg.commands.stand_com_threshold)
        self._walk_steps += walk_cmd.float()
        self._walk_fwd_vel_sum += fwd * walk_cmd.float()

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

        # 49 = 47 proprio + 2 foot-contact (binary)
        # Round-1 obs-contact-feedback: 把真实二值脚接触喂入 actor 观测，
        # 让策略对齐"实际接触"与"步态时钟(sin/cos 相位)"，摆脱静态单脚站立的
        # 局部最优(contact_mask 原本仅存在于 critic privileged obs)。
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12
            dq,  # 12
            self.actions,   # 12
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
            contact_mask.float(),  # 2 ⭐ 真实脚接触反馈 (左/右, 接触力>5N=1)
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
        # V13: 重置 single_foot_contact 历史和 airtime 接触缓冲
        self.single_contact_history[env_ids] = 0.
        self.airtime_contact_prev[env_ids] = False
        # Round-4: 重置 alternating-contact 序列缓冲
        self.alt_contact_prev[env_ids] = False
        self.alt_last_foot[env_ids] = -1
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
        # === Round-1redo gait telemetry logging (行为无关诊断) ===
        st = torch.clamp(self._diag_steps[env_ids], min=1.0)
        ws = torch.clamp(self._walk_steps[env_ids], min=1.0)
        self.extras["episode"]["diag_forward_vel"] = torch.mean(self._fwd_vel_sum[env_ids] / st)
        self.extras["episode"]["diag_abs_forward_vel"] = torch.mean(self._fwd_vel_abs_sum[env_ids] / st)
        self.extras["episode"]["diag_walk_forward_vel"] = torch.mean(self._walk_fwd_vel_sum[env_ids] / ws)
        self.extras["episode"]["diag_single_contact_ratio"] = torch.mean(self._single_c_sum[env_ids] / st)
        self.extras["episode"]["diag_double_contact_ratio"] = torch.mean(self._double_c_sum[env_ids] / st)
        self.extras["episode"]["diag_zero_contact_ratio"] = torch.mean(self._zero_c_sum[env_ids] / st)
        self.extras["episode"]["diag_foot_height_diff"] = torch.mean(self._foot_diff_sum[env_ids] / st)
        # reset diagnostic buffers for reset envs
        self._diag_steps[env_ids] = 0.
        self._fwd_vel_sum[env_ids] = 0.
        self._fwd_vel_abs_sum[env_ids] = 0.
        self._walk_steps[env_ids] = 0.
        self._walk_fwd_vel_sum[env_ids] = 0.
        self._single_c_sum[env_ids] = 0.
        self._double_c_sum[env_ids] = 0.
        self._zero_c_sum[env_ids] = 0.
        self._foot_diff_sum[env_ids] = 0.
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
    # V13: "Minimal Emergence" — 最小涌现设计
    #
    # 核心发现 (van Marum 2024):
    #   tracking + orientation → 只产生跳跃
    #   + single_foot_contact → 产生行走 (步态涌现!)
    #
    # 保留项 (V12→V13):
    #   ① tracking_lin_vel (scale 2.5→1.0)
    #   ② tracking_ang_vel (scale 0.8→0.5)
    #   ③ dof_pos_limits (不变)
    #
    # 新增项 (V13):
    #   ④ single_foot_contact (scale 0.3) — 步态涌现核心
    #   ⑤ feet_airtime (scale 0.3) — 步频正则化
    #   ⑥ orientation (scale 0.5) — 从 stability 拆出
    #   ⑦ base_height (scale 0.2) — 从 stability 拆出
    #   ⑧ torque (scale 0.01) — 温和力矩正则化
    #
    # 移除项 (scale=0):
    #   symmetry, stability, efficiency, landing_impact
    #   及所有步态工程化 reward
    # ============================================================

    def _reward_tracking_lin_vel(self):
        """
        ①-a 线速度跟踪 — 独立 reward，保持独立梯度
        v4: 从 weighted average 拆出，exp 系数降低 k=5→2 扩大梯度区域

        Round-3 (loop iter3) contact-gated-velocity [PIVOT]:
          行走指令下仅单支撑帧(n_contact==1, 当前帧无 grace)赚取速度跟踪奖励。
          bounce 的双脚腾空帧(n_contact≠1)速度跟踪奖励归零 → 逼迫策略转向真实
          单支撑交替行走。这是函数形态改变(scale 不变), 直击两轮证据链确认的
          bounce 稳健吸收态(survival + spoofable speed 双满足)。
          stand 分支(奖励静止)不门控: 站立双脚着地(n_contact==2)是正确的, 且站立
          reward 是"保持静止"非"前进速度", 门控会错误惩罚正确站立。
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        # Round-3 contact-gate (PIVOT): walk 分支乘 single_support (n_contact==1, 当前帧, 无 grace)
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        single_support = (contact.float().sum(dim=1) == 1).float()
        r = r * single_support
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
        ② 对称性 — 镜像对称 Morphological Symmetry (V16)
        
        核心思想: 交替步态中, 左右对应关节偏离默认位的量应等大反向。
        即 l_dev(t) ≈ -r_dev(t) → l_dev + r_dev ≈ 0
        
        数学: mirror_err = Σ_j w_j × (l_dev_j + r_dev_j)²
              reward = exp(-mirror_err / sigma)
        
        为什么 V15 失败而 V16 能区分:
        - V15 (energy-phase): split stance 的 l_energy ≈ r_energy → 高分 (0.876)
        - V16 (mirror):       split stance 的 l+r = 0.35 → 低分 (0.024)
        - 区分度: 40× (sigma=0.5)
        
        梯度特性:
        - ∇_l_dev = 2(l+r)/sigma — 直接推向 l=-r 方向
        - 站立时 l_dev≈0, r_dev≈0 → mirror_err≈0 → reward≈1.0 (无副作用)
        - 大偏差关节自动获得更大梯度 (和 V15 一样)
        
        文献: Ding 2024 [arXiv:2403.10723] morphological symmetry 适配双足
        """
        # 左右各 6 关节偏离默认位
        l_dev = self.dof_pos[:, :6] - self.default_dof_pos[:, :6]   # [N, 6]
        r_dev = self.dof_pos[:, 6:] - self.default_dof_pos[:, 6:]   # [N, 6]
        
        # 镜像偏差: 理想交替步态 l_dev ≈ -r_dev → mirror ≈ 0
        mirror = l_dev + r_dev   # [N, 6]
        
        # 关节权重 — 驱动关节重, 被动关节轻
        # [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]
        weights = torch.tensor([1.0, 0.2, 0.2, 0.8, 0.3, 0.2],
                               device=self.device, dtype=torch.float)
        
        # 加权镜像误差
        mirror_err = torch.sum(weights * (mirror ** 2), dim=1)   # [N]
        
        # sigma=0.5: split stance 得 0.024, 理想步态得 0.987, 区分度 40×
        return torch.exp(-mirror_err / 0.5)

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

    def _reward_swing_step(self):
        """
        ⭐ R2 核心新增 — anti-spoof 跨步 reward (不可欺骗)

        R1redo 诊断：当前 reward 可被欺骗——policy 原地双脚微弹
        (diag_foot_height_diff=6.8mm, zero_contact=45%, 前向速度仅 cmd 的 13-20%)
        即骗取 single_foot_contact + tracking + 存活收益。根因是 reward 栈缺一个
        '不可欺骗的跨步'信号。

        本项同时要求三条件(全 AND)才给分，bounce 全不满足→0：
          ① 步态时钟摆动脚(swing_mask=1)：对齐步态相位
          ② 抬脚高度>min_swing_clearance(0.03m)门控：bounce(6.8mm)判负，stride(5cm)通过
          ③ 该脚向命令方向前进(foot_vx·sign(cmd_x)，clamp 0~0.5)
          ④ 对侧脚确实触地(真单支撑)：bounce 双脚离地→opposite_contact=0→判负
        standing 命令(has_cmd=0)→0，中立不影响站立。

        不可欺骗性数学：
          - bounce: lift=0.0068<0.03→cleared=0 ⇒ reward=0 (条件②失败)
          - 高抬腿原地踏步: foot_vx≈0→fwd=0 ⇒ reward=0 (条件③失败)
          - 双脚同步迈步: opposite_contact=0⇒reward=0 (条件④失败)
          - 真交替跨步: 三条件全满足⇒reward>0
        """
        foot_z = self.rigid_state[:, self.feet_indices, 2]                  # [N,2] 世界 z
        foot_vx = self.rigid_state[:, self.feet_indices, 7]                 # [N,2] 世界 x 线速度
        contact = (self.contact_forces[:, self.feet_indices, 2] > 5.).float()  # [N,2]

        stance_mask = self._get_stance_mask()                               # [N,2] 1=stance
        swing_mask = 1.0 - stance_mask                                      # [N,2] 1=swing

        # 条件②: 抬脚门控 (above-ground 高度)
        lift = foot_z - self.cfg.rewards.feet_to_ankle_distance             # 地面以上高度
        cleared = (lift > self.cfg.rewards.min_swing_clearance).float()     # [N,2]

        # 条件③: 向命令方向前进
        cmd_dir = torch.sign(self.commands[:, 0]).unsqueeze(1)             # [N,1]
        fwd = torch.clamp(foot_vx * cmd_dir, min=0., max=0.5)              # [N,2]
        has_cmd = (torch.abs(self.commands[:, 0]) > 0.05).unsqueeze(1).float()  # [N,1]

        # 条件④: 对侧脚触地 (真单支撑) — 用 flip 取另一只脚的接触
        other_contact = torch.stack([contact[:, 1], contact[:, 0]], dim=1)  # [N,2]

        rew = cleared * fwd * swing_mask * has_cmd * other_contact          # [N,2]
        return torch.sum(rew, dim=1)

    # ============================================================
    # R3: continuous-gait-shape — 把 R2 失败的硬 AND-gate 解耦为三项
    #     连续可微 reward，每项独立梯度，把'门槛'变'斜坡'让策略可探索。
    #     解决 R2-F1(稀疏失效) + R2-F2(bounce 锁定) + 评审要求
    # ============================================================

    def _reward_swing_lift(self):
        """
        ⭐ R3 ⑨-a — 连续抬脚斜坡 (替代 R2 swing_step 硬门控)

        R2 教训: swing_step 硬 AND-gate(抬脚>3cm 才给分)太稀疏，policy 从未触发
        (reward≈0 flat)。本项改用线性连续 ramp，任何抬脚都给正向梯度：

        reward = Σ_f clamp(lift_f / target, 0, 1) × swing_mask_f

        量级 (用 R2 实测 bounce 基线标定):
          - bounce: 摆动脚(swing_mask=1)抬 ~3.4mm → clamp(0.0034/0.05)=0.068
          - 真跨步: 摆动脚抬 5cm → 1.0
          - standing: swing_mask=0 → 0 (中立，_get_stance_mask 在 stand_cmd 时全 stance)
        始终正向梯度：抬得越高分越多，不需跨过门槛。
        """
        foot_z = self.rigid_state[:, self.feet_indices, 2]                    # [N,2] 世界 z
        stance_mask = self._get_stance_mask()                                 # [N,2]
        swing_mask = 1.0 - stance_mask                                        # [N,2]
        lift = foot_z - self.cfg.rewards.feet_to_ankle_distance               # 地面以上高度
        target = self.cfg.rewards.target_feet_height                          # 0.05m
        r = torch.clamp(lift / target, min=0., max=1.0)                       # [N,2] 线性 ramp
        return torch.sum(r * swing_mask, dim=1)

    def _reward_forward_progress(self):
        """
        ⭐ R3 ⑨-b — 不可欺骗的前进 reward (用 base 实际前向速度)

        tracking_lin_vel 是 spoofable 的(exp(-cmd_error) 在振荡时仍高分)，
        是 bounce 的主要收益来源(评审指出)。本项直接奖励 base 实际前向速度，
        振荡骗不到——必须真正前移才得分。

        reward = clamp(vx·sign(cmd_x), 0, cap)/cap × has_cmd

        量级:
          - bounce: vx≈0.15(振荡) → clamp/0.6 ≈ 0.25
          - 真走: vx≈0.6 → 1.0
          - standing: has_cmd=0 → 0 (中立)
        不可欺骗: vx 是 base 实际速度(物理量)，无法靠振荡虚增。

        Round-3 (loop iter3) contact-gated-velocity [PIVOT]:
          再乘 single_support (n_contact==1, 当前帧无 grace)。bounce 双脚腾空帧
          (62% 时间 n_contact≠1)的前向进度奖励归零。standing 段 has_cmd=0 已为0
          不受影响。scale(0.4)不变, 仅改函数形态: reward=f(v)→f(v)·1_single_support。

        Round-6 (loop iter6) fwd-progress-degate [反向消融 R3 gate]:
          R5 关键发现: contact-gate 导致 reward 与真实前向速度脱钩(reward升但vx降)。
          gate 把非单支撑帧(双脚支撑/腾空)的真实vx一律归零, 策略可'维持单支撑相reward
          同时在非单支撑相减少/不做净前移', 即 reward 不再忠实反映 episode 净前移。
          移除 gate(回退 * single_support), 使 reward 忠实计入每帧真实vx, 消除脱钩。
          scale 配套 0.8→0.5(config): ungating 后被 gate 丢弃的~30%相位重新计入,
          raw signal 密度回升, 0.5 维持与 R4(0.8×gated)等效的净梯度量级。
          保留 R3 核心不可欺骗性(base_lin_vel·sign(cmd)·has_cmd), 仅去掉 gate。
          这是 R3 contact-gate 的因果性反向消融: R3 假设'gate切断bounce速度奖励→
          涌现walk', 但 R4(已涌现walk)后 R5 证明 gate 反而压制真实前移 → 现撤销之。

        Round-7 (loop iter7) fwd-net-displacement [形态级修复, 评审指示]:
          R5/R6 联合证伪: '每帧body-vx'形态在gate开/关下都无法兼顾'不脱钩'+'压bounce'
          (R5 gate=on 致脱钩; R6 gate=off 致bounce回升 zero_contact 0.187→0.300)。
          根因: body-frame vx 对bounce振荡敏感(振荡时body vx非零), 且gate丢弃信息。
          本轮换形态: 用 WORLD-frame base 速度(root_states世界xy)投影到'命令方向的世界向量'
          (由base yaw + 机体cmd构建), 得到'每步在世界坐标命令方向上的净前移速度'。
            reward = clamp(v_world · cmd_world_dir, 0, cap)/cap × has_cmd
          (a)不脱钩: v_world是物理净位移速度, 不分支撑相, 忠实反映episode净前移。
          (b)压bounce: bounce在原地振荡, 世界坐标净位移≈0(来回抵消), reward自然低,
              无需contact-gate即可压制bounce——一举解决R5脱钩+R6 bounce回升。
          (c)不可欺骗: v_world是世界系物理量, 无法靠body旋转/振荡虚增(必须真净前移)。
          scale维持0.5(config)。yaw取base_euler_xyz[:,2]。
        """
        # World-frame base xy 速度 (root_states[7:10]是世界系, base_lin_vel是body系)
        v_world = self.root_states[:, 7:9]                                    # [N,2] 世界 xy 速度
        yaw = self.base_euler_xyz[:, 2]                                       # [N] base 偏航
        # 命令方向的世界向量: body前向(cos,sin)·cmd_x + body右向(-sin,cos)·cmd_y
        fwd_x, fwd_y = torch.cos(yaw), torch.sin(yaw)                         # body前向(世界系)
        right_x, right_y = -torch.sin(yaw), torch.cos(yaw)                    # body右向(世界系)
        cmd_x, cmd_y = self.commands[:, 0], self.commands[:, 1]
        dir_x = fwd_x * cmd_x + right_x * cmd_y                               # [N] 命令方向世界x分量
        dir_y = fwd_y * cmd_x + right_y * cmd_y                               # [N] 命令方向世界y分量
        dir_norm = torch.clamp(torch.sqrt(dir_x * dir_x + dir_y * dir_y), min=1e-6)
        dir_x_n = dir_x / dir_norm                                            # 单位向量
        dir_y_n = dir_y / dir_norm
        # 在命令方向上的净前移速度 (世界坐标投影)
        net_fwd = v_world[:, 0] * dir_x_n + v_world[:, 1] * dir_y_n           # [N] m/s
        cap = self.cfg.commands.max_curriculum                                # 0.6
        r = torch.clamp(net_fwd, min=0., max=cap) / cap
        has_cmd = (torch.norm(self.commands[:, :2], dim=1) > 0.05).float()
        return r * has_cmd

    def _reward_no_double_air(self):
        """
        ⭐ R3 ⑨-c — 反 bounce 惩罚 (直接打击双脚同时离地)

        R1redo/R2 诊断: bounce 核心特征是 zero_contact_ratio≈45-52%(双脚同时离地)。
        single_foot_contact 奖励 n_contact==1 但不惩罚 n_contact==0，故 bounce 仍受益。
        本项直接惩罚双脚同时离地。

        reward = both_air (n_contact==0 指示); scale 负 → 惩罚

        量级 (scale -0.4):
          - bounce: zero_contact 52% → raw 0.52 → 罚 -0.21
          - 真走: zero_contact ~5% → 罚 -0.02
          - standing: 双脚着地 n_contact=2 → 0 (不罚)
        直接削弱 bounce 净收益，与 forward_progress 共同把策略推向单支撑交替。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        n_contact = contact.float().sum(dim=1)
        both_air = (n_contact == 0).float()
        return both_air

    def _reward_alternating_contact(self):
        """
        ⭐ R4 (loop iter4) — explicit-alternating-contact [正向 walk 吸引子, 路线转换]

        三轮'削弱bounce'(lr/grace/速度门控)均证明无效, 根因是 walk 吸引子在 reward
        landscape 过弱——single_foot_contact 仅判 n_contact==1、swing_lift 仅判抬脚,
        都不编码'左右交替', 策略无需交替即可在单脚站立相刷分。本项直接用真实接触序列
        (非步态时钟) 编码交替性, 提供 bounce/原地存活此前缺失的强正向竞争对手。

        机制(基于真实接触序列):
          - 落地事件 first_contact = contact & ~alt_contact_prev  (脚从不接触到接触的过渡)
          - 交替判定: 新落地的脚 == 另一只脚(即 != alt_last_foot)
            · 左落 且 上次右落 → 交替 → +1
            · 右落 且 上次左落 → 交替 → +1
          - 不触发的情况:
            · bounce 双脚同时落地(无'上次'另一只脚, 或同脚连落) → 0
            · 原地单脚站立无交替落地事件 → 0
            · 首次落地(alt_last_foot=-1) → 0 (无基准)
          - 更新 alt_last_foot = 本次落地脚(右覆盖左, 故同时落地记录为右、不误判交替)

        不可欺骗性:
          - 同时落地(bounce): first_contact 两脚同时为真, alt_left 需 last==1 且 alt_right 需
            last==0, 二者对同一 last 互斥 → 不触发(除非 last 恰为其中之一且只一只先落, 物理上
            同时落地两脚 first_contact 同帧 → 按右覆盖更新 last, 下次需另一脚单落才触发)。
          - 物理惯性使超快抖动交替难以稳定维持; swing_lift(forward_progress等) 塑造步态质量。
        scale 0.5, 事件驱动稀疏, 步行约每半周期(cycle=0.9s → ~0.45s)一次交替落地。
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.        # [N,2]
        # 落地事件: 当前接触且上一帧不接触
        first_contact = contact & ~self.alt_contact_prev                  # [N,2]
        last = self.alt_last_foot                                          # [N] (-1/0/1)
        # 交替: 新落地的脚 != 上次落地的脚
        alt_left  = first_contact[:, 0] & (last == 1)                      # 上次右、今左 → 交替
        alt_right = first_contact[:, 1] & (last == 0)                      # 上次左、今右 → 交替
        reward = alt_left.float() + alt_right.float()                      # [N] 0 or 1
        # 更新 alt_last_foot: 0=左落, 1=右落 (右覆盖左; 同时落地记右且本次不判交替)
        new_last = last.clone()
        new_last = torch.where(first_contact[:, 0], torch.zeros_like(last), new_last)
        new_last = torch.where(first_contact[:, 1], torch.ones_like(last), new_last)
        self.alt_last_foot = new_last
        self.alt_contact_prev = contact.clone()
        return reward

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

    # ============================================================
    # V13: "Minimal Emergence" 新增 Reward
    # 基于 van Marum 2024 的最小约束设计
    # ============================================================

    def _reward_single_foot_contact(self):
        """
        ⭐ V13 核心新增 — 单脚接触奖励 (van Marum 2024 的关键发现)
        
        核心思想:
        - 跳跃 = 双脚同时离地 → n_contact ≠ 1 → r = 0
        - 行走 = 交替单脚接触 → n_contact = 1 → r = 1
        - 站立命令 → r = 1 (不给偏好, 不阻碍恢复步)
        
        宽限期 0.2s: 在 [t-0.2s, t] 内任一时刻有单脚接触 → r = 1
        允许双支撑阶段, 不要求完美交替
        
        参考: van Marum et al. "Revisiting Reward Design..." (2024)
        他们发现 tracking + orientation 只能产生跳跃, 加上此项即产生行走
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0  # [N, 2]
        n_contact = torch.sum(contact.float(), dim=1)  # [N]
        
        stand_cmd = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        
        # 当前帧单脚接触
        single_now = (n_contact == 1)
        
        # 宽限期: 最近 0.2s 内是否有单脚接触
        single_grace = torch.max(self.single_contact_history, dim=1).values > 0.5  # [N]
        
        # 行走命令时: 单脚接触 or 宽限期内曾单脚接触 → r=1, 否则 → r=0
        r = torch.where(stand_cmd,
                        torch.ones(self.num_envs, device=self.device),
                        torch.where(single_now | single_grace,
                                   torch.ones(self.num_envs, device=self.device),
                                   torch.zeros(self.num_envs, device=self.device)))
        
        # 更新历史 (滚动窗口)
        self.single_contact_history = torch.roll(self.single_contact_history, -1, dims=1)
        self.single_contact_history[:, -1] = single_now.float()
        
        return r

    def _reward_feet_airtime(self):
        """
        ⭐ V13 核心新增 — 空中时间步频正则化 (van Marum 2024)
        
        公式: Σ_f (t_air,f - threshold) · 1_touchdown,f
        - 空中时间 > 0.4s → 正奖励 → 鼓励充分抬脚
        - 空中时间 < 0.4s → 负奖励 → 惩罚过快步频
        - 站立命令 → r = 1.0
        
        只在落地瞬间触发 (稀疏), 但梯度很强
        正则化步频: 没有此奖励, 策略倾向过快步频 (局部最优)
        """
        stand_cmd = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0  # [N, 2]
        
        # 检测刚落地: 当前接触 & 上一帧不接触
        first_contact = contact & ~self.airtime_contact_prev  # [N, 2]
        
        # 累积空中时间
        self.feet_air_time += self.dt
        
        # 落地时计算奖励: (air_time - threshold) * 1_touchdown
        threshold = self.cfg.rewards.airtime_threshold  # 0.4s
        air_rew = torch.sum(
            torch.clamp(self.feet_air_time - threshold, min=-threshold) * first_contact.float(),
            dim=1
        )
        
        # 接触时重置 air_time
        self.feet_air_time *= ~contact
        
        # 保存接触状态
        self.airtime_contact_prev = contact.clone()
        
        # 站立命令: 常量 1.0
        r = torch.where(stand_cmd,
                        torch.ones(self.num_envs, device=self.device),
                        air_rew)
        return r

    def _reward_orientation(self):
        """
        ⭐ V13 新增 — 躯干姿态 (从 V12 stability 拆出)
        
        exp(-||projected_gravity_xy|| × 10)
        - 直立时 g_xy ≈ 0, reward ≈ 1.0
        - 倾斜 5° 时 g_xy ≈ 0.087, reward ≈ 0.42
        - 倾斜 10° 时 g_xy ≈ 0.17, reward ≈ 0.18
        """
        return torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 10)

    def _reward_base_height(self):
        """
        ⭐ V13 新增 — 质心高度稳定 (从 V12 stability 拆出)
        
        exp(-|h - target| × 10)
        - 维持期望站高 0.61m
        - Δh = 2cm 时 reward ≈ 0.82
        - Δh = 5cm 时 reward ≈ 0.61
        """
        stance_mask = self._get_stance_mask()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.clamp(torch.sum(stance_mask, dim=1), min=1.0)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10)

    def _reward_torque(self):
        """
        ⭐ V13 新增 — 温和力矩正则化 (van Marum 用 weight=0.01)

        exp(-Σ|τ| / 100)
        - 归一化因子 100 使 reward 在合理范围内
        - 站立时 τ ≈ 20-40Nm per joint, Σ|τ| ≈ 240-480
        - exp(-300/100) ≈ 0.050 → ×0.01 scale = 0.0005 (微弱信号)
        - 走路时 τ 更大, reward 更低 → 温和倾向减少力矩
        """
        return torch.exp(-torch.sum(torch.abs(self.torques), dim=1) / 100.0)

    def _reward_hip_yaw_reg(self):
        """
        ⭐ V14 新增 — 髋 yaw 正则化 (anti-hip-twist, 根因修复)

        根因: hip_yaw 关节 URDF 限位 ±3.14 rad 近乎无界 → dof_pos_limits
        (soft 0.98, 罚区仅 ±0.063rad) 完全无法约束它。步态日志定量证实
        hip-twist 走时左髋 yaw 均值 +2.2rad (默认 -0.31, err=2.51rad)、
        右髋 yaw 均值 +0.72rad (默认 +0.31)，同时 base_wz 振荡 ±1.9rad/s、
        base_vy ±0.3m/s —— 策略用骨盆扭转代偿前进速度跟踪，纯前进命令
        下 vx 仅达命令的 67%。

        本项把髋 yaw 拉回标称 toed-out 角度 (lhy=-0.31, rhy=+0.31)，
        直接消除 hip-twist 的驱动自由度，迫使策略改用矢状面摆腿前进。

        reward = exp(-((lhy-Δl)² + (rhy-Δr)²) / sigma)
        - 正常走: 髋 yaw ≈ 默认, err≈0.1rad → reward≈0.96
        - hip-twist: 髋 yaw err²≈6.3 → reward≈0.0 (e^-12.9)
        - sigma=0.5: 区分度大, 仍允许转弯时的小幅 yaw 偏移
          (转弯由步态旋转实现, 无需髋 yaw >1rad)
        """
        lhy_err = self.dof_pos[:, 2] - self.default_dof_pos[:, 2]
        rhy_err = self.dof_pos[:, 8] - self.default_dof_pos[:, 8]
        yaw_err = torch.square(lhy_err) + torch.square(rhy_err)
        return torch.exp(-yaw_err / 0.5)

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