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
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
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


import os
import csv
import time
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.envs.x1.x1_dh_stand_env import get_euler_xyz_tensor as _get_euler_xyz_tensor
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from datetime import datetime

from threading import Thread


x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
PLAY_CMD_VEL_X = 0.4
joystick_use = True
joystick_opened = False

if joystick_use:
    try:
        import pygame
        pygame.init()
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄/pygame：{e}")
        joystick_use = False
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False 
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_com = False 
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)


    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)


    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt * env_cfg.control.decimation)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states

    custom_save_path = "/personal/train-more"
    run_name_str = args.run_name if args.run_name is not None else "test"
    os.makedirs(custom_save_path, exist_ok=True)

    _csv_file = None
    if LOG_CSV:
        _JOINT_NAMES = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        ]
        _csv_header = ['timestamp_ns', 'phase_sin', 'phase_cos',
                       'cmd_linear_x', 'cmd_linear_y', 'cmd_angular_z',
                       'left_contact', 'right_contact',
                       'base_euler_x', 'base_euler_y', 'base_euler_z',
                       'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']
        for _jn in _JOINT_NAMES:
            _csv_header += [f'action_{_jn}', f'pos_{_jn}', f'vel_{_jn}', f'effort_{_jn}',
                            f'pos_des_raw_{_jn}', f'pos_des_lpf_{_jn}',
                            f'tau_des_raw_{_jn}', f'tau_des_lpf_{_jn}', f'is_parallel_{_jn}']
        _csv_header += ['clip_count',
                        'imu_quat_w', 'imu_quat_x', 'imu_quat_y', 'imu_quat_z',
                        'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
                        'imu_accel_x', 'imu_accel_y', 'imu_accel_z']
        _csv_header += ['left_foot_contact_force_z', 'right_foot_contact_force_z',
                        'left_foot_contact_force_mag', 'right_foot_contact_force_mag',
                        'feet_contact_force_penalty']
        _csv_header += ['base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z']
        _csv_path = os.path.join(custom_save_path, f'isaac_diag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        _csv_file = open(_csv_path, 'w', newline='')
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(_csv_header)
        print(f'CSV logging to: {_csv_path}')
        _action_scale = env.cfg.control.action_scale
        _ankle_action_scale = getattr(env.cfg.control, 'ankle_action_scale', _action_scale)
        _ankle_indices = [4, 5, 10, 11]
        _ddp = env.default_dof_pos
        _default_dof_pos = (_ddp[robot_index] if _ddp.dim() > 1 else _ddp).cpu()
        _feet_idx = env.feet_indices.detach().cpu().numpy().tolist()
        _left_foot_idx, _right_foot_idx = int(_feet_idx[0]), int(_feet_idx[1])
        _max_contact_force = env.cfg.rewards.max_contact_force
        print(f'feet body indices (left,right) = ({_left_foot_idx},{_right_foot_idx}), '
              f'max_contact_force = {_max_contact_force}N')
        _csv_max_steps = int(10.0 / (env_cfg.sim.dt * env_cfg.control.decimation))
        _step_dt_ns = int(env_cfg.sim.dt * env_cfg.control.decimation * 1e9)
        _t0_ns = time.time_ns()

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(2.0, -2.0, 1.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name_str}.mp4"
        video_filepath = os.path.join(custom_save_path, file_name)
        print(f"Recording video to: {video_filepath}")
        video = cv2.VideoWriter(video_filepath, fourcc, 50.0, (1920, 1080))

    obs = env.get_observations()
    frame_count = 0
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    _obs_off = (env_cfg.env.frame_stack - 1) * env_cfg.env.num_single_obs

    vel_sum = 0.0
    step_accum = 0

    # warmup obs history (frame_stack steps) before evaluation
    _warmup_steps = env_cfg.env.frame_stack
    for _ in range(_warmup_steps):
        _w_actions = policy(obs.detach())
        if FIX_COMMAND:
            env.commands[:, 0] = PLAY_CMD_VEL_X
            env.commands[:, 1] = 0
            env.commands[:, 2] = 0
            env.commands[:, 3] = 0.
        obs, _, _, _, _ = env.step(_w_actions.detach())

    for i in range(10*stop_state_log):
        
        actions = policy(obs.detach()) # * 0.
        
        if FIX_COMMAND:
            env.commands[:, 0] = PLAY_CMD_VEL_X
            env.commands[:, 1] = 0
            env.commands[:, 2] = 0
            env.commands[:, 3] = 0.
            
        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0.

        _current_obs = obs

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        current_vel_x = env.base_lin_vel[0, 0].item()
        vel_sum += current_vel_x
        step_accum += 1

        if LOG_CSV:
            _base_quat = env.root_states[robot_index:robot_index+1, 3:7]
            _euler = _get_euler_xyz_tensor(_base_quat)[0]
            _r, _p, _y = _euler[0].item(), _euler[1].item(), _euler[2].item()
            _lf_force_vec = env.contact_forces[robot_index, _left_foot_idx, :]
            _rf_force_vec = env.contact_forces[robot_index, _right_foot_idx, :]
            _lf_force_z = _lf_force_vec[2].item()
            _rf_force_z = _rf_force_vec[2].item()
            _lf_force_mag = torch.norm(_lf_force_vec).item()
            _rf_force_mag = torch.norm(_rf_force_vec).item()
            _csv_row = [
                _t0_ns + i * _step_dt_ns,
                _current_obs[robot_index, _obs_off + 0].item(),
                _current_obs[robot_index, _obs_off + 1].item(),
                env.commands[robot_index, 0].item(),
                env.commands[robot_index, 1].item(),
                env.commands[robot_index, 2].item(),
                int(_lf_force_z > 5.0),
                int(_rf_force_z > 5.0),
                _r, _p, _y,
                env.base_ang_vel[robot_index, 0].item(),
                env.base_ang_vel[robot_index, 1].item(),
                env.base_ang_vel[robot_index, 2].item(),
            ]
            for _j in range(12):
                _act = actions[robot_index, _j].item()
                _pos = env.dof_pos[robot_index, _j].item()
                _vel = env.dof_vel[robot_index, _j].item()
                _eff = env.torques[robot_index, _j].item()
                _scale = _ankle_action_scale if _j in _ankle_indices else _action_scale
                _pos_des = _act * _scale + _default_dof_pos[_j].item()
                _csv_row += [_act, _pos, _vel, _eff, _pos_des, _pos_des, float('nan'), float('nan'), 0]
            _csv_row += [
                0,
                env.root_states[robot_index, 6].item(),
                env.root_states[robot_index, 3].item(),
                env.root_states[robot_index, 4].item(),
                env.root_states[robot_index, 5].item(),
                env.base_ang_vel[robot_index, 0].item(),
                env.base_ang_vel[robot_index, 1].item(),
                env.base_ang_vel[robot_index, 2].item(),
                float('nan'), float('nan'), float('nan'),
            ]
            _lf_pen = min(max(_lf_force_mag - _max_contact_force, 0.0), 400.0)
            _rf_pen = min(max(_rf_force_mag - _max_contact_force, 0.0), 400.0)
            _csv_row += [
                _lf_force_z, _rf_force_z,
                _lf_force_mag, _rf_force_mag,
                _lf_pen + _rf_pen,
            ]
            _csv_row += [
                env.base_lin_vel[robot_index, 0].item(),
                env.base_lin_vel[robot_index, 1].item(),
                env.base_lin_vel[robot_index, 2].item(),
            ]
            if i < _csv_max_steps:
                _csv_writer.writerow(_csv_row)

        if RENDER:
            frame_count += 1
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)

            if frame_count % 2 == 0:
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                target_vel = env.commands[0, 0].item()
                current_vel_x = env.base_lin_vel[0, 0].item()
                avg_vel = vel_sum / step_accum if step_accum > 0 else 0.0

                if LOG_CSV:
                    left_force = _lf_force_z
                    right_force = _rf_force_z
                else:
                    left_force = env.contact_forces[0, env.feet_indices[0], 2].item()
                    right_force = env.contact_forces[0, env.feet_indices[1], 2].item()

                l_on = left_force > 1.0
                r_on = right_force > 1.0

                img_h, img_w = img.shape[:2]
                base_x = img_w - 1150
                base_y = 60
                line_height = 50

                def draw_outlined_text(image, text, pos, color, scale=0.9):
                    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

                speed_text = f"CMD: {target_vel:.2f} | REAL: {current_vel_x:.2f} | AVG: {avg_vel:.2f}"
                draw_outlined_text(img, speed_text, (base_x, base_y), (255, 255, 0), 1.0)

                l_color = (0, 255, 0) if l_on else (0, 0, 255)
                l_text = f"L-FOOT: {'ON ' if l_on else 'OFF'} ({left_force:.1f} N)"
                draw_outlined_text(img, l_text, (base_x, base_y + line_height), l_color)

                r_color = (0, 255, 0) if r_on else (0, 0, 255)
                r_text = f"R-FOOT: {'ON ' if r_on else 'OFF'} ({right_force:.1f} N)"
                draw_outlined_text(img, r_text, (base_x, base_y + line_height * 2), r_color)

                state_text = "STATE: SINGLE SUPPORT"
                state_color = (200, 200, 200)

                if l_on and r_on:
                    state_text = "STATE: *** DOUBLE SUPPORT ***"
                    state_color = (0, 255, 255)
                elif not l_on and not r_on:
                    state_text = "STATE: >>> FLIGHT PHASE <<<"
                    state_color = (255, 0, 255)

                draw_outlined_text(img, state_text, (base_x, base_y + line_height * 3), state_color, 1.0)

                video.write(img[..., :3])
        real_cmd_x = env.commands[robot_index, 0].item()

        if i > stop_state_log*0.2 and i < stop_state_log:
            _lf_idx = int(env.feet_indices[0].item())
            _rf_idx = int(env.feet_indices[1].item())
            dict = {
                    'base_height' : env.root_states[robot_index, 2].item(),
                    'foot_z_l' : env.rigid_state[robot_index, _lf_idx, 2].item(),
                    'foot_z_r' : env.rigid_state[robot_index, _rf_idx, 2].item(),
                    'foot_forcez_l' : env.contact_forces[robot_index, _lf_idx, 2].item(),
                    'foot_forcez_r' : env.contact_forces[robot_index, _rf_idx, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'command_x': real_cmd_x,
                    'base_vel_y':  env.base_lin_vel[robot_index, 1].item(),
                    'command_y': y_vel_cmd,
                    'base_vel_z':  env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw':  env.base_ang_vel[robot_index, 2].item(),
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, 0].item(),
                    'dof_vel': env.dof_vel[robot_index, 0].item(),
                    'dof_torque': env.torques[robot_index, 0].item(),
                    'command_sin': obs[robot_index, _obs_off + 0].item(),
                    'command_cos': obs[robot_index, _obs_off + 1].item(),
                }

            for _j in range(env_cfg.env.num_actions):
                dict[f'dof_pos_target[{_j}]'] = actions[robot_index, _j].item() * env.cfg.control.action_scale,

            for _j in range(env_cfg.env.num_actions):
                dict[f'dof_pos[{_j}]'] = env.dof_pos[robot_index, _j].item(),

            for _j in range(env_cfg.env.num_actions):
                dict[f'dof_torque[{_j}]'] = env.torques[robot_index, _j].item(),

            for _j in range(env_cfg.env.num_actions):
                dict[f'dof_vel[{_j}]'] = env.dof_vel[robot_index, _j].item(),

            logger.log_states(dict=dict)
        
        elif i == stop_state_log:
            logger.plot_states()

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    if LOG_CSV:
        _csv_file.close()
        print(f'CSV saved to: {_csv_path}')

    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = True
    LOG_CSV = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
