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

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X1DHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66      #all histroy obs num
        short_frame_stack = 5   #short history step
        c_frame_stack = 3  #all histroy privileged obs num
        # Round-1 obs-contact-feedback: 47 proprio + 2 real binary foot-contact
        # Round-9 (loop iter9) obs-base-world-vel [新轴: 观测空间]:
        #   R5-R8 证明 forward-reward 工程是死胡同, 根因是 actor 从未观测真实质心速度
        #   (base_lin_vel 仅在 critic privileged obs), 故前向控制开环、易被 yaw/振荡欺骗。
        #   本轮把 base 世界 xy 速度(2维)加入 actor 观测 → num_single_obs 49→51。
        #   让策略直接感知真实净前移速度, 消除 reward hacking 结构性根基。
        #   reward config 已回退至 R4 Pareto 最优基线(forward_progress=0.4 R3 body-vx)。
        num_single_obs = 51
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24 #episode length in seconds
        use_ref_actions = False
        num_commands = 5 # sin_pos cos_pos vx vy vz

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
        # termination thresholds
        # V10-d: 0.4 rad (22.9°), was 1.5 rad (85.9°)
        # GOOD walking P99: pitch=6.5°, roll=12.3°, max: pitch=7.0°, roll=12.5°
        # 0.4 rad = GOOD max × ~2, gives 2-3× margin for normal gait
        # BAD model hits 22.9° at t=0.23s vs body contact at t=0.35s (earlier termination)
        termination_pitch_threshold = 0.4  # rad (22.9°)
        termination_roll_threshold = 0.4   # rad (22.9°)
        # V11: height termination — 堵死 "慢慢坐下存活" 漏洞
        # GOOD walking z ∈ [0.59, 0.62], default z = 0.70
        # 阈值 0.35m = GOOD 最低值 × 0.6, 足够余量
        termination_height_threshold = 0.35  # m (低于此高度 terminate)


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/x1.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'

        name = "x1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # Round-1 (flat-terrain-gait-emergence): 地形改 'plane'。
        # 历史 16 版改动全是 reward/obs；terrain(mesh_type) 与 curriculum 自首提交从未被碰。
        # 当前 reward 栈(V13e+V14 hip_yaw_reg+R1 接触观测)从未训练过，却起步于粗糙
        # trimesh(坡/阶/离散/波)+满量程 DR——对尚未学会走路的策略过于严苛，且与
        # hip-twist 局部最优相互混淆。改平面让策略在干净场地尝试涌现正常步态，
        # 既建立本 loop 首条基线，又以 terrain 类别切入(与 reward 调参正交)。
        # reward / DR / 课程一律不动，单一变量隔离地形影响。
        mesh_type = 'plane'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 0.3, 
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2, 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0, 
                        "stairs up": 0., 
                        "stairs down": 0.,
                        "discrete": 0.1, 
                        "wave": 0.0,}
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]   # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5 
            ang_vel = 0.2   
            lin_vel = 0.1   
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': 0.4,
            'left_hip_roll_joint': 0.05,
            'left_hip_yaw_joint': -0.31,
            'left_knee_pitch_joint': 0.49,
            'left_ankle_pitch_joint': -0.21,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.05,
            'right_hip_yaw_joint': 0.31,
            'right_knee_pitch_joint': 0.49,
            'right_ankle_pitch_joint': -0.21, 
            'right_ankle_roll_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40,'hip_yaw_joint': 35,
                     'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
        damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0,'hip_yaw_joint': 4, 
                   'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Round-14 (loop iter14) dr-curriculum [结构转向, 内核指示]:
        # 13轮首次触及DR(此前全程满量)。R13证明架构正确但后期fwd_vel回落源于满量DR强制
        # 保守策略。本轮加DR课程: 核心DR项(摩擦/质量/COM/关节friction/damping/gains)的随机
        # 化范围按 dr_strength = min(1, common_step_counter/dr_curriculum_steps) 线性增长。
        # 早期(step0)范围=标称值(无随机), 后期逐步达满量。dr_curriculum_steps=96000(=4000iter×24step)。
        dr_curriculum_steps = 48000  # 0→满量 线性增长的步数 (R15: 96000→48000, 评审建议; iter2000达满量, 步态形成期iter1000-2000显著生效)
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        # push
        push_robots = True
        push_interval_s = 4 # every this second, push robot
        update_step = 2000 * 24 # after this count, increase push_duration index
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25] # increase push duration during training
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3] # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]     # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]
        
        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]
        
        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]
        
        add_imu_lag = False
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]
        
        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 0.6
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25.  # time before command are changed[s]
        gait = ["walk_omnidirectional","stand","walk_omnidirectional"] # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2,6],
                           "walk_lateral": [2,6],
                           "rotate": [2,3],
                           "stand": [2,3],
                           "walk_omnidirectional": [4,6]}

        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05 # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.2, 0.3] # 初始低速起步，curriculum 渐进到 ±0.4/0.8
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61
        foot_min_dist = 0.2
        foot_max_dist = 1.0

        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos
        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
        target_feet_height = 0.05   # 保持 V7 — 抬脚是正确步态，不应降低
        target_feet_height_max = 0.08  # 保持 V7
        # R2: swing_step 门控 — 最小抬脚高度。bounce 实测 lift≈6.8mm，stride≈5cm，
        # 故 0.03m 门控使 bounce 得 0、stride 得分。物理依据：R1redo diag_foot_height_diff=6.8mm。
        min_swing_clearance = 0.03
        feet_to_ankle_distance = 0.041
        # Loop-v2 Round-1 (gait-cadence-slow) [新轴: 步态时钟周期, 正交于reward/obs/DR]:
        # 收割TASK_20260711_135(anneal)后确认持续瓶颈是步态质量而非前向能力——fwd可被推到
        # 0.58-1.0m/s但double_contact始终0.06-0.09(真实行走0.2-0.3)、zero_contact 0.17-0.21
        # (bounce)、single从未破0.8(峰0.789)。cycle_time自首提交0.9s从未被触碰。
        # 0.9→1.0s放缓步频: 更长支撑相→更多双支撑重叠(double_contact↑)、更少腾空(zero_contact↓),
        # 给步态更多时间建立干净交替单支撑。真实人行走stride~1.0-1.2s,X1的0.9s偏快。
        # 保留speed-coupling(k=1.0): cycle_time=1.0下 vx=0→1.0s, vx=0.6(cap)→0.625s(仍合理快步)。
        # airtime_threshold=0.30s 与新cycle仍自洽(max airtime=0.5s,自然airtime0.3-0.35s)。
        cycle_time = 1.0
        # Round-13 (loop iter13) speed-coupled-gait-clock [结构转向, 内核指示]:
        # R12相位跟踪达成步态目标但fwd_vel低(0.201), 根因是固定cycle_time使相位跟踪
        # 与前向reward独立竞争。速度耦合: cycle_time_eff = cycle_time/(1+k·vx_cmd_norm)。
        # 快走→快步频(Cassie/Digit/生物力学), 使相位跟踪与前移物理协同。
        # k=1.0: vx=0→cycle0.9s(慢步), vx=0.6(cap)→0.45s(快步)。基于cmd速度(非实际, 防hack)。
        gait_speed_coupling_k = 1.0
        # Round-16 (loop iter16) fwd-progress-anneal [结构转向: 时间维度reward退火]:
        # R16静态Pareto证明无点同时single>0.8且fwd>0.3。本轮加时间维度:
        # forward_progress有效scale两阶段退火(乘子在_reward_forward_progress内应用):
        #   阶段1(step < warmup): 因子=1.0, 有效scale=0.4(弱前向), gpt=1.0强→步态先建立(R16已证single>0.8)
        #   阶段2(warmup→full): 因子线性1.0→max, 有效scale 0.4→0.4×max→前向驱动力增加推fwd>0.3
        # 假设: 已建立的步态吸引子(gpt1.0)足够稳以抵抗前向增加, 保single>0.8同时fwd>0.3。
        # === Loop-v2 Round-1 (gait-cadence-slow): 回退anneal以隔离cycle_time单变量 ===
        # TASK_20260711_135收割结论: anneal(2.5×)在iter3000-4000温和档确实同时提升single(0.67->0.74)
        # 和fwd(0.31->0.42)(部分打破静态Pareto), 但满档(iter5000+)OVERSHOOT——fwd飙到0.58-1.0而
        # gpt崩0.74->0.22、tracking_lin_vel崩0.13->0.01(无视指令狂奔)、zero 0.20、double 0.06。
        # 把anneal_max设1.0使退火因子恒=1.0(=禁用anneal), 恢复R16 forward_progress(threshold-bonus,
        # scale0.4)。本轮=R16+cycle_time=1.0单变量, 干净归因步频对步态质量的影响。
        rew_fwd_anneal_warmup = 72000   # step (=3000 iter) 阶段1结束, 此前因子=1.0
        rew_fwd_anneal_full = 120000    # step (=5000 iter) 退火完成, 因子=max
        # Loop-v2 Round-2 (moderate-fwd-anneal): anneal_max 1.0->1.5 [单变量, 叠在R1基线上]:
        # R1(cycle_time=1.0)首次达成single>0.8(123点同时fwd>0.3), 但F5显示late期fwd回落
        # 0.20-0.23——强gpt吸引子(0.83)抑制前向速度。F1证明温和退火档(有效fwd scale
        # 0.4-0.65, 即anneal_max≤1.6)能同时提升single和fwd(部分打破Pareto); F2证明满档
        # 2.5×过冲毁步态(gpt崩0.74->0.22)。本轮选1.5×(有效scale 0.4->0.6), 恰在F1验证的
        # 温和工作区上界。在更稳定的cycle_time=1.0基线上(gpt iter3000已达0.754), 温和前向
        # 驱动应把fwd均值稳定拉过0.3同时保single>0.8。
        # 数值估算: vx=0.3时 fwd reward 0.30->0.45(+50%); vx=0.477(峰)时 0.59->0.88≈gpt0.83(竞争但不碾压)。
        # 阶段时序: iter0-3000 factor=1.0(gpt先稳, R1已证iter3000 gpt=0.754); iter3000-5000 ramp 1.0->1.5; iter5000+ 1.5恒定。
        rew_fwd_anneal_max = 1.5        # Loop-v2 R2: 1.0->1.5 温和前向退火(有效scale 0.4->0.6)
        # v6: walk_decay 已移除 — 用 swing scale + stability 降低来平衡
        # walk_decay = 0.3  # v5: removed, caused stability collapse
        # V10: only_positive_rewards=False — 让 penalty 有真实梯度推力
        only_positive_rewards = False
        # tracking reward = exp(-error/sigma)
        tracking_sigma = 0.15  # V13e: 0.25→0.15, 拉大扭胯(0.81)vs正常走(1.0)的reward差距
        # V10: GRF 阈值降低到 300N (≈1.2× 体重), 对齐 Schumacher c_pain
        max_contact_force = 500  # V11-c: 300→500N (3.4×体重), 只惩罚极端冲击, 不与tracking冲突
        compliance_force_threshold = 300  # 保留（函数体仍在，scale=0 不影响）
        # V13: single_foot_contact 宽限期 (秒), van Marum 用 0.2s
        # Round-2 (loop iter2) sfc-grace-0p2-to-0p04: Round-1 TASK_20260709_014 诊断出
        # bounce 的具体成吸引机制 = 此 0.2s 宽限期(20控制帧@dt=0.01s)被策略利用:
        # 一次落地瞬间的单脚接触(实际单脚接触仅占32%时间)让其后整个~0.3s腾空弹跳都领
        # '行走'奖励(reward触发率57%, 比实际单接触高25%)。收紧到 0.04s(4帧):
        # 行走(单脚接触为主导状态)仍持续得 r=1; 弹跳(双脚离地65%)的reward传播窗口
        # 从20帧缩到4帧, 切断其主要收益来源。van Marum 平台无强制步态时钟故0.2s可接受,
        # 本 env 有强制0.9s步态时钟, 行走占空比0.6-0.7 → 单脚接触本就是常态, 无需长grace。
        single_contact_grace = 0.04
        # V13: feet_airtime 最低空中时间阈值 (秒)
        # Round-1 修正 (airtime-cycle-consistency):
        #   van Marum 用 0.4s，但其平台 Digit 无强制步态时钟。本 env 有强制 0.9s
        #   步态时钟(喂入 obs sin/cos 相位、stance_mask、ref 轨迹)，二者须一致。
        #   数学: cycle=0.9s → 最大空中时间=0.45s(50%占空比)；真实行走占空比 0.6-0.7
        #   → 空中时间 0.27-0.36s。原阈值 0.4s 要求占空比≤0.556，故对任何真实步态
        #   airtime reward 恒为负(与"鼓励充分抬脚"意图相反)。
        #   降至 0.30s = cycle×(1-0.6占空比)，使 airtime 成为真正双极正则项:
        #   充分抬脚→正、过快碎步→负。与 0.9s 步态时钟物理自洽。
        airtime_threshold = 0.30
        
        class scales:
            # ============================================================
            # Reward V13: "Minimal Emergence" — 最小涌现
            #
            # 设计哲学: 基于 van Marum (OSU/Digit 2024) 的最小约束奖励
            # 核心发现: 仅 tracking + orientation → 跳跃
            #           + single_foot_contact → 行走 (步态涌现!)
            #
            # 移除所有工程化步态 reward (symmetry, ref_joint_pos, 
            # feet_contact_number, feet_clearance, swing_foot_forward, 
            # efficiency, landing_impact) — 它们互相竞争梯度
            #
            # 参考论文: "Revisiting Reward Design and Evaluation for 
            # Robust Humanoid Standing and Walking" (van Marum 2024)
            # =============================================================
            # ① 任务目标
            tracking_lin_vel = 0.3    # R3: 0.6→0.3, 削弱 bounce 主要收益(spoofable: 0.48@0.6但实际前向仅22%)
            tracking_ang_vel = 0.5    # V12: 0.8→0.5
            # ② 行走涌现 — 单脚接触 (van Marum 的关键发现)
            # R12 (loop iter12) gait-phase-tracking [结构转向]: 禁用此离散事件reward——
            # 11轮证明它可被hacking(n_contact==1瞬时状态可骗), 用gait_phase_tracking(连续相位跟踪)替代。
            single_foot_contact = 0.0
            # ③ 步频正则化 — 空中时间 (van Marum 公式)
            feet_airtime = 0.3        # ⭐ 新增: (t_air - 0.4)·1_td
            # ④ 躯干姿态
            orientation = 0.5         # ⭐ 新增: 从 stability 拆出, exp(-||g_xy||×10)
            # ⑤ 质心高度
            base_height = 0.5         # V13d: 0.2→0.5, 惩罚蹲低 (V13c height=0.17, 策略蹲着走锁死)
            # ⑥ 温和力矩正则化
            torque = 0.01             # ⭐ 新增: exp(-Σ|τ|/100), van Marum 权重 0.01
            # ⑧ V14 anti-hip-twist: 髋 yaw 正则化
            #    hip_yaw 限位 ±3.14rad 无界致 dof_pos_limits 失效 → 加显式正则
            #    正常走 raw≈0.96, hip-twist raw≈0.0; scale 0.4 → 0.38 swing 足以打破局部最优
            hip_yaw_reg = 0.4
            # === R3 continuous-gait-shape: 把 R2 失败的硬 AND-gate(swing_step) 解耦为
            #    三项连续可微 reward，每项独立给梯度，把'门槛'变'斜坡'让策略可探索 ===
            # ⑨-a swing_lift — 连续抬脚斜坡 (替代 swing_step 硬门控)
            #     线性 ramp: clamp(lift/0.05, 0, 1)×swing_mask
            #     bounce(3.4mm) raw≈0.068, 真跨步(5cm) raw≈1.0 — 始终正向梯度鼓励抬脚
            swing_lift = 0.4
            # ⑨-b forward_progress — 不可欺骗的前进 reward (用 base 实际前向速度)
            #     clamp(vx·sign(cmd), 0, 0.6)/0.6 — 必须真正前移才得分，振荡骗不到
            #     bounce(0.15m/s) raw≈0.25, 真走(0.6) raw≈1.0
            forward_progress = 0.4
            # ⑨-c no_double_air — 反 bounce 惩罚 (直接打击双脚同时离地)
            #     both_air 指示 (scale 负 → 惩罚)
            #     bounce zero_contact 52% → raw 0.52 → 罚 -0.21; 真走 5% → 罚 -0.02
            no_double_air = -0.4
            # === R4 (loop iter4) explicit-alternating-contact [正向吸引子, 路线转换] ===
            # 三轮'削弱bounce'失败后,评审指出根因是 walk 吸引子太弱——single_foot_contact
            # 仅判 n_contact==1、swing_lift 仅判抬脚,都不编码'左右交替',策略无需交替即可刷分。
            # 本项用真实接触序列(非步态时钟)检测左右脚交替落地: 每次'新落地的脚 != 上次落地的脚'
            # 触发 +1(落地事件驱动, 稀疏但直接编码交替性)。bounce 双脚同时落地不触发; 原地单脚
            # 站立无交替也不触发。与 bounce/原地存活竞争净收益, 提供此前缺失的强正向 walk 吸引子。
            # scale 0.5 ≈ single_foot_contact(0.8)与 swing_lift(0.4)之间, 量级可观察但不压制。
            # Round-11 (loop iter11) alt-contact-scale-up: R10(里程碑, fwd_vel 0.418)已建前向+步态稳,
            # 但gait一项未完全达标——single_contact 0.692(略低于walk目标0.7), alternating_contact
            # reward 0.075(健康项最弱)。scale 0.5→0.7(+40%)增强交替吸引力, 目标推single>0.7且
            # zero<0.15(物理可信度阈值)。R10 threshold-bonus独立保fwd, 不损前向速度。
            # R12 (loop iter12) gait-phase-tracking [结构转向]: 禁用此离散事件reward——
            # R11证明它可被'双支撑内频繁交替'hacking(double_contact飙0.190)。用gait_phase_tracking替代。
            alternating_contact = 0.0
            # === R12 gait-phase-tracking [相位跟踪替代事件检测, Cassie/Digit范式] ===
            # 连续相位跟踪reward: 用步态时钟(sin_pos)定义期望接触, 奖励实际接触与期望相位的
            # 每帧连续匹配(cosine-consistency)。每帧密集可微, 不可被离散事件hack。
            # 替代 single_foot_contact(0.8) + alternating_contact(0.7) 两项, scale 0.8
            # 对齐原 single_foot_contact 量级(原主步态吸引子)。
            gait_phase_tracking = 1.0  # R16: 0.8→1.0(+25%), 评审首选reward轴, 目标推single_contact>0.8(R12相位跟踪曾达0.805); 增强相位跟踪吸引子
            # ⑩ 安全网
            dof_pos_limits = -10.
            # === V12 遗留 (scale=0, 函数体保留): ===
            symmetry = 0.0
            stability = 0.0
            efficiency = 0.0
            landing_impact = 0.0
            collision = 0.0
            dof_vel_limits = 0.0
            dof_torque_limits = 0.0

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class X1DHStandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'DHOnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims=[256, 128, 64]
        
        #for long_history cnn only
        kernel_size=[6, 4]
        filter_size=[32, 16]
        stride_size=[3, 2]
        lh_output_dim= 64   #long history output dim
        in_channels = X1DHStandCfg.env.frame_stack

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # Round-4 (loop iter4) [评审必须: 抑制三轮稳定的 noise_std 暴涨发散]
        # entropy_coef 0.01→0.005。R1-R3 中 Policy/mean_noise_std 从 1.05 暴涨到 ~4.0-4.4
        # 且与 reward 下滑同步, 表明 entropy_coef=0.01 + adaptive-lr 驱动发散式探索, 会
        # 掩盖新 alternating_contact reward 的效果。降一半以收敛探索。
        entropy_coef = 0.005
        # Round-4 (loop iter4) [评审建议: 干净消融, 去除 R1 起静默生效的 adaptive-lr confound]
        # schedule='fixed' (覆盖 base legged_robot_config 继承的 'adaptive'/desired_kl=0.01)。
        # adaptive-lr 三轮把 lr 在 [5.1e-5, 8.6e-3] 波动(last50 均值~3-4e-4), 无法干净归因。
        # fixed 下 lr=1e-4 真正恒定, 新 reward 项的因果贡献可被干净判定。
        schedule = 'fixed'
        learning_rate = 1e-4
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        if X1DHStandCfg.terrain.measure_heights:
            lin_vel_idx = (X1DHStandCfg.env.single_num_privileged_obs + X1DHStandCfg.terrain.num_height) * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index
        else:
            lin_vel_idx = X1DHStandCfg.env.single_num_privileged_obs * (X1DHStandCfg.env.c_frame_stack - 1) + X1DHStandCfg.env.single_linvel_index

    class runner:
        policy_class_name = 'ActorCriticDH'
        algorithm_class_name = 'DHPPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 20000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'x1_dh_stand'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
