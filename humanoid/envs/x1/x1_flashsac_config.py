# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# FlashSAC + PURE velocity tracking for X1 biped.
#
# Key change vs x1_flashsac_config: removes ALL gait-phase-guided rewards
# (ref_joint_pos, feet_contact_number, feet_clearance, stand_still, etc.)
# and makes velocity tracking the dominant positive reward. This aligns
# with the original FlashSAC paper's environment (IsaacLab pure velocity
# tracking, no reference gait), which is what off-policy SAC needs to
# explore freely and discover walking from scratch.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfgPPO
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg


class X1FlashSACCfg(X1DHStandCfg):
    """Environment config for FlashSAC + pure velocity tracking (no gait guidance)."""

    class env(X1DHStandCfg.env):
        num_envs = 1024

    class asset(X1DHStandCfg.asset):
        # Relaxed termination conditions for off-policy training:
        # higher contact threshold (avoid premature reset on minor bumps)
        # and wider roll/pitch cutoff (give robot room to recover from lean)
        termination_contact_threshold = 20.0   # was 1.0 (hardcoded)
        roll_pitch_threshold = 1.0             # was 1.5 rad (~86° -> ~57°)


class X1FlashSACCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "FlashSACOffPolicyRunner"

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [256]
        actor_num_blocks = 2
        state_estimator_hidden_dims = [128, 64]
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64
        in_channels = X1FlashSACCfg.env.frame_stack
        critic_hidden_dims = [256]
        critic_num_blocks = 2

    class algorithm(LeggedRobotCfgPPO.algorithm):
        gamma = 0.99
        n_step = 3
        tau = 0.01
        actor_lr = 3e-4
        critic_lr = 3e-4
        temp_lr = 1e-4
        init_alpha = 0.2
        target_sigma = 0.3
        actor_update_period = 2
        max_grad_norm = 1.0
        num_bins = 201
        num_qs = 2
        normalized_G_max = 20.0
        normalize_reward = True
        zeta_mu = 2.0
        zeta_max_n = 16
        buffer_max_length = 500_000
        buffer_min_length = 100_000
        batch_size = 2048
        updates_per_interaction_step = 2
        init_random_steps = 0
        bootstrap_on_timeout = False

    class runner:
        policy_class_name = "DualHistoryTanhGaussianActor"
        algorithm_class_name = "FlashSAC"
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 100
        experiment_name = "x1_flashsac"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None


# ====================================================================== #
#  PURE VELOCITY TRACKING REWARD OVERRIDES                               #
# ====================================================================== #
# We override X1DHStandCfg.rewards to strip out gait-phase-guided terms.
# The environment class X1DHStandEnv has _reward_* methods for everything;
# setting a reward's scale to 0 in the config effectively disables it.

class _PureVelocityRewards(X1DHStandCfg.rewards):
    """Rewards for pure velocity-tracking locomotion (no gait phase guidance).

    Strategy:
      - DOMINANT: velocity tracking (lin + ang) — the main learning signal
      - STRONG:   feet_air_time (encourages stepping), orientation, base_height
      - MODERATE: default_joint_pos (stay near nominal posture), feet_distance
      - PENALTY:  torques, dof_acc, collision, action_smoothness, limits
      - DISABLED: all gait-phase-dependent terms (ref_joint_pos, feet_contact_number,
                  feet_clearance, stand_still, swing_foot_forward, foot_landing_pitch)
    """

    soft_dof_pos_limit = 0.98
    soft_dof_vel_limit = 0.9
    soft_torque_limit = 0.9
    base_height_target = 0.61
    foot_min_dist = 0.2
    foot_max_dist = 1.0
    final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
    target_feet_height = 0.05
    target_feet_height_max = 0.08
    landing_pitch_offset = 0.05
    feet_to_ankle_distance = 0.041
    cycle_time = 0.9
    only_positive_rewards = False       # allow negative total (no clipping)
    tracking_sigma = 5
    max_contact_force = 700

    class scales:
        # ===== DISABLED: gait-phase-guided rewards (set to 0) =====
        ref_joint_pos = 0.0              # was 2.2 — depended on sinusoidal ref_dof_pos
        feet_clearance = 0.0            # was 1.5 — depended on phase
        feet_contact_number = 0.0       # was 2.0 — depended on stance_mask
        stand_still = 0.0               # was 2.5 — encouraged "do nothing"
        swing_foot_forward = 0.0        # was 0.5 — depended on phase
        foot_landing_pitch = 0.0        # was 0.3 — depended on phase

        # ===== DOMINANT: velocity tracking (main signal) =====
        tracking_lin_vel = 4.0          # was 1.8 — now the dominant reward
        tracking_ang_vel = 2.5          # was 1.1

        # ===== STRONG: locomotion-enabling rewards (no phase dependency) =====
        feet_air_time = 2.0             # was 1.2 — encourage lifting feet
        orientation = 2.0               # was 1.0 — keep torso upright
        base_height = 0.5               # was 0.2 — keep at target height

        # ===== MODERATE: posture regularizers =====
        default_joint_pos = 1.5         # was 1.0 — stay near nominal angles
        feet_distance = 0.2
        knee_distance = 0.2
        feet_rotation = 0.3
        vel_mismatch_exp = 0.5
        low_speed = 0.2
        track_vel_hard = 0.5
        base_acc = 0.2

        # ===== PENALTY: energy / smoothness / safety =====
        foot_slip = -0.1
        feet_contact_forces = -0.01
        action_smoothness = -0.002
        torques = -8e-9
        dof_vel = -2e-8
        dof_acc = -1e-7
        collision = -1.
        dof_vel_limits = -1
        dof_pos_limits = -10.
        dof_torque_limits = -0.1


# Patch the reward scales onto X1FlashSACCfg.rewards
X1FlashSACCfg.rewards = _PureVelocityRewards
