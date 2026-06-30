# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# FlashSAC training configuration for the AgiBot X1 biped. The *environment*
# config inherits the full X1 robot definition (URDF, PD gains, domain rand,
# reward shaping) from ``X1DHStandCfg``; only the number of parallel envs is
# reduced (off-policy SAC needs far fewer envs than PPO). The *training* config
# points the runner at the FlashSAC algorithm and sets the off-policy
# hyperparameters.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfgPPO
from humanoid.envs.x1.x1_dh_stand_config import X1DHStandCfg


class X1FlashSACCfg(X1DHStandCfg):
    """Environment config for FlashSAC training of X1 (reuses X1 robot)."""

    class env(X1DHStandCfg.env):
        # Off-policy SAC samples i.i.d. from a replay buffer, so it does not need
        # the huge PPO env count. 1024 parallel envs is the FlashSAC default for
        # GPU simulators and keeps GPU memory in check.
        num_envs = 1024


class X1FlashSACCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = "FlashSACOffPolicyRunner"

    class policy:
        # ---- actor (dual-history tanh-Gaussian, residual-block head) ----
        init_noise_std = 0.5
        actor_hidden_dims = [256]
        actor_num_blocks = 2
        state_estimator_hidden_dims = [128, 64]
        # long-history 1D CNN
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64
        in_channels = X1FlashSACCfg.env.frame_stack
        # ---- critic (distributional double-Q, residual-block head) ----
        critic_hidden_dims = [256]
        critic_num_blocks = 2

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # ---- core SAC ----
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
        # ---- distributional critic ----
        num_bins = 201
        num_qs = 2
        normalized_G_max = 20.0
        normalize_reward = True
        # ---- zeta noise repetition (FlashSAC's time-correlated exploration) ----
        zeta_mu = 2.0
        zeta_max_n = 16
        # ---- replay buffer (aligned with FlashSAC IsaacLab config) ----
        buffer_max_length = 500_000
        buffer_min_length = 100_000
        # ---- update cadence ----
        batch_size = 2048
        updates_per_interaction_step = 2
        init_random_steps = 0
        bootstrap_on_timeout = False

    class runner:
        policy_class_name = "DualHistoryTanhGaussianActor"
        algorithm_class_name = "FlashSAC"
        num_steps_per_env = 24          # env steps per logged iteration
        max_iterations = 20000          # number of logged iterations

        # logging
        save_interval = 100
        experiment_name = "x1_flashsac"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
