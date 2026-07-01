# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Off-policy training runner that plugs FlashSAC into the project's
# ``task_registry`` (same constructor / learn / save / load contract as
# ``DHOnPolicyRunner``). It owns the replay buffer and drives the
# collect -> add -> update loop per env step.

import os
import time
import copy
import statistics
from collections import deque
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .flashsac import FlashSAC
from .replay_buffer import ReplayBuffer
from .networks import DualHistoryTanhGaussianActor, DistributionalDoubleQCritic


class FlashSACOffPolicyRunner:
    """Runner for FlashSAC. One *iteration* == ``num_steps_per_env`` env steps
    (per env). Gradient updates happen every env step once the replay buffer
    is warm (``updates_per_interaction_step`` updates per env step)."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env

        # ---- observation / action dimensions ----
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.env.cfg.terrain.measure_heights:
            num_critic_obs = (self.env.cfg.env.c_frame_stack *
                              (self.env.cfg.env.single_num_privileged_obs + self.env.cfg.terrain.num_height))
        self.num_critic_obs = num_critic_obs

        num_short_obs = self.env.num_short_obs
        num_single_obs = self.env.num_single_obs
        num_actions = self.env.num_actions
        num_obs = self.env.num_obs
        self.num_actions = num_actions

        # ---- build actor & critic from the policy config ----
        actor = DualHistoryTanhGaussianActor(
            num_short_obs=num_short_obs,
            num_proprio_obs=num_single_obs,
            num_actions=num_actions,
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [128]),
            state_estimator_hidden_dims=self.policy_cfg.get("state_estimator_hidden_dims", [128, 64]),
            in_channels=self.policy_cfg.get("in_channels", self.env.cfg.env.frame_stack),
            kernel_size=self.policy_cfg.get("kernel_size", [6, 4]),
            filter_size=self.policy_cfg.get("filter_size", [32, 16]),
            stride_size=self.policy_cfg.get("stride_size", [3, 2]),
            lh_output_dim=self.policy_cfg.get("lh_output_dim", 64),
            init_noise_std=self.policy_cfg.get("init_noise_std", 0.5),
            actor_num_blocks=self.policy_cfg.get("actor_num_blocks", 2),
        ).to(self.device)

        critic = DistributionalDoubleQCritic(
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256]),
            num_bins=self.alg_cfg.get("num_bins", 201),
            v_min=-abs(float(self.alg_cfg.get("normalized_G_max", 20.0))),
            v_max=abs(float(self.alg_cfg.get("normalized_G_max", 20.0))),
            num_qs=self.alg_cfg.get("num_qs", 2),
            critic_num_blocks=self.policy_cfg.get("critic_num_blocks", 2),
        ).to(self.device)

        print(f"FlashSAC Actor: {actor}")
        print(f"FlashSAC Critic: {critic}")

        # ---- compute LR schedule steps ----
        max_iters = int(self.cfg.get("max_iterations", 20000))
        steps_per_iter = int(self.cfg.get("num_steps_per_env", 24))
        updates_per_step = float(self.alg_cfg.get("updates_per_interaction_step", 2))
        lr_total_steps = int(max_iters * steps_per_iter * self.env.num_envs * updates_per_step / self.env.num_envs)
        lr_warmup_steps = max(1000, int(0.05 * lr_total_steps))

        # ---- algorithm ----
        self.alg = FlashSAC(
            actor=actor, critic=critic,
            num_envs=self.env.num_envs,
            gamma=self.alg_cfg.get("gamma", 0.99),
            n_step=self.alg_cfg.get("n_step", 3),
            tau=self.alg_cfg.get("tau", 0.01),
            actor_lr=self.alg_cfg.get("actor_lr", 3e-4),
            critic_lr=self.alg_cfg.get("critic_lr", 3e-4),
            temp_lr=self.alg_cfg.get("temp_lr", 1e-4),
            lr_warmup_steps=lr_warmup_steps,
            lr_total_steps=lr_total_steps,
            init_alpha=self.alg_cfg.get("init_alpha", 0.1),
            auto_alpha=self.alg_cfg.get("auto_alpha", False),
            target_sigma=self.alg_cfg.get("target_sigma", 0.3),
            actor_update_period=self.alg_cfg.get("actor_update_period", 2),
            max_grad_norm=self.alg_cfg.get("max_grad_norm", 1.0),
            num_bins=self.alg_cfg.get("num_bins", 201),
            normalized_G_max=self.alg_cfg.get("normalized_G_max", 20.0),
            normalize_reward=self.alg_cfg.get("normalize_reward", True),
            zeta_mu=self.alg_cfg.get("zeta_mu", 2.0),
            zeta_max_n=self.alg_cfg.get("zeta_max_n", 16),
            device=self.device,
        )

        # ---- replay buffer ----
        self.buffer = ReplayBuffer(
            num_envs=self.env.num_envs,
            obs_dim=num_obs,
            critic_obs_dim=num_critic_obs,
            action_dim=num_actions,
            max_length=int(self.alg_cfg.get("buffer_max_length", 1_000_000)),
            n_step=self.alg_cfg.get("n_step", 3),
            gamma=self.alg_cfg.get("gamma", 0.99),
            min_length=int(self.alg_cfg.get("buffer_min_length", 10_000)),
            device=self.device,
        )
        self.alg.attach_buffer(self.buffer)

        # ---- loop / bookkeeping params ----
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 100))
        self.batch_size = int(self.alg_cfg.get("batch_size", 2048))
        self.updates_per_interaction_step = self.alg_cfg.get("updates_per_interaction_step", 1)
        self.init_random_steps = int(self.alg_cfg.get("init_random_steps", 0))
        self.bootstrap_on_timeout = bool(self.alg_cfg.get("bootstrap_on_timeout", False))
        self.action_clip = float(self.env.cfg.normalization.clip_actions) if hasattr(self.env.cfg, "normalization") else 100.0

        # ---- logging ----
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.it = 0

        _, _ = self.env.reset()

    # ------------------------------------------------------------------ #
    def _bootstrap_dones(self, dones, infos):
        """Return the terminal mask used for bootstrap suppression.

        Time-limit truncations should *bootstrap* (not terminate), so we strip
        them from the done mask unless ``bootstrap_on_timeout`` is True.
        """
        done = dones.to(self.device).float().reshape(-1, 1)
        if not self.bootstrap_on_timeout and "time_outs" in infos:
            to = infos["time_outs"].to(self.device).float().reshape(-1, 1)
            done = done * (1.0 - to)
        return done

    # ------------------------------------------------------------------ #
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = (privileged_obs.to(self.device) if privileged_obs is not None else obs)

        self.alg.actor.train()
        self.alg.critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        last_info = {}

        tot_iter = self.current_learning_iteration + num_learning_iterations
        global_step = self.tot_timesteps

        for it in range(self.current_learning_iteration, tot_iter):
            self.it = it
            start = time.time()

            collect_time = 0.0
            learn_time = 0.0
            for _ in range(self.num_steps_per_env):
                global_step += self.env.num_envs

                # ---- action selection (random warmup vs. policy) ----
                if (self.init_random_steps > 0 and global_step < self.init_random_steps) \
                        or not self.buffer.can_sample(self.batch_size):
                    actions = (torch.rand(self.env.num_envs, self.num_actions,
                                          device=self.device) * 2.0 - 1.0)
                else:
                    actions = self.alg.act(obs, deterministic=False)

                # ---- env step ----
                next_obs, next_privileged, rewards, dones, infos = self.env.step(actions)
                next_obs = next_obs.to(self.device)
                next_critic = (next_privileged.to(self.device)
                               if next_privileged is not None else next_obs)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)

                done_flag = self._bootstrap_dones(dones, infos)

                # ---- store transition ----
                self.buffer.add(
                    obs=obs, cobs=critic_obs, action=actions,
                    reward=rewards.reshape(self.env.num_envs, 1),
                    next_obs=next_obs, next_cobs=next_critic, done=done_flag)

                # bookkeeping
                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

                # ---- gradient updates ----
                u_start = time.time()
                if self.buffer.can_sample(self.batch_size):
                    ups = self.updates_per_interaction_step
                    # support fractional updates_per_interaction_step
                    n_full = int(ups)
                    frac = ups - n_full
                    for _ in range(n_full):
                        last_info = self.alg.update(self.batch_size)
                    if frac > 0 and torch.rand(1).item() < frac:
                        last_info = self.alg.update(self.batch_size)
                learn_time += time.time() - u_start

                obs = next_obs
                critic_obs = next_critic

            collect_time = time.time() - start - learn_time
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            if self.log_dir is not None:
                self.log(locals(), last_info, rewbuffer, lenbuffer)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    # ------------------------------------------------------------------ #
    def log(self, locs, info, rewbuffer, lenbuffer, width=80, pad=35):
        iteration_time = locs["collect_time"] + locs["learn_time"]
        self.tot_time += iteration_time

        fps = int(self.num_steps_per_env * self.env.num_envs / (iteration_time + 1e-9))

        self.writer.add_scalar("Loss/critic", info.get("critic_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/actor", info.get("actor_loss", 0.0), locs["it"])
        self.writer.add_scalar("Loss/temperature", info.get("alpha_loss", 0.0), locs["it"])
        self.writer.add_scalar("Policy/q_value", info.get("q_value", 0.0), locs["it"])
        self.writer.add_scalar("Policy/entropy", info.get("entropy", 0.0), locs["it"])
        self.writer.add_scalar("Policy/alpha", info.get("alpha", 0.0), locs["it"])
        self.writer.add_scalar("Buffer/size", len(self.buffer), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collect_time", locs["collect_time"], locs["it"])
        self.writer.add_scalar("Perf/learn_time", locs["learn_time"], locs["it"])

        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(rewbuffer), self.tot_time)

        logstr = (f" \033[1m FlashSAC iteration {locs['it']}/"
                  f"{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m ")
        line = (
            f"""{'#' * width}\n"""
            f"""{logstr.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s """
            f"""(collect: {locs['collect_time']:.3f}s, learn {locs['learn_time']:.3f}s)\n"""
            f"""{'Critic loss:':>{pad}} {info.get('critic_loss', 0.0):.4f}\n"""
            f"""{'Actor loss:':>{pad}} {info.get('actor_loss', 0.0):.4f}\n"""
            f"""{'Alpha (temperature):':>{pad}} {info.get('alpha', 0.0):.4f}\n"""
            f"""{'Mean Q:':>{pad}} {info.get('q_value', 0.0):.4f}\n"""
            f"""{'Entropy:':>{pad}} {info.get('entropy', 0.0):.4f}\n"""
            f"""{'Replay size:':>{pad}} {len(self.buffer)}\n""")
        if len(rewbuffer) > 0:
            line += (
                f"""{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n""")
        line += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n""")
        print(line)

    # ------------------------------------------------------------------ #
    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            dict(
                actor=self.alg.actor.state_dict(),
                critic=self.alg.critic.state_dict(),
                target_critic=self.alg.target_critic.state_dict(),
                log_alpha=self.alg.log_alpha.detach().cpu(),
                actor_opt=self.alg.actor_opt.state_dict(),
                critic_opt=self.alg.critic_opt.state_dict(),
                temp_opt=self.alg.temp_opt.state_dict(),
                reward_normalizer=self.alg.reward_normalizer.state_dict(),
                update_count=self.alg.update_count,
                iter=self.it,
                infos=infos,
            ),
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.load_state_dict(loaded_dict, load_optimizer=load_optimizer)
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", None)

    # ------------------------------------------------------------------ #
    def get_inference_policy(self, device=None):
        self.alg.actor.eval()
        if device is not None:
            self.alg.actor.to(device)
        return self.alg.actor.act_inference
