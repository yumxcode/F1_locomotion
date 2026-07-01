# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Core FlashSAC algorithm (aligned with original paper/repo):
#   * Zeta-distribution action-noise repetition (time-correlated exploration)
#   * Weight normalization (normalize_parameters after every optimizer step)
#   * Warmup-cosine-decay LR schedule
#   * Distributional categorical double-Q critic
#   * Auto-tuned entropy temperature
#   * Running reward normalization + n-step returns

import copy
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import (DualHistoryTanhGaussianActor, DistributionalDoubleQCritic,
                       RunningRewardNormalizer, categorical_td_target,
                       normalize_parameters, build_zeta_cdf, sample_zeta_n)


class FlashSAC:
    """FlashSAC algorithm (paper-aligned)."""

    def __init__(self,
                 actor: DualHistoryTanhGaussianActor,
                 critic: DistributionalDoubleQCritic,
                 num_envs: int,
                 gamma=0.99,
                 n_step=3,
                 tau=0.01,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 temp_lr=3e-4,
                 lr_warmup_steps=1000,
                 lr_total_steps=50000,
                 init_alpha=0.1,
                 auto_alpha=False,
                 target_sigma=0.3,
                 actor_update_period=2,
                 max_grad_norm=1.0,
                 num_bins=201,
                 normalized_G_max=20.0,
                 normalize_reward=True,
                 actor_num_blocks=2,
                 critic_num_blocks=2,
                 zeta_mu=2.0,
                 zeta_max_n=16,
                 device="cpu"):
        self.device = device
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.tau = float(tau)
        self.actor_update_period = int(actor_update_period)
        self.max_grad_norm = float(max_grad_norm)
        self.num_bins = int(num_bins)
        self.normalized_G_max = float(normalized_G_max)
        self.normalize_reward = bool(normalize_reward)
        self.gamma_n = self.gamma ** self.n_step
        self.num_envs = int(num_envs)

        # ---- networks ----
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # ---- optimizers + warmup-cosine-decay LR scheduler ----
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_scheduler = _WarmupCosineDecay(actor_lr, actor_lr * 0.5,
                                                   lr_warmup_steps, lr_total_steps)
        self.critic_scheduler = _WarmupCosineDecay(critic_lr, critic_lr * 0.5,
                                                    lr_warmup_steps, lr_total_steps)

        # ---- entropy temperature ----
        # Alpha auto-tuning consistently collapses on 12-DoF tanh-Gaussian:
        # any negative target_entropy drives alpha → 0 in ~1000 steps.
        # Fix: use a FIXED alpha (no gradient on log_alpha) when auto_alpha=False.
        self.auto_alpha = bool(auto_alpha)
        action_dim = self._infer_action_dim()
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(max(init_alpha, 1e-6)), device=self.device,
                         dtype=torch.float32))
        if self.auto_alpha:
            self.temp_opt = optim.Adam([self.log_alpha], lr=temp_lr)
            self.target_entropy = -0.5 * action_dim
        else:
            self.temp_opt = None
            self.log_alpha.requires_grad_(False)

        self.reward_normalizer = RunningRewardNormalizer(
            normalized_G_max=self.normalized_G_max, device=self.device)

        self.buffer = None
        self.update_count = 0

        self.v_min = critic.v_min
        self.v_max = critic.v_max

        # ---- Zeta noise state (time-correlated exploration) ----
        self._zeta_cdf = build_zeta_cdf(mu=zeta_mu, max_n=zeta_max_n).to(self.device)
        self._zeta_noise = torch.zeros(num_envs, action_dim, device=self.device)
        self._zeta_n = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self._zeta_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # ---- weight normalization on init ----
        normalize_parameters(self.actor)
        normalize_parameters(self.critic)
        normalize_parameters(self.target_critic)

        self._last_actor_loss = 0.0

    # ------------------------------------------------------------------ #
    def _infer_action_dim(self) -> int:
        return int(self.actor.num_actions)

    def attach_buffer(self, buffer):
        self.buffer = buffer

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ #
    # Action sampling with zeta-distribution noise repetition              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Sample action with time-correlated zeta noise (FlashSAC's core)."""
        obs = obs.to(self.device)
        mean, log_std = self.actor.get_mean_logstd(obs)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean)

        # Zeta noise repetition: keep same noise vector for ~n steps
        reinit = (self._zeta_count == 0) | (self._zeta_count >= self._zeta_n)
        if reinit.any():
            new_noise = torch.randn_like(mean)
            new_n = sample_zeta_n(self._zeta_cdf, self.num_envs, self.device)
            self._zeta_noise = torch.where(reinit.unsqueeze(-1), new_noise, self._zeta_noise)
            self._zeta_n = torch.where(reinit, new_n, self._zeta_n)
            self._zeta_count = torch.where(reinit, torch.zeros_like(self._zeta_count), self._zeta_count)

        # temperature = 1.0 during stochastic exploration
        action = torch.tanh(mean + std * self._zeta_noise)
        self._zeta_count += 1
        return action

    # ------------------------------------------------------------------ #
    def update(self, batch_size):
        """One gradient step (critic + delayed actor + temperature + target EMA
        + weight normalization)."""
        assert self.buffer is not None, "Replay buffer not attached"
        batch = self.buffer.sample(batch_size)
        obs = batch["obs"]
        cobs = batch["critic_obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        next_cobs = batch["next_critic_obs"]
        done = batch["done"]

        if self.normalize_reward:
            self.reward_normalizer.update(reward)
            reward = self.reward_normalizer.normalize(reward)

        info = {}

        # ----------------------------- critic ----------------------------- #
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_obs)
            next_vals = self.target_critic.expected_values(next_cobs, next_action)
            next_v = torch.min(torch.cat(next_vals, dim=-1), dim=-1, keepdim=True)[0]
            next_v = next_v - self.alpha.detach() * next_log_prob
            target_dist = categorical_td_target(
                reward, done, next_v, self.critic.support,
                self.v_min, self.v_max, self.num_bins, self.gamma_n, self.device)

        logits_list = self.critic.logits(cobs, action)
        critic_loss = 0.0
        for lg in logits_list:
            log_probs = F.log_softmax(lg, dim=-1)
            critic_loss = critic_loss + -(target_dist.detach() * log_probs).sum(dim=-1).mean()
        critic_loss = critic_loss / len(logits_list)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()
        normalize_parameters(self.critic)          # ← weight normalization
        info["critic_loss"] = float(critic_loss.item())

        with torch.no_grad():
            q_vals = self.critic.expected_values(cobs, action)
            q_diag = torch.min(torch.cat(q_vals, dim=-1), dim=-1, keepdim=True)[0]
            info["q_value"] = float(q_diag.mean().item())

        # ----------------------------- actor ------------------------------ #
        actor_log_prob = None
        if self.update_count % self.actor_update_period == 0:
            new_action, actor_log_prob, _ = self.actor(obs)
            q_vals = self.critic.expected_values(cobs, new_action)
            q = torch.min(torch.cat(q_vals, dim=-1), dim=-1, keepdim=True)[0]
            actor_loss = (self.alpha.detach() * actor_log_prob - q).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()
            normalize_parameters(self.actor)        # ← weight normalization
            info["actor_loss"] = float(actor_loss.item())
            self._last_actor_loss = info["actor_loss"]
        else:
            info["actor_loss"] = self._last_actor_loss

        # --------------------------- temperature -------------------------- #
        if self.auto_alpha:
            if actor_log_prob is None:
                with torch.no_grad():
                    _, actor_log_prob, _ = self.actor(obs)
            alpha_loss = -(self.log_alpha * (actor_log_prob.detach() + self.target_entropy)).mean()
            self.temp_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.temp_opt.step()
            info["alpha_loss"] = float(alpha_loss.item())
        else:
            info["alpha_loss"] = 0.0
        info["alpha"] = float(self.alpha.item())
        if actor_log_prob is not None:
            info["entropy"] = float((-actor_log_prob).mean().item())
        else:
            info["entropy"] = 0.0

        # --------------------------- target EMA --------------------------- #
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        # --------------------------- LR schedule -------------------------- #
        self.actor_scheduler.step(self.actor_opt)
        self.critic_scheduler.step(self.critic_opt)

        self.update_count += 1
        return info

    # ------------------------------------------------------------------ #
    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
            log_alpha=self.log_alpha.detach().cpu(),
            actor_opt=self.actor_opt.state_dict(),
            critic_opt=self.critic_opt.state_dict(),
            temp_opt=self.temp_opt.state_dict(),
            reward_normalizer=self.reward_normalizer.state_dict(),
            update_count=self.update_count,
        )

    def load_state_dict(self, sd, load_optimizer=True):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.target_critic.load_state_dict(sd["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        if load_optimizer:
            self.actor_opt.load_state_dict(sd["actor_opt"])
            self.critic_opt.load_state_dict(sd["critic_opt"])
            self.temp_opt.load_state_dict(sd["temp_opt"])
        self.reward_normalizer.load_state_dict(sd["reward_normalizer"])
        self.update_count = int(sd.get("update_count", 0))


# --------------------------------------------------------------------------- #
# Warmup-cosine-decay learning-rate scheduler                                  #
# --------------------------------------------------------------------------- #
class _WarmupCosineDecay:
    """Linear warmup then cosine decay to end_value."""

    def __init__(self, peak_lr, end_lr, warmup_steps, total_steps):
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.step_count = 0

    def get_lr(self):
        s = self.step_count
        if s < self.warmup_steps:
            return self.peak_lr * (s + 1) / self.warmup_steps
        progress = min(1.0, (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.end_lr + (self.peak_lr - self.end_lr) * cosine

    def step(self, optimizer):
        self.step_count += 1
        lr = self.get_lr()
        for pg in optimizer.param_groups:
            pg["lr"] = lr
