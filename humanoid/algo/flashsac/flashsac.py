# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Core FlashSAC algorithm: off-policy Soft Actor-Critic with
#   * distributional categorical double-Q critic (C51-style TD projection)
#   * auto-tuned entropy temperature (target entropy 0.5*A*ln(2*pi*e*sigma^2))
#   * running reward normalization (value support = +/- normalized_G_max)
#   * EMA target critic (soft update, tau)
#   * delayed actor updates (actor_update_period)
# This class drives gradient updates from the replay buffer. The training loop
# (collect -> add -> update) lives in ``FlashSACOffPolicyRunner``.

import copy
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import (DualHistoryTanhGaussianActor, DistributionalDoubleQCritic,
                       RunningRewardNormalizer, categorical_td_target)


class FlashSAC:
    """FlashSAC algorithm.

    The replay buffer is owned by the runner (it is created from env shapes);
    it is injected here via ``attach_buffer`` so that ``update`` can sample it.
    """

    def __init__(self,
                 actor: DualHistoryTanhGaussianActor,
                 critic: DistributionalDoubleQCritic,
                 gamma=0.99,
                 n_step=3,
                 tau=0.01,                       # EMA target-critic coefficient
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 temp_lr=3e-4,
                 init_alpha=0.01,                # initial entropy temperature
                 target_sigma=0.15,             # target-entropy sigma
                 actor_update_period=2,          # actor updated every N critic updates
                 max_grad_norm=1.0,
                 num_bins=101,
                 normalized_G_max=5.0,           # +/- value support bound
                 normalize_reward=True,
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

        # ---- networks ----
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # ---- optimizers ----
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---- entropy temperature (auto-tuned) ----
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(max(init_alpha, 1e-6)), device=self.device,
                         dtype=torch.float32))
        self.temp_opt = optim.Adam([self.log_alpha], lr=temp_lr)
        # FlashSAC target entropy: 0.5 * A * ln(2*pi*e*sigma^2)
        action_dim = self._infer_action_dim()
        self.target_entropy = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * (target_sigma ** 2))

        self.reward_normalizer = RunningRewardNormalizer(
            normalized_G_max=self.normalized_G_max, device=self.device)

        self.buffer = None
        self.update_count = 0

        # cache value-support bounds for the TD projection
        self.v_min = critic.v_min
        self.v_max = critic.v_max

    # ------------------------------------------------------------------ #
    def _infer_action_dim(self) -> int:
        return int(self.actor.num_actions)

    def attach_buffer(self, buffer):
        self.buffer = buffer

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ #
    def act(self, obs, deterministic=False):
        """Sample an action for the given (batched) actor observation."""
        with torch.no_grad():
            action, _, _ = self.actor(obs.to(self.device), deterministic=deterministic)
        return action

    # ------------------------------------------------------------------ #
    def update(self, batch_size):
        """One gradient step (critic + delayed actor + temperature + target EMA)."""
        assert self.buffer is not None, "Replay buffer not attached"
        batch = self.buffer.sample(batch_size)
        obs = batch["obs"]
        cobs = batch["critic_obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        next_cobs = batch["next_critic_obs"]
        done = batch["done"]

        # reward normalization (update running stats with the raw reward first)
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
        info["critic_loss"] = float(critic_loss.item())

        # q value for diagnostics + actor
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
            info["actor_loss"] = float(actor_loss.item())

        # --------------------------- temperature -------------------------- #
        # need a log_prob for the temperature update; reuse a fresh sample if
        # the actor wasn't updated this step.
        if actor_log_prob is None:
            with torch.no_grad():
                _, actor_log_prob, _ = self.actor(obs)
        alpha_loss = -(self.log_alpha * (actor_log_prob.detach() + self.target_entropy)).mean()
        self.temp_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.temp_opt.step()
        info["alpha_loss"] = float(alpha_loss.item())
        info["alpha"] = float(self.alpha.item())
        info["entropy"] = float((-actor_log_prob).mean().item())

        # --------------------------- target EMA --------------------------- #
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

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
