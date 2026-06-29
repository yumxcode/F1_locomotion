# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Off-policy uniform replay buffer with n-step returns, vectorized over the
# IsaacGym parallel environments (one batched add per env step, uniform random
# sampling for gradient updates).

from collections import deque

import torch


class ReplayBuffer:
    """Uniform circular replay buffer with n-step return computation.

    Layout: each batched ``add`` pushes ``num_envs`` transitions (one per env).
    An internal n-step deque accumulates ``n_step`` consecutive batched
    transitions before flushing a single n-step transition per env into the
    circular storage. The circular buffer is a flat ``[capacity]`` tensor per
    field; ``sample`` draws ``batch_size`` random rows uniformly.
    """

    def __init__(self, num_envs, obs_dim, critic_obs_dim, action_dim,
                 max_length=1_000_000, n_step=3, gamma=0.99,
                 min_length=10_000, device="cpu"):
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.action_dim = int(action_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.min_length = int(min_length)

        # capacity is rounded up to a multiple of num_envs so every batched
        # write always fits contiguously (simplifies wrap-around indexing).
        rem = self.num_envs - (int(max_length) % self.num_envs)
        self.capacity = int(max_length) + (rem % self.num_envs)
        self.device = device

        self.actor_obs = torch.zeros(self.capacity, self.obs_dim, device=device)
        self.critic_obs = torch.zeros(self.capacity, self.critic_obs_dim, device=device)
        self.actions = torch.zeros(self.capacity, self.action_dim, device=device)
        self.rewards = torch.zeros(self.capacity, 1, device=device)
        self.next_actor_obs = torch.zeros(self.capacity, self.obs_dim, device=device)
        self.next_critic_obs = torch.zeros(self.capacity, self.critic_obs_dim, device=device)
        self.dones = torch.zeros(self.capacity, 1, device=device)

        self.ptr = 0
        self.size = 0
        self._initialized = False
        # n-step deque (vectorized over envs)
        self._deque = deque(maxlen=self.n_step)

    # ------------------------------------------------------------------ #
    def _init_dims_from(self, obs, cobs, action):
        if not self._initialized:
            # tolerate obs dims slightly differing from constructor hint
            self._initialized = True

    def add(self, obs, cobs, action, reward, next_obs, next_cobs, done):
        """Add a batched single-step transition (all shaped [num_envs, *]).

        After ``n_step`` such calls the first batched n-step transition is
        flushed into storage.
        """
        self._init_dims_from(obs, cobs, action)
        # store tensors on the buffer device
        self._deque.append((
            obs.to(self.device), cobs.to(self.device), action.to(self.device),
            reward.to(self.device).reshape(self.num_envs, 1),
            done.to(self.device).reshape(self.num_envs, 1),
            next_obs.to(self.device), next_cobs.to(self.device),
        ))
        if len(self._deque) < self.n_step:
            return

        obs0, cobs0, act0 = self._deque[0][0], self._deque[0][1], self._deque[0][2]

        # n-step discounted return truncated at first terminal within window
        n_rew = torch.zeros(self.num_envs, 1, device=self.device)
        discount = 1.0
        any_done = torch.zeros(self.num_envs, 1, device=self.device)
        for i in range(self.n_step):
            r = self._deque[i][3]
            d = self._deque[i][4]
            n_rew += discount * r * (1.0 - any_done)
            any_done = torch.max(any_done, d)
            discount *= self.gamma

        next_obs_n = self._deque[-1][5]
        next_cobs_n = self._deque[-1][6]
        done_flag = any_done  # bootstrap is disabled whenever any terminal hit

        self._store(obs0, cobs0, act0, n_rew, next_obs_n, next_cobs_n, done_flag)

    def _store(self, obs0, cobs0, act0, n_rew, next_obs_n, next_cobs_n, done_flag):
        n = self.num_envs
        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.actor_obs[sl] = obs0
            self.critic_obs[sl] = cobs0
            self.actions[sl] = act0
            self.rewards[sl] = n_rew
            self.next_actor_obs[sl] = next_obs_n
            self.next_critic_obs[sl] = next_cobs_n
            self.dones[sl] = done_flag
        else:
            # split wrap-around (only when capacity not a multiple of num_envs)
            first = self.capacity - self.ptr
            self._write_split(self.ptr, first, obs0, cobs0, act0, n_rew,
                              next_obs_n, next_cobs_n, done_flag, offset=0)
            self._write_split(0, n - first, obs0, cobs0, act0, n_rew,
                              next_obs_n, next_cobs_n, done_flag, offset=first)
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def _write_split(self, start, length, obs0, cobs0, act0, n_rew,
                     next_obs_n, next_cobs_n, done_flag, offset):
        sl = slice(start, start + length)
        o = slice(offset, offset + length)
        self.actor_obs[sl] = obs0[o]
        self.critic_obs[sl] = cobs0[o]
        self.actions[sl] = act0[o]
        self.rewards[sl] = n_rew[o]
        self.next_actor_obs[sl] = next_obs_n[o]
        self.next_critic_obs[sl] = next_cobs_n[o]
        self.dones[sl] = done_flag[o]

    # ------------------------------------------------------------------ #
    def can_sample(self, batch_size=None):
        need = self.min_length if batch_size is None else max(batch_size, self.min_length)
        return self.size >= need

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return dict(
            obs=self.actor_obs[idx],
            critic_obs=self.critic_obs[idx],
            action=self.actions[idx],
            reward=self.rewards[idx],
            next_obs=self.next_actor_obs[idx],
            next_critic_obs=self.next_critic_obs[idx],
            done=self.dones[idx],
        )

    def __len__(self):
        return self.size

    # ---- (de)serialization ----
    def save(self, path):
        torch.save(dict(
            actor_obs=self.actor_obs[:self.size].clone(),
            critic_obs=self.critic_obs[:self.size].clone(),
            actions=self.actions[:self.size].clone(),
            rewards=self.rewards[:self.size].clone(),
            next_actor_obs=self.next_actor_obs[:self.size].clone(),
            next_critic_obs=self.next_critic_obs[:self.size].clone(),
            dones=self.dones[:self.size].clone(),
            size=self.size, capacity=self.capacity, ptr=self.ptr,
            num_envs=self.num_envs, n_step=self.n_step, gamma=self.gamma,
            min_length=self.min_length,
        ), path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        size = sd["size"]
        self.capacity = max(self.capacity, size)
        self.actor_obs[:size] = sd["actor_obs"]
        self.critic_obs[:size] = sd["critic_obs"]
        self.actions[:size] = sd["actions"]
        self.rewards[:size] = sd["rewards"]
        self.next_actor_obs[:size] = sd["next_actor_obs"]
        self.next_critic_obs[:size] = sd["next_critic_obs"]
        self.dones[:size] = sd["dones"]
        self.size = size
        self.ptr = sd.get("ptr", size % self.capacity)
