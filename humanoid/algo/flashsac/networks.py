# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# Based on:
#   - legged_gym / rsl_rl (BSD-3-Clause, NVIDIA, ETH Zurich)
#   - AgiBot F1_locomotion (AgiBot Inc.)
#   - FlashSAC algorithm design (Kim et al., RSS'26, arXiv:2604.04539,
#     Holiday Robotics) — re-implemented here to fit the legacy IsaacGym /
#     rsl_rl codebase of F1_locomotion.
# SPDX-License-Identifier: BSD-3-Clause
#
# Neural-network modules for the FlashSAC port.
#   * RunningRewardNormalizer  : running mean/var reward normalization
#   * DualHistoryTanhGaussianActor : tanh-squashed Gaussian policy that reuses
#                                    the X1 "dual-history" feature extractor
#                                    (state-estimator MLP + long-history 1D CNN)
#   * DistributionalDoubleQCritic  : FlashSAC's hallmark categorical double-Q
#                                    critic (N bins) with EMA target support

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reward normalization (global running mean / variance, Welford batch update)  #
# --------------------------------------------------------------------------- #
class RunningRewardNormalizer:
    """Maintain running mean/var of rewards and rescale them.

    Mirrors FlashSAC's reward normalizer: rewards are divided by the running
    standard deviation so that the categorical value support
    ``[-normalized_G_max, +normalized_G_max]`` stays well scaled.
    """

    def __init__(self, normalized_G_max=5.0, eps=1e-6, device="cpu"):
        self.normalized_G_max = float(normalized_G_max)
        self.eps = float(eps)
        self.device = device
        self.mean = torch.zeros((), device=device, dtype=torch.float32)
        self.var = torch.ones((), device=device, dtype=torch.float32)
        self.count = torch.tensor(self.eps, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, reward: torch.Tensor):
        """Update running stats from a batch of (raw) rewards."""
        r = reward.detach().to(self.device).reshape(-1)
        if r.numel() == 0:
            return
        batch_mean = r.mean()
        batch_var = r.var(unbiased=False)
        batch_count = float(r.numel())

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + (delta ** 2) * self.count * batch_count / tot) / tot
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot)

    @torch.no_grad()
    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var.clamp(min=self.eps))
        return reward / (std + self.eps)

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, count=self.count,
                    normalized_G_max=self.normalized_G_max)

    def load_state_dict(self, sd):
        self.mean.copy_(sd["mean"])
        self.var.copy_(sd["var"])
        self.count.copy_(sd["count"])
        self.normalized_G_max = float(sd.get("normalized_G_max", self.normalized_G_max))


# --------------------------------------------------------------------------- #
# Tanh-squashed Gaussian actor with the X1 "dual-history" feature extractor     #
# --------------------------------------------------------------------------- #
class DualHistoryTanhGaussianActor(nn.Module):
    """SAC actor for the X1 biped.

    Preserves the observation pipeline of the original ``ActorCriticDH``:
      actor_obs  ->  [ short_history | state_estimator(short_history) |
                       long_history_CNN(actor_obs) ]  -> mean / log_std heads
    but emits a *tanh-squashed Gaussian* distribution (the SAC requirement)
    instead of the unbounded Gaussian used by PPO.

    Observation layout for X1 (see X1DHStandCfg):
      * num_obs         = frame_stack * num_single_obs            (full actor obs)
      * num_short_obs   = short_frame_stack * num_single_obs      (tail slice)
      * num_proprio_obs = num_single_obs                          (CNN time dim)
      * in_channels     = frame_stack                             (CNN channels)
    """

    def __init__(self,
                 num_short_obs: int,
                 num_proprio_obs: int,
                 num_actions: int,
                 actor_hidden_dims=(512, 256, 128),
                 state_estimator_hidden_dims=(256, 128, 64),
                 in_channels=66,
                 kernel_size=(6, 4),
                 filter_size=(32, 16),
                 stride_size=(3, 2),
                 lh_output_dim=64,
                 init_noise_std=1.0,
                 log_std_min=-5.0,
                 log_std_max=2.0,
                 activation=None,
                 **kwargs):
        super().__init__()
        if kwargs:
            print("DualHistoryTanhGaussianActor ignoring unexpected kwargs: "
                  + str(list(kwargs.keys())))

        if activation is None:
            activation = nn.ELU()

        self.num_short_obs = int(num_short_obs)
        self.num_proprio_obs = int(num_proprio_obs)
        self.in_channels = int(in_channels)
        self.num_actions = int(num_actions)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # ---- state-estimator MLP (mirrors ActorCriticDH) ----
        se_layers = [nn.Linear(self.num_short_obs, state_estimator_hidden_dims[0]), activation]
        for l in range(len(state_estimator_hidden_dims)):
            if l == len(state_estimator_hidden_dims) - 1:
                se_layers.append(nn.Linear(state_estimator_hidden_dims[l], 3))
            else:
                se_layers.append(nn.Linear(state_estimator_hidden_dims[l],
                                           state_estimator_hidden_dims[l + 1]))
                se_layers.append(activation)
        self.state_estimator = nn.Sequential(*se_layers)

        # ---- long-history 1D CNN (mirrors ActorCriticDH) ----
        lh_layers = []
        in_ch = self.in_channels
        cnn_out = self.num_proprio_obs
        out_ch_last = in_ch
        for out_ch, ks, st in zip(filter_size, kernel_size, stride_size):
            lh_layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                       kernel_size=ks, stride=st))
            lh_layers.append(nn.ReLU())
            cnn_out = (cnn_out - ks + st) // st
            in_ch = out_ch
            out_ch_last = out_ch
        cnn_out *= out_ch_last
        lh_layers.append(nn.Flatten())
        lh_layers.append(nn.Linear(cnn_out, 128))
        lh_layers.append(nn.ELU())
        lh_layers.append(nn.Linear(128, lh_output_dim))
        self.long_history = nn.Sequential(*lh_layers)

        # ---- mean head ----
        feat_dim = self.num_short_obs + 3 + lh_output_dim
        mean_layers = [nn.Linear(feat_dim, actor_hidden_dims[0]), activation]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                mean_layers.append(nn.Linear(actor_hidden_dims[l], self.num_actions))
            else:
                mean_layers.append(nn.Linear(actor_hidden_dims[l],
                                             actor_hidden_dims[l + 1]))
                mean_layers.append(activation)
        self.mean_net = nn.Sequential(*mean_layers)

        # learnable per-dimension log-std
        init_log_std = math.log(max(init_noise_std, 1e-3))
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(init_log_std)))

    # ------------------------------------------------------------------ #
    def extract_feature(self, obs: torch.Tensor) -> torch.Tensor:
        short_history = obs[..., -self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed = self.long_history(obs.view(-1, self.in_channels, self.num_proprio_obs))
        return torch.cat((short_history, es_vel, compressed), dim=-1)

    def get_mean_logstd(self, obs: torch.Tensor):
        feat = self.extract_feature(obs)
        mean = self.mean_net(feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mean)
        return mean, log_std

    def forward(self, obs, deterministic=False, with_logprob=True):
        mean, log_std = self.get_mean_logstd(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = mean if deterministic else normal.rsample()  # reparameterized
        action = torch.tanh(x_t)

        log_prob = None
        if with_logprob:
            lp = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
            # tanh correction (numerically stable)
            corr = (2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))).sum(dim=-1, keepdim=True)
            log_prob = lp - corr
        return action, log_prob, mean

    @torch.no_grad()
    def act_inference(self, obs):
        """Deterministic deployment action (tanh(mean)). JIT-export friendly."""
        mean, _ = self.get_mean_logstd(obs)
        return torch.tanh(mean)


# --------------------------------------------------------------------------- #
# Distributional categorical double-Q critic (FlashSAC hallmark)               #
# --------------------------------------------------------------------------- #
class DistributionalDoubleQCritic(nn.Module):
    """Double-Q critic that predicts a categorical distribution over a value
    support ``[v_min, v_max]`` with ``num_bins`` atoms.

    Inputs are the *privileged* critic observation and the action. Two Q-heads
    (clipped double-Q) are computed in parallel. The expected value is the
    mean of the categorical distribution.
    """

    def __init__(self,
                 num_critic_obs: int,
                 num_actions: int,
                 critic_hidden_dims=(768, 256, 128),
                 num_bins=101,
                 v_min=-5.0,
                 v_max=5.0,
                 num_qs=2,
                 activation=None,
                 **kwargs):
        super().__init__()
        if kwargs:
            print("DistributionalDoubleQCritic ignoring unexpected kwargs: "
                  + str(list(kwargs.keys())))
        if activation is None:
            activation = nn.ELU()

        self.num_bins = int(num_bins)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_qs = int(num_qs)
        self.delta = (self.v_max - self.v_min) / (self.num_bins - 1)
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.num_bins))

        self.qs = nn.ModuleList([
            self._build(num_critic_obs, num_actions, critic_hidden_dims, activation)
            for _ in range(self.num_qs)
        ])

    def _build(self, dc, da, dims, act):
        layers = [nn.Linear(dc + da, dims[0]), act]
        for l in range(len(dims)):
            if l == len(dims) - 1:
                layers.append(nn.Linear(dims[l], self.num_bins))
            else:
                layers.append(nn.Linear(dims[l], dims[l + 1]))
                layers.append(act)
        return nn.Sequential(*layers)

    def logits(self, critic_obs, action):
        """Return a list of [B, num_bins] logits, one per Q-head."""
        x = torch.cat([critic_obs, action], dim=-1)
        return [q(x) for q in self.qs]

    def expected_values(self, critic_obs, action):
        """Return a list of [B, 1] expected values (mean of each distribution)."""
        out = []
        for lg in self.logits(critic_obs, action):
            probs = F.softmax(lg, dim=-1)
            out.append((probs * self.support.unsqueeze(0)).sum(dim=-1, keepdim=True))
        return out

    def min_expected_value(self, critic_obs, action):
        vals = self.expected_values(critic_obs, action)
        return torch.min(torch.cat(vals, dim=-1), dim=-1, keepdim=True)[0]


@torch.no_grad()
def categorical_td_target(reward, done, next_value, support, v_min, v_max,
                          num_bins, gamma_n, device):
    """Project scalar bootstrap targets onto the categorical value support.

    target atom value:  Tz = reward + gamma_n * (1 - done) * next_value
    Each sample carries unit mass placed at its (clamped) Tz and distributed
    between the two neighbouring bins (C51 projection).

    Args:
        reward:     [B, 1]
        done:       [B, 1]  (1 => true terminal, no bootstrap)
        next_value: [B, 1]  (target-critic value minus alpha*entropy)
    Returns:
        [B, num_bins] target probability distribution.
    """
    reward = reward.reshape(-1)
    done = done.reshape(-1)
    next_value = next_value.reshape(-1)

    delta = (v_max - v_min) / (num_bins - 1)
    Tz = reward + gamma_n * (1.0 - done) * next_value
    Tz = Tz.clamp(v_min, v_max)
    b = (Tz - v_min) / delta                       # [B] fractional bin index
    l = b.floor().long().clamp(0, num_bins - 1)
    u = b.ceil().long().clamp(0, num_bins - 1)

    B = reward.shape[0]
    proj = torch.zeros(B, num_bins, device=device, dtype=reward.dtype)
    idx = torch.arange(B, device=device)

    lower_w = (u.float() - b)
    upper_w = (b - l.float())
    proj[idx, l] += lower_w
    proj[idx, u] += upper_w

    # integer-bin edge case: place full mass on the single bin
    both = (l == u)
    if both.any():
        proj[idx[both], l[both]] = 1.0
    return proj
