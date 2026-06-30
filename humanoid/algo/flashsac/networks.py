# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# Based on:
#   - legged_gym / rsl_rl (BSD-3-Clause, NVIDIA, ETH Zurich)
#   - AgiBot F1_locomotion (AgiBot Inc.)
#   - FlashSAC algorithm design (Kim et al., RSS'26, arXiv:2604.04539,
#     Holiday Robotics) — re-implemented here to fit the legacy IsaacGym /
#     rsl_rl codebase of F1_locomotion.
# SPDX-License-Identifier: BSD-3-Clause
#
# Neural-network modules for the FlashSAC port (aligned with original paper):
#   * UnitRMSNorm            : feature-level RMS normalization
#   * ResidualBlock          : residual MLP block (FlashSAC's architecture)
#   * normalize_parameters   : weight-level RMS normalization after each step
#   * Zeta noise helpers     : time-correlated exploration (FlashSAC's core)
#   * RunningRewardNormalizer: running mean/var reward normalization
#   * DualHistoryTanhGaussianActor : tanh-Gaussian policy with X1 dual-history
#   * DistributionalDoubleQCritic  : categorical double-Q critic

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# UnitRMSNorm — feature normalization (used inside residual blocks)            #
# --------------------------------------------------------------------------- #
class UnitRMSNorm(nn.Module):
    """Normalize features to unit RMS (paper: UnitRMSNorm)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


# --------------------------------------------------------------------------- #
# ResidualBlock — FlashSAC's building block                                    #
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """Pre-activation residual block with ELU activation."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = UnitRMSNorm(dim)

    def forward(self, x):
        h = F.elu(x)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.fc2(h)
        return self.norm(x + h)


# --------------------------------------------------------------------------- #
# Weight normalization — applied after every optimizer step                   #
# --------------------------------------------------------------------------- #
def normalize_parameters(module: nn.Module):
    """Apply unit-RMS weight normalization to all Linear/Conv layers.

    After every optimizer step, each weight tensor w is rescaled so that
    RMS(w) = 1, i.e. w ← w / sqrt(mean(w²) + eps). This is FlashSAC's explicit
    norm bounding that prevents off-policy bootstrapping error accumulation.
    """
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                rms = m.weight.data.pow(2).mean().add(1e-6).rsqrt()
                m.weight.data.mul_(rms)


# --------------------------------------------------------------------------- #
# Zeta-distribution action-noise repetition (time-correlated exploration)      #
# --------------------------------------------------------------------------- #
def build_zeta_cdf(mu: float = 2.0, max_n: int = 16) -> torch.Tensor:
    """Build truncated zeta distribution CDF.

    Samples how many consecutive steps the same noise vector persists.
    mu=2.0 → n=1 most likely (~61%), decreasing for larger n.
    """
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns.pow(-mu)
    pmf = pmf / pmf.sum()
    return torch.cumsum(pmf, dim=0)


def sample_zeta_n(cdf: torch.Tensor, batch_size: int, device) -> torch.Tensor:
    """Sample n (noise persistence length) for each env from the zeta CDF."""
    u = torch.rand(batch_size, device=device)
    idx = (u.unsqueeze(-1) < cdf.unsqueeze(0).to(device)).float().argmax(dim=-1)
    return (idx + 1).long()


# --------------------------------------------------------------------------- #
# Reward normalization (global running mean / variance, Welford batch update)  #
# --------------------------------------------------------------------------- #
class RunningRewardNormalizer:
    """Maintain running mean/var of rewards and rescale them."""

    def __init__(self, normalized_G_max=5.0, eps=1e-6, device="cpu"):
        self.normalized_G_max = float(normalized_G_max)
        self.eps = float(eps)
        self.device = device
        self.mean = torch.zeros((), device=device, dtype=torch.float32)
        self.var = torch.ones((), device=device, dtype=torch.float32)
        self.count = torch.tensor(self.eps, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, reward: torch.Tensor):
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
# Helper: build a residual MLP head                                            #
# --------------------------------------------------------------------------- #
def _build_residual_head(input_dim, hidden_dim, output_dim, num_blocks, activation=None):
    """Linear(input→hidden) → N× ResidualBlock(hidden) → Linear(hidden→output)."""
    if activation is None:
        activation = nn.ELU()
    layers = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Tanh-squashed Gaussian actor with X1 dual-history feature extractor          #
# --------------------------------------------------------------------------- #
class DualHistoryTanhGaussianActor(nn.Module):
    """SAC actor for X1 with residual-block policy head (FlashSAC-aligned).

    Preserves the X1 dual-history pipeline:
      actor_obs → [ short_history | state_estimator(short) | long_history_CNN ] → residual head
    """

    def __init__(self,
                 num_short_obs: int,
                 num_proprio_obs: int,
                 num_actions: int,
                 actor_hidden_dims=(128,),
                 state_estimator_hidden_dims=(128, 64),
                 in_channels=66,
                 kernel_size=(6, 4),
                 filter_size=(32, 16),
                 stride_size=(3, 2),
                 lh_output_dim=64,
                 init_noise_std=1.0,
                 actor_num_blocks=2,
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

        # ---- mean head: residual blocks (FlashSAC-aligned) ----
        feat_dim = self.num_short_obs + 3 + lh_output_dim
        actor_hidden_dim = actor_hidden_dims[0] if isinstance(actor_hidden_dims, (list, tuple)) else actor_hidden_dims
        self.mean_net = _build_residual_head(
            input_dim=feat_dim,
            hidden_dim=actor_hidden_dim,
            output_dim=self.num_actions,
            num_blocks=actor_num_blocks,
            activation=activation,
        )

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

        x_t = mean if deterministic else normal.rsample()
        action = torch.tanh(x_t)

        log_prob = None
        if with_logprob:
            lp = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
            corr = (2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))).sum(dim=-1, keepdim=True)
            log_prob = lp - corr
        return action, log_prob, mean

    @torch.no_grad()
    def act_inference(self, obs):
        """Deterministic deployment action (tanh(mean))."""
        mean, _ = self.get_mean_logstd(obs)
        return torch.tanh(mean)


# --------------------------------------------------------------------------- #
# Distributional categorical double-Q critic (FlashSAC-aligned residual arch)  #
# --------------------------------------------------------------------------- #
class DistributionalDoubleQCritic(nn.Module):
    """Double-Q critic predicting categorical distribution over value support."""

    def __init__(self,
                 num_critic_obs: int,
                 num_actions: int,
                 critic_hidden_dims=(256,),
                 num_bins=101,
                 v_min=-5.0,
                 v_max=5.0,
                 num_qs=2,
                 critic_num_blocks=2,
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

        critic_hidden_dim = critic_hidden_dims[0] if isinstance(critic_hidden_dims, (list, tuple)) else critic_hidden_dims
        self.qs = nn.ModuleList([
            _build_residual_head(
                input_dim=num_critic_obs + num_actions,
                hidden_dim=critic_hidden_dim,
                output_dim=self.num_bins,
                num_blocks=critic_num_blocks,
                activation=activation,
            )
            for _ in range(self.num_qs)
        ])

    def logits(self, critic_obs, action):
        x = torch.cat([critic_obs, action], dim=-1)
        return [q(x) for q in self.qs]

    def expected_values(self, critic_obs, action):
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
    """Project scalar bootstrap targets onto the categorical value support (C51)."""
    reward = reward.reshape(-1)
    done = done.reshape(-1)
    next_value = next_value.reshape(-1)

    delta = (v_max - v_min) / (num_bins - 1)
    Tz = reward + gamma_n * (1.0 - done) * next_value
    Tz = Tz.clamp(v_min, v_max)
    b = (Tz - v_min) / delta
    l = b.floor().long().clamp(0, num_bins - 1)
    u = b.ceil().long().clamp(0, num_bins - 1)

    B = reward.shape[0]
    proj = torch.zeros(B, num_bins, device=device, dtype=reward.dtype)
    idx = torch.arange(B, device=device)

    lower_w = (u.float() - b)
    upper_w = (b - l.float())
    proj[idx, l] += lower_w
    proj[idx, u] += upper_w

    both = (l == u)
    if both.any():
        proj[idx[both], l[both]] = 1.0
    return proj
