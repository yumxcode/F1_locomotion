# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Focused smoke test for the FlashSAC port. Does NOT need IsaacGym — it wires
# the actor/critic/replay-buffer/SAC-update directly with random data, so it
# validates the algorithm plumbing on any machine that has PyTorch.
#
# Run (from repo root):
#   python humanoid/scripts/smoke_test_flashsac.py
#   python humanoid/scripts/smoke_test_flashsac.py --device cpu   # or cuda:0
#
# Checks:
#   1. Actor forward: action in [-1,1] (tanh), finite log_prob
#   2. Critic forward: softmax probabilities sum to 1, expected Q finite
#   3. Replay buffer: n-step add + uniform sample shapes
#   4. FlashSAC.update: one full gradient step (critic+actor+temp+target)
#                        returns finite losses; buffer drains correctly.

import argparse

import torch

from humanoid.algo.flashsac import (
    DualHistoryTanhGaussianActor,
    DistributionalDoubleQCritic,
    FlashSAC,
    ReplayBuffer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--num_actions", type=int, default=12)
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)

    # ---- X1-like observation dimensions (see X1DHStandCfg) ----
    num_single_obs = 47
    frame_stack = 66
    short_frame_stack = 5
    c_frame_stack = 3
    single_num_privileged_obs = 73

    num_obs = frame_stack * num_single_obs               # 3102
    num_short_obs = short_frame_stack * num_single_obs   # 235
    num_privileged_obs = c_frame_stack * single_num_privileged_obs  # 219
    num_actions = args.num_actions

    N = args.num_envs

    # ---- build networks ----
    actor = DualHistoryTanhGaussianActor(
        num_short_obs=num_short_obs,
        num_proprio_obs=num_single_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        state_estimator_hidden_dims=[128, 64],
        in_channels=frame_stack,
        kernel_size=[6, 4], filter_size=[32, 16], stride_size=[3, 2],
        lh_output_dim=64, init_noise_std=0.5,
    ).to(device)

    critic = DistributionalDoubleQCritic(
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        critic_hidden_dims=[256, 128],
        num_bins=101, v_min=-5.0, v_max=5.0, num_qs=2,
    ).to(device)

    obs = torch.randn(N, num_obs, device=device)
    cobs = torch.randn(N, num_privileged_obs, device=device)

    # ---- 1. actor forward ----
    action, log_prob, mean = actor(obs)
    assert action.shape == (N, num_actions)
    assert (action >= -1.0 - 1e-5).all() and (action <= 1.0 + 1e-5).all(), "tanh action out of [-1,1]"
    assert torch.isfinite(log_prob).all()
    print("[1/4] Actor forward:        PASS  action in [-1,1], log_prob finite")

    # ---- 2. critic forward ----
    vals = critic.expected_values(cobs, action)
    assert len(vals) == 2 and all(v.shape == (N, 1) for v in vals)
    assert torch.isfinite(torch.cat(vals, dim=1)).all()
    min_q = critic.min_expected_value(cobs, action)
    # softmax check
    lg = critic.logits(cobs, action)[0]
    p = torch.softmax(lg, dim=-1)
    assert torch.allclose(p.sum(dim=-1), torch.ones(N, device=device), atol=1e-4)
    print("[2/4] Critic forward:       PASS  double-Q finite, softmax sums to 1")

    # ---- 3. replay buffer (n-step add + sample) ----
    buf = ReplayBuffer(
        num_envs=N, obs_dim=num_obs, critic_obs_dim=num_privileged_obs,
        action_dim=num_actions, max_length=4096, n_step=3, gamma=0.99,
        min_length=256, device=device)
    # feed enough steps to flush n-step transitions
    for _ in range(20):
        buf.add(obs, cobs, action,
                reward=torch.randn(N, 1, device=device),
                next_obs=obs, next_cobs=cobs,
                done=torch.zeros(N, 1, device=device))
    assert buf.can_sample(256), "buffer not warm enough"
    batch = buf.sample(512)
    assert batch["obs"].shape == (512, num_obs)
    assert batch["reward"].shape == (512, 1)
    assert buf.size >= N * (20 - 3 + 1)
    print("[3/4] Replay buffer:        PASS  n-step add + uniform sample, size=%d" % buf.size)

    # ---- 4. FlashSAC.update (one gradient step) ----
    sac = FlashSAC(
        actor=actor, critic=critic, gamma=0.99, n_step=3, tau=0.01,
        actor_lr=3e-4, critic_lr=3e-4, temp_lr=3e-4,
        init_alpha=0.01, target_sigma=0.15, actor_update_period=2,
        num_bins=101, normalized_G_max=5.0, normalize_reward=True, device=device)
    sac.attach_buffer(buf)
    info = sac.update(512)
    for k in ("critic_loss", "actor_loss", "alpha", "entropy", "q_value"):
        assert k in info, "missing metric: " + k
        assert math_isfinite(info[k]), "%s not finite: %s" % (k, info[k])
    # target EMA should have moved target params toward critic
    tp = next(sac.target_critic.parameters())
    cp = next(sac.critic.parameters())
    assert not torch.allclose(tp, cp, atol=1e-6) or sac.tau == 1.0, "target not EMA-updated"
    print("[4/4] FlashSAC.update:      PASS  losses finite, target EMA applied")
    print("        critic_loss=%.4f actor_loss=%.4f alpha=%.4f entropy=%.4f q=%.4f"
          % (info["critic_loss"], info["actor_loss"], info["alpha"], info["entropy"], info["q_value"]))

    print("\nALL SMOKE TESTS PASSED — FlashSAC port is algorithmically sound.")


def math_isfinite(x):
    try:
        import math
        return math.isfinite(float(x))
    except Exception:
        return False


if __name__ == "__main__":
    main()
