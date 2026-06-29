# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# Export a trained FlashSAC actor to a deployment-friendly JIT module.
#
# The deployed policy takes the raw X1 actor observation and returns the
# deterministic SAC action tanh(mean), reusing the dual-history feature
# extractor (state-estimator MLP + long-history 1D CNN).
#
# Usage:
#   python scripts/export_policy_flashsac.py --task=x1_flashsac --load_run=-1

import os
import copy
from datetime import datetime

import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import get_load_path, class_to_dict
from humanoid.algo.flashsac import DualHistoryTanhGaussianActor


class ExportedFlashSAC(torch.nn.Module):
    """Deployment module: obs -> tanh(mean) via the dual-history extractor.

    This mirrors the original ``ExportedDH`` wrapper but applies the tanh
    squashing of the SAC policy and is fully TorchScript-compatible.
    """

    def __init__(self, mean_net, long_history, state_estimator,
                 num_short_obs, in_channels, num_proprio_obs):
        super().__init__()
        self.mean_net = copy.deepcopy(mean_net).cpu()
        self.long_history = copy.deepcopy(long_history).cpu()
        self.state_estimator = copy.deepcopy(state_estimator).cpu()
        self.num_short_obs = int(num_short_obs)
        self.in_channels = int(in_channels)
        self.num_proprio_obs = int(num_proprio_obs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        short_history = observations[..., -self.num_short_obs:]
        es_vel = self.state_estimator(short_history)
        compressed = self.long_history(
            observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed), dim=-1)
        mean = self.mean_net(actor_obs)
        return torch.tanh(mean)

    def export(self, path):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def export_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]

    num_critic_obs = env_cfg.env.num_privileged_obs
    if env_cfg.terrain.measure_heights:
        num_critic_obs = (env_cfg.env.c_frame_stack *
                          (env_cfg.env.single_num_privileged_obs + env_cfg.terrain.num_height))
    num_short_obs = env_cfg.env.short_frame_stack * env_cfg.env.num_single_obs

    actor = DualHistoryTanhGaussianActor(
        num_short_obs=num_short_obs,
        num_proprio_obs=env_cfg.env.num_single_obs,
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
        state_estimator_hidden_dims=policy_cfg.get("state_estimator_hidden_dims", [256, 128, 64]),
        in_channels=policy_cfg.get("in_channels", env_cfg.env.frame_stack),
        kernel_size=policy_cfg.get("kernel_size", [6, 4]),
        filter_size=policy_cfg.get("filter_size", [32, 16]),
        stride_size=policy_cfg.get("stride_size", [3, 2]),
        lh_output_dim=policy_cfg.get("lh_output_dim", 64),
        init_noise_std=policy_cfg.get("init_noise_std", 0.5),
    )

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported_data')
    model_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    print("Load model from:", model_path)

    loaded_dict = torch.load(model_path, map_location="cpu")
    actor.load_state_dict(loaded_dict["actor"])

    exported = ExportedFlashSAC(
        mean_net=actor.mean_net,
        long_history=actor.long_history,
        state_estimator=actor.state_estimator,
        num_short_obs=num_short_obs,
        in_channels=policy_cfg["in_channels"],
        num_proprio_obs=env_cfg.env.num_single_obs,
    )

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                             train_cfg.runner.experiment_name, 'exported_policies', now)
    os.makedirs(root_path, exist_ok=True)
    path = os.path.join(root_path, "policy_flashsac.jit")
    exported.export(path)
    print("Export policy to:", path)


if __name__ == '__main__':
    args = get_args()
    if args.load_run is None:
        args.load_run = -1
    if args.checkpoint is None:
        args.checkpoint = -1
    export_policy(args)
