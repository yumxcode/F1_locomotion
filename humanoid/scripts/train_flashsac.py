# SPDX-FileCopyrightText: Copyright (c) 2026 (FlashSAC port for F1_locomotion).
# SPDX-License-Identifier: BSD-3-Clause
#
# FlashSAC training entry point for the X1 biped.
#
# Usage:
#   python scripts/train_flashsac.py --task=x1_flashsac --headless
#   python scripts/train_flashsac.py --task=x1_flashsac --headless --num_envs 2048 \
#       --max_iterations 30000 --seed 0

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env, name=args.task, args=args)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=False)


if __name__ == '__main__':
    args = get_args()
    # default to the FlashSAC X1 task if none given
    if getattr(args, 'task', None) in (None, 'XBotL_free'):
        args.task = 'x1_flashsac'
    train(args)
