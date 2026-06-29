# FlashSAC for AgiBot X1 (双足训练)

本项目将 [F1_locomotion](https://github.com/yumxcode/F1_locomotion)（AgiBot X1 双足机器人，原基于 IsaacGym + rsl_rl **PPO**）改造为 **FlashSAC**（离策略 Soft Actor-Critic）训练框架，用于 X1 双足行走训练。

> **算法来源**：FlashSAC (Kim et al., *FlashSAC: Fast and Stable Off-Policy RL for High-Dimensional Robot Control*, RSS'26, arXiv:2604.04539, Holiday Robotics)。其官方实现是面向 **IsaacLab** 的独立框架，无法直接套用到本项目所依赖的 **legacy IsaacGym + rsl_rl** 架构。因此本仓库在保留 F1 原始 X1 环境（URDF、PD 增益、域随机化、奖励塑形、双历史观测）的前提下，**在 rsl_rl 框架内重新实现了 FlashSAC 的核心算法**，并接入项目的 `task_registry`。

## 一、新增了什么

```
humanoid/
├── algo/flashsac/                       # ★ FlashSAC 算法实现
│   ├── networks.py                      #   奖励归一化 + 双历史 tanh-Gaussian actor + 分布式 double-Q critic
│   ├── replay_buffer.py                 #   n-step 均匀回放缓冲（GPU 向量化）
│   ├── flashsac.py                      #   算法类：分布式 TD target + 自动温度 + 软目标更新
│   ├── off_policy_runner.py             #   离策略 runner（适配 task_registry 契约）
│   └── __init__.py
├── envs/x1/x1_flashsac_config.py        # ★ X1 FlashSAC 配置（复用 X1 机器人，num_envs=1024）
├── scripts/
│   ├── train_flashsac.py                # ★ 训练入口
│   ├── export_policy_flashsac.py        # ★ 导出 JIT 部署模型
│   └── smoke_test_flashsac.py           # ★ 算法链路自测（不需要 IsaacGym，仅需 torch）
```

修改的既有文件（仅追加，不破坏原 PPO 流程）：
- `humanoid/algo/__init__.py` — 导出 flashsac 包
- `humanoid/utils/task_registry.py` — 导入 `FlashSACOffPolicyRunner` 使其类名可被 `eval` 解析
- `humanoid/envs/__init__.py` — 注册新任务 `x1_flashsac`

> 原始 PPO 任务 `x1_dh_stand` **完全不受影响**，仍可照常训练。

## 二、FlashSAC vs 原始 PPO（关键差异）

| 维度 | 原始 PPO (DHPPO) | FlashSAC (本实现) |
|---|---|---|
| 策略类型 | 在策略 | **离策略** |
| 数据存储 | RolloutStorage（每次迭代后丢弃） | **n-step 均匀回放缓冲**（GPU，复用） |
| 更新频率 | 收集 N 步后批量更新 | **每个环境步多次梯度更新**（`updates_per_interaction_step=2`）|
| Actor | 无界高斯（固定 std） | **tanh-squashed 高斯**（学习 log_std） |
| Critic | 单值函数 + GAE | **分布式 double-Q critic**（101 bins），EMA 目标网络 |
| 熵 | 固定系数熵正则 | **自动调谐温度 α** |
| 奖励 | 原始 | **运行均值/方差归一化** |
| 并行环境数 | 4096 | **1024**（离策略不需要海量环境）|

## 三、安装

依赖与原项目完全相同（Python 3.8 + PyTorch 1.13 + Isaac Gym Preview 4），无需额外安装：

```bash
conda create -n myenv python=3.8
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy=1.23
# 安装 Isaac Gym Preview 4（见原 README）
cd isaacgym/python && pip install -e .
# 安装本项目
cd F1_locomotion && pip install -e .
```

## 四、算法自测（推荐先跑）

无需 IsaacGym，仅需 PyTorch，验证整个算法链路（网络前向 → buffer 存取 → 一次梯度更新）无 off-by-one 等错误：

```bash
python humanoid/scripts/smoke_test_flashsac.py --device cuda:0
# 预期输出：[1/4]...[4/4] ALL SMOKE TESTS PASSED
```

## 五、训练 X1 双足

```bash
# 训练（无头模式）
python humanoid/scripts/train_flashsac.py --task=x1_flashsac --headless

# 自定义参数
python humanoid/scripts/train_flashsac.py --task=x1_flashsac --headless \
    --num_envs 2048 --max_iterations 30000 --seed 0
```

模型保存在 `logs/x1_flashsac/exported_data/<日期><run_name>/model_<iter>.pt`。

用 TensorBoard 查看训练曲线：
```bash
tensorboard --logdir logs/x1_flashsac
```

## 六、回放与导出部署模型

```bash
# 回放（需有屏幕）
python humanoid/scripts/play.py --task=x1_flashsac --load_run=<日期>

# 导出 JIT 模型（部署用，输出 tanh(mean) 确定性动作）
python humanoid/scripts/export_policy_flashsac.py --task=x1_flashsac --load_run=-1
# 模型保存到 logs/x1_flashsac/exported_policies/<日期>/policy_flashsac.jit
```

> 导出的 JIT 模型与原 PPO 模型**输入输出接口一致**（输入 X1 actor 观测，输出 12 维动作），可直接用于 sim2sim 和真机部署。唯一区别：FlashSAC 动作经过 tanh 压缩到 `[-1,1]`，更贴合 SAC 的有界动作空间。

## 七、关键超参数（`humanoid/envs/x1/x1_flashsac_config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gamma` | 0.99 | 折扣因子 |
| `n_step` | 3 | n-step 回报 |
| `tau` | 0.01 | 目标 critic EMA 系数 |
| `actor_lr` / `critic_lr` / `temp_lr` | 3e-4 | 三个学习率 |
| `init_alpha` | 0.01 | 初始熵温度 |
| `target_sigma` | 0.15 | 目标熵 = 0.5·A·ln(2πeσ²) |
| `num_bins` | 101 | 分布式 critic 值支持 bin 数 |
| `normalized_G_max` | 5.0 | 值支持范围 ±5.0 |
| `buffer_max_length` | 2,000,000 | 回放缓冲容量 |
| `buffer_min_length` | 20,000 | 开始训练的最小缓冲量 |
| `batch_size` | 2048 | 每次梯度更新采样数 |
| `updates_per_interaction_step` | 2 | 每环境步的梯度更新次数 |
| `num_envs` | 1024 | 并行环境数 |

## 八、注意事项

1. **GPU 显存**：回放缓冲（200 万条）+ 1024 环境需要约 8-12GB 显存。如显存不足，调小 `buffer_max_length` 或 `num_envs`。
2. **收敛预期**：SAC 离策略通常比 PPO 样本效率更高（按真实环境交互步数计），但前 `buffer_min_length` 步用随机/策略动作填充缓冲，初期 reward 会偏低。
3. **sim2sim**：导出的 JIT 模型可直接用原 `scripts/sim2sim.py`（MuJoCo）验证，接口兼容。
4. **恢复训练**：`--task=x1_flashsac --resume --load_run=<日期> --checkpoint=<iter>`。
