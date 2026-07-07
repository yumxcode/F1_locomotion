# Task Spec — X1 人形机器人行走控制开发

## 目标
为 X1 人形机器人开发稳定、物理可信的前向行走 RL 控制策略。通过 reward 设计与训练流程的可重复迭代，在仿真中达到稳健、可重复的行走步态，为 sim2real 部署奠定基础。每轮围绕本 spec 推进，产出可验证 findings。

## 里程碑
- M1 Reward 基线：训练稳定，mean reward 单调上升无 collapse，能持续行走。
- M2 稳健步态：连续行走不倒（episode 存活达 episode_length_s），稳定跟踪速度命令。
- M3 物理可信：无 reward hacking，reward 各项分解物理合理，步态自然、能耗与关节使用合理。
- M4 鲁棒性：在地形变化/外部扰动下保持稳定行走。

## 成功标准（deploy 候选）
| 指标 | 目标 |
|------|------|
| episode 存活时长 | ≥ 24 s（episode_length_s）不倒 |
| 线速度跟踪误差 | < 0.3 m/s |
| mean reward | 稳定收敛、无 collapse（终止后回落 ≤ 30%）|
| reward 分解 | 各项物理合理，无单项漏洞主导 |
| 步态 | 自然对称，无明显畸形/抖动 |

## 约束
- 训练在远端 gradmotion 执行（codeType=2 git 拉代码），代码改动须 git push 后方可被任务拉取。
- reward 高但物理不可信 = 死路（reward hacking 为首要风险）。
- 遵循 OMA 流程：读码算数 → 搜论文 → 设计 → 实现。

## 当前状态
- 代码：humanoid/（基于 legged_gym/IsaacGym 的 X1 行走 env，reward/config 在 humanoid/envs/x1/）。
- 设计：.oma/designs/design-unified-reward.md（统一 reward 主线）。
- 历史：见 ledger/（findings/directions）与 .oma/experiments/、.oma/trajectory.jsonl。