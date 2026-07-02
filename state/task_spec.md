# Task Spec — X1 双足行走 RL（大一统 Reward 主线）

_Project: agibot_x1_train_oma — AgiBot X1 人形机器人行走_
_Created: 2026-07-02T09:54:45Z_
_Sources: requirements.md (DRAFT 模板), designs/design-unified-reward.md (V13 Minimal Emergence), logs/exploration_investigation.md, logs/exploration_refutation.md_
_Mode: STANDALONE (stage=design)_
_Robot: AgiBot X1, 12-DOF 双足, base_height_target ≈ 0.61 m_

---

## 1. 目标 (Goal)

训练 AgiBot X1 人形机器人通过强化学习实现稳定的交替双足行走（omnidirectional walking + stand），核心方法是 V13 “Minimal Emergence” 大一统 reward 设计（灵感 van Marum 2024, OSU/Digit：以 single_foot_contact 驱动步态涌现）。

当前唯一阻塞问题（已坐实）：策略出现 reward hacking 退化步态——扭胯走（hip-yaw 主导的骨盆扭转前移）与蹲着走（base_height 坍缩到 0.17 m）。这些步态满足现有全部 reward 项，却并非真正的交替迈步行走。single_foot_contact 对扭胯走零区分度（得满分 1.0）。

> 本编排循环的任务 = 在 max_iterations=10 轮内，通过“读码算数 → 假设 → 单变量改动 → 训练验证 → 记录”迭代，消除 reward hacking，涌现出物理可信的交替行走步态，达到成功标准后进入 tune 精调。

硬性原则（来自 memory）：
- 高奖励若缺乏物理可信度即为死路；不得仅凭 reward 数值断言策略有效。
- 先读码/算数（grep+read 完整实现、查历史分支值、python 估算）再动手设计任何数值（reward scale / curriculum / 超参）。
- 算法设计前先 paper_search 复核业界方案。
- reward-based symmetry 已 4 版失败（V12/V14/V15/V16），不得从“奖励”角度再提对称性。

---

## 2. 里程碑 (Milestones)

排序依据：先堵漏洞 → 再提质量 → 最后重武器，且单变量改动便于消融归因。

| # | 里程碑 | 交付物 | 状态 |
|---|--------|--------|------|
| M0 | 跑一次 V13 gait replay（CSV），目视判定 C1（交替步态 vs 扭胯/蹲走）——最低成本、一锤定音 | replay 报告 + C1 判定 | pending |
| M1 | 消除 design 文档与 config 的权重漂移（8 项中 5 项不符）：把 config 实际值回写 design §2.2，或反向做对照实验 | 更新后的 design + config 对照表 | pending |
| M2 | 实现髋 yaw / 腿外展正则化（hip-yaw abduction penalty, V14a, scale -0.08, 带 has_forward_cmd mask）——零梯度冲突，直击扭胯 hack | 代码改动 + 训练 + hip_yaw_amp/hip_pitch_amp 对比 | pending |
| M3 | （若 M2 不足）叠加 feet_position / 站立宽度奖励（scale -5 到 -10）形成关节+足位双层护轨 | 代码 + 训练对比 | pending |
| M4 | 步态正常化后提升质量：自适应步频 / 步长奖励 / action jerk 正则 / CAM_z 正则（任选） | 消融记录 | pending |
| M5 | 达到 §3 成功标准 → 锁定最佳配置，移交 tune 精调 | best 配置 + 验证报告 | pending |

---

## 3. 成功标准 (Success Criteria)

全部条目须同时满足（合取门）。任一 reward-hacking 步态（跳跃/蹲走/扭胯）出现即判定未达标。

### 3.1 主指标（progress.json 的 best_metric 追踪对象）

| 指标 | 定义 | 当前值 | 成功阈值 | 方向 |
|------|------|--------|---------|------|
| gait_twist_ratio | hip_yaw_amp / hip_pitch_amp（前进帧） | 2.0–3.0（退化） | < 0.5 | 越低越好 |

物理含义：真正交替迈步时 hip-pitch（屈髋迈步）摆幅远大于 hip-yaw（扭转）摆幅，比值 ≈ 0.1–0.5；扭胯走时 yaw 摆幅是 pitch 的数倍。

### 3.2 门控指标（必须全部达标）

| 门控指标 | 目标 | 验证方法 |
|---------|------|---------|
| C1 步态涌现 | CSV 回放显示左右脚接触交替占主导，非跳跃/蹲走/扭胯 | gait replay 目视（M0 的终判） |
| 速度跟踪 | 前进命令下 base 线速度稳定 > 0.3 m/s | tracking_lin_vel reward 曲线 |
| single_foot_contact | reward 均值 > 0.8（且非靠扭胯刷满分） | reward 曲线 + C1 交叉验证 |
| 站立稳定 | stand 命令下质心不漂移 | stand phase 分析 |
| Episode 长度 | mean_episode_length > 500 steps (≈10 s) | mean_episode_length 曲线 |
| 力矩安全 | 峰值力矩在硬件容许范围（torque reward 不爆） | torque reward 曲线 |

### 3.3 终止判据 (success_threshold)

success_threshold = (gait_twist_ratio < 0.5) AND (tracking > 0.3 m/s) AND (single_foot_contact > 0.8) AND (episode_length > 500 steps) AND C1(交替步态通过目视)

---

## 4. 编排循环参数 (Orchestration Loop)

| 参数 | 值 |
|------|-----|
| max_iterations | 10 |
| 每轮产出 | 1 个假设 + 1 个单变量改动 + 训练/回放验证 + 1 条 finding |
| 消融纪律 | 每轮只改一个变量；记录 hip_yaw_amp/hip_pitch_amp、single_foot_contact 均值、tracking 均值、episode_length 四项，与 V13e baseline 对比 |
| 长任务 | 训练任务不阻塞等待——通过 auto_orch_pause_external 挂起，待外部结果 |
| 训练资源 | 1× 4090D·24G (ESKU000001), 镜像 Isaac Gym:preview-4 (BJX00000001, V000124) |
| 算法 | PPO (DHPPO), lr=1e-5, entropy_coef ≈0.003–0.005, gamma=0.994, lambda=0.9 |

---

## 5. 当前基线 (Baseline)

| 项 | 值 |
|----|-----|
| 设计 | V13 Minimal Emergence，8 reward terms（design 文档值，非 config 实际值） |
| 实际 config | V13e：tracking 0.6 / single_foot_contact 0.8 / orientation 0.5 / base_height 0.5 / feet_airtime 0.3 / torque 0.01 / sigma=0.15 |
| 训练任务 | TASK_20260612_050 |
| 分支/commit | v9-smooth-landing / 1be71f8（worktree 当前: sub/subtask-99e24c72/code @ 91320b9） |
| 实际控制频率 | 100 Hz（self.dt=0.01 s, legged_robot.py:1270；文档误写 50 Hz） |
| 已知退化步态 | 蹲走(V13c, height 坍缩到 0.17 m) / 扭胯走(V13e, hip-yaw 峰值偏移 3.42 rad) |

---

## 6. 不做的事 (Non-Goals, this cycle)

- 不从“奖励”角度再提 symmetry（已 4 版失败）。
- 不在 C1 未验证前盲目进入 tune 精调（reward hacking 会被放大而非消除）。
- 不凭 reward 标量数值断言步态有效（须 C1 目视 + gait_twist_ratio 双重确认）。
- sim-to-real / 硬件部署留待 deploy，本轮只在仿真内达成物理可信步态。

---
_Provenance: requirements.md + design-unified-reward.md(V13) + logs/exploration_*.md_
_Next: M0 — V13 gait replay → C1 终判 → M2 hip-yaw 正则_
