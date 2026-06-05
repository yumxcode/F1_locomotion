# 双线推进方案 (Dual-Track Plan)

_创建时间: 2026-06-03_
_状态: ACTIVE_

---

## 背景

X1 双足机器人行走训练已迭代至 v5.7（main 分支，29 reward），在 iter 6100 出现平台期。为突破瓶颈，同时探索精简 reward 设计，确立两条并行推进线。

---

## Track A: 原 29-Reward 路线（main 分支）

### 基本信息
| 字段 | 值 |
|------|-----|
| 分支 | `main` |
| 最新 commit | `9e462c5` — chore: remove large media |
| 训练任务 | `TASK_20260602_134`（已停止，iter 6100） |
| Reward 数量 | 29 项 |
| 资源 | 1×4090D·24G (`ESKU000001`) |
| 镜像 | Isaac Gym:preview-4 (`BJX00000001`, v`V000124`) |

### Reward 结构（29 项）
```
步态引导:   ref_joint_pos(2.2), feet_clearance(1.5), feet_contact_number(2.0),
            feet_air_time(1.2), swing_foot_forward(0.5), foot_landing_pitch(0.3)
接触:       foot_slip(-0.1), feet_distance(0.2), knee_distance(0.2)
速度跟踪:   tracking_lin_vel(1.8), tracking_ang_vel(1.1), vel_mismatch_exp(0.5),
            low_speed(0.2), track_vel_hard(0.5)
姿态:       default_joint_pos(1.0), orientation(1.0), feet_rotation(0.3),
            base_height(0.2), base_acc(0.2)
能效:       action_smoothness(-0.002), torques(-8e-9), dof_vel(-2e-8), dof_acc(-1e-7)
接触力:     feet_contact_forces(-0.01)
静止:       stand_still(2.5)
安全:       collision(-1.0), dof_vel_limits(-1), dof_pos_limits(-10), dof_torque_limits(-0.1)
```

### 当前进展
- 训练在 iter ~6100 进入平台期
- 62 个 checkpoint 已保存（iter 100~6100）
- 0.9s 步态周期，swing forward，landing pitch 特性已集成

### 下一步方向
1. **从 checkpoint 恢复训练**：从 iter 6100 继续，观察是否突破平台期
2. **调参**：调整 entropy_coef、learning_rate 等 PPO 超参
3. **增加域随机化**：提升 sim-to-real 鲁棒性
4. **细调 reward 权重**：逐项消融分析主导 reward 分量

---

## Track B: 精简 10-Reward 路线（reward-simplify-29to10 分支）

### 基本信息
| 字段 | 值 |
|------|-----|
| 分支 | `reward-simplify-29to10` |
| 最新 commit | `1710254` — feat: simplify reward 29→10 |
| 训练任务 | `TASK_20260603_016`（运行中） |
| Reward 数量 | 10 项（从 29 精简） |
| 资源/镜像 | 同 Track A |

### 设计理念：第一性原理 → 四层架构

```
① 速度跟踪 (velocity_tracking = 3.0)
   唯一任务目标。合并: lin_vel, ang_vel, vel_mismatch, low_speed, track_vel_hard, stand_still
   → 统一信号，消除 reward 间竞争

② 步态引导 (ref_joint_pos=1.5, feet_contact_number=2.0, feet_clearance=1.2)
   参考轨迹 + 相位对齐，不单独拆分 air_time/landing_pitch/swing_forward

③ 稳定性 (stability = 2.0)
   合并: orientation, base_height, base_acc, default_joint_pos, feet_rotation
   → "不摔倒 + 质心稳定"，5→1 信号

④ 能效 (efficiency = -0.02)
   机械功率 / 速度 = Cost of Transport
   替代: torques², dof_vel, dof_acc, action_smoothness, contact_forces
   → 物理意义明确，|τ·q̇| 比 τ² 更准确

⑤ 脚底打滑 (foot_slip = -0.1)
⑥ 安全硬约束 (collision=-1, dof_pos_limits=-10, dof_vel_limits=-1, dof_torque_limits=-0.1)
```

### PPO 调整
- `entropy_coef`: 0.001 → **0.005**（5×，防止 10 reward 信号稀疏导致探索不足）
- `learning_rate`: 1e-5（保持不变）
- 其余参数保持一致

### 当前进展
- 训练已启动，待观察

### 下一步方向
1. **监控收敛曲线**：观察 10 reward 是否能正常收敛
2. **对比 29 reward**：相同 iter 下 reward 曲线对比
3. **仿真回放**：检查步态质量是否退化
4. **消融**：如收敛，逐项增减验证每层贡献

---

## 对比矩阵

| 维度 | Track A (29 Reward) | Track B (10 Reward) |
|------|---------------------|---------------------|
| 设计哲学 | 细粒度工程调优 | 第一性原理精简 |
| Reward 项数 | 29 | 10 |
| 能效指标 | τ² + q̇² + q̈² (proxy) | \|τ·q̇\|/v (CoT, 物理精确) |
| 调参复杂度 | 高（29 权重相互耦合） | 低（10 权重独立） |
| 收敛风险 | 平台期（已验证） | 未知（待验证） |
| Sim-to-Real 潜力 | 高（细粒度控制） | 可能更高（减少 reward hacking） |
| 探索多样性 | 低 (entropy=0.001) | 较高 (entropy=0.005) |

---

## 决策节点

1. **Track B 收敛失败** → 回到 Track A，从 checkpoint 继续
2. **Track B 收敛且步态质量 ≥ Track A** → 主力转 Track B，进入 $tune
3. **两条线均平台期** → 分析 sim-to-real gap，考虑 $design 迭代
4. **一条线显著优于另一条** → 合并最优策略，单线推进

---

_最后更新: 2026-06-03_
