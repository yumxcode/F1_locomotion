# Design: X1 大一统 Reward — 步态涌现架构

_Created: 2026-06-08_
_Design ID: unified-reward_
_Status: ACTIVE — 训练主线_
_Provenance: V1→V10 迭代 + Schumacher 2025 iScience_

---

## 1. 核心理念

**自然步态 = 速度跟踪 + 能效最优 + 疼痛避免 的涌现结果。**

不需要参考轨迹、步态时钟、抬脚高度、接触调度等显式步态指令。当 reward 收敛时，策略**一定**不会拖脚走——因为拖脚是低效的，高力矩必然被能效惩罚压制。

### 1.1 理论基础

灵感来源：Schumacher et al. 2025, *"Biologically plausible objectives generate naturalistic locomotion"*, iScience.

核心发现：
- 用 `r = r_forward - α·effort - β·pain` 三个 reward 训练的人形机器人，**自发涌现**人类步态特征
- α(费力系数) 和 β(疼痛系数) 需要足够大才能压制非自然步态
- **不需要**任何步态参考轨迹或显式步态参数

### 1.2 我们的映射

| Schumacher | X1 大一统 Reward | 实现 |
|------------|-----------------|------|
| r_forward  | `tracking_lin_vel × 2.5` + `tracking_ang_vel × 0.8` | 速度跟踪 |
| -α·effort  | `efficiency × -2e-4` | τ² 能效惩罚 |
| -β·pain    | `landing_impact × -0.3` | GRF > 300N 惩罚 |
| 生存       | `stability × 0.5` + termination | 姿态/高度 + episode终止 |

---

## 2. Reward 架构

### 2.1 层级设计

```
┌─────────────────────────────────────────────────────┐
│              大一统 Reward = Σ(scale_i × r_i)       │
├──────────────┬──────────────┬────────────────────────┤
│  ① 任务目标  │  ② 物理约束  │  ③ 安全硬约束          │
│  "往前走"    │  "省力+不痛" │  "别撞/别断"           │
├──────────────┼──────────────┼────────────────────────┤
│ tracking_    │ efficiency   │ collision              │
│  lin_vel 2.5 │  -2e-4       │ -1.0                   │
│ tracking_    │ landing_     │ dof_pos_limits  -10    │
│  ang_vel 0.8 │  impact -0.3 │ dof_vel_limits  -1    │
│ stability    │              │ dof_torque_limits -0.1 │
│  0.5         │              │                        │
└──────────────┴──────────────┴────────────────────────┘
```

### 2.2 Reward 项详细规格

#### ① 任务目标 — "往前走"

**Component: tracking_lin_vel**
```
Formula:   exp(-Σ(cmd[:2] - base_lin_vel[:2])² / σ)     [行走]
           exp(-Σ(base_lin_vel[:2])² / σ)               [站立]
Scale:     2.5
σ:         0.5 (tracking_sigma)
Purpose:   核心任务信号 — 策略必须跟踪目标速度
Risk:      过高权重导致步态僵硬（all-in on velocity）
Stand:     当 ||cmd[:3]|| < 0.05 时切换到站立模式
```

**Component: tracking_ang_vel**
```
Formula:   exp(-(cmd[2] - base_ang_vel[2])² / σ)        [行走]
           exp(-(base_ang_vel[2])² / σ)                  [站立]
Scale:     0.8
Purpose:   转弯跟踪 — 独立于线速度的梯度
Risk:      与 lin_vel 竞争时可能引起晃动
```

**Component: stability**
```
Formula:   (exp(-||projected_gravity[:2]|| × 20) + 
            exp(-|base_height - 0.61| × 100)) / 2
Scale:     0.5
Purpose:   保持直立 + 质心高度稳定
Risk:      过高权重抑制腿部运动幅度（不走路 = 最稳定）
设计决策:   V10 从 2.0 降到 0.5，因为 episode终止是摔倒的最终保险，
            stability 只需提供"保持直立"的梯度，不需要高压
```

#### ② 物理约束 — "省力 + 不痛"

**Component: efficiency**
```
Formula:   Σ(τ²)  (sum of squared torques across 12 joints)
Scale:     -2e-4
Purpose:   步态涌现的核心驱动力 — 拖脚走需要大力矩 → 被惩罚 → 策略学会抬脚
Range:     站立 ~10Nm → penalty ≈ -0.24 (15% of forward reward)
           正常走 ~30Nm → penalty ≈ -2.16 (58% of forward reward)
           大力走 ~50Nm → penalty ≈ -6.00 (182% of forward reward)
Risk:      过强可能抑制探索，策略不敢动
调参历史:   V2=-0.02(CoT) → 崩溃, V3=-8e-9(τ²) → 太弱, V10=-2e-4 → 激进但有效
```

**Component: landing_impact**
```
Formula:   Σ max(cfz - 300, 0)² / 300²  (per foot)
Scale:     -0.3
Purpose:   疼痛信号 — GRF 超过 300N (≈1.2× 体重) 时惩罚
Threshold: 300N = Schumacher c_pain 的 1.2×BW 对应值
Risk:      阈值太低导致策略不敢落地 → 脚永远悬空
设计决策:   V9 从 500N 降到 300N，更敏感但更接近生物学疼痛信号
```

#### ③ 安全硬约束 — "别撞/别断"

```
Component: collision
Formula:   Σ(||contact_forces[penalised]|| > 0.1)
Scale:     -1.0
Purpose:   躯干碰撞 → 不允许

Component: dof_pos_limits
Formula:   Σ(max(0, lower - q) + max(0, q - upper))
Scale:     -10.0
Purpose:   关节超限保护

Component: dof_vel_limits
Formula:   Σ(max(0, |dq| - 0.9 × limit))
Scale:     -1.0
Purpose:   关节超速保护

Component: dof_torque_limits
Formula:   Σ(max(0, |τ| - 0.9 × limit))
Scale:     -0.1
Purpose:   力矩过载保护
```

### 2.3 显式移除的 Reward（及其移除理由）

| 移除项 | 原 Scale | 移除理由 |
|--------|---------|---------|
| ref_joint_pos | 1.5→0 | 步态从物理涌现，不需要参考轨迹引导 |
| feet_contact_number | 2.0→0 | 相位对齐是自然步态的结果，不是目标 |
| feet_clearance | 1.2→0 | 抬脚高度由能效自然决定（不抬脚→高GRF→惩罚） |
| swing_foot_forward | 0.5→0 | 已被 tracking_lin_vel 的速度梯度覆盖 |
| foot_slip | -0.1→0 | 滑动浪费能量，efficiency 自然惩罚 |
| landing_compliance | 0→0 | 屈曲吸收是 landing_impact 的最优解（涌现） |
| dof_vel | -2e-8→0 | 被 efficiency τ² 覆盖 |
| dof_acc | -1e-7→0 | 被 efficiency τ² 覆盖 |
| action_smoothness | -0.002→0 | 被 efficiency τ² 覆盖 |
| feet_contact_forces | -0.01→0 | 被 landing_impact 覆盖 |
| base_acc | 0.2→0 | 不摔=终止，稳定=stability，加速度是结果 |
| default_joint_pos | 1.0→0 | 被 efficiency 覆盖（偏离默认=大力矩） |
| feet_rotation | 0.3→0 | 细粒度控制不必要 |
| feet_air_time | 1.2→0 | 被 efficiency + landing_impact 隐含覆盖 |

---

## 3. 为什么收敛时不会拖脚走

这是设计文档最核心的论证。

### 3.1 数学推导

设策略 π 选择"拖脚走"模式：

```
拖脚的力矩代价:
  - 脚底摩擦力 F_friction = μ × N ≈ 0.6 × 150N = 90N
  - 拖动需要的关节力矩 τ ≈ F × L ≈ 90 × 0.4 = 36 Nm/joint
  - efficiency penalty = -2e-4 × Σ(τ²) = -2e-4 × 12 × 36² ≈ -3.11

正常抬脚走的力矩代价:
  - 抬腿需要克服重力做功，平均 τ ≈ 20 Nm/joint
  - efficiency penalty = -2e-4 × 12 × 20² ≈ -0.96

差值: -3.11 - (-0.96) = -2.15/step
```

**每一步，拖脚走比抬脚走多付出 2.15 的 penalty。** 对于总正向 reward ≈ 3.3（速度跟踪+稳定性）来说，这是一个不可忽视的代价。策略会自然学会抬脚。

### 3.2 能效的自动调节机制

```
efficiency 的有效惩罚与力矩的平方成正比:

  如果策略选择:  低力矩 (小步幅) → penalty 小 → reward 高 → 但速度跟踪差
  如果策略选择:  高力矩 (大步幅) → penalty 大 → reward 低 → 但速度快

策略必须找到:   最优力矩 → 最大速度 / 最小力矩 的平衡点
这个平衡点 = 自然步态
```

### 3.3 与 Schumacher 2025 的对应

Schumacher 的关键发现是 α(t) 需要自适应增大到很大的值：
- 我们的 V10-a 使用固定的 α=-2e-4，等价于 Schumacher 的 α≈200（缩放到我们的力矩量级）
- 如果训练中 reward 不收敛，说明 α 还不够大，需要继续增大到 -5e-4 或 -1e-3

---

## 4. 观测/动作空间

### 4.1 观测空间 (47 dim × 66 frames = 3102 dim)

```
obs = [
  sin_pos, cos_pos           (2,)     # 步态相位编码
  cmd_x, cmd_y, cmd_yaw      (3,)     # 速度指令 (scaled)
  q - q_default              (12,)    # 关节偏差 (scaled by 1.0)
  dq                         (12,)    # 关节速度 (scaled by 0.05)
  last_action                (12,)    # 上一步动作
  base_ang_vel               (3,)     # IMU 角速度 (scaled by 1.0)
  base_euler                 (3,)     # 姿态角 (scaled by 1.0)
]
Total = 47 × 66 frames = 3102
```

### 4.2 特权观测 (73 dim × 3 frames = 219 dim)

```
privileged = obs + [
  ref_diff                   (12,)    # 关节偏差参考轨迹
  base_lin_vel               (3,)     # 真实线速度
  base_ang_vel_full          (3,)     # 完整角速度
  base_euler_full            (3,)     # 完整姿态
  push_force                 (2,)     # 外扰力
  push_torque                (3,)     # 外扰力矩
  friction                   (1,)     # 地面摩擦
  body_mass                  (1,)     # 质量扰动
  stance_mask                (2,)     # 步态相位
  contact_mask               (2,)     # 接触状态
]
Total = 73 × 3 frames = 219
```

### 4.3 动作空间

```
Type:          关节位置增量 (Δq)
Action scale:  0.5 rad
PD control:    Kp ∈ {30-100}, Kd ∈ {0.5-10} (joint-specific)
Control freq:  50 Hz (decimation=10, dt=0.001)
```

---

## 5. 策略网络

```
Actor:   MLP [512, 256, 128], init_noise_std=1.0, activation=ELU
Critic:  MLP [768, 256, 128], with state estimator [256, 128, 64]
CNN:     Long-history encoder: kernel=[6,4], filter=[32,16], stride=[3,2], output=64
         Input: 66 frames × 47 dim obs → 64 dim latent
```

---

## 6. 算法 & 超参

```
Algorithm:    PPO (DHPPO variant)
lr:           1e-5 (constant)
entropy_coef: 0.005 (5× 标准值，补偿 reward 稀疏性)
gamma:        0.994
lambda (GAE): 0.9
epochs:       2
mini-batches: 4
steps/env:    24
max_iter:     15000-20000
```

---

## 7. Domain Randomization

| Parameter | Sim Default | Range | Distribution |
|-----------|------------|-------|-------------|
| Kp multiplier | 1.0 | [0.8, 1.2] | uniform |
| Kd multiplier | 1.0 | [0.8, 1.2] | uniform |
| Base mass | ~actual | ±3 kg | uniform |
| COM offset | 0 | ±5 cm each axis | uniform |
| Friction | 0.6 | [0.2, 1.3] | uniform |
| Joint friction | 1.0 | [0.01, 1.15] | uniform |
| Motor offset | 0 | ±0.035 rad | uniform |
| Torque multiplier | 1.0 | [0.8, 1.2] | uniform |
| DOF lag | 0 | [0, 40] steps | uniform |
| Push vel xy | 0 | ±0.2 m/s | uniform |
| Push ang vel | 0 | ±0.2 rad/s | uniform |

---

## 8. Curriculum

```
速度指令 curriculum (基于 common_step_counter):
  Phase 1: steps 0→20k      cmd_x: [−0.15, 0.3]   低速适应
  Phase 2: steps 20k→60k    cmd_x: [−0.2, 0.45]   逐步提速
  Phase 3: steps 60k+       cmd_x: [−0.3, 0.6]    目标速度

步态类型 curriculum (基于 episode 进度):
  gait = [walk_omnidirectional, stand, walk_omnidirectional]
  比例随机采样，自然过渡

地形 curriculum:
  flat: 0.3, rough_flat: 0.2, slope: 0.4, discrete: 0.1
  curriculum=False (当前不做自适应地形难度提升)
```

---

## 9. 训练历史 & 迭代经验

### 9.1 版本迭代链

```
V1 (29 reward, main分支)
  → iter 6100 平台期
  → 核心问题: reward hacking, 29项权重相互耦合

V2 (10 reward, τ·dq CoT)
  → efficiency=-0.02 (CoT) → 训练崩溃
  → 教训: CoT 公式 ∑|τ·dq| 对力矩符号敏感，不是好的 penalty

V3 (τ² 效率)
  → efficiency=-8e-9 → 太弱，拖脚走无惩罚
  → 加了 forward propulsion 补偿 → vel hacking

V4 (split velocity + curriculum)
  → 速度拆分 + curriculum → vel hacking 修复
  → 但效率惩罚仍然太弱

V5 (stand/walk 切换)
  → 加 stand_still → 站立行为修复
  → walk_decay 导致 stability 崩溃

V6 (去掉 walk_decay)
  → walk_decay 移除 → stability 不再被衰减
  → 但 swing_foot_forward 线性增长导致 stability 被碾压

V7 (swing cap + scale rebalance)
  → swing max=0.5 → 步态更稳定
  → 但落地冲击仍然大（heel-strike 无吸收）

V8 (heel-toe landing)
  → ankle dorsiflexion + smoothstep ramp → 落地柔和
  → GRF 仍有 600-800N 峰值

V9 (smooth landing)
  → sin² 下降轨迹 → 接地时速度→0
  → landing_impact=-0.3, threshold=500N → GRF 降低
  → landing_compliance → 鼓励屈曲吸收

V10 (大一统: 最小 reward, 步态涌现)
  → 灵感: Schumacher 2025 iScience
  → 核心假设: 步态从速度+能效+疼痛涌现
  → 移除 ref_joint_pos, feet_contact_number, feet_clearance 等
  → efficiency=-8e-9 (初始) → 太弱
  
V10-a (efficiency 放大)
  → efficiency=-2e-4 (25,000×)
  → 8 reward terms, GRF 300N
  → 当前训练中: TASK_20260608_071
```

### 9.2 已验证的失败模式 (Dead Ends)

| 方向 | 失败原因 | 版本 |
|------|---------|------|
| CoT 能效 (τ·dq) | 对符号敏感，正负力矩抵消 | V2 |
| walk_decay | 衰减 stability 导致姿态崩溃 | V5 |
| 效率太弱 (-8e-9) | 拖脚走无惩罚 | V3-V10 |
| swing_foot_forward 无上限 | 线性增长碾压 stability | V6 |
| efficiency 主导 (V2=-0.02) | 压制一切运动 | V2 |

---

## 10. 消融计划 (Ablation Plan)

| 决策变量 | 候选值 | 假设 | 优先级 |
|---------|--------|------|--------|
| efficiency scale | [-5e-5, -1e-4, -2e-4, -5e-4] | -2e-4 是起步点，可能需要调整 | 高 |
| landing_impact scale | [0, -0.1, -0.3, -0.5] | -0.3 对齐 Schumacher β | 高 |
| GRF threshold | [200, 300, 500] N | 300N ≈ 1.2×BW 是生物学阈值 | 中 |
| tracking_lin_vel scale | [1.5, 2.5, 3.5] | 2.5 是当前值，需验证敏感度 | 中 |
| stability scale | [0.2, 0.5, 1.0, 2.0] | 过高抑制运动 | 中 |
| 网络宽度 | [512, 768] actor hidden | 512 是当前值 | 低 |
| entropy_coef | [0.001, 0.005, 0.01] | 0.005 补偿 reward 稀疏 | 低 |

---

## 11. 设计风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| efficiency=-2e-4 过强，策略不动 | 中 | 训练崩溃 | 回调到 -5e-5 |
| 步态涌现但质量差（碎步、高频） | 中 | 不可部署 | 加 foot_slip 惩罚 |
| 移除 ref_joint_pos 后步态失稳 | 中 | 左右不对称 | 加 symmetric reward |
| sim-to-real gap: 仿真步态不能迁移 | 高 | 需要额外 DR 迭代 | $deploy 阶段处理 |
| 搜索空间太激进导致不收敛 | 低 | 浪费 GPU 时间 | 从 -5e-5 开始二分搜索 |

---

## 12. 当前状态 & 下一步

### 12.1 当前训练
- **Task**: TASK_20260608_071
- **Branch**: v9-smooth-landing
- **Commit**: fdac1e7
- **Config**: efficiency=-2e-4, 8 reward terms, GRF 300N, 15k iter
- **Status**: 运行中

### 12.2 验证清单
训练完成后，需要验证以下指标：

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| 步态涌现 | 抬脚走，不拖脚 | CSV replay 目视检查 |
| GRF 峰值 | < 500N (单脚) | contact force 日志 |
| 速度跟踪 | > 0.4 m/s 稳定 | mean_reward 曲线 |
| 能效 | τ² 总量稳定 | efficiency reward 曲线 |
| 站立稳定 | 站立时不漂移 | stand phase 分析 |

### 12.3 调参路径
```
V10-a (-2e-4) 结果:
  → 如果收敛 + 步态正常 → ✅ 成功，进入 $tune 精调
  → 如果收敛 + 步态碎/高频 → 加 foot_slip, 调低 tracking
  → 如果不收敛 / 崩溃 → 回调到 -5e-5 (V10-b)
  → 如果收敛但拖脚 → 继续增大到 -5e-4 (V10-c)
```

---

## 13. 替代方案

### 方案 B: V9 基线（保留 ref_joint_pos + 显式步态引导）
如果 V10 步态涌现失败，回退到 V9 的半涌现方案：
- 保留 ref_joint_pos (scale=1.0)、feet_contact_number (scale=1.5)
- 但使用 V10 的 efficiency=-2e-4 和 landing_impact=-0.3
- 10-12 个 reward terms

### 方案 C: 完全 Schumacher 复现
完全按照 Schumacher 2025 的公式：
- r = r_forward - α(t)·Σ(τ²) - β(t)·Σ(max(GRF-threshold, 0))
- α(t) 和 β(t) 从小值线性增大到目标值
- 最小化我们的自定义修改

---

_Design committed: unified-reward_
_Last updated: 2026-06-08_
_Next: V10-a 训练结果验证 → 调参 → $tune_
