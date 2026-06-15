# Design: X1 大一统 Reward — 最小涌现架构 (V13)

_Created: 2026-06-08_
_Updated: 2026-06-12_
_Design ID: unified-reward_
_Status: ACTIVE — V13 训练中_
_Provenance: V1→V13 迭代 + van Marum 2024 (OSU/Digit) + Schumacher 2025 iScience_

---

## 0. 设计范式转变 (V10→V13)

V10-V12 基于 **Schumacher 2025** 的"efficiency + pain → 步态涌现"路线，经过 6 轮迭代证明**不可行**：

| 版本 | 核心思路 | 结果 |
|------|---------|------|
| V10-c | Schumacher 三件套 (tracking+efficiency+landing) | 0.95s 倒地，移除 stability 后无安全网 |
| V11 | σ=0.08 梯度消失 | tracking 662 iter 仍≈0 |
| V11-b | σ=0.2 修复梯度 | tracking 64%/走路 100%，差距 36% |
| V11-c | stability ×10, landing 300→500N | surrogate≈0, reward 横盘 72-80, 正反馈循环 |
| V12 | 新增 symmetry reward | 待观察 |
| V15 | 相位调制能量对称性 | 学会 split stance（单腿主导），非交替步态 |
| V16 | 镜像 morphological symmetry | 逻辑正确但方向冲突仍存在 |

**根因诊断**：
1. **梯度冲突** — efficiency / landing_impact 与 tracking 方向冲突，surrogate≈0
2. **tracking 权重过高** (2.5) → 压制其他所有 reward 的梯度
3. **efficiency |τ·q̇| 过于复杂** — 站立≈0 走路≈360，scale 极难调
4. **symmetry 设计反复失败** — V13/V14/V15/V16 四版均未解决交替步态问题

**V13 范式转变**：基于 **van Marum 2024** (Oregon State / Digit) 的最小约束设计：

> **tracking + orientation → 只产生跳跃**
> **+ single_foot_contact → 产生行走（步态涌现！）**

不需要 symmetry、不需要 efficiency、不需要 landing_impact、不需要参考轨迹。一个简单的单脚接触检测就够了。

---

## 1. 核心理念 (V13)

**步态 = single_foot_contact 驱动的涌现行为。**

van Marum 在 Digit 人形机器人上验证：
- 仅有 tracking + orientation reward 时，策略学会**跳跃**前进
- 加上一个简单的 `single_foot_contact`（当前是否有恰好 1 只脚接触地面）后，策略自发学会**交替行走**
- 不需要步态时钟、不需要参考轨迹、不需要对称性约束

### 1.1 理论基础

**van Marum et al. 2024** *"Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking"*, Oregon State University.

核心发现：
1. 最小 reward 集：velocity tracking + orientation + single_foot_contact + feet_airtime
2. single_foot_contact 是从跳跃到行走的**决定性 reward**
3. 所有权重很小（0.01–0.15），不压制任何方向
4. feet_airtime 正则化步频：(t_air - 0.4) × 1_touchdown

### 1.2 V13 与 van Marum 的映射

| van Marum (Digit) | V13 (X1) | Scale |
|-------------------|----------|-------|
| velocity_tracking_xy × 0.15 | tracking_lin_vel × 1.0 | 放大 6.7× (X1 12DOF 更高维) |
| orientation × 0.2 | orientation × 0.5 | 放大 2.5× |
| **feet_contact × 0.1** | **single_foot_contact × 0.3** | 放大 3× |
| feet_airtime × 1.0 | feet_airtime × 0.3 | 缩小 3.3× (稀疏 reward) |
| base_height × 0.05 | base_height × 0.2 | 放大 4× |
| torque × 0.01 | torque × 0.01 | 一致 |

---

## 2. Reward 架构 (V13)

### 2.1 层级设计

```
┌───────────────────────────────────────────────────────────┐
│            V13 Reward = Σ(scale_i × r_i)                  │
├──────────────┬──────────────────┬─────────────────────────┤
│  ① 任务目标  │  ② 步态涌现      │  ③ 安全硬约束            │
│  "往前走"    │  "走路而非跳跃"   │  "别断"                  │
├──────────────┼──────────────────┼─────────────────────────┤
│ tracking_    │ single_foot_     │ dof_pos_limits  -10      │
│  lin_vel 1.0 │  contact 0.3     │                         │
│ tracking_    │ feet_airtime 0.3 │                         │
│  ang_vel 0.5 │                  │                         │
│ orientation  │                  │                         │
│  0.5         │                  │                         │
│ base_height  │                  │                         │
│  0.2         │                  │                         │
│ torque       │                  │                         │
│  0.01        │                  │                         │
└──────────────┴──────────────────┴─────────────────────────┘
```

### 2.2 Reward 项详细规格

#### ① 任务目标 — "往前走 + 直立"

**Component: tracking_lin_vel**
```
Formula:   exp(-Σ(cmd[:2] - base_lin_vel[:2])² / σ)     [行走]
           exp(-Σ(base_lin_vel[:2])² / σ)               [站立]
Scale:     1.0
σ:         0.25 (tracking_sigma)
Purpose:   核心任务信号 — 策略必须跟踪目标速度
Risk:      过高权重导致梯度压制（V12 教训：2.5 压制所有其他 reward）
Stand:     当 ||cmd[:3]|| < 0.05 时切换到站立模式
History:   V10=2.5 → V13=1.0 (降低 2.5×, 消除梯度压制)
```

**Component: tracking_ang_vel**
```
Formula:   exp(-(cmd[2] - base_ang_vel[2])² / σ)        [行走]
           exp(-(base_ang_vel[2])² / σ)                  [站立]
Scale:     0.5
Purpose:   转弯跟踪 — 独立于线速度的梯度
Risk:      与 lin_vel 竞争时可能引起晃动
```

**Component: orientation**
```
Formula:   exp(-||projected_gravity[:2]|| × 10)
Scale:     0.5
Purpose:   保持躯干直立
Range:     直立=1.0, 倾斜5°=0.42, 倾斜10°=0.18
Risk:      无（从 V12 stability 拆出，更简洁）
History:   V12 stability 包含 orientation+height, V13 拆为独立项
```

**Component: base_height**
```
Formula:   exp(-|h - 0.61| × 10)
Scale:     0.2
Purpose:   维持期望站高 0.61m
Range:     Δh=0: 1.0, Δh=2cm: 0.82, Δh=5cm: 0.61
Risk:      与 tracking 轻微冲突（走路时高度有波动）
```

**Component: torque**
```
Formula:   exp(-Σ|τ| / 100)
Scale:     0.01
Purpose:   温和力矩正则化 — van Marum 用相同权重 0.01
Range:     站立 Σ|τ|≈300 → r≈0.050 × 0.01 = 0.0005 (微弱)
Risk:      极低风险 — 0.01 的 scale 使其几乎不影响决策
History:   替代 V12 的 efficiency (|τ·q̇| × -1e-4), 后者过于复杂
```

#### ② 步态涌现 — "走路而非跳跃" ⭐

**Component: single_foot_contact** ⭐
```
Formula:   1.0 if n_contact == 1 (with 0.2s grace), 1.0 if stand_cmd, 0.0 otherwise
Scale:     0.3
Purpose:   从跳跃到行走的决定性 reward (van Marum 核心发现)
Mechanism: 
  - 跳跃 = 双脚同时离地 → n_contact ≠ 1 → r = 0
  - 行走 = 交替单脚接触 → n_contact = 1 → r = 1
  - 站立命令 → r = 1 (不给偏好, 允许恢复步)
  - 宽限期 0.2s: 允许双支撑阶段, 不要求完美交替
Risk:      宽限期太长可能容忍过多的双脚离地
History:   V13 新增 — 替代所有步态工程化 (symmetry, feet_contact_number, etc.)
```

**Component: feet_airtime** ⭐
```
Formula:   Σ_f (t_air,f - 0.4) × 1_touchdown,f     [行走]
           1.0                                              [站立]
Scale:     0.3
Purpose:   步频正则化 — 防止策略走太快碎步
Mechanism:
  - 空中时间 > 0.4s → 正奖励 → 鼓励充分抬脚
  - 空中时间 < 0.4s → 负奖励 → 惩罚过快步频
  - 只在落地瞬间触发（稀疏），但梯度很强
Risk:      阈值 0.4s 可能不适合 X1 的腿长（van Marum 针对 Digit 调参）
History:   V13 新增 — van Marum 公式
```

#### ③ 安全硬约束

```
Component: dof_pos_limits
Formula:   Σ(max(0, lower - q) + max(0, q - upper))
Scale:     -10.0
Purpose:   关节超限保护
```

### 2.3 显式移除的 Reward（V12→V13 及移除理由）

| 移除项 | V12 Scale | 移除理由 |
|--------|----------|---------|
| symmetry | 1.0 | single_foot_contact 自然产生交替步态，无需额外约束 |
| stability | 1.0 | 拆分为 orientation + base_height，更简洁 |
| efficiency | -1e-4 | |τ·q̇| 过于复杂（站立≈0 走路≈360），van Marum 只用温和 exp(-\|τ\|) |
| landing_impact | -0.3 | 与 tracking 方向冲突（V11-c 根因），van Marum 不用 |
| ref_joint_pos | (0) | 参考轨迹约束阻碍自然步态涌现 |
| feet_contact_number | (0) | 相位依赖，被 single_foot_contact（无相位）替代 |
| feet_clearance | (0) | 相位依赖，被 feet_airtime 替代 |
| swing_foot_forward | (0) | tracking 已提供前进动力 |
| foot_slip | (0) | 不在最小集合中 |

---

## 3. 为什么 single_foot_contact 能驱动步态涌现

### 3.1 van Marum 的实验发现

在 Digit 人形机器人上：
1. **仅 tracking + orientation** → 策略学会**双脚同时跳**前进（效率最高的"违规"路径）
2. **加入 single_foot_contact** → 策略立即转向**交替单脚行走**

原因：跳跃时双脚同时离地，n_contact = 0，reward = 0。行走时总有一只脚在地面上，n_contact = 1，reward = 1。这是最简单的物理约束，不需要任何步态时钟或相位信息。

### 3.2 与 Schumacher 路线的对比

| 维度 | Schumacher (V10-V12) | van Marum (V13) |
|------|----------------------|-----------------|
| 步态产生机制 | efficiency 压制拖脚 → 抬脚涌现 | single_foot_contact 禁止跳跃 → 行走涌现 |
| Reward 项数 | 7+ | 8（但 3 项极弱） |
| 方向冲突 | efficiency ↔ tracking, landing ↔ tracking | **无冲突** |
| Tracking 权重 | 2.5（高压） | 1.0（低压） |
| 需要参考轨迹？ | V10 移除后失稳 | 完全不需要 |
| 需要对称性？ | V12-V16 四版失败 | 完全不需要 |
| 验证平台 | 理论（iScience 论文） | **实际部署** (Digit, 4英里徒步) |

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

### 9.1 完整版本迭代链

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

─── Schumacher 路线 (V10-V12): efficiency+pain→步态涌现 ───

V10 (大一统: 最小 reward, 步态涌现)
  → 灵感: Schumacher 2025 iScience
  → 移除 ref_joint_pos, feet_contact_number, feet_clearance 等
  → 结果: 0.95s 倒地，移除 stability 后无安全网

V10-a (efficiency 放大)
  → efficiency=-2e-4 (25,000×) → 训练稳定但步态质量差

V10-c (V10-a 调参)
  → 站立仍能存活但走路能力差

V11 (σ=0.08)
  → tracking_sigma 0.5→0.08 → 梯度完全消失
  → 662 iter 后 tracking≈0, 策略不学习
  → 教训: exp(-E/σ) 的 σ 过小 → 初始误差在梯度死区

V11-b (σ=0.2 修复)
  → tracking_sigma 0.08→0.2 → 梯度恢复
  → 站立 tracking 64%, 走路 100%, 差距 36%

V11-c (stability + landing 修复)
  → stability ×20→×10, ×100→×10 (降低陡度留梯度)
  → landing_impact 300→500N (减少冲突)
  → 结果: surrogate≈0, reward 横盘 72-80, std持续涨 → 正反馈循环
  → 根因: efficiency/landing 与 tracking 方向冲突

V12 (symmetry reward)
  → 新增 symmetry=1.0 (exp(-mirror_err/0.5))
  → tracking 2.5→2.5 (不变), sigma 0.2→0.3
  → 待验证

─── van Marum 路线 (V13): single_foot_contact→步态涌现 ───

V13 (Minimal Emergence) ⭐ 当前
  → 灵感: van Marum 2024 (OSU/Digit)
  → 新增 single_foot_contact(0.3) + feet_airtime(0.3)
  → tracking 2.5→1.0, sigma 0.3→0.25
  → 移除 symmetry/stability/efficiency/landing_impact
  → 新增 orientation(0.5) + base_height(0.2) + torque(0.01)
  → Task: TASK_20260612_050, commit 1be71f8
  → 早期(iter 4): mean_reward 0.71→1.06, all rewards 有有效梯度
```

### 9.2 已验证的失败模式 (Dead Ends)

| 方向 | 失败原因 | 版本 | 证据 |
|------|---------|------|------|
| CoT 能效 (τ·dq) | 对符号敏感，正负力矩抵消 | V2 | 训练崩溃 |
| walk_decay | 衰减 stability 导致姿态崩溃 | V5 | stability→0 |
| 效率太弱 (-8e-9) | 拖脚走无惩罚 | V3-V10 | 步态拖脚 |
| efficiency 主导 (-0.02) | 压制一切运动 | V2 | 策略不动 |
| swing_foot_forward 无上限 | 线性增长碾压 stability | V6 | stability 被压制 |
| σ 过小 (0.08) | 初始误差在梯度死区 | V11 | tracking≈0 |
| tracking 权重过高 (2.5) | 压制其他 reward 梯度 | V11-c | surrogate≈0 |
| efficiency/landing 与 tracking 冲突 | 方向对抗导致正反馈循环 | V11-c | reward 横盘 |
| symmetry (energy-phase) | 只检查能量平衡，split stance 得高分 | V15 | split=0.876 |
| symmetry (mirror) | 逻辑正确但整体方向冲突仍存在 | V16 | 待验证 |
| Schumacher 三件套整体路线 | efficiency+pain 方向冲突不可调和 | V10-V12 | 6轮失败 |

### 9.3 已验证的工作模式

| 模式 | 条件 | 增益 | 来源 |
|------|------|------|------|
| σ=0.2–0.3 | exp(-E/σ) 类 reward | 从梯度消失→有效梯度 | V11-b |
| 独立 reward 拆分 | 合并→拆出 | 梯度方向清晰 | V4 |
| single_foot_contact | walking cmd 时 n_contact==1 | 跳跃→行走涌现 | V13 (van Marum) |

---

## 10. 消融计划 (Ablation Plan)

| 决策变量 | 候选值 | 假设 | 优先级 |
|---------|--------|------|--------|
| single_foot_contact scale | [0.1, 0.3, 0.5] | 0.3 是 van Marum 0.1 的合理放大 | 高 |
| feet_airtime threshold | [0.2, 0.4, 0.6] s | 0.4s 可能需适配 X1 腿长 | 高 |
| tracking_lin_vel scale | [0.5, 1.0, 1.5] | 1.0 消除梯度压制 | 高 |
| tracking_sigma | [0.15, 0.25, 0.35] | 0.25 对齐 Berkeley HT-2 | 中 |
| single_contact_grace | [0.1, 0.2, 0.4] s | 0.2s 允许双支撑 | 中 |
| orientation scale | [0.2, 0.5, 1.0] | 0.5 与 tracking 平衡 | 低 |
| base_height scale | [0.1, 0.2, 0.5] | 0.2 温和维持高度 | 低 |

---

## 11. 设计风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| single_foot_contact 宽限期太长 | 中 | 策略学会"跳跃+偶尔接触" | 缩短 grace 到 0.1s |
| feet_airtime 0.4s 不适配 X1 腿长 | 中 | 步频过慢或惩罚过重 | 调整 threshold |
| X1 12DOF 比 Digit 复杂度高 | 中 | 涌现步态质量差 | 可适当加回 foot_slip |
| 移除所有惩罚后力矩过大 | 低 | 硬件风险 | torque 0.01 温和正则化 |
| sim-to-real gap | 高 | 仿真步态不能迁移 | $deploy 阶段处理 |

---

## 12. 当前状态 & 下一步

### 12.1 当前训练
- **Task**: TASK_20260612_050
- **Branch**: v9-smooth-landing
- **Commit**: 1be71f8
- **Config**: V13 "Minimal Emergence", 8 reward terms, 15k iter
- **Status**: 运行中 (iter 4, mean_reward 0.71→1.06, 所有 reward 有有效梯度)

### 12.2 验证清单
训练完成后，需要验证以下指标：

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| 步态涌现 | 交替行走，非跳跃 | CSV replay 目视检查 |
| single_foot_contact 奖励 | > 0.8 | reward 曲线 |
| 速度跟踪 | > 0.3 m/s 稳定 | tracking reward 曲线 |
| 站立稳定 | 站立时不漂移 | stand phase 分析 |
| Episode length | > 500 steps (10s) | mean_episode_length 曲线 |

### 12.3 调参路径
```
V13 结果:
  → 如果步态涌现 + 正常行走 → ✅ 成功，进入 $tune 精调
  → 如果步态涌现但步频太快 → 增大 airtime_threshold 到 0.6s
  → 如果仍然跳跃（无单脚接触） → 增大 single_foot_contact scale 到 0.5
  → 如果摔倒频繁 → 增大 orientation scale 到 1.0
  → 如果步态但力矩过大 → 加回 foot_slip 或增大 torque scale
```

---

## 13. 替代方案

### 方案 B: V12 基线（symmetry + stability 引导）
如果 V13 步态涌现失败，回退到 V12 的半工程化方案：
- 保留 symmetry=1.0, stability=1.0
- 但使用 V13 的 tracking scale (1.0) 和 sigma (0.25)
- 7 reward terms

### 方案 C: 强化 single_foot_contact
在 V13 基础上进一步强化步态约束：
- single_foot_contact scale 提升到 0.5-1.0
- 新增 feet_position reward（van Marum 的 exp(-3·\|p-c\|)）
- 适度加回 landing_impact (scale=-0.1, threshold=500N)

---

_Design committed: unified-reward_
_Last updated: 2026-06-12_
_Next: V13 训练结果验证 → 调参 → $tune_
