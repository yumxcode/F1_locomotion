# Exploration Investigation: 未尝试的步态/控制机制候选方向

_Project: agibot_x1_train_oma (X1 双足行走)_
_Investigator: 子代理 (exploration 调研)_
_Date: 2026-07-02_
_Scope: 调研尚未出现在 `directions_tried` 的步态/控制机制 + 最新论文思路，列候选方向与依据。写域限定本文件。_

> **重要声明 — `paper_search` 不可用**
> 本会话未提供 `paper_search` 工具（PATH 上也不存在该 CLI）。因此本文引用的论文来自：(a) 设计文档已确认的参考文献；(b) 调研者对双足/四足 RL 步态文献的既有知识。**所有论文标题/作者/年份均为待核实**——进入 `$design`/`$tune` 前，请用 `paper_search` 复核每一条引用，确认其结论仍适用于当前 X1 设定。候选方向本身的价值不依赖单一引用，证据已由本仓库源码与 `gait_logs` 实测数据独立支撑。

---

## 0. 执行摘要 (TL;DR)

1. **发现一个已被实测坐实、但 `directions_tried` 从未触及的失败模式：扭胯走（hip-yaw 为主驱动的骨盆扭转步态）。** 在 V13e 最新 4 个 `gait_log` 中，hip-yaw 摆幅是 hip-pitch（迈步）摆幅的 **2.0×–3.0×**；左髋 yaw 关节相对默认位偏移峰值达 **3.42 rad（≈110°）**，而当前所有 reward 项（tracking / single_foot_contact / feet_airtime / orientation / base_height / torque）**无一约束髋 yaw / 腿外展**。这是 `single_foot_contact` 涌现路线的典型 reward hack——策略用"扭转骨盆 + 单脚始终接地"满足所有奖励，却不真正迈步。
2. **最高优先级、最低风险、零梯度冲突的候选：髋 yaw 外展正则化（hip-yaw / leg-abduction penalty）。** 它直接惩罚被 hack 的那个量，与现有 reward 正交，scale 估算清晰（见 §2.1）。
3. **第二优先级：足部位置 / 站立宽度奖励（feet_position, van Marum 设计中列为"方案C"但从未落地）**——从几何角度堵同一条漏洞。
4. **结构性新范式（未尝试）：自适应步频 gait clock（CPG-RL）、AMP/DeepMimic 运动先验、质心角动量(CAM)正则、动作 jerk 正则**。
5. **历史教训直接相关的避坑**：reward-based symmetry 已 4 次失败（V12/V14/V15/V16），不要再从"奖励"角度碰对称性；若要做对称，应走"对称策略结构化"（权重共享/等变网络）这条未走过的路。

---

## 1. 方法论与证据

### 1.1 `directions_tried` 已穷尽方向（边界快照）

综合 `design-unified-reward.md` §9（V1→V13 迭代链）、§9.2（Dead Ends）、§2.3（显式移除项）以及 `x1_dh_stand_config.py` 的 `scales` 注释，**已经尝试过**的 reward/控制机制如下（用于排除，避免重复造轮子）：

| 类别 | 已尝试机制 | 代表版本 | 状态 |
|------|-----------|---------|------|
| 能效 | CoT `|τ·q̇|`、τ²、dof_vel/acc penalty | V2/V3/Track-A | Dead End（符号敏感/拖脚） |
| 步态轨迹 | ref_joint_pos 参考轨迹跟踪、feet_clearance、feet_contact_number(相位对齐)、feet_air_time(相位版) | V1–V10 | V13 全部移除 |
| 落地 | landing_impact(GRF)、landing_compliance、heel-toe、sin² 下降 | V8/V9/V11-c | 与 tracking 冲突，移除 |
| 前进 | swing_foot_forward(±cap)、forward_propulsion 补偿 | V3/V6/V7 | 被移除/合并 |
| 稳定性 | stability(orientation+height 合并)、base_acc、default_joint_pos、feet_rotation | V12/Track-A | 拆分或移除 |
| 对称 | symmetry × 4 版（镜像差/能量相位/anti_phase/morphological mirror） | V12/V14/V15/V16 | **4 次全失败** |
| 涌现 | single_foot_contact、feet_airtime(van Marum) | V13 当前 | 涌现了，但步态退化 |
| 姿态/高度 | orientation、base_height（独立） | V13 | 当前在用 |
| 力矩 | torque `exp(-Σ|τ|/100)` | V13 | 当前在用（弱） |
| 速度 | split velocity、vel_mismatch、low_speed、track_vel_hard、stand_still | V3–V5 | 合并为 tracking |
| 滑步 | foot_slip、feet_distance、knee_distance | Track-A | V13 scale=0 |
| 安全 | collision、dof_pos/vel/torque_limits | 全版本 | dof_pos_limits 仍在 |
| 控制 | 固定 cycle_time=0.9 s 正弦相位时钟、速度 curriculum(3 段)、gait-type 混排、域随机化(摩擦/质量/COM/增益/力矩/lag/关节阻尼/armature/电机偏置/库仑)、push 扰动、pitch/roll/height 终止 | 全版本 | 当前在用 |
| 观测 | 长历史 CNN(66 帧)、特权 critic+状态估计器、sin/cos 相位、IMU | 全版本 | 当前在用 |

**关键缺口（以下均不在 `directions_tried` 中，是本文候选重点）：**
髋 yaw/腿外展正则、足部位置奖励、步长奖励、自适应步频、运动先验(AMP/DeepMimic)、质心角动量正则、动作 jerk 正则、对称策略结构化、地形 curriculum、奖励权重自动 curriculum(PBT)。

### 1.2 新坐实的失败模式：扭胯走（hip-yaw 主导）

对最近 4 个 `gait_logs/*.csv`（仅取 `cmd_vx > 0.1 m/s` 的前进帧）做定量统计：

| 日志 | `hip_yaw_amp / hip_pitch_amp` | 备注 |
|------|------------------------------|------|
| 0610_175841 | **2.03** | yaw 摆幅 1.9 rad vs pitch 0.95 rad |
| 0611_131931 | **2.62** | base_yaw pp=0.64 rad，明显扭转向 |
| 0612_101906 | **2.24** | |
| 0616_135100 | **2.98** | 最新模型，最严重 |

最新模型 0616 的关节偏移（相对默认位）：

| 关节 | 默认值 (rad) | 均值 (rad) | 峰值偏移 (rad) | 峰值偏移² (rad²) |
|------|------------|-----------|---------------|-----------------|
| left_hip_yaw | −0.31 | +2.204 | **3.42** | **11.7** |
| right_hip_yaw | +0.31 | +0.716 | 0.81 | 0.66 |
| left_hip_roll | +0.05 | −0.167 | 0.43 | 0.18 |
| right_hip_roll | −0.05 | −0.199 | 0.29 | 0.08 |

**解读**：左髋 yaw 被驱动到距默认 **3.42 rad（≈110°）**——脚几乎横着伸出，靠扭转骨盆+单脚接地前移。hip-roll 偏移小（<0.5 rad），故是 **yaw 主导，不是 roll 外八**。脚抬升 0.087–0.091 m（>target 0.05 m），所以**脚有在抬，问题不在 clearance，而在水平推进被 yaw 接管**。

**为什么现有 reward 拦不住**：tracking 只看 base 线速度（任何方式达到都给分）；single_foot_contact 只数接触脚数；orientation 只罚 pitch/roll（不管 yaw，不管腿外展）；base_height/torque 都不触及 yaw。→ 策略找到"扭胯"这条满足全部奖励的捷径。这是 van Marum 最小涌现路线在 12-DOF 全向 X1 上的**结构化漏洞**。

> 副作用记录：config 注释还显示另一个涌现 hack——"蹲着走"（V13c height=0.17 锁死），已用 base_height 0.2→0.5 部分压制。`orientation` 用 `projected_gravity[:2]`（pitch+roll），未管 yaw。

---

## 2. 候选方向（未尝试，分级）

每项给出：**机制 / 依据 / 实现(公式·单位·scale 估算) / 与现有 reward 冲突分析 / 风险**。

### Tier 1 —— 直击已坐实失败模式（最高价值，应优先实验）

#### 2.1 ★★★ 髋 yaw / 腿外展正则化（hip-yaw abduction penalty）
- **机制**：直接惩罚髋 yaw（及可选 roll）偏离默认位，把"扭胯"这条 hack 路径堵掉。
- **依据**：实测 §1.2——左髋 yaw 峰值偏移 3.42 rad；humanoid RL 普遍把"自然关节构型先验"作为基本约束（脚应落在髋下而非甩到外侧）。
- **实现**：
  ```python
  # 关节索引: left_hip_yaw=2, right_hip_yaw=8 (0-based, 6/leg)
  yaw_dev = (dof_pos[:,2]-default[:,2])**2 + (dof_pos[:,8]-default[:,8])**2   # rad^2, [N]
  # 可选并入 roll: + (dof_pos[:,1]-default[:,1])**2 + (dof_pos[:,7]-default[:,7])**2
  reward = -yaw_dev      # 二次惩罚(护轨型)
  ```
  - **scale 估算（带单位）**：峰值 yaw_dev² ≈ 11.7 rad²（单腿，0616）。期望在 3.42 rad 极端处产生与 single_foot_contact(0.8×1=0.8) 同量级的惩罚（≈−1.0）。→ `scale ≈ −1.0/11.7 ≈ −0.085`，取 **scale = −0.08**。
  - 验算：偏移 1.0 rad → 惩罚 −0.08（温和，不压制正常步态的轻微 yaw）；偏移 3.42 rad → −0.94（强力护轨）。渐进式，符合"软约束"。
- **冲突分析**：**零梯度冲突**。tracking 只管 base 速度，不要求 yaw；与 single_foot_contact/feet_airtime 正交。这是它最大的优点——不像 efficiency/landing 那样与 tracking 抢方向。
- **风险**：低。唯一需注意 yaw 在"原地转向 cmd(wz)"或"侧向走 cmd_y"时本就该动——可加 `mask = has_forward_cmd`（`|cmd[0]|>0.05`），或仅在前进命令时启用，避免压制合理的转弯/侧步。
- **优先级**：🔴 P0，强烈建议作为 V14 的首个改动。

#### 2.2 ★★★ 足部位置 / 站立宽度奖励（feet_position, van Marum "方案C"）
- **机制**：从几何角度——脚应在髋下方（限制站立宽度），并向前迈过质心（防 shuffle）。`design-unified-reward.md` §13 方案C 列了 van Marum 的 `exp(−3·|p−c|)` 但**代码从未实现**（属"已知但未试"）。
- **依据**：van Marum 2024 的最小集实际包含 feet_position；本设计当时简化掉了。可与 §2.1 互补（一个罚关节角、一个罚末端足位）。
- **实现**：
  ```python
  # 脚相对 base 的横向偏移 (限制外八/劈腿)
  foot_y = rigid_state[:, feet_indices, 1] - root_states[:,1]          # m, [N,2]
  stance_width_pen = torch.sum(torch.clamp(torch.abs(foot_y)-W_hip, min=0)**2, dim=1)  # m^2
  # W_hip ≈ 髋宽/2 ≈ 0.10 m (需按 URDF 核实)
  reward = -stance_width_pen
  ```
  - **scale 估算**：劈腿时 `foot_y` 可达 ≈0.25 m，超出 W_hip≈0.10 m 的部分 ≈0.15 m，平方 ≈0.0225 m²；`scale = −5.0` → 惩罚 ≈−0.11（温和护轨）。可调到 −10 加大力度。
- **冲突分析**：低。与 tracking 弱相关（脚位不直接决定 base 速度），与 single_foot_contact 互补。
- **风险**：低。需核实 X1 髋宽与合理站立宽度阈值。
- **优先级**：🔴 P0，与 2.1 二选一或组合。

#### 2.3 ★★ 步长 / 前迈足速奖励（anti-shuffle, 重设计版）
- **机制**：要求摆动脚在空中向前移动一定距离/速度，压制"原地震荡+扭胯"的 shuffle。
- **区别于已试**：`swing_foot_forward`（V6/V7）只做"前进动力"且被 cap/移除；本项是**最小步长门槛**（foot 位移下限），是不同的失败维度。
- **实现**：在 first_contact（落地瞬间）统计本次摆动脚的 `Δx_foot`，reward = `clamp(Δx_foot − L_min, 0)` × first_contact；`L_min ≈ 0.05 m`（低速）/ 与 cmd 同步缩放。
- **冲突**：中低。与 tracking 略同向（都鼓励前进），不会反向。
- **风险**：中。L_min 设太大可能在低速命令时过严——务必与 cmd 速度耦合。
- **优先级**：🟡 P1。

---

### Tier 2 —— 结构性新范式（未尝试，中等价值，需更大改动）

#### 2.4 ★★ 自适应步频 gait clock（CPG-RL / 频率自适应）
- **机制**：当前 `cycle_time` **固定 0.9 s**，相位时钟是开环正弦。固定周期可能正是"扭胯走"的诱因之一——命令速度高时，0.9 s 的固定步频让策略无法用正常迈步达到速度，被迫改用 yaw 扭转。改为可学习/随速度自适应的步频。
- **依据（待核实）**：Bellegarda & Ijspeert, *"CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion"* (2022, IEEE RA-L)；Choi et al., context-aware CPG 双足 (2023)。这类工作让步频随速度/地形自适应。
- **实现**：把 `cycle_time` 从常量改为策略头输出的标量（或由 state-estimator 估计），相位积分用自适应频率。obs 已含 sin/cos 相位编码，改动集中在 `_get_phase`。
- **冲突**：低（结构改动，不改 reward 方向）。
- **风险**：中高。改动触及 obs/动作空间与训练稳定性；建议在 §2.1 见效后，作为质量提升而非救火。
- **优先级**：🟡 P1（在步态已正常后再上）。

#### 2.5 ★★ 运动先验 AMP / DeepMimic
- **机制**：用判别器从参考运动（哪怕粗糙的离线轨迹/人类 MoCap 重定向）学一个"人形步态先验"，替代手工 reward 调出"自然性"。
- **依据（待核实）**：Peng et al. *"AMP: Adversarial Motion Priors"* (2021, ACM TOG)；*"DeepMimic"* (2018, ACM TOG)。直接解决"涌现但难看/不自然"。
- **实现**：加一个 discriminator（reward = style reward + 任务 reward）；需参考运动数据。X1 若无可重定向的 MoCap，可用早期"好步态"的 rollout 轨迹当先验。
- **冲突**：与任务 reward 需仔细平衡权重（AMP 失衡会压制任务）。
- **风险**：高（数据依赖 + 训练不稳 + 显著增算力）。
- **优先级**：🟢 P2（重武器，留作 sim-to-real 阶段的自然性兜底）。

#### 2.6 ★★ 动作 jerk（二阶动作平滑）正则
- **机制**：惩罚动作的二阶差分（jerk），抑制颤抖/高频抖动。`action_smoothness`（一阶 Δa）曾在 Track-A，但 V13 移除了**所有**平滑项；二阶 jerk 从未单独尝试。
- **实现**：`penalty = Σ((a_t − 2a_{t−1} + a_{t−2})/dt)²`；`scale ≈ −1e-4`（按 V7 action_smoothness −0.002 的二阶版降一个量级起估，需实测标定）。
- **冲突**：低。纯正则，与任务方向不抢。
- **风险**：低，但收益视是否真有颤抖而定——可与 §1.2 数据交叉验证（当前 logs 未单独记录动作，需补）。
- **优先级**：🟡 P1（低成本保险项）。

#### 2.7 ★★ 质心角动量(CAM)正则，尤其 yaw 分量
- **机制**：最小化全身角动量，特别是绕 z 轴（yaw）分量——从动量角度压制扭胯（扭胯必伴随大 yaw 角动量）。
- **依据（待核实）**：trajectory optimization 与 quadruped/biped RL 普遍用 CAM 最小化（如 quadruped 整体控制类工作）；MIT/CMU humanoid whole-body control。它是 §2.1 在动力学层面的镜像。
- **实现**：`L_z = Σ_b r_b × (m_b·v_b)`（绕质心 z 分量）；reward = `−|L_z|/L_ref`，`L_ref` 按质量×尺度估（X1 ≈20 kg，腿长 ≈0.6 m，参考 L_ref ≈1 kg·m²·s⁻¹，需标定）。
- **冲突**：低（与 tracking 正交，与 §2.1 互补）。
- **风险**：中。计算量稍大；需物理一致的角动量估计。
- **优先级**：🟡 P1。

---

### Tier 3 —— 鲁棒性/工程增强（不直接修步态质量，但提升 sim-to-real）

#### 2.8 ★ 自适应地形 curriculum
- 现状：`terrain.curriculum = False`，地形比例固定。开启成功驱动的难度提升是未尝试的标准鲁棒性手段。不影响步态 hack 本身。

#### 2.9 ★ 侧向 base 振荡惩罚（直线命令时）
- 直线前进命令时罚 `|base_vy|` 与 `|base_wz|` 的抖动，间接抑制扭胯带来的横向/偏航晃动。低风险。

#### 2.10 ★ 奖励权重自动 curriculum（PBT / 自动权重）
- V13b→e 已手调 4 轮，仍在 reward 权重上打转。用 Population-Based Training 或自动权重 curriculum 把"找 scale"自动化，减少人工试错。依据（待核实）：Jaderberg et al. PBT (2017)。

#### 2.11 ★ 对称策略结构化（**不是 reward**）
- 历史教训：reward-based symmetry 4 次失败。改走**结构对称**——左右腿权重共享/等变网络，让"左右交替"成为归纳偏置而非奖励。依据（待核实）：Heess 2017 (rich env) 的对称运动；等变 RL 系列。属未走过的路，风险中等、收益不确定，列为探索项。

---

## 3. 推荐实验序列

依据"先堵漏洞、再提质量、最后重武器"，且严格遵循 OMA 原则（先读码算数再动手，§2.1/§2.2 已给单位化 scale 估算）：

1. **P0 — V14a：髋 yaw 外展正则（§2.1，scale −0.08，带 has_forward_cmd mask）**
   预期：`hip_yaw_amp/hip_pitch_amp` 从 2–3× 回落到 <0.5×；若 base_vx tracking 同时不掉，即确认扭胯被堵。**单变量改动，便于消融归因。**
2. **P0 — V14b（若 §2.1 不足）：叠加 feet_position/站立宽度（§2.2，scale −5 ~ −10）**
   从末端几何二次约束。与 §2.1 形成关节+足位双层护轨。
3. **P1 — V15：在步态已正常后上自适应步频（§2.4）或步长奖励（§2.3）**，提升步态质量/速度覆盖。
4. **P1 — 旁路：动作 jerk 正则（§2.6）/ CAM_z 正则（§2.7）**，低成本保险，可与上述并行小规模 ablation。
5. **P2 — sim-to-real 阶段：AMP 运动先验（§2.5）** 兜底自然性；PBT（§2.10）自动调权。

**消融纪律**：每步只改一个变量，记录 `hip_yaw_amp/hip_pitch_amp`、`single_foot_contact` 均值、tracking 均值、episode_length 四项，与 V13e baseline 对比。

---

## 4. 待核实引用清单（进入 `$design` 前用 `paper_search` 复核）

| 候选方向 | 引用（待核实） | 核实重点 |
|---------|--------------|---------|
| 自适应步频 | Bellegarda & Ijspeert, CPG-RL (2022); Choi et al. context-CPG 双足 (2023) | 双足频率自适应是否优于固定周期 |
| AMP/DeepMimic | Peng et al. AMP (2021); DeepMimic (2018) | 无 MoCap 时能否用自生成轨迹当先验 |
| CAM 正则 | quadruped/biped whole-body control 系 | 双足 yaw-CAM 最小化的标准实现与权重 |
| PBT 自动权重 | Jaderberg et al. (2017) | reward 权重自动搜索在 PPO 单卡 4090D 的可行性 |
| 等变/对称策略 | Heess 2017; Equivariant RL 系列 | 左右权重共享对 12-DOF 的工程成本 |
| feet_position | van Marum 2024 原文 | `exp(−3·|p−c|)` 的精确形式与权重 |

_本文件仅做候选方向调研，不修改任何源码或 .oma 状态。_