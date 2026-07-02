# 假设证伪分析 — V13 主线最可能被推翻的假设

_Created: 2026-07-02_
_Analysis mode: Red-team / pre-mortem (read-only, no code change)_
_Target: .oma/designs/design-unified-reward.md (V13 "Minimal Emergence")_
_Evidence: design doc + 实际源码 (x1_dh_stand_env.py, x1_dh_stand_config.py, legged_robot.py) + gait logs_

---

## 0. 执行摘要 (TL;DR)

V13 主线**最承重、且最可能被推翻**的假设是 **H1**：

> 一个简单的 single_foot_contact 检测（n_contact == 1）就足以把 X1 从"跳跃"切换到"交替行走"的步态涌现，无需对称性、效率、参考轨迹等任何工程化约束。

这是整个 V13 范式转变的支点（design §0/§1）。**本分析认为 H1 已经在实际迭代中被证伪，只是尚未在文档中承认。**

三条独立证据链共同指向这一结论：

1. **内部实验证据（最致命）**：实际 config 中残留的 V13b/c/d/e 调参注释明确记录了两个 H1 本应阻止的退化步态——**蹲着走（V13c, base_height 跌至 0.17 m，锁死）**和**扭胯走（V13e）**。修复它们靠的不是 single_foot_contact，而是反复加码的工程化手段（base_height 0.2→0.5、tracking 0.4→0.6、σ 0.25→0.15）。
2. **设计文档与实际代码系统性漂移**：design §2.2 宣称的 8 项 reward 权重，与 config 实际值在 5/8 项上不符；文档把 single_foot_contact 当作 0.3 的小步态项，实际它已被抬到 0.8 成为主导项。
3. **机制缺陷 + 缺失验证**：0.2 s 宽限期 = 20 个控制帧，使 single_foot_contact 成为"必要非充分"约束；且 repo 中**无任何 V13 步态回放数据**可佐证"步态涌现"，文档 §12.1 的乐观状态（"iter 4, all rewards 有有效梯度"）停留在 reward 数值，从未被步态目视验证。

**结论：应将 H1 降级为"已部分证伪"，并把 single_foot_contact 从"步态涌现的决定性 reward"重新定位为"仅排除双脚同时离地这一单一反例的弱约束"。**

---

## 1. 被审主线与证伪目标假设

### 1.1 主线设计
V13 "Minimal Emergence" 基于 van Marum et al. 2024 (OSU / Digit)。其范式转变（design §0）断言：

> tracking + orientation → 只产生跳跃
> + single_foot_contact → 产生行走（步态涌现！）
> 不需要 symmetry、不需要 efficiency、不需要 landing_impact、不需要参考轨迹。

### 1.2 目标假设 H1（最承重）
**H1**：single_foot_contact 是步态涌现的**决定性、自足的** reward——在 X1 上，仅靠它就能从"跳跃"局部最优切换到"正常交替行走"局部最优，无需额外约束。

这是证伪分析的正确靶点，因为：
- 它是 V13 区别于 V1–V12 的**唯一**核心理念。若 H1 假，V13 就退化为"另一套工程化 reward"，失去其设计正当性。
- design §3.1 将其列为"van Marum 的实验发现"，§9.3 将其列为"已验证的工作模式"——但**该"验证"来自 Digit 论文，非 X1 自身实验**。

---

## 2. 为什么 H1 是最承重的假设（负载分析）

逐项排除其他候选，确认 H1 的负载最高：

| 候选假设 | 承重度 | 证据强度 | 判断 |
|---------|-------|---------|------|
| **H1: single_foot_contact 单独驱动涌现** | **最高**（V13 范式支点） | **强**（V13c/e 失败已记录） | **证伪目标** |
| H2: feet_airtime 0.4 s 阈值适配 X1 | 中（仅步频项） | 中（design §11 已自述风险） | 次要 |
| H3: van Marum 路线"无梯度冲突" | 中 | 中（base_height/σ 调参显露出冲突） | 顺带 |
| H4: "完全不需要参考轨迹" | 低–中 | 中（critic 仍吃 ref_diff） | 附录 |

H1 一旦假，H2/H3/H4 的讨论失去前提；反之 H2 假不危及 H1。故 H1 是证伪分析的**唯一正确靶点**。

---

## 3. 证伪证据

### 3.1 致命证据：V13b/c/d/e 已记录的失败模式（来自实际 config 注释）

x1_dh_stand_config.py 的 class scales 残留了 V13 之后的真实迭代痕迹，与 design §12.1 的乐观状态直接矛盾：

| 现象 | 证据（config 注释原文） | 位置 | 对 H1 的含义 |
|------|----------------------|------|-------------|
| **蹲着走 + 锁死** | base_height = 0.5 (V13d: 0.2→0.5, 惩罚蹲低; V13c height=0.17, 策略蹲着走锁死) | line 367 | base_height 跌到 0.17 m（目标 0.61 m，**偏低 72 %**）。H1 声称的"涌现行走"坍缩成"蹲着挪动"。修复靠 base_height 加码，非 single_foot_contact。 |
| **扭胯走** | tracking_lin_vel = 0.6 (V13e: 0.4→0.6, 增大推力让策略放弃扭胯走) | line 358 | 策略学会扭胯前进（degenerate gait）。 |
| **single_foot_contact 无法区分扭胯** | tracking_sigma = 0.15 (V13e: 0.25→0.15, 拉大扭胯(0.81) vs 正常走(1.0)的 reward 差距) | line 333 | **最关键**：扭胯步态拿 single_foot_contact **满分**，唯一能区分它的是 tracking 的 σ 调参。 |
| **single_foot_contact 原始权重失效** | single_foot_contact = 0.8 (V13c: 1.0→0.8, 二值 reward 密度修正后与 tracking 平衡) | line 361 | design 写 0.3，实际经历过 0.3→1.0→0.8——说明 design 值根本不足以驱动涌现。 |
| **探索不稳定** | entropy_coef = 0.003 (V13d: 0.001→0.003, 恢复探索; V13c 0.001→noise_std 0.34 冻结; V13b 0.005→4.69 暴涨) | line 411 | "最小涌现"对 entropy 极度敏感（0.001 冻结 / 0.005 暴涨），并非"最小"。 |

**关键推论（来自 line 333 注释）**：扭胯步态在 single_foot_contact 上得 **1.0 满分**，正常行走也得 1.0——**single_foot_contact 对两者零区分度**。真正把策略从扭胯里拉出来的是 tracking_sigma 0.25→0.15 的收紧——而注释中的 0.81/1.0 正是 **tracking_lin_vel 取值**（exp 形式 0–1）：即扭胯步态仍能拿 0.81 的速度跟踪分、在 single_foot_contact 上又拿满分，故唯一可用的选择信号只剩 tracking 的 σ。换言之，H1（"一个简单单脚接触检测就够了"）在 X1 上**已被自身的迭代实践否定**：决定步态质量的是 tracking 的工程化调参，不是 single_foot_contact——一个对所有步态都给满分的 reward 项，**结构上无法充当选择信号**。

### 3.2 系统性漂移：design 文档值 ≠ 实际 config 值

| Reward 项 | design §2.2 / §1.2 宣称 | config 实际值 | 漂移 |
|----------|----------------------|-------------|------|
| tracking_lin_vel | 1.0 | **0.6** (line 358) | 0.6× |
| single_foot_contact | 0.3 | **0.8** (line 361) | **2.67×**（且经 0.3→1.0→0.8） |
| base_height | 0.2 | **0.5** (line 367) | 2.5× |
| tracking_sigma | 0.25 | **0.15** (line 333) | 0.6×（收紧） |
| orientation | 0.5 | 0.5 (line 365) | 一致 ✓ |
| feet_airtime | 0.3 | 0.3 (line 363) | 一致 ✓ |
| torque | 0.01 | 0.01 (line 369) | 一致 ✓ |

8 项中 5 项漂移。design 把 single_foot_contact 描述为"0.3，van Marum 0.1 的合理放大"的小步态项；实际它已是**主导项**。这意味着 design 文档描述的"低压 tracking + 弱步态约束"架构与真实代码的"single_foot_contact 主导 + tracking 工程化收紧"架构已经**根本不同**——文档在描述一个并不存在的系统。

> 另注：design §4.3 称 "Control freq: 50 Hz (decimation=10, dt=0.001)"，但 legged_robot.py:1270 self.dt = decimation × sim_dt = 10 × 0.001 = 0.01 s → 实际 **100 Hz**。文档的基础物理参数也有误。

### 3.3 实现机制缺陷：grace 窗口稀释约束力

_reward_single_foot_contact（x1_dh_stand_env.py:876-913）：

```
single_now = (n_contact == 1)
single_grace = torch.max(self.single_contact_history, dim=1).values > 0.5
r = where(stand_cmd, 1, where(single_now | single_grace, 1, 0))
```

定量分析（含单位）：
- 控制频率 = 100 Hz → self.dt = 0.01 s（legged_robot.py:1270）。
- grace_frames = int(0.2 s / 0.01 s) = 20 帧（x1_dh_stand_env.py:119）。
- 步态周期 cycle_time = 0.9 s；单腿支撑相 ≈ 0.45 s。
- **宽限期 0.2 s = 单腿支撑相的 44 %**。
- 后果：策略只需在每 20 个控制帧内出现**任意一帧** n_contact==1，single_foot_contact 即饱和为 1.0。这是一个极弱约束——扭胯、低抬腿、甚至间歇拖脚都能轻松满足。接触阈值仅 5.0 N（line 891），任何轻微单脚承重即计数。

→ single_foot_contact 实际退化为"最近 0.2 s 内脚没全离地"的检测器，**而非"持续交替单脚支撑"**。它只能排除"双脚同时离地（跳跃）"这一个反例，对扭胯/蹲走这类"双脚基本不离地"的退化步态毫无约束力——与 §3.1 的 V13c/e 失败完全自洽。

> 权重细节（含单位，供 rigor）：compute_reward 对每项乘以 self.dt = 0.01（legged_robot.py:999）。有效每步权重：single_foot_contact 0.008 > tracking_lin_vel 0.006 > orientation = base_height 0.005 > feet_airtime 0.003 > torque 1e-4。即便如此主导，它仍阻止不了扭胯——再次说明"主导 ≠ 决定性"。

### 3.4 缺失验证：无 V13 步态回放数据

compute_reward（legged_robot.py:362-368）通过遍历非零 scale 的 reward 函数生成 rew_ 列。检查全部 9 份 gait_logs（gait_20260605 至 gait_20260616）：

- 每一份都只含 **旧 reward 列**：rew_stability, rew_swing_foot_forward, rew_feet_contact_number, rew_feet_clearance, rew_efficiency, rew_collision。
- **无任何一份**含 V13 列（rew_single_foot_contact, rew_feet_airtime, rew_orientation, rew_base_height, rew_torque）。

结论：repo 中**不存在 V13 步态涌现的目视证据**。design §12.1 的"iter 4, mean_reward 0.71→1.06, all rewards 有有效梯度"仅是 reward 标量早期趋势，而 config 注释（V13b–e，晚于 iter 4）已显示真实轨迹是蹲走→扭胯→反复调参。"步态涌现"在 X1 上**从未被验证**，已被间接证伪。

---

## 4. 机制根因：为什么 single_foot_contact 在 X1 上无法复现 Digit 的涌现

### 4.1 single_foot_contact 是"必要非充分"约束
van Marum 的涌现效应依赖一个前提：在 Digit 上，前进的**最低成本违规路径**是"双脚同时离地的跳跃"。single_foot_contact 精准封死了这唯一捷径，故策略被迫落入行走局部最优。
但在 X1 上，前进的低成本违规路径**不止跳跃**：扭胯走、蹲着挪动、低抬腿拖步都满足 n_contact==1（或经 0.2 s 宽限期内偶尔满足）。single_foot_contact 对这些捷径**零约束力**。X1 的局部最优景观比 Digit 宽，单一约束不足以收窄到"正常行走"。

### 4.2 Digit→X1 形态学迁移假设未被检验
- van Marum 在 Digit（约 1.6 m 全尺寸人形）验证；X1 的 base_height_target = 0.61 m，是更小形态。
- design §11 仅把风险表述为"X1 12DOF 比 Digit 复杂度高"——**方向可能搞反**：X1 腿短，自然摆动相更短，0.4 s 的 feet_airtime 阈值（H2）很可能惩罚正常步频；同时更易触发蹲走局部最优（V13c 已验证）。
- design §9.3 把 single_foot_contact 列为"已验证工作模式"，证据列却是"V13 (van Marum)"——**引用的是论文平台，不是 X1 实验**，属循环论证。

### 4.3 终止条件的潜在漏洞（需进一步确认）
check_termination（legged_robot.py:206-223）在 root z < 0.35 m 时复位。但 V13c 注释记录 base_height 跌至 0.17 m（≈ root z，偏低 72 %）。两种可能：(a) V13c 当时终止阈值不同/被旁路；(b) 策略被频繁复位却仍收敛到蹲走局部最优（即每次 reset 都快速跌回蹲姿）。无论哪种，都说明"无安全网的最小 reward 集"无法阻止姿态坍缩——正是 V10 当初被批评的同一缺陷（design §0：移除 stability 后无安全网）。

---

## 5. 证伪判据（可证伪性 / 何时确认 H1 彻底死亡）

为避免主观判断，给出**明确的量化确认判据**。任一条满足即确认 H1 在 X1 上为假：

| 判据 | 阈值 | 现状 |
|------|------|------|
| C1：完整 V13 训练（≥15 k iter）后，CSV 回放显示交替步态 | 左右 contact 交替占主导、非扭胯 | **未达**（无 V13 回放） |
| C2：扭胯/蹲走退化步态消失，且**无需**收紧 σ / 加码 base_height | single_foot_contact 单独即可 | **已违反**（V13e 靠 σ 收紧） |
| C3：single_foot_contact reward 与步态质量（对称性、抬脚高度）正相关 | 相关系数 > 0.5 | **反向**（扭胯拿满分） |
| C4：设计文档权重（§2.2）可直接运行出涌现步态 | 一次成型 | **已违反**（需 5/8 项改值） |

当前 4 条判据中 C2、C4 已被违反，C3 反向，C1 待验。按证据强度，**H1 应判定为"部分证伪（preponderance of evidence），待 C1 终判"**。

---

## 6. 影响与建议

### 6.1 直接影响
- 若 H1 假，V13 的"最小涌现"叙事不成立：single_foot_contact 只是个弱过滤项，真正塑造步态的是 tracking σ、base_height 等被 design 宣称"不需要"的工程化项。
- design §9.3 "已验证工作模式"表中的 single_foot_contact 行应标注"仅 Digit 验证，X1 未成"。
- design §12.3 调参路径的首选（"步态涌现 → 进入 $tune"）前提不成立，需先解决退化步态。

### 6.2 建议的下一步（最小动作，优先级排序）
1. **跑一次 V13 gait replay（CSV）并目视判定 C1**——这是当前唯一能一锤定音的实验，且成本最低。若无交替步态，H1 终判为假。
2. **承认并固化漂移**：把 config 实际权重回写 design §2.2，或反向把 config 改回 design 值做对照实验，消除"文档描述一个不存在的系统"的状态。
3. **重定位 single_foot_contact**：从"决定性涌现项"降级为"弱约束"，并评估是否需要把 design §13 方案 C（加回 symmetry / foot_slip）或 V16 镜像对称作为主约束——注意 memory 已记录 symmetry 四版失败（V13–V16），需先诊断为何 mirror symmetry 也未能阻止扭胯。
4. **若坚持 van Marum 路线**：缩短宽限期到 0.1 s（design §11 自述缓解方案）并去掉二值 grace（改用连续 single-foot ratio），做对照消融——验证是"宽限期稀释"还是"约束本身必要非充分"。

### 6.3 不建议的方向
- 不要在未验证 C1 的情况下盲目进入 $tune 精调——reward hacking（蹲走/扭胯）会在精调中被放大而非消除（memory 警示：高奖励若缺物理可信度即死路）。

---

## 7. 附：次要假设（顺带记录，不展开）

- **H2（feet_airtime 0.4 s）**：X1 腿短，自然摆动相可能 < 0.4 s → 对正常步频施加负奖励。design §11 已自述风险，建议 C1 回放时同时看 swing phase 占空比。
- **H3（"无梯度冲突"）**：V13c 靠 base_height 加码、V13e 靠 σ 收紧，本身就是 reward 间博弈的痕迹，design §3.2 "无冲突"的对比表已被实践推翻。
- **H4（"完全不需要参考轨迹"）**：compute_observations（x1_dh_stand_env.py:384, 396-403）仍每步计算完整 ref 轨迹并喂给 critic（ref_diff 12 dim），compute_ref_state 仍生成 sin/cos 相位 + V9 非对称摆动轨迹。虽然 use_ref_actions=False、ref_joint_pos scale=0，但 critic 仍隐式依赖参考相位信息。design §2.3 "完全不需要参考轨迹"与代码不符。

---

## 附录 A：证据溯源（行号索引）

| 证据 | 文件与行 | 类型 |
|------|--------|------|
| H1 原文（范式支点） | design-unified-reward.md §0/§1 | 文档 |
| single_foot_contact 实现 | x1_dh_stand_env.py 876-913 | 源码 |
| grace_frames = 20 计算 | x1_dh_stand_env.py 118-120 | 源码 |
| 接触阈值 5.0 N | x1_dh_stand_env.py 891, 928 | 源码 |
| V13c 蹲走失败 (base_height→0.5) | x1_dh_stand_config.py 367 | 源码注释 |
| V13e 扭胯失败 (tracking→0.6, σ→0.15) | x1_dh_stand_config.py 333, 358 | 源码注释 |
| single_foot_contact 0.3→1.0→0.8 | x1_dh_stand_config.py 361 | 源码注释 |
| entropy 不稳定 | x1_dh_stand_config.py 411 | 源码注释 |
| 实际权重 (×dt) | legged_robot.py 999, 362-367 | 源码 |
| self.dt = 0.01 s (100 Hz) | legged_robot.py 1270 | 源码 |
| 终止条件 root z < 0.35 m | legged_robot.py 206-223 | 源码 |
| reward 列动态生成 | legged_robot.py 351-368 | 源码 |
| 9 份 gait_logs 全为旧列 | gait_logs 内 CSV 头部 | 数据 |
| symmetry 四版失败 | design-unified-reward.md §9.2; memory | 文档/记忆 |

## 附录 B：分析假设与局限

- 本分析为**只读** pre-mortem，未运行任何训练/仿真，所有结论基于源码静态分析 + config 注释。
- config 注释（V13b–e）是对已发生实验的二手记录，其准确度依赖原作者注释质量；最稳妥的终判仍是 C1（实跑 V13 回放）。
- 形态学差异（Digit vs X1）的定量论证（§4.2）为推断，未实测 X1 自然步频。
- 终止条件与蹲走存活的交互（§4.3）未能从 V13c git 历史完全确认机制，已标注为"需进一步确认"。

_分析人: Meta-Agent (exploration refutation)_
_复核 (2026-07-02 read-only re-verify): 全部证据已据源码逐条核对 —— self.dt = 0.01 s（100 Hz）@legged_robot.py:1270（纠正 design §4.3 的"50 Hz"）；reward×dt @:999；height 终止 0.35 m @:223；contact 阈值 5.0 N @env:891；9 份 gait CSV 表头均无 V13 reward 列。H1 状态确认为"部分证伪，待 C1（V13 gait replay）终判"。_
_状态: H1 判定为"部分证伪"，建议执行 C1（V13 gait replay）做终判_
