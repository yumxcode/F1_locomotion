# X1 行走控制：两次 Loop 研究历史合并总结

> 合并了两次自主研究 loop 的全部进展与结论，供新 graph loop 继承。
> Loop v1：x1-walking-control-v1（2026-07-09 ~ 07-11，14 轮结算，约 $10.9）
> Loop v2：x1-walking-research-v2-v1（2026-07-13 ~ 07-14，3 轮结算 + 1 轮停泊，约 $2.1）
> 训练仓库 github.com/yumxcode/F1_locomotion，分支 v9-smooth-landing；远端 Gradmotion 训练。

## 0. 当前状态快照（新 loop 的起点）

- **目标**（同一训练 epoch 同时达成）：single_contact > 0.8、前向速度 > 0.3 m/s、
  zero_contact < 0.15、episode > 500 步、步态物理可信（无 bounce/hip-twist/yaw-cheat）。
- **最好成绩**：cycle_time=1.0 基线上已有 123 个数据点同时满足 single>0.8 ∧ fwd>0.3
  （v2-R1）；swing_foot_forward 实验的最优窗口 iter7000-8000 达 single 0.800 /
  fwd 0.664 / zero 0.159——三目标几乎全部命中，只差把系统"停"在这个窗口。
- **当前范式判断**：标量化奖励工程范式已把 Pareto 墙撞了三轮，v2 轮 4 已转向
  **参考步态模板范式**（use_ref_actions 残差策略 + ref_joint_pos DeepMimic 式跟踪，
  基线回退到干净的 cycle_time=1.0）。
- **进行中实验**：TASK_20260714_134（reference-gait-template）已提交 Gradmotion，
  loop 停在等待收割。**开新 loop 前先 `gm task info/data get` 收割或终止它**——
  它是"范式转变是否让步态更早建立"的直接证据。

## 1. 两次 loop 各自做了什么（一句话版）

**v1（14 轮）**：从"bounce 局部最优"出发，逐步证明问题在 reward landscape 而非
优化器；发明了交替接触吸引子、超线性门槛 bonus、步态时钟相位跟踪、速度耦合时钟
四个关键机制，三目标曾在单一窗口内同时达成，最后精确刻画出"静态 reward 配比
无法同时满足 single>0.8 与 fwd>0.3"的 Pareto 前沿。

**v2（3 轮 + 停泊）**：收割 v1 遗留的时间退火实验，确认"前向能力已充足、瓶颈在
步态质量"；用 cycle_time 0.9→1.0s 单变量取得突破（single 首次稳定破 0.8）；
温和退火 1.5× 把 Pareto 右移；激活 swing_foot_forward 打破 Pareto 并找到最优
窗口靶点；随后判定标量奖励范式整体到顶，转向参考步态模板。

## 2. 完整方向清单与结果

### Loop v1（方向 → 结果）

| 方向 | 结果 |
|---|---|
| PPO lr 1e-5→1e-4 | ❌ 证伪"欠训练"假设：存活 16×、reward 快涨，仍 65% 时间双脚腾空弹跳（bounce）。附带发现 adaptive-lr 隐藏 confound（后用 schedule='fixed' 根治） |
| single_contact_grace 0.2→0.04s | ❌ 堵漏洞无效：bounce 是 survival+tracking 双奖励下的稳健吸收态，削单项推不出盆地 |
| 新增显式交替接触奖励 | 🎯 突破：bounce 崩塌（zero 0.57→0.16），single 0.40→0.77。配套 entropy 0.005 + lr fixed 根治发散 |
| forward_progress 线性 2×/4× | ❌ 2× 只减缓前向退化；4× 步态崩（zero 飙 0.51）。线性形态低速区梯度太弱 |
| world-frame 净位移形态 | ❌ 早期同步好（r=0.90）、中后期脱钩复发（r=-0.43）：策略调 yaw 骗世界投影 |
| actor 观测加 base 世界 xy 速度 | ✅ 消除 yaw-cheat；步态质量当时最佳，但暴露关键诊断：**低速是策略真实偏好而非 hacking**（线性前向奖励低速区梯度弱） |
| 超线性门槛 bonus（1+2·max(0,vx-0.05)） | 🏆 里程碑：首次 fwd 0.418>0.3 且步态健康。通用洞察：**梯度结构（而非 scale）决定行为** |
| alternating_contact scale 0.5→0.7 | ❌ 事件检测奖励可被"双支撑内左右快速交替"hack：single 反降、fwd 崩 |
| 步态时钟相位跟踪（gait_phase_tracking） | 🏆 范式升级：连续相位匹配不可 hack，single 0.805 / zero 0.131 双达标，14000 iter 无退化 |
| 速度耦合步态时钟（cycle/(1+k·vx_cmd)） | 🏆 三目标首次同窗达成（iter3000-3900：single 0.73-0.76 / zero 0.134 / fwd 0.40-0.42）。洞察：**物理耦合优于独立加权** |
| gpt scale 0.8→1.0 | 步态历史最佳（single 0.80 / zero 0.11）但 fwd 降 0.27；精确刻画 Pareto 斜率 Δsingle/Δfwd≈-0.5，**静态配比无解** |

（另：轮 5-8 因运行环境断网无有效产出，非研究失败。）

### Loop v2（方向 → 结果）

| 方向 | 结果 |
|---|---|
| 收割 v1 遗留的两阶段时间退火（fwd 有效 scale 0.4→1.0） | 温和档（0.4-0.65）部分打破 Pareto ✅；满档 2.5× 严重过冲 ❌：fwd 飙 0.58-1.0 但 gpt 崩 0.74→0.22、无视速度指令、bounce 回升。**结论：前向能力已充足，瓶颈在步态质量** |
| cycle_time 0.9→1.0s（步频放慢，正交新轴） | 🏆 v2 最大突破：single **首次稳定破 0.8**、gpt 升到 0.828 且全程不崩、123 个数据点同时 single>0.8∧fwd>0.3。机制修正：不是"更多双支撑重叠"（double 反而降），而是**慢步频让相位跟踪变得可达且稳定**。残留问题：late 期 fwd 回落 0.20-0.23（强 gpt 吸引子抑制前向） |
| 温和退火 anneal_max 1.5×（叠加 cycle_time=1.0） | ✅ fwd late 期从 0.20 提到 0.27-0.29（+43%）、步态代价极小；消融干净（前段轨迹与基线逐点重合）。但 fwd 均值仍未稳定>0.3——**Pareto 右移但未消除** |
| 激活 swing_foot_forward（摆动脚前向速度奖励，V13 清理时被禁用） | 🎯 打破 Pareto：combo 点 568 个（6.8×）、fwd 持续加速不再被 gpt 压制。但与 1.5× 退火叠加过冲：gpt 缓降、tracking 崩、bounce 回升。**关键产出：最优窗口 iter7000-8000（single 0.800 / fwd 0.664 / zero 0.159）——若能把前向驱动强度停在该窗口水平，三目标可稳定达成** |
| 参考步态模板（use_ref_actions 残差 + ref_joint_pos 跟踪） | ⏳ 进行中（TASK_20260714_134，范式转变：从"接触奖励涌现步态"变为"跟随已知好模板，只学平衡与转向"；基线回退 cycle_time=1.0 保干净归因） |

## 3. 合并后的核心结论（新 loop 应默认采纳）

**已证死路（勿重试，除非有新证据）**
1. 只调 PPO 超参解 bounce——优化器不是瓶颈。
2. 堵单项 reward 漏洞（grace 收紧等）——推不出吸收态盆地，要给更强的正向吸引子。
3. 线性前向奖励加 scale（2×/4×/满档退火 2.5×）——要么梯度不足要么过冲毁步态。
4. world-frame 位移形态——脱钩会复发。
5. 离散接触事件类步态奖励（含加大 alternating_contact）——天然可 hack，已被相位跟踪取代。
6. 静态 reward 配比同时满足 single>0.8 且 fwd>0.3——Pareto 无解，必须用时间调度、
   新奖励轴（swing_foot_forward）或换范式。

**已验证有效（默认基线）**
1. 稳定性三件套：entropy_coef 0.005 + schedule='fixed'（lr 1e-4 恒定）+ 负趋势即止损。
2. 观测加 base 世界 xy 速度（防 yaw-cheat）。
3. 步态时钟连续相位跟踪（gpt scale 1.0）+ 速度耦合（k=1.0）。
4. **cycle_time = 1.0s**（v2 突破，慢步频使相位跟踪可达）。
5. 超线性门槛 bonus 形态（thresh 0.05, k 2）；前向增强用**温和**时间退火（≤1.5×）。
6. swing_foot_forward（scale 0.4）能打破前向-步态 Pareto，但需控制总前向驱动强度。

**开放问题（按优先级）**
1. 参考步态模板范式是否成立——TASK_20260714_134 的收割结论（进行中，最高优先）。
2. 如何把系统"停在"最优窗口（iter7000-8000 水平的前向驱动强度）：候选手段有
   swing_foot_forward 降 scale、退火目标值调低、或按步态质量指标动态调节前向驱动。
3. DR 课程学习（先低随机化建步态、再渐进加满）——v1 轮 14 的 pivoter 指令，至今未试。
4. double_contact 始终偏低（0.06-0.13 vs 真实行走 0.2-0.3）：步频/相位期望的双支撑
   占比可再精调（cycle_time 已从 0.9 调到 1.0，更长周期或改相位窗口未试）。

## 4. 归档指引

本文件是两次 loop 的唯一可读总结。逐轮原始账本（含全部数值证据、judge 裁决、
成本审计）备份于 `.oma/loop-archive/raw/`（v1）；v2 的 `.loop/x1-walking-research-v2-v1`
删除前请先收割 TASK_20260714_134。`.oma/memory.md` 的表格基于 v1 整理，本文件
已包含 v2 修订（cycle_time 假设已验证、退火假设已收割），以本文件为准。
