# Algorithm Memory
_Project: AgiBot X1 行走控制（PPO/DHPPO）_
_Last updated: 2026-07-13_

<!--
  USAGE: This file is written ONLY by $consolidate.
  Other skills must READ this file at session start but must NOT modify it.
  Append new rows to tables; never delete existing rows.

  2026-07-13: consolidated from loop x1-walking-control-v1 (14 rounds, 11
  committed findings). Full narrative + raw ledger: .oma/loop-archive/.
-->

## Dead Ends
Directions proven not to work. Do not re-propose entries in this table without a compelling new experimental reason.

| Direction | Why Failed | Seeds Tested | Evidence Experiments | Date Added |
|-----------|-----------|-------------|---------------------|-----------|
| 仅调 PPO 超参解 bounce（lr 1e-5→1e-4 等） | 优化器非瓶颈：存活 16×、reward 快涨，策略仍 65% zero-contact 弹跳；bounce 由 reward landscape 吸引 | 1 | TASK_20260709_014 | 2026-07-13 |
| 收紧 single_contact_grace 堵 bounce 漏洞（0.2→0.04s） | zero_contact 仅 0.65→0.62；bounce 是 survival+tracking 双奖励下的稳健吸收态，削单项推不出盆地；4 帧 grace 仍有 34% reward 泄漏 | 1 | TASK_20260709_023 | 2026-07-13 |
| 线性 forward_progress scale 放大（2×/4×）治前向退化 | 2× 只减缓（-62%→-33%）；4× 致步态代价（zero_contact 飙 0.509）；线性形态低速区梯度弱，scale 换不来定向推力 | 2 | TASK_20260709_056, R8(未入账,见归档§6) | 2026-07-13 |
| world-frame 净位移 reward 形态治脱钩 | 早期同步好（r=0.903）但中后期脱钩复发（r=-0.43）：策略调 yaw 使世界投影最大化同时减少真实前移 | 1 | TASK_20260709_109 | 2026-07-13 |
| 加大 alternating_contact scale 精调步态（0.5→0.7） | 事件检测 reward 可被"双支撑内左右频繁交替"hack：single 反降 0.692→0.603，fwd 崩至 0.11；reward 持平掩盖行为退化 | 1 | TASK_20260710_053 | 2026-07-13 |
| 离散接触事件检测类步态 reward（范式级） | 天然可 hack（grace 泄漏、双支撑交替欺骗）；被连续相位跟踪范式全面取代 | 5+ | R1-R11 链条, TASK_20260710_079 对照 | 2026-07-13 |
| 静态 reward 配比同时满足 single>0.8 且 fwd>0.3 | Pareto 前沿实测：scale0.8→(0.73,0.41)，scale1.0→(0.80,0.27)，Δsingle/Δfwd≈-0.5；单一静态 scale 无解，需时间维度调度或双目标解耦 | 2 | TASK_20260711_024, TASK_20260711_074 | 2026-07-13 |

## Working Patterns
Directions with confirmed positive effect. Use as defaults in $design and $tune.

| Pattern | Conditions | Median Gain | Evidence Experiments | Date Added |
|---------|-----------|------------|---------------------|-----------|
| 显式交替接触正向吸引子（打破 bounce 的解法原型） | 缺乏 walk 吸引子时 | single_contact 0.40→0.765；bounce 崩塌 | TASK_20260709_053 | 2026-07-13 |
| entropy_coef 0.005 + schedule='fixed'（lr 1e-4 恒定） | 所有轮，作为稳定性基线 | noise_std 4.4→0.6 收敛；消除 adaptive-lr confound | TASK_20260709_053 起全部轮复现 | 2026-07-13 |
| actor 观测加 base 世界 xy 速度 | 消除 yaw-cheat 类 fwd hacking | 步态质量本 loop 最佳（double 0.227）；正交调节手段 | TASK_20260710_011 | 2026-07-13 |
| 超线性门槛 bonus：reward·(1+k·max(0,vx-thresh))，thresh=0.05,k=2 | 需把策略推出低速盆地 | fwd_vel 0.19→0.418（首破 0.3）；梯度结构>scale 的通用洞察 | TASK_20260710_042 | 2026-07-13 |
| 步态时钟连续相位跟踪替代事件检测（gait_phase_tracking） | 步态质量目标 | single 0.805 / zero 0.131 双 PASS；14000 iter 无退化、不可 hack | TASK_20260710_079 | 2026-07-13 |
| 速度耦合步态时钟 cycle_time_eff=cycle_time/(1+k·vx_cmd)，k=1 | 步态与前移需协同 | 三目标首次同窗达成（single0.73-0.76/zero0.134/fwd0.40-0.42 @iter3000-3900）；"物理耦合优于独立加权" | TASK_20260711_024 | 2026-07-13 |
| 负趋势即止损（reward 峰后持续下滑立即 stop） | 所有训练监控 | 多轮验证判断正确，省算力 | R1/R2/R5/R7 等 | 2026-07-13 |

## Open Hypotheses
Directions suggested by evidence but not yet tested. Prioritize high-priority items in next $tune.

| Hypothesis | Source | Priority | Estimated Gain | Status |
|-----------|--------|----------|---------------|--------|
| forward_progress 两阶段时间退火（iter0-3000 因子1.0 先建步态→3000-5000 线性升至 2.5 推前向）可突破静态 Pareto 前沿 | round15 direction 草稿；R13 最优窗口证明能力可行 | HIGH | single>0.8 且 fwd>0.3 同时达成 | in-flight：TASK_20260711_135 已提交未收割，先查曲线再决定重跑 |
| DR 课程学习（当前全程满量 randomization 从未被质疑；先低 DR 建步态再渐进加满） | 轮 14 pivoter 指令（未执行） | HIGH | 减少探索期干扰，可能改善三目标收敛 | untested |
| cycle_time 0.9→1.0s 精调改善 double_contact 偏低（0.06-0.09 vs 真实行走 0.2-0.3） | R12/R16 findings | MED | 步频放缓、双支撑过渡更自然 | untested |
| gpt 与 fwd 的 scale 分离调度（非单一 scale 权衡） | R16 Pareto 刻画 | MED | 绕开 Δsingle/Δfwd≈-0.5 的直线权衡 | untested |
| 门槛 bonus 形态推广到其他涌现目标（更高抬脚、更慢步频） | R10 finding-5 洞察 | LOW | 待定 | untested |

## Budget Tracker
| Item | Value |
|------|-------|
| Total experiment budget (GPU-hrs) | 见 requirements.md（loop v1 预算：20 rounds / $100） |
| Consumed to date | 14 rounds / $10.93（loop v1，已归档） |
| Remaining | 新 loop 从零计 |
| Experiments run | 13 次远端训练（11 入账 + R8 + TASK_20260711_135 未收割） |
| Current best metric (val) | single_contact 0.8053（R16）；三目标最优同窗：single 0.73-0.76 / zero 0.134 / fwd 0.40-0.42（R13 iter3000-3900） |
| Current best metric (test) | not yet evaluated（未做部署侧评估） |
| Deploy gate status | closed |
