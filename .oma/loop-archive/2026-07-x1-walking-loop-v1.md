# X1 行走控制 Loop 归档（x1-walking-control-v1，2026-07-09 ~ 07-11）

> 本文是 `.loop/x1-walking-control-v1` 全部研究进展的无损汇总，供删除 `.loop/`、
> `.meta-agent/` 后新 loop 继承。原始账本逐字节备份在 `raw/` 子目录。
> 提炼结论（Dead Ends / Working Patterns / Open Hypotheses）已合并进
> `.oma/memory.md`——新 loop 的 worker 从那里读。

## 1. 概览

| 项 | 值 |
|---|---|
| 实例 | x1-walking-control-v1（charter v1，research 语义） |
| 目标 | X1 稳定自然前进行走：episode>500 步、single_foot_contact>0.8、前向>0.3 m/s、步态物理可信 |
| 已结算轮次 | 14（另有 round 15 停泊中，见 §5） |
| 总成本 | $10.93 |
| best_metric | 0.8053（single_contact 口径） |
| 训练分支 | v9-smooth-landing（github.com/yumxcode/F1_locomotion） |
| 状态 | 未达 goal_satisfied；三目标已在最优窗口同时达成过（R13，iter 3000-3900） |

## 2. 轮次时间线

| 轮 | 模式 | 方向 | 路由 | 备注 |
|---|---|---|---|---|
| 1 | normal | ppo-lr-1e5-to-1e4 | continue | 6 findings，证伪"欠训练"假设 |
| 2 | normal | sfc-grace-0p2-to-0p04 | pivot | 证伪 grace-exploit 主因假设 |
| 3 | pivot | explicit-alternating-contact | continue | 🎯 突破：bounce 崩塌，交替步态涌现 |
| 4 | normal | fwd-progress-scale-2x | continue | 方向对、力度不足 |
| 5–8 | — | （笔记本合盖断网事故） | 5:continue 6,7:pivot 8:escalate | 无有效产出；escalate 后人工 ack（reason="合盖断网误升级，恢复"，resetMeters=[stale_count]） |
| 9 | normal | fwd-net-displacement | continue | world-frame 形态：早期同步、后期脱钩复发 |
| 10 | normal | obs-base-world-vel | continue | 步态本 loop 最佳，但 fwd 崩溃→"低速是真实偏好"关键诊断 |
| 11 | normal | fwd-speed-threshold-bonus | pivot* | 🏆 里程碑：首次 fwd>0.3 + 步态双达标（best 0.418） |
| 12 | pivot | alt-contact-scale-up | pivot | ❌ 证伪；暴露 alternating_contact 可被双支撑内交替 hack |
| 13 | pivot | gait-phase-tracking | pivot | 🏆 范式转换：相位跟踪，步态两目标首次全 PASS |
| 14 | pivot | speed-coupled-gait-clock → gpt-scale-up | pivot | 🏆 三目标首次同窗达成（R13 iter3000-3900）；Pareto 前沿刻画 |

\* 轮 11-14 连续 pivot 是 charter v1 的 `results_improved` 观测键缺陷（judge 未必
输出该键 → stale_count 冻结在 2 → pivot 绊线每轮触发）。该缺陷已在新版运行时
结构性修复（义务注入 + 三态观测），归档时按事实记录。

## 3. 主线因果链（11 条入账 findings 的浓缩叙事）

1. **bounce 不是欠训练**（R1/ppo-lr）：lr 提到 1e-4 后存活 16×、reward 快涨，
   策略仍选双脚离地弹跳（zero_contact 0.65）——问题在 reward landscape。
   附带发现隐藏 confound：adaptive-lr schedule 自 base config 静默生效于全部历史轮。
2. **堵漏洞无效**（R2/grace）：grace 0.2→0.04s，bounce 仅降 3pp。bounce 是
   survival+tracking 双奖励下的稳健吸收态，"修单项"推不出盆地。
3. **正向吸引子才是解**（轮 3/explicit-alternating-contact）：新增左右交替落地
   奖励后 single_contact 0.40→0.765，bounce 崩塌。配套 entropy 0.01→0.005 +
   schedule='fixed' 根治了 noise_std 暴涨与 lr confound。
4. **前向退化的三轮排查**（fwd-2x、fwd-4x、net-displacement）：线性 scale 加倍
   只减缓不根治；world-frame 形态早期同步（r=0.903）后期脱钩复发（r=-0.43）；
   跨形态一致的退化模式指向更深的原因。
5. **关键诊断**（R9/obs-base-world-vel）：加入 base 世界速度观测后步态最佳但
   fwd 归零——**低速是策略在当前 reward 下的真实偏好，不是 hacking**。线性
   forward_progress 低速区梯度太弱。
6. **超线性门槛根治**（R10/threshold-bonus）：`reward·(1+2·max(0,vx-0.05))`
   把策略推出低速盆地，首次 fwd 0.418>0.3 且步态健康。**洞察：梯度结构（而非
   scale）是驱动行为的核心，可推广到其他涌现目标。**
7. **事件检测范式可 hack**（R11/alt-contact-scale-up 证伪 + R12/gait-phase-tracking
   验证）：离散接触事件 reward 可被"双支撑内频繁交替"欺骗；改用步态时钟连续
   相位跟踪后 single 0.805 / zero 0.131，两步态目标首次全 PASS 且 14000 iter 无退化。
8. **物理耦合优于独立加权**（R13/speed-coupled-gait-clock）：步频=f(速度指令)
   使相位跟踪与前移从竞争变协同，iter 3000-3900 窗口**三目标首次同时达成**
   （single 0.73-0.76 / zero 0.134 / fwd 0.40-0.42）。
9. **Pareto 前沿刻画**（R16/gpt-scale-up）：gpt scale 0.8→1.0 换来 single 0.80 /
   zero 0.110（历史最佳步态）但 fwd 0.27；两点定斜率 Δsingle/Δfwd≈-0.5。
   静态 reward 无法同时满足 single>0.8 与 fwd>0.3——但 R13 窗口证明能力上可以，
   矛盾指向**训练动力学**，引出 round 15 的时间退火方向。

## 4. 结构性转向指令记录（pivoter 产出）

- 轮 3 前：速度奖励改"单支撑相门控"形态（contact-gate）。
- 轮 12 前：从接触事件检测转向步态时钟相位跟踪（Cassie/Digit/MIT Cheetah 类比）。
- 轮 13 前：步态时钟速度耦合（cycle_time_eff = cycle_time/(1+k·vx_cmd)）。
- 轮 14 前：dr-curriculum（DR 课程学习——13 轮来首次质疑环境分布假设，**未执行**，
  被 gpt-scale-up 占用了该轮；仍是有效的待试方向）。
- 轮 15 前：reward-curriculum-annealing（两阶段时间退火，见 §5）。

## 5. 中断现场（round 15，未结算）

- 方向：`fwd-progress-anneal`——forward_progress 有效 scale 两阶段退火
  （iter0-3000 因子 1.0 先建步态；iter3000-5000 线性升至 2.5 推前向），
  假设已建立的步态吸引子足以抵抗前向增强。完整 rationale 见
  `raw/round15-direction-draft.json`。
- 远端训练：`TASK_20260711_135`（Gradmotion），提交后停泊于 self_timer
  （原计划 2026-07-11 ~16:56 唤醒收割，检查退火阶段 2 是否推 fwd>0.3 同时保
  single>0.8）。**删除 .loop 前请先确认该任务已手工收割或终止**（gm task info /
  task data get / task stop），其曲线结论值得作为新 loop 的第一轮输入。
- 提交段成本 $0.31 未入账（新 loop 预算从零计，无影响）。

## 6. 数据完整性备注

- `fwd-progress-scale-4x` 在 directions 中有记录但无入账 finding（该轮评审未
  通过或产出未提交；其结果在 R9 finding 中被间接引用："R8 曾飙至 0.509"）。
- `fwd-net-displacement` 有入账 finding 但不在 directions 去重表中（旧版方向
  登记的一个缺口，新版已由 Artifact 事务修复）。
- 轮 5-8 的空产出源于运行环境断网（笔记本合盖），非研究失败；escalate 被人工
  ack 并重置 stale_count。
- `.meta-agent/` 仅含会话自动 checkpoint（运行时恢复状态），无研究知识，可删。

## 7. raw/ 清单

findings.jsonl（11 条完整入账 findings，含全部 evidence 数值与
baseline_metrics_for_comparison）、directions.json（11 方向 + rationale）、
rounds.jsonl（14 轮审计）、progress.json、pending_round.json（round 15 现场）、
round15-direction-draft.json、charter.frozen.json（v1 冻结章程全文，含 worker/
judge/pivoter/finalizer prompt）、attention_report.md（轮 8 升级报告）、
lifecycle.jsonl（人工 ack 记录）。
