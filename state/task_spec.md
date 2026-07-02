# Task Spec - X1 Humanoid Robot Walking RL

_Created: 2026-07-02 (cold-start bootstrap)_
_Provenance: extracted from design-unified-reward.md (V13) and dual-track-plan.md_

> NOTE: requirements.md and knowledge.md are still in DRAFT/template
> form (unfilled placeholders). This spec is derived from the ACTIVE design documents,
> which carry the concrete research goals. If/when requirements.md is later locked,
> reconcile any conflicts and update this file.

---

## Research Goal

Train the X1 bipedal humanoid robot (12 DOF) to perform robust omnidirectional
walking and standing via reinforcement learning (PPO, DHPPO variant), using a
minimal reward design that produces emergent alternating gait.

The current active design is V13 "Minimal Emergence", based on
van Marum et al. 2024 (Oregon State University, Digit robot):
  tracking + orientation -> only produces hopping.
  adding single_foot_contact -> walking emerges.

This replaces the failed Schumacher 2025 (efficiency + pain -> emergence) route
iterated through V10-V12 (6 rounds, proven infeasible due to unreconcilable reward
gradient conflicts).

Core design hypothesis: A single physical contact constraint
(single_foot_contact, reward when exactly 1 foot touches ground) is sufficient to
make the policy abandon hopping and self-emerge alternating gait -- without gait
clocks, reference trajectories, or symmetry penalties.

---

## Platform and Algorithm

| Item | Value |
|------|-------|
| Robot | X1 bipedal humanoid, 12 DOF |
| Simulator | Isaac Gym (preview-4) |
| Algorithm | PPO (DHPPO variant) |
| Actor | MLP [512, 256, 128], ELU, init_noise_std=1.0 |
| Critic | MLP [768, 256, 128] + state estimator [256, 128, 64] |
| History encoder | CNN kernel=[6,4], filter=[32,16], stride=[3,2], out=64 |
| Control freq | 50 Hz (decimation=10, dt=0.001 s) |
| Resource | 1 x 4090D 24G (ESKU000001) |
| Image | Isaac Gym preview-4 (BJX00000001, v V000124) |

---

## Current Design Snapshot (V13)

Reward (8 terms, 3 tiers):
  (1) Task: tracking_lin_vel x1.0 (sigma=0.25), tracking_ang_vel x0.5, orientation x0.5, base_height x0.2 (h-star=0.61 m), torque x0.01
  (2) Gait emergence: single_foot_contact x0.3 (n_contact==1, 0.2 s grace), feet_airtime x0.3 (target 0.4 s)
  (3) Safety hard constraint: dof_pos_limits x-10.0

Explicitly removed (vs V12): symmetry, stability, efficiency, landing_impact,
ref_joint_pos, feet_contact_number, feet_clearance, swing_foot_forward, foot_slip.

---

## Success and Validation Criteria (from design 12.2)

| Metric | Target | Verification |
|--------|--------|--------------|
| Gait emergence | alternating walk, NOT hopping | CSV replay visual check |
| single_foot_contact reward | > 0.8 | reward curve |
| Velocity tracking | stable > 0.3 m/s | tracking reward curve |
| Standing stability | no drift during stand phase | stand-phase analysis |
| Episode length | > 500 steps (10 s) | mean_episode_length curve |

---

## Milestones

| ID | Milestone | Status | Notes |
|----|-----------|--------|-------|
| M1 | V13 training and gait-emergence validation | IN-PROGRESS | Task TASK_20260612_050, commit 1be71f8, branch v9-smooth-landing; early iter 4 mean_reward 0.71 to 1.06, all rewards show effective gradients |
| M2 | Reward ablation and hyperparameter tuning ($tune) | PENDING | Ablation plan in design 10: single_foot_contact scale [0.1,0.3,0.5], feet_airtime threshold [0.2,0.4,0.6] s, tracking scale [0.5,1.0,1.5], etc. |
| M3 | Best-policy selection and final test evaluation | PENDING | leader-board from sweep; held-out test eval |
| M4 | Sim-to-real and hardware validation ($deploy) | PENDING | Highest risk per design 11 (sim-to-real gap) |

### Decision and tuning branches after M1 (design 12.3)
- Gait emerges + normal walking -> success -> $tune
- Gait emerges but cadence too fast -> raise airtime_threshold to 0.6 s
- Still hopping (no single-foot contact) -> raise single_foot_contact scale to 0.5
- Frequent falling -> raise orientation scale to 1.0
- Gait OK but excessive torque -> add back foot_slip or raise torque scale

---

## Known Dead Ends (do NOT re-propose)

From design 9.2 (mirrors memory.md Dead Ends, currently empty -- to be populated by $consolidate):
- CoT efficiency tau-dot-q (sign-cancelling), walk_decay, efficiency too weak (-8e-9),
  efficiency dominant (-0.02), swing_foot_forward unbounded, sigma too small (0.08),
  tracking weight too high (2.5), efficiency-vs-landing vs tracking conflict,
  symmetry (energy-phase and mirror), Schumacher three-piece route (V10-V12).
