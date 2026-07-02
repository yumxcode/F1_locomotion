# van Marum 2024 — Humanoid Walking Reward Design & Single-Foot-Contact

## Question
Extract, from van Marum 2024 (Oregon State University), the reward-function design with focus on the **single-foot-contact** term: title/authors, year/venue, all reward terms and their weights, what behaviors emerge with a minimal reward set, and the key conclusion about single-foot contact and gait emergence.

## Key Findings

1. **The paper exists and is exactly the target.** Title: *"Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking."* Authors: **Bart van Marum, Aayam Shrestha, Helei Duan, Pranay Dugar, Jeremy Dao, Alan Fern** (Dynamic Robotics and Artificial Intelligence Laboratory, Oregon State University, Corvallis, OR, USA). Published **2024**, venue **IEEE/RSJ IROS 2024**. arXiv **2404.19173** (v1 30 Apr 2024, v2 30 Aug 2024). Robot: **Digit** humanoid (20 actuators).

2. **Minimal tracking reward → HOPPING, not walking.** Training with only the three command-following terms (x/y-velocity, yaw orientation) produces a locomotion behavior where the robot *moves by jumping with both feet* (a hop). It satisfies the velocity commands but is not walking — so additional terms are mandatory.

3. **Single-foot-contact is the key term that flips hopping → walking** — and it needs *no tuning*. Verbatim: *"We found that the most reliable and unconstrained way to produce walking instead of hopping is via the single foot contact reward, which also does not require tuning."* Among the alternatives tested (base-height reward, clock-based rewards, exploration-noise tuning, feet-contact-transition reward), single-foot-contact won.

4. **Definition of the single-foot-contact term (Feet contact, weight 0.1):**
   - During any **non-standing** command: reward = **1** at a timestep if *exactly one* foot is in contact with the ground; a **0.2 s grace period** is applied so that if single contact occurred *at least once* in the last 0.2 s, the reward is granted (otherwise 0). This allows natural stance/swing overlap.
   - During the **standing** command (cs): reward is a **constant 1** (no preference for foot contact).
   - **Why standing is NOT rewarded with double-foot contact:** rewarding double-foot contact would *penalize the recovery steps* needed to reject disturbances (which require lifting a foot), and during walk→stand transitions it makes the policy pick the closest stance rather than the most stable one. Standing is instead rewarded *implicitly* because most other reward terms are larger when both feet are planted and stationary.

5. **No clocks / no reference trajectories.** The whole design deliberately avoids clock-based rewards and inputs, reference-motion rewards, footstep planners, or demonstrations. This removes the transition-engineering problem between standing and walking and leaves the policy free to move feet arbitrarily to stay upright under disturbance.

6. **Architecture/training:** (64,64) two-layer **LSTM** policy; input = robot state + user command cu=[cx,cy,cyaw] (standing = [0,0,0]); output = joint-space PD setpoints; **PPO** with a **mirror loss** for symmetry; controller at 50 Hz, PD at 2 kHz. Episodes 16 s; new command sampled every 2–6 s from {stand, sagittal walk, lateral walk, in-place rotate, omnidirectional}. Command ranges cx∈[-0.5,2.0] m/s, cy∈[-0.5,0.5] m/s, cyaw∈[-0.5,0.5] rad/s. Random push 1% chance/frame, 200–800 N for one 20 ms step.

## Full Reward Table (Table I) — Terms, Definitions, Weights

Notation: `cs` = standing command; `qd(·,·)` = quaternion distance; `nc,t*` = number of feet in contact at time t*.

| # | Reward term | Definition (kernel) | Weight |
|---|-------------|---------------------|--------|
| 1 | **x, y velocity** (tracking) | `e^{-5·(v_xy − c_xy)}` if standing; else `e^{-5·(v_xy − c_xy)²}` | **0.15, 0.15** (one per axis) |
| 2 | **Yaw orientation** | `e^{-300·qd(q_yaw, c_yaw)}` | **0.1** |
| 3 | **Roll, pitch orientation** | `e^{-30·qd(q_rp, c_rp)}` | **0.2** |
| 4 | **Feet contact (single-foot-contact)** | `1` if standing; `1` if `nc,t*=1` for any t*∈[t−0.2, t]; `0` else | **0.1** |
| 5 | **Base height** | `e^{-20·|p_z − c_h|}` | **0.05** |
| 6 | **Feet air time** | `1` if standing; else `Σ_{f∈(l,r)} (t_air,f − 0.4)·1_{td,f}` | **1.0 †** |
| 7 | **Feet orientation** | `e^{-Σ|r_feet,rp − c_feet,rp|}` if |cyaw|>0 else `e^{-Σ|r_feet,rpy − c_feet,rpy|}` | **0.05** |
| 8 | **Feet position** | `e^{-3·|p_feet − c_feet|}` if standing; else `1` | **0.05** |
| 9 | **Arm** | `e^{-3·‖θ_arm − c_arm‖}` | **0.03** |
| 10 | **Base acceleration** | `e^{-0.01·Σ|b_xyz|}` | **0.1** |
| 11 | **Action difference** (smoothing) | `e^{-0.02·Σ|a_t − a_{t-1}|}` | weight in truncation zone — **[partially verified; not read]** |
| 12 | **Torque** (minimize torque) | penalizes torque usage | weight + exact kernel in truncation zone — **[partially verified; not read]** |

**On the † footnote (row 6):** the `1.0†` marks the feet-airtime term. The paper notes this term is *not* a normalized [0,1] exponential kernel like the others: it is `Σ(t_air − 0.4)` at each touchdown, i.e. it can be **negative** (a 0.4 penalty at touchdown offset by positive airtime). This is why it is flagged differently. The exact wording of the † footnote itself fell in the 100 KB truncation and could not be read verbatim — **[footnote wording unverified]**.

**Design rationale for tracking terms (rows 1–3):** the squared-error kernel `e^{-5·Δ²}` is used while *walking* (smooth, rewards proximity symmetrically) but switches to the one-sided `e^{-5·Δ}` kernel while *standing* so that moving away from zero velocity in the commanded direction isn't symmetrically rewarded. Yaw uses a 10× sharper kernel (300) than roll/pitch (30) — heading tracking is prioritized over upright within tight tolerances.

**Command-following is "satisfied" by a hop:** rows 1–3 alone make the velocity/orientation rewards achievable by jumping with both feet, which is why row 4 (single-foot-contact) is the structural fix that enforces an alternating stance/swing gait.

## Experimental Results — Behaviors & Numbers

- **Disturbance rejection (standing):** the trained controller recovers from impulses applied via a DIY electromagnetic impulse device (rope attached to Digit at 122 cm height). From Figure 1: it survives a **lateral push of 150 N for 500 ms** and a **sagittal push of 200 N for 500 ms**. Metric is *Standing Fall Percentage* = % of trials the robot does not fall, swept over push weight & duration in ±x and ±y.
- **Seamless stand↔walk transitions:** the single policy natively switches between standing (both feet planted, cu=[0,0,0]) and walking without any clock/heuristic blending — demonstrated in project video clips (command transitions, shaking, pulling, random pushes, strong backward push).
- **Benchmark-guided improvement (Section V-B, qualitative):** benchmarking revealed the training-time push distribution (uniform 200–800 N, single 20 ms step) was **insufficient**; adopting a **more diverse push distribution** improved real-world disturbance rejection. (Author note: early ad-hoc testing had suggested more diversity did not help in sim; the real-world benchmark proved otherwise.)
- **Comparison set:** evaluated against the **manufacturer-supplied Digit controller** and a **state-of-the-art** reward-based controller, on command following (velocity accuracy, in-place rotation accuracy + lateral drift) and energy efficiency (estimated actuator work W=∫τ·ω dt, reported per meter traveled).

> **Limitation note (transparency):** The arXiv HTML page is ~110 KB and is returned truncated at 100 KB by the fetch tool, which cuts the page exactly at reward-row 11 (Action difference) and therefore *before Section V (Evaluation Results) and the Summary*. So: (a) the **exact weights/kernels of the Action-difference and Torque terms** and (b) the **full numeric tables of Section V** (fall-percentage grids, rotation-error numbers, velocity-error numbers, energy J/m values, and the head-to-head comparison vs the SOTA controller) could **not** be read from the source. The disturbance numbers (150 N / 200 N for 500 ms) come from the Figure 1 caption, which *was* read. The Section-V results above are therefore marked **[qualitative / partially verified]**. OpenAlex (503) and Semantic Scholar (429) were rate-limited and provided no additional data.

## Key Conclusion (single-foot-contact → gait emergence)

A bipedal humanoid trained with nothing but velocity + orientation tracking converges to **hopping** (jumping with both feet), which technically maximizes the tracking reward but is not walking. **Adding a single-foot-contact reward — binary 1 when exactly one foot is grounded, with a 0.2 s grace window, weight 0.1 — is the minimal, tuning-free, clock-free mechanism that makes an alternating-contact walking gait emerge.** During standing the term is a neutral constant 1 (so it never penalizes the recovery steps needed for disturbance rejection). This is the paper's central reward-design contribution and is the recommended starting point for a minimally-constraining humanoid SaW reward.

## Synthesis & Recommendation

- **For reward design (per OMA memory: read code/values before designing):** if building a humanoid SaW reward, begin with the three tracking kernels above (rows 1–3, weights 0.15/0.15/0.1/0.2) and immediately add the **single-foot-contact term (row 4, weight 0.1, 0.2 s grace)** as the gait-forcing term — this avoids the hop local optimum without clocks or reference motion.
- **Standing strategy:** do *not* reward double-foot contact during standing; keep foot-contact reward constant (1) during standing and let tracking + style terms implicitly favor a stable two-foot stance, so disturbance-recovery steps remain free.
- **Avoid prescriptive constraints:** no clocks, no reference trajectories, no footstep planner → cleaner stand/walk transitions and better disturbance rejection (the policy can move feet arbitrarily).
- **Push training distribution:** a single-step 200–800 N uniform push is too narrow; benchmark-driven diversity in push magnitude/duration improved real recovery.
- **Caveat to act on:** before locking weights, retrieve the *exact* Action-difference and Torque weights and the full Section-V numeric tables from the PDF (the HTML truncation blocked these) — re-fetch from a PDF text extraction or the published IROS 2024 proceedings.

## Sources (one section per source)

### Source 1 — arXiv 2404.19173 (primary, full-text HTML read up to 100 KB truncation)
- **Title:** Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking
- **Authors:** Bart van Marum, Aayam Shrestha, Helei Duan, Pranay Dugar, Jeremy Dao, Alan Fern (Oregon State University)
- **Year/Venue:** 2024, IROS 2024
- **URL:** https://arxiv.org/abs/2404.19173v2 (HTML: https://arxiv.org/html/2404.19173v2)
- **Takeaway:** single-foot-contact is the minimal, tuning-free, clock-free term that converts a hopping policy into a walking one on Digit.
- **What was read:** Abstract, §I–IV-B (including the full Table I reward table up to row 10, and the complete single-foot-contact definition/rationale), and the Figure-1 disturbance numbers. **Not read (100 KB truncation):** end of Table I (rows 11–12 weights), §V Evaluation Results numeric tables, §VI Summary.
- **Backing excerpts:** see `sources_markdown`.

### Source 2 — Project website (supplementary)
- **URL:** https://b-vm.github.io/Robust-SaW/
- **Takeaway:** confirms authors/affiliation, provides the impulse-device parts list & construction, and video clips of stand/walk transitions and disturbance rejection. No numeric result tables.