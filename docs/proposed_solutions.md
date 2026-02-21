# Proposed Generalizable Solutions for CIRC-RL Pipeline Limitations

This document describes five generalizable improvements identified from the
Acrobot-v1 benchmark analysis. Each solution addresses a specific failure mode
of the current v2 pipeline and is applicable to any environment, not just
Acrobot.

---

## 1. MPPI/CEM Planning (Replace iLQR for Non-Convex Problems)

**Problem**: iLQR is a local trajectory optimizer that linearizes dynamics
around a nominal trajectory. For tasks requiring global strategies (swing-up,
manipulation, locomotion), the cost landscape has many local minima and iLQR
converges to suboptimal solutions.

**Solution**: Model Predictive Path Integral (MPPI) or Cross-Entropy Method
(CEM) as alternative planners. These sampling-based methods:

- Sample K action sequences from a noise distribution around a mean
- Roll out all K trajectories through the dynamics model
- Weight trajectories by softmax of reward (MPPI) or select elite percentile
  (CEM)
- Update the mean action sequence as the weighted/elite average
- Repeat for several iterations

**Mathematical formulation** (MPPI):

    mean_{i+1} = sum_k w_k * actions_k
    w_k = exp(-1/lambda * cost_k) / sum_j exp(-1/lambda * cost_j)

where lambda is the temperature parameter controlling exploration vs
exploitation.

**Key advantages over iLQR**:
- No linearization required -- handles non-convex cost landscapes
- Discovers energy-pumping strategies through random exploration
- Works with any dynamics model (even black-box)
- Naturally parallelizable (K independent rollouts)
- No Jacobian computation needed

**Generalizability**: Any task where the optimal trajectory is non-obvious
(manipulation, locomotion, swing-up). iLQR remains preferred for tasks with
locally quadratic cost landscapes (tracking, stabilization).

**Expected impact**: Significant improvement on swing-up tasks. For Acrobot,
MPPI with K=512 samples should discover the energy-pumping strategy that iLQR
cannot find through local optimization.

**Implementation**: New `MPPISolver` class in `circ_rl/analytic_policy/`
with vectorized numpy batched rollout.

**Status**: COMPLETE. `circ_rl/analytic_policy/mppi_solver.py` (413 lines).
Features: colored noise (beta-spectrum), vectorized K-sample rollouts, warm
starting from previous solution, ILQRSolution-compatible output. MPPI alone
(with Euler dynamics) achieved mean=-649 on 15 OOD Acrobot envs due to
dynamics model divergence. Combined with Solution 3 (RK4 dynamics), achieved
mean=-147.7, 100% goal rate.

---

## 2. Energy-Based Cost Shaping (Automatic from Dynamics Model)

**Problem**: Position-based cost functions (e.g., tip height for Acrobot)
create deceptive gradients. The optimal swing-up strategy requires temporarily
moving away from the goal (to build energy), but position-based costs penalize
this necessary intermediate behavior.

**Solution**: For any mechanical system with validated dynamics, automatically
compute an energy-based cost:

1. Compute total mechanical energy E(state) from the dynamics model:

       E = T(q, qdot) + V(q)

   where T is kinetic energy and V is potential energy.

2. Compute target energy E* = E(goal_state).

3. Use a composite cost function:

       cost = w_energy * (E - E*)^2 + w_position * position_cost

   with automatic weight scheduling: when |E - E*| is large, the energy term
   dominates (guides energy pumping). When E is near E*, the position term
   takes over (guides goal approach).

**Generalizability**: All mechanical systems (pendulums, multi-link arms,
locomotion) have conserved energy at the dynamics level. The energy function
can be extracted from PySR dynamics expressions symbolically. For
non-mechanical systems, a Lyapunov-like function serves the same purpose.

**Expected impact**: Eliminates the deceptive gradient problem. Combined with
MPPI, should enable rapid swing-up by guiding the planner to pump energy
first, then approach the goal.

**Implementation**: Energy functions added to
`circ_rl/hypothesis/lagrangian_decomposition.py`: `evaluate_coefficients()`,
`compute_mechanical_energy()`, `compute_mechanical_energy_batched()`,
`compute_goal_energy()`. Energy-shaped reward factory functions in
`experiments/acrobot_v2.py` activated via `--energy` flag.

**Status**: COMPLETE. Energy computed from Lagrangian EL coefficients:
T = 0.5 * qdot^T * M(q) * qdot, V = -g_sin1*cos(phi_0) - g_sin12*cos(phi_0+phi_1).
Goal energy E* = g_sin1 + g_sin12 (upright, stationary). Reward formulation:
tip_height - energy_weight * ((E - E*) / |E*|)^2 - 0.01*u^2 - 0.001*(w1^2 + w2^2).
Results: mean=-134.6 (vs -147.7 without energy), 100% goal rate, avg 113 steps
(vs 141 without energy). 8.9% return improvement, 20% fewer steps.

---

## 3. Lagrangian Decomposition for PySR

**Problem**: PySR operates per-dimension independently and cannot discover
dynamics that involve matrix inversion or coupled equations. Multi-body
Lagrangian systems have dynamics of the form:

    M(q) * qddot = tau - C(q, qdot) * qdot - G(q)

where M is the mass matrix. The angular accelerations involve M^(-1), which
creates rational functions of trigonometric terms that PySR's sum-of-products
search space cannot represent.

**Solution**: Decompose the dynamics regression into structural components:

1. **Detect multi-body structure** from the causal graph (multiple angular
   dimensions, coupled accelerations).

2. **Regress individual components** separately:
   - Mass matrix elements M_ij(q) -- functions of positions only (simpler)
   - Coriolis/centrifugal terms C_ij(q, qdot) -- functions of positions and
     velocities
   - Gravity vector G_i(q) -- functions of positions only
   - Control input matrix B(q) -- maps actions to generalized forces

3. **Compose analytically** via qddot = M^(-1)(tau - C*qdot - G). The matrix
   inversion is applied analytically (closed-form for small systems), not
   discovered by PySR.

Each component is a much simpler expression. For Acrobot:

    M_11 = m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(t2)) + I1 + I2

This is a polynomial in cos(t2) with env params -- easily discoverable by
PySR.

**Generalizability**: Any system governed by Lagrangian/Hamiltonian mechanics
(all mechanical systems, the primary domain of classical control benchmarks).
Detection of "this is a multi-body system" can be automated from the causal
graph structure.

**Expected impact**: Dramatically improved dynamics R2 for multi-body systems
(from R2=0.92 to R2>0.99 on acceleration dimensions).

**Implementation**: New `LagrangianDecomposer` class that takes the causal
graph and automates the M/C/G decomposition, running PySR on each component.

**Status**: COMPLETE. `circ_rl/hypothesis/lagrangian_decomposition.py` (994
lines). Features: automatic 2-DOF structure detection, EL feature matrix
construction, per-env NLS regression with RK4 forward model, parametric
template fitting (polynomial functions of m1, m2, l1, l2), symbolic
composition via M^(-1). Vectorized RK4 dynamics for MPPI rollouts (Euler
diverges at dt=0.2). Results: per-env NLS R2=1.0000 (all 10 envs), composed
R2=1.0000 (both acceleration dims).

---

## 4. Dynamics Quality Gating with Hybrid Models

**Problem**: PySR expressions with R2<0.95 produce unreliable multi-step
predictions. With R2=0.92 per step, a 100-step trajectory prediction diverges
significantly from reality within 20-30 steps, making iLQR/MPPI plans
unreliable beyond the first few replanning intervals.

**Solution**: Establish a quality gate in the pipeline based on expression R2:

| R2 Range | Strategy |
|----------|----------|
| >= 0.99 + structural consistency | Use PySR expression (current behavior) |
| >= 0.95, no structural consistency | Use PySR + shortened horizon + frequent replan |
| < 0.95 | Fall back to environment-as-model or learned correction |

The key insight: an expression with R2=0.92 is useful for computing
approximate gradients (for iLQR backward pass), but should not be trusted for
multi-step forward prediction. Use it for the backward pass (gradient
computation) but use the real environment for the forward pass.

For dimensions where PySR fails entirely, options include:
- **Environment-as-model**: step the actual env for dynamics prediction
- **Neural ODE**: train a small neural network on the transition data
- **Ensemble correction**: PySR expression + learned residual

**Generalizability**: Applies to any pipeline dimension where PySR produces
low-R2 expressions. The quality threshold automatically adapts to problem
complexity.

**Expected impact**: Prevents compounding model errors from degrading planning
quality. Most impactful when combined with MPPI (which tolerates model errors
better than iLQR).

**Implementation**: Add R2-based gating logic to the
`AnalyticPolicyDerivationStage`, with configurable thresholds.

---

## 5. Phase-Based Planning with Automatic Mode Detection

**Problem**: Different planning strategies are optimal for different phases of
a task. For swing-up: the energy-pumping phase benefits from MPPI (global
search), while the stabilization phase benefits from iLQR (local quadratic
optimization). Using a single planner for the entire task is suboptimal.

**Solution**: Automatically detect planning phases from the dynamics model and
current state:

1. **Energy deficit phase** (E << E*): Use MPPI with energy-augmented cost.
   The system needs to build energy through oscillations.

2. **Approach phase** (E near E*, far from goal configuration): Use MPPI with
   position cost. Energy is sufficient but the system needs to be steered
   toward the goal.

3. **Stabilization phase** (near goal state): Switch to iLQR with quadratic
   cost. iLQR excels at local stabilization around a target.

Phase transitions are triggered by state-dependent conditions computed from
the dynamics model:

    if |E(state) - E_goal| > threshold_energy:
        phase = ENERGY_PUMPING
    elif |state - goal_state| > threshold_position:
        phase = APPROACH
    else:
        phase = STABILIZATION

**Generalizability**: Most non-trivial control tasks have distinct phases.
The energy-based phase detection works for any mechanical system. For
non-mechanical systems, the "distance to goal in reachable set" serves the
same purpose.

**Expected impact**: Combines the best of both planners. MPPI handles the
globally non-convex phases, iLQR handles the locally quadratic stabilization.

**Implementation**: New `PhasePlanner` class in
`circ_rl/analytic_policy/phase_planner.py` that wraps multiple solvers and
switches between them based on automatically detected phase conditions.
Activated via `--phase` flag in `experiments/acrobot_v2.py`.

**Status**: COMPLETE. `circ_rl/analytic_policy/phase_planner.py` (~130 lines).
Features: state-dependent solver switching via `use_local_fn` callable,
warm-start resizing (truncate/pad) for horizon mismatches between solvers,
`ILQRSolution`-compatible output from both phases.

Phase detector for Acrobot: switch to iLQR when `energy_deficit_rel < 0.2`
AND `tip_height > 0.8`. iLQR solver uses numerical finite-difference
Jacobians from the Lagrangian dynamics function.

Results on Acrobot: mean=-135.8, 100% goal rate, avg 110 steps. The phase
detector did not trigger iLQR on any test env -- MPPI alone handled both
swing-up and stabilization. This confirms MPPI is sufficient for Acrobot.
The PhasePlanner infrastructure remains valuable for environments where MPPI
struggles near the goal (e.g., precision positioning, contact-rich tasks).

---

## Results Summary

Acrobot-v1 benchmark: 10 training envs, 15 OOD test envs (varied m1, m2, l1, l2).

| Configuration | Mean Return | Goal Rate | Avg Steps |
|---------------|------------|-----------|-----------|
| iLQR (PySR dynamics) | -394 | ~60% | -- |
| MPPI (PySR Euler dynamics) | -649 | ~20% | -- |
| MPPI + Lagrangian RK4 (Solutions 1+3) | -147.7 | 100% | 141 |
| MPPI + Lagrangian RK4 + Energy (Solutions 1+2+3) | **-134.6** | **100%** | **113** |
| MPPI + Lagrangian RK4 + Energy + Phase (Solutions 1+2+3+5) | -135.8 | 100% | 110 |

OpenAI Gym leaderboard #1: -42.37 (discrete actions, single fixed env).
Our result generalizes across 15 OOD environments with continuous torque.
Best single-env result: -59.3 (55 steps).

---

## Priority Ranking

| Priority | Solution | Status | Impact |
|----------|----------|--------|--------|
| 1 | MPPI Planning | **DONE** | Solved swing-up failure |
| 2 | Energy-Based Cost | **DONE** | 8.9% return improvement, 20% fewer steps |
| 3 | Lagrangian Decomposition | **DONE** | Fixed multi-body dynamics (R2=1.0) |
| 4 | Quality Gating | TODO | Prevents model error compounding |
| 5 | Phase-Based Planning | **DONE** | Infrastructure ready; MPPI sufficient for Acrobot |

All implemented solutions (1, 2, 3, 5) combined achieve 100% goal rate on
15 OOD Acrobot environments. The primary bottleneck is now the environment
itself (Acrobot dynamics complexity) rather than the planning algorithm.
Phase-based switching will be most impactful on environments with distinct
stabilization requirements (e.g., balancing near an unstable equilibrium).
