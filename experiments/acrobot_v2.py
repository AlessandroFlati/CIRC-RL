# ruff: noqa: T201
"""CIRC-RL v2 pipeline on Acrobot-v1: symbolic dynamics + iLQR control.

Validates that the v2 pipeline (PySR symbolic regression -> falsification ->
iLQR) generalizes to a second, harder benchmark: the 2-link Acrobot with
coupled Lagrangian dynamics, 4D canonical state, and continuous torque.

The experiment:
1. Wraps Acrobot-v1 with continuous actions + shaped reward
2. Creates environment families with varied link masses/lengths
3. Runs the full v2 pipeline (causal discovery -> hypothesis falsification)
4. Builds iLQR policy from validated symbolic dynamics
5. Evaluates on 50 OOD test environments (continuous + discrete)
6. Compares against random and zero-torque baselines

Usage::

    uv run python experiments/acrobot_v2.py

Requires PySR: ``uv sync --extra symbolic``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from loguru import logger

from circ_rl.analytic_policy.ilqr_solver import (
    ILQRConfig,
    ILQRSolver,
    make_quadratic_terminal_cost,
)
from circ_rl.analytic_policy.mppi_solver import MPPIConfig, MPPISolver
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.hypothesis.derived_features import DerivedFeatureSpec
from circ_rl.hypothesis.symbolic_regressor import SymbolicRegressionConfig
from circ_rl.orchestration.stages import (
    CausalDiscoveryStage,
    FeatureSelectionStage,
    HypothesisFalsificationStage,
    HypothesisGenerationStage,
    ObservationAnalysisStage,
    TransitionAnalysisStage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.hypothesis.expression import SymbolicExpression
    from circ_rl.hypothesis.hypothesis_register import HypothesisEntry
    from circ_rl.orchestration.stages import _ILQRAnalyticPolicy


# ---------------------------------------------------------------------------
# Continuous Acrobot wrapper
# ---------------------------------------------------------------------------


def _rk4(
    derivs: Callable[[np.ndarray], tuple[float, ...]],
    y0: np.ndarray,
    t: list[float],
) -> np.ndarray:
    """4th-order Runge-Kutta integrator (matches Gymnasium's rk4).

    :param derivs: ODE right-hand side, ``y' = f(y)``.
    :param y0: Initial state vector.
    :param t: Time points ``[t0, t1]``.
    :returns: State at final time, first 4 components only.
    """
    ny = len(y0)
    yout = np.zeros((len(t), ny), dtype=np.float64)
    yout[0] = y0

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dt2 = dt / 2.0
        y = yout[i]

        k1 = np.asarray(derivs(y))
        k2 = np.asarray(derivs(y + dt2 * k1))
        k3 = np.asarray(derivs(y + dt2 * k2))
        k4 = np.asarray(derivs(y + dt * k3))
        yout[i + 1] = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return yout[-1][:4]


def _wrap(x: float, low: float, high: float) -> float:
    """Wrap x into [low, high) (circular)."""
    diff = high - low
    while x > high:
        x -= diff
    while x < low:
        x += diff
    return x


class ContinuousAcrobotEnv(AcrobotEnv):
    """Acrobot with continuous torque and shaped reward.

    Subclasses Gymnasium's ``AcrobotEnv`` and overrides ``step()`` to:
    - Accept continuous actions in ``Box(-1, 1, (1,))``
    - Apply continuous torque directly (bypassing ``AVAIL_TORQUE``)
    - Compute a shaped reward: tip height minus action/velocity penalties

    The shaped reward is::

        R = -cos(theta1) - cos(theta1 + theta2)
            - 0.01 * torque^2
            - 0.001 * (dtheta1^2 + dtheta2^2)

    Physics parameters (``LINK_MASS_*``, ``LINK_LENGTH_*``, etc.) are
    inherited from ``AcrobotEnv`` and can be set via ``setattr``.
    """

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step with continuous torque and shaped reward.

        :param action: Continuous torque in ``[-1, 1]``, shape ``(1,)``.
        :returns: ``(obs, reward, terminated, truncated, info)``.
        """
        s = self.state
        assert s is not None, "Call reset before using ContinuousAcrobotEnv."
        torque = float(np.clip(action[0], -1.0, 1.0))

        # Add noise if configured
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max,
            )

        # RK4 integration with continuous torque
        s_augmented = np.append(s, torque)
        ns = _rk4(self._dsdt, s_augmented, [0, self.dt])

        # Wrap angles, clip velocities (same as AcrobotEnv)
        ns[0] = _wrap(ns[0], -pi, pi)
        ns[1] = _wrap(ns[1], -pi, pi)
        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        # Terminal check: tip height > 1.0 above pivot
        terminated = bool(-cos(ns[0]) - cos(ns[0] + ns[1]) > 1.0)

        # Shaped reward: tip height - action penalty - velocity penalty
        tip_height = -cos(ns[0]) - cos(ns[0] + ns[1])
        reward = float(
            tip_height
            - 0.01 * torque ** 2
            - 0.001 * (ns[2] ** 2 + ns[3] ** 2)
        )

        if self.render_mode == "human":
            self.render()

        obs = self._get_ob()
        return obs, reward, terminated, False, {}


# ---------------------------------------------------------------------------
# Environment family factories
# ---------------------------------------------------------------------------

# Parameters to vary (all settable via setattr on AcrobotEnv).
# Gravity is hardcoded in _dsdt, so we skip it.
ACROBOT_PARAM_DISTS_TRAIN: dict[str, tuple[float, float]] = {
    "LINK_MASS_1": (0.7, 1.5),
    "LINK_MASS_2": (0.7, 1.5),
    "LINK_LENGTH_1": (0.7, 1.5),
    "LINK_LENGTH_2": (0.7, 1.5),
}

ACROBOT_PARAM_DISTS_TEST: dict[str, tuple[float, float]] = {
    "LINK_MASS_1": (0.5, 2.0),
    "LINK_MASS_2": (0.5, 2.0),
    "LINK_LENGTH_1": (0.5, 2.0),
    "LINK_LENGTH_2": (0.5, 2.0),
}


def _make_continuous_acrobot(params: dict[str, float]) -> ContinuousAcrobotEnv:
    """Factory for EnvironmentFamily: create and configure ContinuousAcrobotEnv."""
    env = ContinuousAcrobotEnv()
    for attr, value in params.items():
        setattr(env, attr, value)
    # Keep COM proportional to link length
    if "LINK_LENGTH_1" in params:
        env.LINK_COM_POS_1 = params["LINK_LENGTH_1"] / 2.0
    if "LINK_LENGTH_2" in params:
        env.LINK_COM_POS_2 = params["LINK_LENGTH_2"] / 2.0
    return env


def _make_acrobot_families(
    n_train: int = 25,
    n_test: int = 50,
    seed: int = 42,
) -> tuple[EnvironmentFamily, EnvironmentFamily]:
    """Create training and test environment families.

    :returns: ``(train_family, test_family)``.
    """
    train_family = EnvironmentFamily(
        env_factory=_make_continuous_acrobot,
        param_distributions=ACROBOT_PARAM_DISTS_TRAIN,
        n_envs=n_train,
        seed=seed,
    )
    test_family = EnvironmentFamily(
        env_factory=_make_continuous_acrobot,
        param_distributions=ACROBOT_PARAM_DISTS_TEST,
        n_envs=n_test,
        seed=seed + 1000,
    )
    return train_family, test_family


# ---------------------------------------------------------------------------
# v2 pipeline runner (stages 1-6)
# ---------------------------------------------------------------------------


def _run_pipeline(
    train_family: EnvironmentFamily,
    seed: int = 42,
) -> tuple[
    dict[str, HypothesisEntry],
    HypothesisEntry | None,
    list[str],
    list[str],
    list[DerivedFeatureSpec],
    dict[str, Any],
    dict[str, Any],
    list[str],
    dict[str, Any],
]:
    """Run v2 pipeline stages 1-6 on training envs.

    :returns: ``(best_dynamics, best_reward, dynamics_state_names,
        action_names, reward_derived_features, hf_output, oa_output,
        obs_state_names, cd_output)``.
    """
    n_transitions = 5000

    # PySR config tuned for Acrobot: need sin/cos for Lagrangian dynamics
    sr_config = SymbolicRegressionConfig(
        max_complexity=35,
        n_iterations=100,
        populations=30,
        binary_operators=("+", "-", "*", "/"),
        unary_operators=("sin", "cos", "square"),
        parsimony=0.0003,
        timeout_seconds=600,
        deterministic=True,
        random_state=seed,
        nested_constraints={
            "*": {"+": 0, "-": 0},
            "/": {"+": 0, "-": 0},
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
        },
        complexity_of_operators={"square": 1, "sin": 2, "cos": 2},
        constraints={"sin": 5, "cos": 5},
        max_samples=25000,
        n_sr_runs=3,
    )

    # Stage 1: Causal discovery
    print("\n  [1/6] Running causal discovery...")
    cd_stage = CausalDiscoveryStage(
        env_family=train_family,
        n_transitions_per_env=n_transitions,
        discovery_method="pc",
        alpha=0.05,
        seed=seed,
        include_env_params=True,
        ep_correlation_threshold=0.05,
    )
    cd_output = cd_stage.run({})
    state_names = cd_output["state_names"]
    print(f"    State names: {state_names}")

    # Stage 2: Feature selection
    print("  [2/6] Running feature selection...")
    fs_stage = FeatureSelectionStage(
        epsilon=0.15,
        min_ate=0.01,
        use_mechanism_invariance=True,
    )
    fs_output = fs_stage.run({"causal_discovery": cd_output})

    # Stage 3: Transition analysis
    print("  [3/6] Running transition analysis...")
    ta_stage = TransitionAnalysisStage()
    ta_output = ta_stage.run({
        "causal_discovery": cd_output,
        "feature_selection": fs_output,
    })

    # Stage 4: Observation analysis (circle constraint detection)
    print("  [4/6] Running observation analysis...")
    oa_stage = ObservationAnalysisStage()
    oa_output = oa_stage.run({"causal_discovery": cd_output})
    analysis_result = oa_output.get("analysis_result")
    if analysis_result is not None:
        print(f"    Canonical coords: {oa_output['canonical_state_names']}")
        for c in analysis_result.constraints:
            print(f"    Constraint: {c.constraint_type} on dims {c.involved_dims}")
    else:
        print("    No canonical mappings found (dynamics in obs space)")

    # Stage 5: Hypothesis generation (PySR)
    print("  [5/6] Running symbolic regression (PySR)...")
    hg_stage = HypothesisGenerationStage(
        include_env_params=True,
        sr_config=sr_config,
        reward_sr_config=sr_config,
    )
    hg_output = hg_stage.run({
        "causal_discovery": cd_output,
        "feature_selection": fs_output,
        "transition_analysis": ta_output,
        "observation_analysis": oa_output,
    })

    register = hg_output["register"]
    print(f"    Discovered {len(register.entries)} hypotheses")

    # Stage 6: Hypothesis falsification
    print("  [6/6] Running hypothesis falsification...")
    hf_stage = HypothesisFalsificationStage(
        structural_p_threshold=0.01,
        structural_min_relative_improvement=0.01,
        ood_confidence=0.99,
        held_out_fraction=0.2,
    )
    hf_output = hf_stage.run({
        "hypothesis_generation": hg_output,
        "causal_discovery": cd_output,
        "observation_analysis": oa_output,
    })

    best_dynamics = hf_output["best_dynamics"]
    best_reward = hf_output["best_reward"]
    reward_derived_features = hf_output.get("reward_derived_features", [])

    hf_result = hf_output["falsification_result"]
    print(
        f"    Tested: {hf_result.n_tested}, "
        f"Validated: {hf_result.n_validated}, "
        f"Falsified: {hf_result.n_falsified}"
    )

    if best_dynamics:
        for target, entry in best_dynamics.items():
            print(
                f"    [{target}] {entry.expression.expression_str}"
                f"  (R2={entry.training_r2:.4f})"
            )
    if best_reward:
        print(
            f"    [reward] {best_reward.expression.expression_str}"
            f"  (R2={best_reward.training_r2:.4f})"
        )

    action_names = hg_output.get("action_names", ["action"])
    dynamics_state_names = hf_output.get("dynamics_state_names", state_names)

    return (
        best_dynamics,
        best_reward,
        dynamics_state_names,
        action_names,
        reward_derived_features,
        hf_output,
        oa_output,
        state_names,
        cd_output,
    )


# ---------------------------------------------------------------------------
# iLQR policy builder
# ---------------------------------------------------------------------------


def _acrobot_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """Hand-crafted Acrobot reward: tip height - penalties.

    :param state: Canonical state ``[theta1, theta2, dtheta1, dtheta2]``.
    :param action: Continuous torque ``[u]``.
    :returns: Shaped reward.
    """
    t1, t2, dt1, dt2 = state[0], state[1], state[2], state[3]
    tip_height = -np.cos(t1) - np.cos(t1 + t2)
    return float(
        tip_height
        - 0.01 * action[0] ** 2
        - 0.001 * (dt1 ** 2 + dt2 ** 2)
    )


def _build_ilqr_policy(
    test_family: EnvironmentFamily,
    best_dynamics: dict[str, HypothesisEntry],
    state_names: list[str],
    action_names: list[str],
    oa_output: dict[str, Any] | None = None,
    cd_output: dict[str, Any] | None = None,
    adaptive_replan_multiplier: float = 3.0,
    min_replan_interval: int = 3,
) -> _ILQRAnalyticPolicy:
    """Build iLQR policy for Acrobot test environments.

    Uses hand-crafted reward (tip height) since Acrobot's sparse reward
    is unusable for iLQR cost optimization.

    Includes Section 7.2 improvements (calibration, spurious detection,
    adaptive replanning).

    :returns: An ``_ILQRAnalyticPolicy`` with ``get_action``/``reset``.
    """
    from circ_rl.orchestration.stages import (
        _build_dynamics_fn,
        _build_dynamics_jacobian_fns,
        _ILQRAnalyticPolicy,
    )

    max_action = 1.0

    # Canonical space from observation analysis
    analysis_result = None
    obs_to_canonical_fn = None
    angular_dims: tuple[int, ...] = ()
    if oa_output is not None:
        analysis_result = oa_output.get("analysis_result")
        if analysis_result is not None:
            obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
            angular_dims = analysis_result.angular_dims

    # Parse dynamics expressions by dim index
    state_dim = len(state_names)
    dynamics_expressions: dict[int, SymbolicExpression] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        dynamics_expressions[dim_idx] = entry.expression

    # -- Section 7.2: Spurious term detection --
    spurious_dataset = cd_output.get("dataset") if cd_output is not None else None
    if spurious_dataset is not None:
        from circ_rl.hypothesis.spurious_detection import SpuriousTermDetector

        detector = SpuriousTermDetector(r2_contribution_threshold=0.005)
        spvar_names = list(state_names) + list(action_names)
        if test_family.param_names:
            spvar_names.extend(test_family.param_names)

        sp_dataset = spurious_dataset
        if oa_output is not None:
            oa_canonical = oa_output.get("canonical_dataset")
            if oa_canonical is not None:
                sp_dataset = oa_canonical

        for dim_idx, expr in list(dynamics_expressions.items()):
            try:
                sp_result = detector.detect(
                    expr, sp_dataset, dim_idx, spvar_names,
                    wrap_angular=(dim_idx in angular_dims),
                )
                if sp_result.n_spurious > 0:
                    from circ_rl.hypothesis.expression import (
                        SymbolicExpression as _SPE,
                    )
                    dynamics_expressions[dim_idx] = _SPE.from_sympy(
                        sp_result.pruned_expr,
                    )
                    print(
                        f"    Pruned {sp_result.n_spurious} spurious terms "
                        f"from dim {dim_idx} "
                        f"(R2: {sp_result.original_r2:.6f} -> "
                        f"{sp_result.pruned_r2:.6f})"
                    )
            except Exception:
                logger.debug(
                    "Spurious detection skipped for dim {} (evaluation error)",
                    dim_idx,
                )

    # -- Section 7.2: Coefficient calibration (pooled) --
    pooled_cal: dict[int, tuple[float, float]] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        if entry.pooled_calibration is not None:
            pooled_cal[dim_idx] = entry.pooled_calibration
    cal_arg = pooled_cal or None
    if cal_arg:
        print(f"    Calibration: {len(cal_arg)} dims with pooled (alpha, beta)")

    # -- Section 7.2: Adaptive replanning threshold --
    dynamics_mse_sum = sum(e.training_mse for e in best_dynamics.values())
    adaptive_tau: float | None = None
    if dynamics_mse_sum > 0:
        adaptive_tau = adaptive_replan_multiplier * np.sqrt(dynamics_mse_sum)
    if adaptive_tau is not None:
        print(f"    Adaptive replan threshold: tau={adaptive_tau:.4f}")

    ilqr_config = ILQRConfig(
        horizon=100,
        gamma=0.99,
        max_action=max_action,
        n_random_restarts=10,
        restart_scale=0.3,
        replan_interval=10,
        adaptive_replan_threshold=adaptive_tau,
        min_replan_interval=min_replan_interval,
    )

    ilqr_solvers: dict[int, ILQRSolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        dynamics_fn = _build_dynamics_fn(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=None,
            obs_high=None,
            angular_dims=angular_dims,
            calibration_coefficients=cal_arg,
        )

        jac_state_fn, jac_action_fn = _build_dynamics_jacobian_fns(
            dynamics_expressions,
            state_names,
            action_names,
            state_dim,
            env_params,
            obs_low=None,
            obs_high=None,
            calibration_coefficients=cal_arg,
        )

        terminal_cost_fn = make_quadratic_terminal_cost(
            reward_fn=_acrobot_reward_fn,
            action_dim=1,
            gamma=0.99,
            state_dim=state_dim,
            scale_override=10.0,
        )

        ilqr_solvers[env_idx] = ILQRSolver(
            config=ilqr_config,
            dynamics_fn=dynamics_fn,
            reward_fn=_acrobot_reward_fn,
            dynamics_jac_state_fn=jac_state_fn,
            dynamics_jac_action_fn=jac_action_fn,
            terminal_cost_fn=terminal_cost_fn,
        )

    first_entry = next(iter(best_dynamics.values()))
    action_dim = 1

    return _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=None,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=test_family.n_envs,
        ilqr_solvers=ilqr_solvers,
        action_low=-max_action * np.ones(action_dim),
        action_high=max_action * np.ones(action_dim),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )


# ---------------------------------------------------------------------------
# MPPI policy builder
# ---------------------------------------------------------------------------


def _acrobot_batched_reward_fn(
    states: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    """Vectorized Acrobot reward: (K, 4), (K, 1) -> (K,).

    :param states: Canonical states ``[theta1, theta2, dtheta1, dtheta2]``.
    :param actions: Continuous torques.
    :returns: Shaped rewards for each sample.
    """
    t1 = states[:, 0]  # (K,)
    t2 = states[:, 1]  # (K,)
    dt1 = states[:, 2]  # (K,)
    dt2 = states[:, 3]  # (K,)
    u = actions[:, 0]  # (K,)

    tip_height = -np.cos(t1) - np.cos(t1 + t2)  # (K,)
    return tip_height - 0.01 * u**2 - 0.001 * (dt1**2 + dt2**2)  # (K,)


def _make_energy_reward_fn(
    coeffs: dict[str, float],
    e_goal: float,
    energy_weight: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create a scalar energy-shaped reward function for a specific env.

    Composite reward: position_reward - energy_penalty - action/velocity costs.
    The energy penalty guides MPPI to pump energy toward E* before focusing
    on tip position.

    :param coeffs: EL coefficient values for this environment.
    :param e_goal: Target mechanical energy (upright, stationary).
    :param energy_weight: Weight for the energy deficit penalty.
    :returns: Reward function ``(state, action) -> float``.
    """
    from circ_rl.hypothesis.lagrangian_decomposition import (
        compute_mechanical_energy,
    )

    _coeffs = coeffs
    _eg = e_goal
    _ew = energy_weight

    def reward_fn(state: np.ndarray, action: np.ndarray) -> float:
        t1, t2, dt1, dt2 = state[0], state[1], state[2], state[3]
        tip_height = -np.cos(t1) - np.cos(t1 + t2)

        # Energy deficit: normalized squared error
        e_current = compute_mechanical_energy(state, _coeffs)
        e_deficit = ((e_current - _eg) / max(abs(_eg), 1e-6)) ** 2

        return float(
            tip_height
            - _ew * e_deficit
            - 0.01 * action[0] ** 2
            - 0.001 * (dt1 ** 2 + dt2 ** 2)
        )

    return reward_fn


def _make_energy_batched_reward_fn(
    coeffs: dict[str, float],
    e_goal: float,
    energy_weight: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create a vectorized energy-shaped reward for K-parallel MPPI rollouts.

    :param coeffs: EL coefficient values for this environment.
    :param e_goal: Target mechanical energy.
    :param energy_weight: Weight for energy deficit penalty.
    :returns: Reward function ``(states, actions) -> rewards``
        where shapes are ``(K, 4), (K, 1) -> (K,)``.
    """
    from circ_rl.hypothesis.lagrangian_decomposition import (
        compute_mechanical_energy_batched,
    )

    _coeffs = coeffs
    _eg = e_goal
    _ew = energy_weight

    def batched_reward_fn(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        t1 = states[:, 0]  # (K,)
        t2 = states[:, 1]  # (K,)
        dt1 = states[:, 2]  # (K,)
        dt2 = states[:, 3]  # (K,)
        u = actions[:, 0]  # (K,)

        tip_height = -np.cos(t1) - np.cos(t1 + t2)  # (K,)

        e_current = compute_mechanical_energy_batched(states, _coeffs)  # (K,)
        e_deficit = ((e_current - _eg) / max(abs(_eg), 1e-6)) ** 2  # (K,)

        return (
            tip_height
            - _ew * e_deficit
            - 0.01 * u**2
            - 0.001 * (dt1**2 + dt2**2)
        )  # (K,)

    return batched_reward_fn


def _build_mppi_policy(
    test_family: EnvironmentFamily,
    best_dynamics: dict[str, HypothesisEntry],
    state_names: list[str],
    action_names: list[str],
    oa_output: dict[str, Any] | None = None,
    cd_output: dict[str, Any] | None = None,
    adaptive_replan_multiplier: float = 3.0,
    min_replan_interval: int = 3,
    lagrangian_templates: dict[str, Any] | None = None,
    use_energy_shaping: bool = False,
    energy_weight: float = 1.0,
) -> _ILQRAnalyticPolicy:
    """Build MPPI policy for Acrobot test environments.

    Uses sampling-based trajectory optimization instead of iLQR.
    Vectorized dynamics via numpy broadcasting for fast K-parallel rollouts.

    :returns: An ``_ILQRAnalyticPolicy`` wrapping MPPI solvers.
    """
    from circ_rl.analytic_policy.fast_dynamics import build_batched_dynamics_fn
    from circ_rl.orchestration.stages import (
        _build_dynamics_fn,
        _ILQRAnalyticPolicy,
    )

    max_action = 1.0

    # Canonical space from observation analysis
    analysis_result = None
    obs_to_canonical_fn = None
    angular_dims: tuple[int, ...] = ()
    if oa_output is not None:
        analysis_result = oa_output.get("analysis_result")
        if analysis_result is not None:
            obs_to_canonical_fn = analysis_result.obs_to_canonical_fn
            angular_dims = analysis_result.angular_dims

    # Parse dynamics expressions by dim index
    state_dim = len(state_names)
    dynamics_expressions: dict[int, SymbolicExpression] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        dynamics_expressions[dim_idx] = entry.expression

    # -- Spurious term detection (same as iLQR) --
    spurious_dataset = cd_output.get("dataset") if cd_output is not None else None
    if spurious_dataset is not None:
        from circ_rl.hypothesis.spurious_detection import SpuriousTermDetector

        detector = SpuriousTermDetector(r2_contribution_threshold=0.005)
        spvar_names = list(state_names) + list(action_names)
        if test_family.param_names:
            spvar_names.extend(test_family.param_names)

        sp_dataset = spurious_dataset
        if oa_output is not None:
            oa_canonical = oa_output.get("canonical_dataset")
            if oa_canonical is not None:
                sp_dataset = oa_canonical

        for dim_idx, expr in list(dynamics_expressions.items()):
            try:
                sp_result = detector.detect(
                    expr, sp_dataset, dim_idx, spvar_names,
                    wrap_angular=(dim_idx in angular_dims),
                )
                if sp_result.n_spurious > 0:
                    from circ_rl.hypothesis.expression import (
                        SymbolicExpression as _SPE,
                    )
                    dynamics_expressions[dim_idx] = _SPE.from_sympy(
                        sp_result.pruned_expr,
                    )
                    print(
                        f"    Pruned {sp_result.n_spurious} spurious terms "
                        f"from dim {dim_idx} "
                        f"(R2: {sp_result.original_r2:.6f} -> "
                        f"{sp_result.pruned_r2:.6f})"
                    )
            except Exception:
                logger.debug(
                    "Spurious detection skipped for dim {} (evaluation error)",
                    dim_idx,
                )

    # -- Coefficient calibration (pooled) --
    pooled_cal: dict[int, tuple[float, float]] = {}
    for target, entry in best_dynamics.items():
        dim_name = target.removeprefix("delta_")
        dim_idx = state_names.index(dim_name)
        if entry.pooled_calibration is not None:
            pooled_cal[dim_idx] = entry.pooled_calibration
    cal_arg = pooled_cal or None
    if cal_arg:
        print(f"    Calibration: {len(cal_arg)} dims with pooled (alpha, beta)")

    # -- Adaptive replanning threshold --
    dynamics_mse_sum = sum(e.training_mse for e in best_dynamics.values())
    adaptive_tau: float | None = None
    if dynamics_mse_sum > 0:
        adaptive_tau = adaptive_replan_multiplier * np.sqrt(dynamics_mse_sum)
    if adaptive_tau is not None:
        print(f"    Adaptive replan threshold: tau={adaptive_tau:.4f}")

    mppi_config = MPPIConfig(
        horizon=100,
        n_samples=512,
        temperature=0.1,
        noise_sigma=0.5,
        n_iterations=3,
        gamma=0.99,
        max_action=max_action,
        colored_noise_beta=1.0,
        replan_interval=5,
        adaptive_replan_threshold=adaptive_tau,
        min_replan_interval=min_replan_interval,
    )

    mppi_solvers: dict[int, MPPISolver] = {}
    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        # Use Lagrangian RK4 dynamics if templates available
        if lagrangian_templates is not None:
            from circ_rl.hypothesis.lagrangian_decomposition import (
                build_lagrangian_batched_dynamics_fn,
                build_lagrangian_scalar_dynamics_fn,
            )
            dynamics_fn = build_lagrangian_scalar_dynamics_fn(
                lagrangian_templates, env_params,
            )
            batched_dynamics_fn = build_lagrangian_batched_dynamics_fn(
                lagrangian_templates, env_params,
            )
        else:
            # Scalar dynamics for final rollout
            dynamics_fn = _build_dynamics_fn(
                dynamics_expressions,
                state_names,
                action_names,
                state_dim,
                env_params,
                obs_low=None,
                obs_high=None,
                angular_dims=angular_dims,
                calibration_coefficients=cal_arg,
            )

            # Vectorized dynamics for K-parallel MPPI rollouts
            batched_dynamics_fn = build_batched_dynamics_fn(
                dynamics_expressions,
                state_names,
                action_names,
                state_dim,
                env_params,
                angular_dims=angular_dims,
                calibration_coefficients=cal_arg,
            )

        # Choose reward functions (energy-shaped or standard)
        if use_energy_shaping and lagrangian_templates is not None:
            from circ_rl.hypothesis.lagrangian_decomposition import (
                compute_goal_energy,
                evaluate_coefficients,
            )
            env_coeffs = evaluate_coefficients(
                lagrangian_templates, env_params,
            )
            e_goal = compute_goal_energy(env_coeffs)
            scalar_reward_fn = _make_energy_reward_fn(
                env_coeffs, e_goal, energy_weight,
            )
            batched_reward_fn = _make_energy_batched_reward_fn(
                env_coeffs, e_goal, energy_weight,
            )
        else:
            scalar_reward_fn = _acrobot_reward_fn
            batched_reward_fn = _acrobot_batched_reward_fn

        mppi_solvers[env_idx] = MPPISolver(
            config=mppi_config,
            dynamics_fn=dynamics_fn,
            reward_fn=scalar_reward_fn,
            batched_dynamics_fn=batched_dynamics_fn,
            batched_reward_fn=batched_reward_fn,
        )

    first_entry = next(iter(best_dynamics.values()))
    action_dim = 1

    return _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=None,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=test_family.n_envs,
        ilqr_solvers=mppi_solvers,  # type: ignore[arg-type]
        action_low=-max_action * np.ones(action_dim),
        action_high=max_action * np.ones(action_dim),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )


# ---------------------------------------------------------------------------
# Phase-based policy builder (MPPI + iLQR switching)
# ---------------------------------------------------------------------------


def _make_numerical_jacobian_fns(
    dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state_dim: int,
    action_dim: int,
    eps: float = 1e-5,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]:
    """Create numerical Jacobian functions from a dynamics function.

    Uses forward finite differences for df/dx and df/du.

    :param dynamics_fn: ``(state, action) -> next_state``.
    :param state_dim: State dimensionality.
    :param action_dim: Action dimensionality.
    :param eps: Finite difference step size.
    :returns: ``(jac_state_fn, jac_action_fn)`` where each returns
        a numpy array of the appropriate Jacobian shape.
    """
    _eps = eps

    def jac_state_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        f0 = dynamics_fn(state, action)  # (S,)
        jac = np.zeros((state_dim, state_dim))
        for i in range(state_dim):
            s_plus = state.copy()
            s_plus[i] += _eps
            jac[:, i] = (dynamics_fn(s_plus, action) - f0) / _eps
        return jac

    def jac_action_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        f0 = dynamics_fn(state, action)  # (S,)
        jac = np.zeros((state_dim, action_dim))
        for i in range(action_dim):
            a_plus = action.copy()
            a_plus[i] += _eps
            jac[:, i] = (dynamics_fn(state, a_plus) - f0) / _eps
        return jac

    return jac_state_fn, jac_action_fn


def _build_phase_policy(
    test_family: EnvironmentFamily,
    best_dynamics: dict[str, HypothesisEntry],
    state_names: list[str],
    action_names: list[str],
    lagrangian_templates: dict[str, Any],
    oa_output: dict[str, Any] | None = None,
    cd_output: dict[str, Any] | None = None,
    adaptive_replan_multiplier: float = 3.0,
    min_replan_interval: int = 3,
    energy_weight: float = 1.0,
    energy_threshold: float = 0.2,
    tip_threshold: float = 0.8,
) -> _ILQRAnalyticPolicy:
    """Build phase-based policy: MPPI for energy pumping, iLQR for stabilization.

    Creates per-env ``PhasePlanner`` instances that automatically switch
    between MPPI (with energy-shaped reward) and iLQR (with position-based
    reward) based on the system's mechanical energy and tip height.

    Phase detection: switch to iLQR when both:
    - Energy deficit |E - E*|/|E*| < ``energy_threshold``
    - Tip height > ``tip_threshold``

    :param lagrangian_templates: Required. Parametric EL coefficient templates.
    :param energy_threshold: Relative energy deficit threshold for iLQR switch.
    :param tip_threshold: Tip height threshold for iLQR switch.
    :returns: An ``_ILQRAnalyticPolicy`` wrapping PhasePlanner instances.
    """
    from circ_rl.analytic_policy.phase_planner import PhasePlanner
    from circ_rl.hypothesis.lagrangian_decomposition import (
        build_lagrangian_batched_dynamics_fn,
        build_lagrangian_scalar_dynamics_fn,
        compute_goal_energy,
        compute_mechanical_energy,
        evaluate_coefficients,
    )
    from circ_rl.orchestration.stages import _ILQRAnalyticPolicy

    max_action = 1.0
    state_dim = len(state_names)

    # Canonical space from observation analysis
    obs_to_canonical_fn = None
    if oa_output is not None:
        analysis_result = oa_output.get("analysis_result")
        if analysis_result is not None:
            obs_to_canonical_fn = analysis_result.obs_to_canonical_fn

    # -- Adaptive replanning threshold --
    dynamics_mse_sum = sum(e.training_mse for e in best_dynamics.values())
    adaptive_tau: float | None = None
    if dynamics_mse_sum > 0:
        adaptive_tau = adaptive_replan_multiplier * np.sqrt(dynamics_mse_sum)
    if adaptive_tau is not None:
        print(f"    Adaptive replan threshold: tau={adaptive_tau:.4f}")

    # MPPI config: global exploration with energy-shaped reward
    mppi_config = MPPIConfig(
        horizon=100,
        n_samples=512,
        temperature=0.1,
        noise_sigma=0.5,
        n_iterations=3,
        gamma=0.99,
        max_action=max_action,
        colored_noise_beta=1.0,
        replan_interval=5,
        adaptive_replan_threshold=adaptive_tau,
        min_replan_interval=min_replan_interval,
    )

    # iLQR config: local stabilization near goal
    ilqr_config = ILQRConfig(
        horizon=50,
        gamma=0.99,
        max_action=max_action,
        n_random_restarts=3,
        restart_scale=0.1,
        replan_interval=5,
        adaptive_replan_threshold=adaptive_tau,
        min_replan_interval=min_replan_interval,
    )

    phase_planners: dict[int, PhasePlanner] = {}

    for env_idx in range(test_family.n_envs):
        env_params = test_family.get_env_params(env_idx)

        # Lagrangian-based dynamics (RK4)
        dynamics_fn = build_lagrangian_scalar_dynamics_fn(
            lagrangian_templates, env_params,
        )
        batched_dynamics_fn = build_lagrangian_batched_dynamics_fn(
            lagrangian_templates, env_params,
        )

        # Energy functions for this env
        env_coeffs = evaluate_coefficients(lagrangian_templates, env_params)
        e_goal = compute_goal_energy(env_coeffs)

        # --- MPPI solver (energy-shaped reward) ---
        energy_reward = _make_energy_reward_fn(
            env_coeffs, e_goal, energy_weight,
        )
        batched_energy_reward = _make_energy_batched_reward_fn(
            env_coeffs, e_goal, energy_weight,
        )
        mppi_solver = MPPISolver(
            config=mppi_config,
            dynamics_fn=dynamics_fn,
            reward_fn=energy_reward,
            batched_dynamics_fn=batched_dynamics_fn,
            batched_reward_fn=batched_energy_reward,
        )

        # --- iLQR solver (position-based reward for stabilization) ---
        jac_state_fn, jac_action_fn = _make_numerical_jacobian_fns(
            dynamics_fn, state_dim, 1,
        )
        terminal_cost_fn = make_quadratic_terminal_cost(
            reward_fn=_acrobot_reward_fn,
            action_dim=1,
            gamma=0.99,
            state_dim=state_dim,
            scale_override=10.0,
        )
        ilqr_solver = ILQRSolver(
            config=ilqr_config,
            dynamics_fn=dynamics_fn,
            reward_fn=_acrobot_reward_fn,
            dynamics_jac_state_fn=jac_state_fn,
            dynamics_jac_action_fn=jac_action_fn,
            terminal_cost_fn=terminal_cost_fn,
        )

        # --- Phase detector ---
        _coeffs = env_coeffs
        _eg = e_goal
        _et = energy_threshold
        _tt = tip_threshold

        def use_local(
            state: np.ndarray,
            coeffs: dict[str, float] = _coeffs,
            eg: float = _eg,
            et: float = _et,
            tt: float = _tt,
        ) -> bool:
            e_curr = compute_mechanical_energy(state, coeffs)
            e_deficit_rel = abs(e_curr - eg) / max(abs(eg), 1e-6)
            tip = -np.cos(state[0]) - np.cos(state[0] + state[1])
            return bool(e_deficit_rel < et and tip > tt)

        phase_planners[env_idx] = PhasePlanner(
            global_solver=mppi_solver,
            local_solver=ilqr_solver,
            use_local_fn=use_local,
        )

    first_entry = next(iter(best_dynamics.values()))
    action_dim = 1

    print(f"    Phase thresholds: energy_deficit<{energy_threshold}, "
          f"tip_height>{tip_threshold}")

    return _ILQRAnalyticPolicy(
        dynamics_hypothesis=first_entry,
        reward_hypothesis=None,
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=test_family.n_envs,
        ilqr_solvers=phase_planners,  # type: ignore[arg-type]
        action_low=-max_action * np.ones(action_dim),
        action_high=max_action * np.ones(action_dim),
        obs_to_canonical_fn=obs_to_canonical_fn,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_continuous(
    test_family: EnvironmentFamily,
    policy: _ILQRAnalyticPolicy,
    max_steps: int = 500,
) -> dict[int, dict[str, float]]:
    """Evaluate iLQR policy on continuous Acrobot envs.

    :returns: Dict mapping env_idx to ``{"return": ..., "steps": ...,
        "reached_goal": ...}``.
    """
    results: dict[int, dict[str, float]] = {}

    for env_idx in range(test_family.n_envs):
        env = test_family.make_env(env_idx)
        obs, _ = env.reset(seed=42)

        policy.reset(env_idx)
        total_reward = 0.0
        reached_goal = False

        for step in range(max_steps):
            action = policy.get_action(obs, env_idx)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated:
                reached_goal = True
                results[env_idx] = {
                    "return": total_reward,
                    "steps": float(step + 1),
                    "reached_goal": 1.0,
                }
                break
            if truncated:
                break

        if env_idx not in results:
            results[env_idx] = {
                "return": total_reward,
                "steps": float(max_steps),
                "reached_goal": 0.0,
            }

        env.close()

        params = test_family.get_env_params(env_idx)
        goal_str = "GOAL" if reached_goal else "----"
        print(
            f"  [{env_idx:3d}] R={total_reward:7.1f} "
            f"steps={results[env_idx]['steps']:3.0f} "
            f"{goal_str}  "
            f"m1={params['LINK_MASS_1']:.2f} m2={params['LINK_MASS_2']:.2f} "
            f"l1={params['LINK_LENGTH_1']:.2f} l2={params['LINK_LENGTH_2']:.2f}"
        )

    return results


def _discretize_action(continuous_action: np.ndarray) -> int:
    """Map continuous torque to discrete Acrobot action.

    :param continuous_action: Shape ``(1,)``, in ``[-1, 1]``.
    :returns: 0 (-1 torque), 1 (0 torque), or 2 (+1 torque).
    """
    u = float(continuous_action[0])
    if u < -0.33:
        return 0
    elif u > 0.33:
        return 2
    else:
        return 1


def _eval_discrete(
    test_family: EnvironmentFamily,
    policy: _ILQRAnalyticPolicy,
    max_steps: int = 500,
) -> dict[int, dict[str, float]]:
    """Evaluate iLQR policy on real discrete Acrobot-v1.

    Maps continuous actions to discrete via threshold discretization.

    :returns: Dict mapping env_idx to ``{"return": ..., "steps": ...,
        "reached_goal": ...}``.
    """
    results: dict[int, dict[str, float]] = {}

    for env_idx in range(test_family.n_envs):
        params = test_family.get_env_params(env_idx)
        env = gym.make("Acrobot-v1", max_episode_steps=max_steps)
        for attr, value in params.items():
            setattr(env.unwrapped, attr, value)
        # Match COM proportional to length
        if "LINK_LENGTH_1" in params:
            env.unwrapped.LINK_COM_POS_1 = params["LINK_LENGTH_1"] / 2.0  # type: ignore[union-attr]
        if "LINK_LENGTH_2" in params:
            env.unwrapped.LINK_COM_POS_2 = params["LINK_LENGTH_2"] / 2.0  # type: ignore[union-attr]

        obs, _ = env.reset(seed=42)

        policy.reset(env_idx)
        total_reward = 0.0
        reached_goal = False

        for step in range(max_steps):
            continuous_action = policy.get_action(obs, env_idx)
            continuous_action = np.clip(continuous_action, -1.0, 1.0)
            discrete_action = _discretize_action(continuous_action)

            obs, reward, terminated, truncated, _ = env.step(discrete_action)
            total_reward += float(reward)
            if terminated:
                reached_goal = True
                results[env_idx] = {
                    "return": total_reward,
                    "steps": float(step + 1),
                    "reached_goal": 1.0,
                }
                break
            if truncated:
                break

        if env_idx not in results:
            results[env_idx] = {
                "return": total_reward,
                "steps": float(max_steps),
                "reached_goal": 0.0,
            }

        env.close()

    return results


def _eval_baselines(
    test_family: EnvironmentFamily,
    max_steps: int = 500,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Evaluate random and zero-torque baselines on discrete Acrobot-v1.

    :returns: ``(random_results, zero_results)``.
    """
    random_results: dict[int, dict[str, float]] = {}
    zero_results: dict[int, dict[str, float]] = {}

    for env_idx in range(test_family.n_envs):
        params = test_family.get_env_params(env_idx)

        for baseline, results_dict in [
            ("random", random_results),
            ("zero", zero_results),
        ]:
            env = gym.make("Acrobot-v1", max_episode_steps=max_steps)
            for attr, value in params.items():
                setattr(env.unwrapped, attr, value)
            if "LINK_LENGTH_1" in params:
                env.unwrapped.LINK_COM_POS_1 = params["LINK_LENGTH_1"] / 2.0  # type: ignore[union-attr]
            if "LINK_LENGTH_2" in params:
                env.unwrapped.LINK_COM_POS_2 = params["LINK_LENGTH_2"] / 2.0  # type: ignore[union-attr]

            obs, _ = env.reset(seed=42)
            total_reward = 0.0
            rng = np.random.default_rng(env_idx)

            for step in range(max_steps):
                if baseline == "random":
                    action = int(rng.integers(0, 3))
                else:
                    action = 1  # zero torque

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated:
                    results_dict[env_idx] = {
                        "return": total_reward,
                        "steps": float(step + 1),
                        "reached_goal": 1.0,
                    }
                    break
                if truncated:
                    break

            if env_idx not in results_dict:
                results_dict[env_idx] = {
                    "return": total_reward,
                    "steps": float(max_steps),
                    "reached_goal": 0.0,
                }

            env.close()

    return random_results, zero_results


def _print_results_table(
    label: str,
    results: dict[int, dict[str, float]],
) -> None:
    """Print summary statistics for evaluation results."""
    returns = np.array([r["return"] for r in results.values()])
    steps = np.array([r["steps"] for r in results.values()])
    goal_rate = np.mean([r["reached_goal"] for r in results.values()])

    print(f"  {label:<25} "
          f"mean={returns.mean():8.1f}  "
          f"med={np.median(returns):8.1f}  "
          f"worst={returns.min():8.1f}  "
          f"std={returns.std():7.1f}  "
          f"goal={goal_rate * 100:5.1f}%  "
          f"avg_steps={steps.mean():5.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CIRC-RL v2 pipeline on Acrobot-v1",
    )
    parser.add_argument(
        "--n-train", type=int, default=25,
        help="Number of training environments (default: 25).",
    )
    parser.add_argument(
        "--n-test", type=int, default=50,
        help="Number of OOD test environments (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode (default: 500).",
    )
    parser.add_argument(
        "--skip-discrete", action="store_true",
        help="Skip discrete Acrobot-v1 evaluation.",
    )
    parser.add_argument(
        "--solver", choices=["ilqr", "mppi"], default="ilqr",
        help="Solver to use: ilqr (default) or mppi.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 10 train, 15 test envs, skip discrete.",
    )
    parser.add_argument(
        "--lagrangian", action="store_true",
        help="Use Lagrangian decomposition for velocity dynamics "
        "(replaces PySR-discovered expressions for coupled DOFs).",
    )
    parser.add_argument(
        "--energy", action="store_true",
        help="Use energy-based cost shaping (requires --lagrangian). "
        "Adds (E - E*)^2 penalty to guide energy pumping.",
    )
    parser.add_argument(
        "--phase", action="store_true",
        help="Use phase-based planning (requires --lagrangian). "
        "Switches between MPPI (energy pumping) and iLQR (stabilization).",
    )
    return parser.parse_args()


def main() -> None:
    """Run CIRC-RL v2 pipeline on Acrobot-v1."""
    args = _parse_args()

    # --fast overrides
    if args.fast:
        args.n_train = 10
        args.n_test = 15
        args.skip_discrete = True

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    output_dir = "experiments/outputs"
    os.makedirs(output_dir, exist_ok=True)

    solver_name = args.solver.upper()
    lag_tag = " + LAGRANGIAN" if args.lagrangian else ""
    energy_tag = " + ENERGY" if args.energy else ""
    phase_tag = " + PHASE" if args.phase else ""
    print("=" * 70)
    print(f"CIRC-RL v2: ACROBOT-v1 BENCHMARK "
          f"({solver_name}{lag_tag}{energy_tag}{phase_tag})")
    print(f"  {args.n_train} training envs, {args.n_test} OOD test envs")
    print(f"  Varied: LINK_MASS_1/2, LINK_LENGTH_1/2")
    reward_type = "energy-shaped" if args.energy else "tip height"
    print(f"  Continuous torque + shaped reward ({reward_type})")
    print(f"  Solver: {solver_name}{lag_tag}{energy_tag}{phase_tag}")
    print("=" * 70)

    t0 = time.time()

    # ================================================================
    # Create environment families
    # ================================================================
    print("\n[1/5] Creating environment families...")
    train_family, test_family = _make_acrobot_families(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    print(f"  Train: {train_family}")
    print(f"  Test:  {test_family}")

    # ================================================================
    # Run v2 pipeline on training envs
    # ================================================================
    print("\n[2/5] Running v2 pipeline (PySR + falsification)...")
    t_pipeline = time.time()
    (
        best_dynamics,
        best_reward,
        dynamics_state_names,
        action_names,
        reward_derived_features,
        hf_output,
        oa_output,
        obs_state_names,
        cd_output,
    ) = _run_pipeline(train_family, seed=args.seed)

    t_pipeline_total = time.time() - t_pipeline
    print(f"  Pipeline total: {t_pipeline_total:.1f}s")

    # ================================================================
    # Lagrangian decomposition (optional, replaces velocity dynamics)
    # ================================================================
    lag_templates: dict[str, Any] | None = None
    if args.lagrangian:
        print("\n  [Lagrangian] Running EL decomposition on velocity dims...")
        from circ_rl.hypothesis.expression import SymbolicExpression
        from circ_rl.hypothesis.hypothesis_register import (
            HypothesisEntry,
            HypothesisStatus,
        )
        from circ_rl.hypothesis.lagrangian_decomposition import (
            LagrangianDecomposer,
        )

        # Get canonical dataset and angular dims
        canonical_dataset = oa_output.get("canonical_dataset")
        if canonical_dataset is None:
            canonical_dataset = cd_output["dataset"]
        analysis_result = oa_output.get("analysis_result")
        angular_dims: tuple[int, ...] = ()
        if analysis_result is not None:
            angular_dims = analysis_result.angular_dims

        decomposer = LagrangianDecomposer()
        lag_result = decomposer.decompose(
            canonical_dataset,
            dynamics_state_names,
            action_names,
            list(train_family.param_names),
            angular_dims,
        )

        if lag_result is not None:
            print(f"    Per-env NLS R2: "
                  f"{[f'{r2:.4f}' for r2 in lag_result.per_env_r2.values()]}")
            print(f"    Composed R2: "
                  f"{[f'dim_{k}={v:.4f}' for k, v in lag_result.composed_r2.items()]}")

            # Store templates for RK4-based policy dynamics
            lag_templates = lag_result.parametric_templates

            # Replace velocity-dim dynamics entries with Lagrangian expressions
            for dim_idx, sym_expr in lag_result.dynamics_expressions.items():
                target = f"delta_{dynamics_state_names[dim_idx]}"
                r2 = lag_result.composed_r2.get(dim_idx, 0.99)
                entry = HypothesisEntry(
                    hypothesis_id=f"lagrangian_{target}",
                    target_variable=target,
                    expression=sym_expr,
                    complexity=sym_expr.complexity,
                    training_r2=r2,
                    training_mse=max(0.0, 1.0 - r2),
                    status=HypothesisStatus.VALIDATED,
                    pooled_calibration=None,
                )
                best_dynamics[target] = entry
                print(f"    Replaced {target} with Lagrangian expression "
                      f"(complexity={sym_expr.complexity})")
        else:
            print("    No multi-DOF structure detected, using PySR dynamics")

    if not best_dynamics:
        print("\n  ERROR: No validated dynamics hypotheses.")
        print("  PySR may need different config for Acrobot dynamics.")
        print("  Try re-running (PySR is stochastic).")
        return

    print(f"\n  Validated dynamics for {len(best_dynamics)} dimensions:")
    for target, entry in best_dynamics.items():
        print(f"    {target}: {entry.expression.expression_str}")

    # ================================================================
    # Build policy (iLQR or MPPI)
    # ================================================================
    build_label = solver_name + phase_tag
    print(f"\n[3/5] Building {build_label} policy for test envs...")
    t_build = time.time()
    if args.phase:
        if lag_templates is None:
            print("  ERROR: --phase requires --lagrangian (need energy function).")
            return
        policy = _build_phase_policy(
            test_family,
            best_dynamics,
            dynamics_state_names,
            action_names,
            lagrangian_templates=lag_templates,
            oa_output=oa_output,
            cd_output=cd_output,
        )
    elif args.solver == "mppi":
        policy = _build_mppi_policy(
            test_family,
            best_dynamics,
            dynamics_state_names,
            action_names,
            oa_output=oa_output,
            cd_output=cd_output,
            lagrangian_templates=lag_templates,
            use_energy_shaping=args.energy,
        )
    else:
        policy = _build_ilqr_policy(
            test_family,
            best_dynamics,
            dynamics_state_names,
            action_names,
            oa_output=oa_output,
            cd_output=cd_output,
        )
    t_build_total = time.time() - t_build
    print(f"  {build_label} build: {t_build_total:.1f}s")

    # ================================================================
    # Evaluate
    # ================================================================
    print(f"\n[4/5] Evaluating on {args.n_test} OOD test envs...")

    # Continuous evaluation
    print("\n  --- Continuous Acrobot (shaped reward) ---")
    t_eval = time.time()
    continuous_results = _eval_continuous(
        test_family, policy, max_steps=args.max_steps,
    )
    t_eval_cont = time.time() - t_eval

    # Discrete evaluation
    discrete_results: dict[int, dict[str, float]] = {}
    if not args.skip_discrete:
        print("\n  --- Discrete Acrobot-v1 (sparse reward) ---")
        t_eval_d = time.time()
        discrete_results = _eval_discrete(
            test_family, policy, max_steps=args.max_steps,
        )
        t_eval_disc = time.time() - t_eval_d
    else:
        t_eval_disc = 0.0

    # Baselines
    print("\n[5/5] Evaluating baselines...")
    t_base = time.time()
    random_results, zero_results = _eval_baselines(
        test_family, max_steps=args.max_steps,
    )
    t_base_total = time.time() - t_base

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print("ACROBOT-v1 RESULTS")
    print(f"{'=' * 70}")

    print("\n  Continuous Acrobot (shaped reward):")
    _print_results_table(f"v2 ({solver_name})", continuous_results)

    if discrete_results:
        print("\n  Discrete Acrobot-v1 (sparse reward: -1/step):")
        _print_results_table(f"v2 ({solver_name}, discretized)", discrete_results)
        _print_results_table("Random baseline", random_results)
        _print_results_table("Zero-torque baseline", zero_results)

    print(f"\n  Timings:")
    print(f"    Pipeline (PySR + falsification): {t_pipeline_total:7.1f}s")
    print(f"    {solver_name} build:                      {t_build_total:7.1f}s")
    print(f"    Continuous eval:                  {t_eval_cont:7.1f}s")
    if not args.skip_discrete:
        print(f"    Discrete eval:                    {t_eval_disc:7.1f}s")
    print(f"    Baselines:                        {t_base_total:7.1f}s")
    print(f"    Total:                            {elapsed:7.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
