"""Robust MPC via scenario-based worst-case planning.

Samples coefficient scenarios from per-environment uncertainty
estimates, runs iLQR independently per scenario, then selects the
maximin action sequence (highest worst-case reward).

The approach:

1. Sample ``n_scenarios`` sets of ``(alpha, beta)`` per dynamics
   dimension from the calibration covariance.
2. Run iLQR for each scenario to get a candidate action sequence.
3. Cross-evaluate: simulate each candidate's actions under ALL
   scenario dynamics.
4. Select the candidate with the highest worst-case (minimum across
   scenarios) total reward.

Scenario 0 is always the nominal (mean) coefficients; the remaining
scenarios are drawn from the multivariate normal defined by the OLS
covariance matrix.

See ``CIRC-RL_Framework.md`` Section 7.2 (Robust MPC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from circ_rl.analytic_policy.coefficient_calibrator import (
        CoefficientUncertainty,
    )
    from circ_rl.analytic_policy.ilqr_solver import ILQRSolution, ILQRSolver


@dataclass(frozen=True)
class RobustMPCConfig:
    """Configuration for robust MPC.

    :param n_scenarios: Total number of coefficient scenarios,
        including the nominal. Must be >= 2. Default 5.
    :param confidence_level: Confidence level for the coefficient
        region. Controls the scale of sampled scenarios. Default 0.95.
    :param min_uncertainty_threshold: Minimum total coefficient
        uncertainty (sum of trace(cov) across dims) below which
        robust MPC degrades gracefully to standard MPC. Default 1e-6.
    :param reduced_restarts: Number of random restarts per scenario
        (reduced from normal iLQR to keep computation tractable).
        Default 2.
    """

    n_scenarios: int = 5
    confidence_level: float = 0.95
    min_uncertainty_threshold: float = 1e-6
    reduced_restarts: int = 2

    def __post_init__(self) -> None:
        if self.n_scenarios < 2:
            raise ValueError(
                f"n_scenarios must be >= 2, got {self.n_scenarios}"
            )
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), "
                f"got {self.confidence_level}"
            )
        if self.min_uncertainty_threshold < 0:
            raise ValueError(
                f"min_uncertainty_threshold must be >= 0, "
                f"got {self.min_uncertainty_threshold}"
            )
        if self.reduced_restarts < 0:
            raise ValueError(
                f"reduced_restarts must be >= 0, "
                f"got {self.reduced_restarts}"
            )


class ScenarioSampler:
    """Sample coefficient scenarios from calibration uncertainty.

    Each scenario is a mapping from dynamics dimension index to
    ``(alpha, beta)`` coefficients. Scenario 0 is always the nominal
    (mean) coefficients; the rest are sampled from the per-env OLS
    covariance.

    :param config: Robust MPC configuration.
    """

    def __init__(self, config: RobustMPCConfig) -> None:
        self._config = config

    def sample_scenarios(
        self,
        per_dim_uncertainty: dict[int, CoefficientUncertainty],
        env_idx: int,
        rng: np.random.Generator,
    ) -> list[dict[int, tuple[float, float]]]:
        """Sample coefficient scenarios for a specific environment.

        :param per_dim_uncertainty: Calibration uncertainty per dynamics
            dimension, from ``CoefficientCalibrator.calibrate()``.
        :param env_idx: Environment index for per-env covariance.
        :param rng: Random number generator.
        :returns: List of ``n_scenarios`` dicts mapping dim_idx to
            ``(alpha, beta)``.
        """
        n = self._config.n_scenarios

        # Scenario 0: nominal (per-env if available, else pooled)
        nominal: dict[int, tuple[float, float]] = {}
        for dim_idx, unc in per_dim_uncertainty.items():
            if env_idx in unc.per_env:
                cal = unc.per_env[env_idx]
                nominal[dim_idx] = (cal.alpha, cal.beta)
            else:
                nominal[dim_idx] = (unc.pooled_alpha, unc.pooled_beta)

        scenarios: list[dict[int, tuple[float, float]]] = [nominal]

        # Sample remaining scenarios from per-env covariance
        for _ in range(n - 1):
            scenario: dict[int, tuple[float, float]] = {}
            for dim_idx, unc in per_dim_uncertainty.items():
                if env_idx in unc.per_env:
                    cal = unc.per_env[env_idx]
                    mean = np.array([cal.beta, cal.alpha])  # (2,)
                    cov = cal.covariance  # (2, 2)
                else:
                    mean = np.array(
                        [unc.pooled_beta, unc.pooled_alpha],
                    )  # (2,)
                    cov = unc.pooled_covariance  # (2, 2)

                # Ensure covariance is positive semi-definite
                eigvals = np.linalg.eigvalsh(cov)
                if np.any(eigvals < 0):
                    cov = cov + np.eye(2) * (abs(float(eigvals.min())) + 1e-10)

                sample = rng.multivariate_normal(mean, cov)  # (2,)
                scenario[dim_idx] = (float(sample[1]), float(sample[0]))

            scenarios.append(scenario)

        return scenarios


def build_scenario_dynamics_fn(
    base_dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    scenario_coeffs: dict[int, tuple[float, float]],
    state_dim: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a dynamics function with scenario-specific coefficients.

    Wraps the base dynamics such that each dimension's delta is
    scaled by the scenario's ``(alpha, beta)``:

    .. math::

        \\delta'_i = \\alpha_i \\cdot \\delta_i^{\\text{base}} + \\beta_i

    where :math:`\\delta_i = x_{t+1,i} - x_{t,i}`.

    :param base_dynamics_fn: Base dynamics callable
        ``(state, action) -> next_state``.
    :param scenario_coeffs: Mapping from dim_idx to ``(alpha, beta)``.
    :param state_dim: Number of state dimensions.
    :returns: Wrapped dynamics callable.
    """

    def scenario_fn(
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        next_state = base_dynamics_fn(state, action)  # (state_dim,)
        for dim_idx, (alpha, beta) in scenario_coeffs.items():
            if dim_idx < state_dim:
                delta = next_state[dim_idx] - state[dim_idx]
                next_state[dim_idx] = (
                    state[dim_idx] + alpha * delta + beta
                )
        return next_state

    return scenario_fn


def build_scenario_jacobian_fns(
    base_jac_state_fn: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ] | None,
    base_jac_action_fn: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ] | None,
    scenario_coeffs: dict[int, tuple[float, float]],
    state_dim: int,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
]:
    """Build scenario-specific Jacobian functions.

    For the state Jacobian:

    .. math::

        A^{\\text{scenario}} = I + \\alpha \\cdot (A^{\\text{base}} - I)

    For the action Jacobian:

    .. math::

        B^{\\text{scenario}} = \\alpha \\cdot B^{\\text{base}}

    Only rows corresponding to dimensions with scenario coefficients
    are scaled; other rows pass through unchanged.

    :param base_jac_state_fn: Base state Jacobian or None.
    :param base_jac_action_fn: Base action Jacobian or None.
    :param scenario_coeffs: Mapping from dim_idx to ``(alpha, beta)``.
    :param state_dim: Number of state dimensions.
    :returns: Tuple of (scenario_jac_state_fn, scenario_jac_action_fn).
    """
    if base_jac_state_fn is None and base_jac_action_fn is None:
        return None, None

    def scenario_jac_state(
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        if base_jac_state_fn is None:
            return np.eye(state_dim)
        A = base_jac_state_fn(state, action).copy()  # (sd, sd)
        for dim_idx, (alpha, _beta) in scenario_coeffs.items():
            if dim_idx < state_dim:
                # A_scenario[i,:] = I[i,:] + alpha * (A_base[i,:] - I[i,:])
                row = A[dim_idx, :]
                identity_row = np.zeros(state_dim)
                identity_row[dim_idx] = 1.0
                A[dim_idx, :] = (
                    identity_row + alpha * (row - identity_row)
                )
        return A

    def scenario_jac_action(
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        if base_jac_action_fn is None:
            raise ValueError(
                "base_jac_action_fn is None but was expected"
            )
        B = base_jac_action_fn(state, action).copy()  # (sd, ad)
        for dim_idx, (alpha, _beta) in scenario_coeffs.items():
            if dim_idx < state_dim:
                B[dim_idx, :] *= alpha
        return B

    jac_state_out = scenario_jac_state if base_jac_state_fn is not None else None
    jac_action_out = scenario_jac_action if base_jac_action_fn is not None else None

    return jac_state_out, jac_action_out


class RobustILQRPlanner:
    """Scenario-based robust iLQR planner.

    Plans under multiple coefficient scenarios and selects the maximin
    action sequence (highest worst-case reward across all scenarios).

    :param config: Robust MPC configuration.
    :param base_solver: The nominal iLQR solver (used for config and
        reward/terminal cost).
    :param scenario_solvers: Per-scenario iLQR solvers with
        scenario-specific dynamics.
    :param scenario_dynamics: Per-scenario dynamics functions for
        cross-evaluation.
    """

    def __init__(
        self,
        config: RobustMPCConfig,
        base_solver: ILQRSolver,
        scenario_solvers: list[ILQRSolver],
        scenario_dynamics: list[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ],
    ) -> None:
        if len(scenario_solvers) != config.n_scenarios:
            raise ValueError(
                f"Expected {config.n_scenarios} scenario solvers, "
                f"got {len(scenario_solvers)}"
            )
        if len(scenario_dynamics) != config.n_scenarios:
            raise ValueError(
                f"Expected {config.n_scenarios} scenario dynamics, "
                f"got {len(scenario_dynamics)}"
            )
        self._config = config
        self._base_solver = base_solver
        self._scenario_solvers = scenario_solvers
        self._scenario_dynamics = scenario_dynamics

    def plan(
        self,
        initial_state: np.ndarray,
        action_dim: int,
        warm_start: np.ndarray | None = None,
    ) -> ILQRSolution:
        """Plan under all scenarios and select the maximin solution.

        Phase 1: Run iLQR independently per scenario.
        Phase 2: Cross-evaluate each solution's actions under ALL
            scenario dynamics.
        Phase 3: Select the candidate with the highest worst-case
            (minimum across scenarios) total reward.

        :param initial_state: Starting state, shape ``(state_dim,)``.
        :param action_dim: Number of action dimensions.
        :param warm_start: Optional initial action sequence.
        :returns: The robust (maximin) solution.
        """
        # Phase 1: Plan per scenario
        solutions: list[ILQRSolution] = []
        for solver in self._scenario_solvers:
            sol = solver.plan(initial_state, action_dim, warm_start)
            solutions.append(sol)

        # Phase 2: Cross-evaluate
        reward_fn = self._base_solver._reward_fn
        gamma = self._base_solver._config.gamma
        horizon = self._base_solver._config.horizon

        # worst_case_rewards[i] = min reward of solution i across
        # all scenario dynamics
        worst_case_rewards: list[float] = []

        for sol_idx, sol in enumerate(solutions):
            scenario_rewards: list[float] = []
            for dyn_fn in self._scenario_dynamics:
                total_r = self._rollout_reward(
                    initial_state, sol.nominal_actions,
                    dyn_fn, reward_fn, gamma, horizon,
                )
                scenario_rewards.append(total_r)
            worst_case_rewards.append(min(scenario_rewards))

        # Phase 3: Maximin selection
        best_idx = int(np.argmax(worst_case_rewards))

        logger.debug(
            "Robust MPC: {} scenarios, worst-case rewards = {}, "
            "selected scenario {}",
            self._config.n_scenarios,
            [f"{r:.2f}" for r in worst_case_rewards],
            best_idx,
        )

        return solutions[best_idx]

    @staticmethod
    def _rollout_reward(
        initial_state: np.ndarray,
        actions: np.ndarray,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        gamma: float,
        horizon: int,
    ) -> float:
        """Roll out actions under given dynamics and compute total reward.

        :returns: Total discounted reward.
        """
        state = initial_state.copy()
        total_reward = 0.0
        n_actions = min(len(actions), horizon)

        for t in range(n_actions):
            r = reward_fn(state, actions[t])
            total_reward += (gamma ** t) * r
            state = dynamics_fn(state, actions[t])

        return total_reward


def check_uncertainty_significant(
    per_dim_uncertainty: dict[int, CoefficientUncertainty],
    env_idx: int,
    threshold: float,
) -> bool:
    """Check if total coefficient uncertainty warrants robust planning.

    Sums ``trace(covariance)`` across all dynamics dimensions for the
    given environment. Returns ``True`` if above threshold.

    :param per_dim_uncertainty: Calibration uncertainty per dimension.
    :param env_idx: Environment index.
    :param threshold: Minimum total trace to be considered significant.
    :returns: ``True`` if robust planning is warranted.
    """
    total_trace = 0.0
    for dim_idx, unc in per_dim_uncertainty.items():
        if env_idx in unc.per_env:
            total_trace += float(np.trace(unc.per_env[env_idx].covariance))
        else:
            total_trace += float(np.trace(unc.pooled_covariance))
    return total_trace > threshold
