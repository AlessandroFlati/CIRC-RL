# ruff: noqa: ANN001 ANN201

"""Unit tests for robust MPC with coefficient uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.robust_mpc import (
    RobustILQRPlanner,
    RobustMPCConfig,
    ScenarioSampler,
    build_scenario_dynamics_fn,
    build_scenario_jacobian_fns,
    check_uncertainty_significant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCalibrationResult:
    """Minimal CalibrationResult mock."""

    def __init__(
        self,
        env_id: int,
        alpha: float,
        beta: float,
        covariance: np.ndarray,
    ):
        self.env_id = env_id
        self.alpha = alpha
        self.beta = beta
        self.alpha_se = float(np.sqrt(covariance[1, 1]))
        self.beta_se = float(np.sqrt(covariance[0, 0]))
        self.covariance = covariance
        self.mse = 0.01
        self.n_samples = 100


class _FakeUncertainty:
    """Minimal CoefficientUncertainty mock."""

    def __init__(
        self,
        per_env: dict[int, _FakeCalibrationResult],
        pooled_alpha: float = 1.0,
        pooled_beta: float = 0.0,
        pooled_covariance: np.ndarray | None = None,
    ):
        self.per_env = per_env
        self.pooled_alpha = pooled_alpha
        self.pooled_beta = pooled_beta
        self.pooled_covariance = (
            pooled_covariance
            if pooled_covariance is not None
            else np.eye(2) * 0.01
        )


def _make_per_dim_uncertainty(
    n_envs: int = 3,
    alpha: float = 1.0,
    beta: float = 0.0,
    cov_scale: float = 0.01,
) -> dict[int, _FakeUncertainty]:
    """Create mock per-dimension uncertainty for a single dimension."""
    cov = np.eye(2) * cov_scale
    per_env = {}
    for env_id in range(n_envs):
        per_env[env_id] = _FakeCalibrationResult(
            env_id=env_id,
            alpha=alpha,
            beta=beta,
            covariance=cov,
        )
    return {
        0: _FakeUncertainty(
            per_env=per_env,
            pooled_alpha=alpha,
            pooled_beta=beta,
            pooled_covariance=cov,
        ),
    }


def _simple_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """1D linear dynamics: x' = x + 0.1*u."""
    return np.array([state[0] + 0.1 * action[0]])


def _simple_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Simple reward: -(x^2 + 0.1*u^2)."""
    return -float(state[0] ** 2 + 0.1 * action[0] ** 2)


def _simple_jac_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.array([[1.0]])


def _simple_jac_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.array([[0.1]])


# ---------------------------------------------------------------------------
# RobustMPCConfig tests
# ---------------------------------------------------------------------------

class TestRobustMPCConfig:
    def test_default_values(self):
        cfg = RobustMPCConfig()
        assert cfg.n_scenarios == 5
        assert cfg.confidence_level == 0.95
        assert cfg.min_uncertainty_threshold == 1e-6
        assert cfg.reduced_restarts == 2

    def test_invalid_n_scenarios_raises(self):
        with pytest.raises(ValueError, match="n_scenarios"):
            RobustMPCConfig(n_scenarios=1)

    def test_invalid_confidence_level_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            RobustMPCConfig(confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            RobustMPCConfig(confidence_level=1.0)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="min_uncertainty_threshold"):
            RobustMPCConfig(min_uncertainty_threshold=-1.0)

    def test_invalid_restarts_raises(self):
        with pytest.raises(ValueError, match="reduced_restarts"):
            RobustMPCConfig(reduced_restarts=-1)


# ---------------------------------------------------------------------------
# ScenarioSampler tests
# ---------------------------------------------------------------------------

class TestScenarioSampler:
    def test_samples_correct_count(self):
        config = RobustMPCConfig(n_scenarios=5)
        sampler = ScenarioSampler(config)
        unc = _make_per_dim_uncertainty()

        rng = np.random.default_rng(42)
        scenarios = sampler.sample_scenarios(unc, env_idx=0, rng=rng)

        assert len(scenarios) == 5

    def test_first_scenario_is_nominal(self):
        config = RobustMPCConfig(n_scenarios=3)
        sampler = ScenarioSampler(config)
        unc = _make_per_dim_uncertainty(alpha=2.0, beta=0.5)

        rng = np.random.default_rng(42)
        scenarios = sampler.sample_scenarios(unc, env_idx=0, rng=rng)

        # First scenario should be nominal
        alpha, beta = scenarios[0][0]
        assert abs(alpha - 2.0) < 1e-10
        assert abs(beta - 0.5) < 1e-10

    def test_remaining_scenarios_differ_from_nominal(self):
        config = RobustMPCConfig(n_scenarios=5)
        sampler = ScenarioSampler(config)
        unc = _make_per_dim_uncertainty(alpha=1.0, beta=0.0, cov_scale=0.1)

        rng = np.random.default_rng(42)
        scenarios = sampler.sample_scenarios(unc, env_idx=0, rng=rng)

        # At least some non-nominal scenarios should differ
        nominal = scenarios[0]
        some_differ = False
        for sc in scenarios[1:]:
            if sc[0] != nominal[0]:
                some_differ = True
                break
        assert some_differ

    def test_falls_back_to_pooled_for_unknown_env(self):
        config = RobustMPCConfig(n_scenarios=2)
        sampler = ScenarioSampler(config)
        # Create uncertainty with only env_id=0
        unc = _make_per_dim_uncertainty(n_envs=1)

        rng = np.random.default_rng(42)
        # Request env_idx=5 (not in per_env)
        scenarios = sampler.sample_scenarios(unc, env_idx=5, rng=rng)

        assert len(scenarios) == 2
        # Nominal should use pooled values
        alpha, beta = scenarios[0][0]
        assert abs(alpha - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# build_scenario_dynamics_fn tests
# ---------------------------------------------------------------------------

class TestBuildScenarioDynamics:
    def test_nominal_coefficients_unchanged(self):
        """With alpha=1, beta=0 the dynamics should be unchanged."""
        sc_fn = build_scenario_dynamics_fn(
            _simple_dynamics,
            scenario_coeffs={0: (1.0, 0.0)},
            state_dim=1,
        )

        state = np.array([1.0])
        action = np.array([2.0])
        expected = _simple_dynamics(state, action)
        result = sc_fn(state, action)

        np.testing.assert_allclose(result, expected)

    def test_alpha_scales_delta(self):
        """Alpha should scale the state delta."""
        sc_fn = build_scenario_dynamics_fn(
            _simple_dynamics,
            scenario_coeffs={0: (2.0, 0.0)},
            state_dim=1,
        )

        state = np.array([1.0])
        action = np.array([2.0])
        base_next = _simple_dynamics(state, action)
        base_delta = base_next[0] - state[0]

        result = sc_fn(state, action)
        expected_delta = 2.0 * base_delta
        np.testing.assert_allclose(
            result[0], state[0] + expected_delta,
        )

    def test_beta_adds_offset(self):
        """Beta should add a constant offset to the delta."""
        sc_fn = build_scenario_dynamics_fn(
            _simple_dynamics,
            scenario_coeffs={0: (1.0, 0.5)},
            state_dim=1,
        )

        state = np.array([1.0])
        action = np.array([2.0])
        base_next = _simple_dynamics(state, action)

        result = sc_fn(state, action)
        np.testing.assert_allclose(
            result[0], base_next[0] + 0.5,
        )


# ---------------------------------------------------------------------------
# build_scenario_jacobian_fns tests
# ---------------------------------------------------------------------------

class TestBuildScenarioJacobians:
    def test_nominal_jacobians_unchanged(self):
        """With alpha=1, Jacobians should be unchanged."""
        jac_s, jac_a = build_scenario_jacobian_fns(
            _simple_jac_state, _simple_jac_action,
            scenario_coeffs={0: (1.0, 0.0)},
            state_dim=1,
        )

        state = np.array([1.0])
        action = np.array([2.0])

        assert jac_s is not None
        assert jac_a is not None
        np.testing.assert_allclose(
            jac_s(state, action),
            _simple_jac_state(state, action),
        )
        np.testing.assert_allclose(
            jac_a(state, action),
            _simple_jac_action(state, action),
        )

    def test_alpha_scales_action_jacobian(self):
        """Alpha should scale the action Jacobian."""
        jac_s, jac_a = build_scenario_jacobian_fns(
            _simple_jac_state, _simple_jac_action,
            scenario_coeffs={0: (3.0, 0.0)},
            state_dim=1,
        )

        state = np.array([1.0])
        action = np.array([2.0])

        assert jac_a is not None
        expected_B = 3.0 * _simple_jac_action(state, action)
        np.testing.assert_allclose(
            jac_a(state, action),
            expected_B,
        )

    def test_none_inputs_return_none(self):
        jac_s, jac_a = build_scenario_jacobian_fns(
            None, None,
            scenario_coeffs={0: (2.0, 0.0)},
            state_dim=1,
        )
        assert jac_s is None
        assert jac_a is None


# ---------------------------------------------------------------------------
# check_uncertainty_significant tests
# ---------------------------------------------------------------------------

class TestCheckUncertaintySignificant:
    def test_high_uncertainty_returns_true(self):
        unc = _make_per_dim_uncertainty(cov_scale=1.0)
        assert check_uncertainty_significant(unc, env_idx=0, threshold=0.1)

    def test_low_uncertainty_returns_false(self):
        unc = _make_per_dim_uncertainty(cov_scale=1e-10)
        assert not check_uncertainty_significant(
            unc, env_idx=0, threshold=1e-6,
        )

    def test_unknown_env_uses_pooled(self):
        unc = _make_per_dim_uncertainty(n_envs=1, cov_scale=1.0)
        # env_idx=5 not in per_env, should use pooled
        assert check_uncertainty_significant(
            unc, env_idx=5, threshold=0.01,
        )


# ---------------------------------------------------------------------------
# RobustILQRPlanner tests
# ---------------------------------------------------------------------------

class TestRobustILQRPlanner:
    def test_plan_returns_solution(self):
        """Robust planner should return a valid solution."""
        from circ_rl.analytic_policy.ilqr_solver import (
            ILQRConfig,
            ILQRSolver,
        )

        config = RobustMPCConfig(n_scenarios=2, reduced_restarts=0)

        # Create identical scenario solvers (for simplicity)
        solver_config = ILQRConfig(
            horizon=10,
            max_iterations=5,
            gamma=0.99,
            max_action=2.0,
        )

        base_solver = ILQRSolver(
            config=solver_config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
            dynamics_jac_state_fn=_simple_jac_state,
            dynamics_jac_action_fn=_simple_jac_action,
        )

        # Two scenarios with slightly different dynamics
        sc1_fn = build_scenario_dynamics_fn(
            _simple_dynamics, {0: (1.0, 0.0)}, state_dim=1,
        )
        sc2_fn = build_scenario_dynamics_fn(
            _simple_dynamics, {0: (1.1, 0.01)}, state_dim=1,
        )

        sc_solver1 = ILQRSolver(
            config=solver_config,
            dynamics_fn=sc1_fn,
            reward_fn=_simple_reward,
        )
        sc_solver2 = ILQRSolver(
            config=solver_config,
            dynamics_fn=sc2_fn,
            reward_fn=_simple_reward,
        )

        planner = RobustILQRPlanner(
            config=config,
            base_solver=base_solver,
            scenario_solvers=[sc_solver1, sc_solver2],
            scenario_dynamics=[sc1_fn, sc2_fn],
        )

        sol = planner.plan(
            initial_state=np.array([1.0]),
            action_dim=1,
        )

        assert sol.nominal_actions.shape[0] == 10
        assert np.isfinite(sol.total_reward)

    def test_wrong_solver_count_raises(self):
        from circ_rl.analytic_policy.ilqr_solver import (
            ILQRConfig,
            ILQRSolver,
        )

        config = RobustMPCConfig(n_scenarios=3)
        solver_config = ILQRConfig(
            horizon=5, gamma=0.99, max_action=1.0,
        )
        base_solver = ILQRSolver(
            config=solver_config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
        )

        with pytest.raises(ValueError, match="scenario solvers"):
            RobustILQRPlanner(
                config=config,
                base_solver=base_solver,
                scenario_solvers=[base_solver],  # Only 1, need 3
                scenario_dynamics=[_simple_dynamics],
            )

    def test_wrong_dynamics_count_raises(self):
        from circ_rl.analytic_policy.ilqr_solver import (
            ILQRConfig,
            ILQRSolver,
        )

        config = RobustMPCConfig(n_scenarios=2)
        solver_config = ILQRConfig(
            horizon=5, gamma=0.99, max_action=1.0,
        )
        base_solver = ILQRSolver(
            config=solver_config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
        )

        with pytest.raises(ValueError, match="scenario dynamics"):
            RobustILQRPlanner(
                config=config,
                base_solver=base_solver,
                scenario_solvers=[base_solver, base_solver],
                scenario_dynamics=[_simple_dynamics],  # Only 1, need 2
            )
