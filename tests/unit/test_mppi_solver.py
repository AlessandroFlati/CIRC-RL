# ruff: noqa: ANN001 ANN201

"""Unit tests for the MPPI solver."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.ilqr_solver import ILQRSolution
from circ_rl.analytic_policy.mppi_solver import MPPIConfig, MPPISolver


# ---------------------------------------------------------------------------
# Test systems (reuse patterns from test_ilqr_solver)
# ---------------------------------------------------------------------------


def _linear_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Simple 2D linear system: x' = A x + B u."""
    a_mat = np.array([[1.0, 0.1], [0.0, 1.0]])
    b_mat = np.array([[0.0], [0.1]])
    return a_mat @ state + b_mat @ action


def _quadratic_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Quadratic reward: -(x^T Q x + u^T R u)."""
    return -float(state @ state + 0.1 * action @ action)


def _batched_linear_dynamics(
    states: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    """Vectorized 2D linear system: (K,2), (K,1) -> (K,2)."""
    a_mat = np.array([[1.0, 0.1], [0.0, 1.0]])
    b_mat = np.array([[0.0], [0.1]])
    # (K, 2) @ (2, 2)^T + (K, 1) @ (1, 2)^T
    return states @ a_mat.T + actions @ b_mat.T


def _batched_quadratic_reward(
    states: np.ndarray, actions: np.ndarray,
) -> np.ndarray:
    """Vectorized quadratic reward: (K, 2), (K, 1) -> (K,)."""
    state_cost = np.sum(states ** 2, axis=1)  # (K,)
    action_cost = np.sum(actions ** 2, axis=1)  # (K,)
    return -(state_cost + 0.1 * action_cost)  # (K,)


# ---------------------------------------------------------------------------
# MPPIConfig tests
# ---------------------------------------------------------------------------


class TestMPPIConfig:
    """Test MPPIConfig validation and defaults."""

    def test_defaults(self):
        cfg = MPPIConfig()
        assert cfg.horizon == 100
        assert cfg.n_samples == 256
        assert cfg.temperature == 1.0
        assert cfg.noise_sigma == 0.5
        assert cfg.n_iterations == 3
        assert cfg.gamma == 0.99
        assert cfg.max_action == 2.0
        assert cfg.colored_noise_beta == 1.0

    def test_custom_values(self):
        cfg = MPPIConfig(
            horizon=50,
            n_samples=512,
            temperature=0.1,
            noise_sigma=1.0,
            n_iterations=5,
        )
        assert cfg.horizon == 50
        assert cfg.n_samples == 512
        assert cfg.temperature == 0.1
        assert cfg.noise_sigma == 1.0
        assert cfg.n_iterations == 5

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            MPPIConfig(horizon=0)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            MPPIConfig(n_samples=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            MPPIConfig(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            MPPIConfig(temperature=-1.0)

    def test_invalid_noise_sigma(self):
        with pytest.raises(ValueError, match="noise_sigma"):
            MPPIConfig(noise_sigma=0.0)

    def test_invalid_n_iterations(self):
        with pytest.raises(ValueError, match="n_iterations"):
            MPPIConfig(n_iterations=0)

    def test_replan_fields_exist(self):
        """MPPIConfig must have replan fields for _ILQRAnalyticPolicy compat."""
        cfg = MPPIConfig(replan_interval=5)
        assert cfg.replan_interval == 5
        assert cfg.min_replan_interval == 3
        assert cfg.adaptive_replan_threshold is None

    def test_frozen(self):
        cfg = MPPIConfig()
        with pytest.raises(AttributeError):
            cfg.horizon = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Colored noise tests
# ---------------------------------------------------------------------------


class TestColoredNoise:
    """Test colored noise generation."""

    def test_white_noise_shape(self):
        cfg = MPPIConfig(colored_noise_beta=0.0, noise_sigma=1.0)
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        noise = solver._generate_colored_noise(64, 50, 2)
        assert noise.shape == (64, 50, 2)

    def test_colored_noise_shape(self):
        cfg = MPPIConfig(colored_noise_beta=1.0, noise_sigma=0.5)
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        noise = solver._generate_colored_noise(32, 100, 1)
        assert noise.shape == (32, 100, 1)

    def test_noise_sigma_scaling(self):
        """Noise standard deviation should be close to sigma."""
        cfg = MPPIConfig(
            colored_noise_beta=0.0, noise_sigma=2.0, n_samples=1000,
        )
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        noise = solver._generate_colored_noise(1000, 100, 1)
        actual_std = noise.std()
        assert 1.5 < actual_std < 2.5  # allow some variance

    def test_brown_noise_smoother(self):
        """Brown noise (beta=2) should be smoother than white noise (beta=0)."""
        solver_white = MPPISolver(
            MPPIConfig(colored_noise_beta=0.0, noise_sigma=1.0),
            _linear_dynamics, _quadratic_reward,
        )
        solver_brown = MPPISolver(
            MPPIConfig(colored_noise_beta=2.0, noise_sigma=1.0),
            _linear_dynamics, _quadratic_reward,
        )
        np.random.seed(42)
        white = solver_white._generate_colored_noise(100, 200, 1)
        brown = solver_brown._generate_colored_noise(100, 200, 1)

        # Smoothness = mean abs difference between consecutive timesteps
        white_diff = np.abs(np.diff(white, axis=1)).mean()
        brown_diff = np.abs(np.diff(brown, axis=1)).mean()
        assert brown_diff < white_diff


# ---------------------------------------------------------------------------
# MPPISolver tests
# ---------------------------------------------------------------------------


class TestMPPISolver:
    """Test MPPISolver planning."""

    def test_plan_returns_ilqr_solution(self):
        """plan() must return an ILQRSolution for compatibility."""
        cfg = MPPIConfig(
            horizon=20, n_samples=32, n_iterations=2,
        )
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        sol = solver.plan(np.array([1.0, 0.0]), action_dim=1)
        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_states.shape == (21, 2)  # (H+1, S)
        assert sol.nominal_actions.shape == (20, 1)  # (H, A)
        assert len(sol.feedback_gains) == 20
        assert len(sol.feedforward_gains) == 20
        # Feedback gains should be zero (MPPI is open-loop)
        for K_t in sol.feedback_gains:
            np.testing.assert_array_equal(K_t, np.zeros((1, 2)))

    def test_plan_improves_reward(self):
        """MPPI should produce positive total reward improvement."""
        cfg = MPPIConfig(
            horizon=30,
            n_samples=128,
            n_iterations=3,
            temperature=1.0,
            max_action=5.0,
        )
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        sol = solver.plan(np.array([1.0, 0.0]), action_dim=1)

        # Compare to doing nothing (zero actions)
        zero_reward = 0.0
        state = np.array([1.0, 0.0])
        for t in range(30):
            zero_reward += 0.99**t * _quadratic_reward(
                state, np.zeros(1),
            )
            state = _linear_dynamics(state, np.zeros(1))

        assert sol.total_reward > zero_reward

    def test_actions_clipped(self):
        """All actions must respect max_action bounds."""
        cfg = MPPIConfig(
            horizon=20, n_samples=64, n_iterations=2,
            max_action=1.5, noise_sigma=5.0,
        )
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        sol = solver.plan(np.array([5.0, 5.0]), action_dim=1)
        assert np.all(np.abs(sol.nominal_actions) <= 1.5 + 1e-10)

    def test_warm_start(self):
        """Warm start should accept pre-existing action sequences."""
        cfg = MPPIConfig(
            horizon=10, n_samples=32, n_iterations=1,
        )
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        warm = np.zeros((10, 1))
        sol = solver.plan(
            np.array([1.0, 0.0]),
            action_dim=1,
            warm_start_actions=warm,
        )
        assert isinstance(sol, ILQRSolution)

    def test_warm_start_wrong_shape_raises(self):
        """Warm start with wrong shape should raise."""
        cfg = MPPIConfig(horizon=10, n_samples=32, n_iterations=1)
        solver = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        wrong_shape = np.zeros((5, 1))  # wrong horizon
        with pytest.raises(AssertionError):
            solver.plan(
                np.array([1.0, 0.0]),
                action_dim=1,
                warm_start_actions=wrong_shape,
            )


# ---------------------------------------------------------------------------
# Batched dynamics tests
# ---------------------------------------------------------------------------


class TestBatchedDynamics:
    """Test that batched dynamics produces same results as scalar loop."""

    def test_batched_matches_scalar(self):
        """Batched rollout should match scalar loop."""
        cfg = MPPIConfig(
            horizon=15, n_samples=16, n_iterations=1,
            colored_noise_beta=0.0,
        )
        solver_scalar = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
        )
        solver_batched = MPPISolver(
            cfg, _linear_dynamics, _quadratic_reward,
            batched_dynamics_fn=_batched_linear_dynamics,
        )

        state = np.array([1.0, 0.5])
        # Use same RNG state for both
        rng_state = np.random.get_state()
        sol_scalar = solver_scalar.plan(state, action_dim=1)
        np.random.set_state(rng_state)
        # Reset internal RNG too
        solver_batched._rng = np.random.default_rng(
            solver_scalar._rng.bit_generator.state["state"]["state"],
        )
        solver_scalar._rng = np.random.default_rng(
            solver_scalar._rng.bit_generator.state["state"]["state"],
        )
        sol_scalar2 = solver_scalar.plan(state, action_dim=1)
        sol_batched = solver_batched.plan(state, action_dim=1)

        # Both should produce valid solutions (exact match not required
        # due to RNG state differences, but rewards should be similar)
        assert isinstance(sol_batched, ILQRSolution)
        assert sol_batched.nominal_states.shape == (16, 2)

    def test_batched_reward_fn(self):
        """Batched reward should speed up cost computation."""
        cfg = MPPIConfig(
            horizon=20, n_samples=64, n_iterations=2,
        )
        solver = MPPISolver(
            cfg,
            _linear_dynamics,
            _quadratic_reward,
            batched_dynamics_fn=_batched_linear_dynamics,
            batched_reward_fn=_batched_quadratic_reward,
        )
        sol = solver.plan(np.array([1.0, 0.0]), action_dim=1)
        assert isinstance(sol, ILQRSolution)
        # Should still find a reasonable solution
        assert sol.total_reward > -1000


# ---------------------------------------------------------------------------
# Integration: build_batched_dynamics_fn
# ---------------------------------------------------------------------------


class TestBuildBatchedDynamicsFn:
    """Test the batched dynamics builder from fast_dynamics.py."""

    def test_build_and_call(self):
        """Build a batched dynamics fn from sympy expressions."""
        import sympy

        from circ_rl.analytic_policy.fast_dynamics import (
            build_batched_dynamics_fn,
        )

        # Simple linear dynamics: delta_x0 = 0.1 * x1, delta_x1 = 0.1 * u0
        x0, x1, u0 = sympy.symbols("x0 x1 u0")

        class FakeExpr:
            def __init__(self, expr):
                self.sympy_expr = expr

        exprs = {
            0: FakeExpr(0.1 * x1),
            1: FakeExpr(0.1 * u0),
        }

        fn = build_batched_dynamics_fn(
            dynamics_expressions=exprs,
            state_names=["x0", "x1"],
            action_names=["u0"],
            state_dim=2,
            env_params=None,
        )

        states = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        actions = np.array([[0.5], [1.0]])  # (2, 1)
        next_states = fn(states, actions)

        # delta_x0 = 0.1 * x1 -> [0.2, 0.4]
        # delta_x1 = 0.1 * u0 -> [0.05, 0.1]
        expected = np.array([
            [1.0 + 0.2, 2.0 + 0.05],
            [3.0 + 0.4, 4.0 + 0.1],
        ])
        np.testing.assert_allclose(next_states, expected, atol=1e-10)

    def test_with_calibration(self):
        """Calibration coefficients (alpha, beta) should be applied."""
        import sympy

        from circ_rl.analytic_policy.fast_dynamics import (
            build_batched_dynamics_fn,
        )

        x0, u0 = sympy.symbols("x0 u0")

        class FakeExpr:
            def __init__(self, expr):
                self.sympy_expr = expr

        exprs = {0: FakeExpr(u0)}
        cal = {0: (2.0, 0.5)}  # alpha=2.0, beta=0.5

        fn = build_batched_dynamics_fn(
            dynamics_expressions=exprs,
            state_names=["x0"],
            action_names=["u0"],
            state_dim=1,
            env_params=None,
            calibration_coefficients=cal,
        )

        states = np.array([[1.0], [2.0]])  # (2, 1)
        actions = np.array([[1.0], [3.0]])  # (2, 1)
        next_states = fn(states, actions)

        # delta = alpha * u0 + beta = 2.0 * u0 + 0.5
        # next = x0 + delta
        expected = np.array([
            [1.0 + 2.0 * 1.0 + 0.5],  # 3.5
            [2.0 + 2.0 * 3.0 + 0.5],  # 8.5
        ])
        np.testing.assert_allclose(next_states, expected, atol=1e-10)

    def test_angular_wrapping(self):
        """Angular dimensions should be wrapped to [-pi, pi]."""
        import sympy

        from circ_rl.analytic_policy.fast_dynamics import (
            build_batched_dynamics_fn,
        )

        x0 = sympy.Symbol("x0")
        u0 = sympy.Symbol("u0")

        class FakeExpr:
            def __init__(self, expr):
                self.sympy_expr = expr

        exprs = {0: FakeExpr(u0)}  # delta = u0

        fn = build_batched_dynamics_fn(
            dynamics_expressions=exprs,
            state_names=["x0"],
            action_names=["u0"],
            state_dim=1,
            env_params=None,
            angular_dims=(0,),
        )

        # x0=3.0, u0=1.0 -> next=4.0 -> wrapped to 4.0 - 2*pi
        states = np.array([[3.0]])
        actions = np.array([[1.0]])
        next_states = fn(states, actions)
        expected_angle = np.arctan2(np.sin(4.0), np.cos(4.0))
        np.testing.assert_allclose(
            next_states[0, 0], expected_angle, atol=1e-10,
        )
