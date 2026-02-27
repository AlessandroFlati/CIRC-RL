"""Tests for BatchedNumpyILQRSolver."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.batched_ilqr_solver import (
    BatchedNumpyILQRSolver,
    _batched_cholesky_safe,
    build_batched_jacobian_fns,
)
from circ_rl.analytic_policy.ilqr_solver import ILQRConfig, ILQRSolver


# -- Simple linear dynamics for testing --


def _make_linear_dynamics(dt: float = 0.05) -> tuple:
    """Build 2D integrator: x' = x + dt*v, v' = v + dt*u.

    Returns (batched_dyn, scalar_dyn, batched_jac_s, batched_jac_a,
    reward_fn, reward_derivs_fn).
    """

    def scalar_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, v = state
        u = action[0]
        return np.array([x + dt * v, v + dt * u])

    def batched_dynamics(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        # (B, 2), (B, 1) -> (B, 2)
        next_s = states.copy()
        next_s[:, 0] += dt * states[:, 1]
        next_s[:, 1] += dt * actions[:, 0]
        return next_s

    def batched_jac_state(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        B = states.shape[0]
        jac = np.zeros((B, 2, 2))
        jac[:, 0, 0] = 1.0
        jac[:, 0, 1] = dt
        jac[:, 1, 1] = 1.0
        return jac

    def batched_jac_action(
        states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        B = states.shape[0]
        jac = np.zeros((B, 2, 1))
        jac[:, 1, 0] = dt
        return jac

    def reward_fn(state: np.ndarray, action: np.ndarray) -> float:
        x, v = state
        u = action[0]
        return -(x**2 + 0.1 * v**2 + 0.01 * u**2)

    def reward_derivatives_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> tuple:
        x, v = state
        u = action[0]
        r_x = np.array([-2.0 * x, -0.2 * v])
        r_u = np.array([-0.02 * u])
        r_xx = np.array([[-2.0, 0.0], [0.0, -0.2]])
        r_uu = np.array([[-0.02]])
        r_ux = np.array([[0.0, 0.0]])
        return r_x, r_u, r_xx, r_uu, r_ux

    return (
        batched_dynamics,
        scalar_dynamics,
        batched_jac_state,
        batched_jac_action,
        reward_fn,
        reward_derivatives_fn,
    )


class TestBatchedCholesky:
    """Tests for _batched_cholesky_safe."""

    def test_scalar_positive(self) -> None:
        matrices = np.array([[[4.0]], [[9.0]], [[1.0]]])
        cho, ok = _batched_cholesky_safe(matrices)
        assert ok.all()
        np.testing.assert_allclose(cho[:, 0, 0], [2.0, 3.0, 1.0])

    def test_scalar_negative(self) -> None:
        matrices = np.array([[[4.0]], [[-1.0]], [[9.0]]])
        cho, ok = _batched_cholesky_safe(matrices)
        assert ok[0] and not ok[1] and ok[2]

    def test_2x2_batch(self) -> None:
        # Two positive definite 2x2 matrices
        m1 = np.array([[2.0, 0.5], [0.5, 1.0]])
        m2 = np.array([[3.0, 1.0], [1.0, 2.0]])
        matrices = np.stack([m1, m2])
        cho, ok = _batched_cholesky_safe(matrices)
        assert ok.all()
        for i in range(2):
            np.testing.assert_allclose(
                cho[i] @ cho[i].T, matrices[i], atol=1e-10,
            )


class TestBatchedSolver:
    """Tests for BatchedNumpyILQRSolver."""

    def test_basic_optimization(self) -> None:
        """Solver reduces cost from random initial state."""
        (
            batched_dyn, _, batched_jac_s, batched_jac_a,
            reward_fn, reward_derivs,
        ) = _make_linear_dynamics()

        config = ILQRConfig(
            horizon=30,
            max_iterations=20,
            n_random_restarts=5,
            max_action=2.0,
            gamma=0.99,
        )

        solver = BatchedNumpyILQRSolver(
            config=config,
            batched_dynamics_fn=batched_dyn,
            reward_fn=reward_fn,
            batched_jac_state_fn=batched_jac_s,
            batched_jac_action_fn=batched_jac_a,
            reward_derivatives_fn=reward_derivs,
        )

        initial_state = np.array([1.0, 0.5])
        sol = solver.plan(initial_state, action_dim=1)

        assert sol.nominal_states.shape == (31, 2)
        assert sol.nominal_actions.shape == (30, 1)
        assert len(sol.feedback_gains) == 30
        assert len(sol.feedforward_gains) == 30
        # Must improve over doing nothing
        assert sol.total_reward > -50.0

    def test_matches_scalar_solver(self) -> None:
        """Batched solver achieves similar reward to scalar solver."""
        (
            batched_dyn, scalar_dyn, batched_jac_s, batched_jac_a,
            reward_fn, reward_derivs,
        ) = _make_linear_dynamics()

        config = ILQRConfig(
            horizon=30,
            max_iterations=30,
            n_random_restarts=10,
            max_action=2.0,
            gamma=0.99,
        )

        # Scalar solver
        scalar_jac_s = lambda s, a: batched_jac_s(  # noqa: E731
            s[None], a[None],
        )[0]
        scalar_jac_a = lambda s, a: batched_jac_a(  # noqa: E731
            s[None], a[None],
        )[0]

        scalar_solver = ILQRSolver(
            config=config,
            dynamics_fn=scalar_dyn,
            reward_fn=reward_fn,
            dynamics_jac_state_fn=scalar_jac_s,
            dynamics_jac_action_fn=scalar_jac_a,
            reward_derivatives_fn=reward_derivs,
        )

        batched_solver = BatchedNumpyILQRSolver(
            config=config,
            batched_dynamics_fn=batched_dyn,
            reward_fn=reward_fn,
            batched_jac_state_fn=batched_jac_s,
            batched_jac_action_fn=batched_jac_a,
            reward_derivatives_fn=reward_derivs,
        )

        initial_state = np.array([1.0, 0.5])
        np.random.seed(42)
        scalar_sol = scalar_solver.plan(initial_state, action_dim=1)
        np.random.seed(42)
        batched_sol = batched_solver.plan(initial_state, action_dim=1)

        # Both should achieve similar rewards (not identical due to
        # different iteration termination per restart)
        assert abs(scalar_sol.total_reward - batched_sol.total_reward) < 5.0

    def test_warm_start(self) -> None:
        """Solver uses warm start actions."""
        (
            batched_dyn, _, batched_jac_s, batched_jac_a,
            reward_fn, reward_derivs,
        ) = _make_linear_dynamics()

        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            n_random_restarts=3,
            max_action=2.0,
            gamma=0.99,
        )

        solver = BatchedNumpyILQRSolver(
            config=config,
            batched_dynamics_fn=batched_dyn,
            reward_fn=reward_fn,
            batched_jac_state_fn=batched_jac_s,
            batched_jac_action_fn=batched_jac_a,
            reward_derivatives_fn=reward_derivs,
        )

        initial_state = np.array([0.5, 0.0])
        warm_start = np.zeros((20, 1))
        sol = solver.plan(initial_state, action_dim=1, warm_start_actions=warm_start)

        assert sol.nominal_states.shape == (21, 2)
        assert sol.total_reward > -20.0

    def test_config_property(self) -> None:
        """Config is accessible via property."""
        (
            batched_dyn, _, batched_jac_s, batched_jac_a,
            reward_fn, _,
        ) = _make_linear_dynamics()

        config = ILQRConfig(horizon=10, n_random_restarts=2)
        solver = BatchedNumpyILQRSolver(
            config=config,
            batched_dynamics_fn=batched_dyn,
            reward_fn=reward_fn,
            batched_jac_state_fn=batched_jac_s,
            batched_jac_action_fn=batched_jac_a,
        )
        assert solver.config is config
        assert solver.config.horizon == 10
        assert solver.config.replan_interval is None


class TestBuildBatchedJacobianFns:
    """Tests for build_batched_jacobian_fns."""

    def test_linear_dynamics(self) -> None:
        """Jacobians match known analytic values for linear dynamics."""
        import sympy

        # x' = x + dt*v -> delta_x = dt*v
        # v' = v + dt*u -> delta_v = dt*u
        dt = 0.05

        class MockExpr:
            def __init__(self, expr: sympy.Expr) -> None:
                self.sympy_expr = expr

        x, v, u = sympy.symbols("x v u")
        exprs = {
            0: MockExpr(dt * v),   # delta_x = dt*v
            1: MockExpr(dt * u),   # delta_v = dt*u
        }

        result = build_batched_jacobian_fns(
            dynamics_expressions=exprs,
            state_names=["x", "v"],
            action_names=["u"],
            state_dim=2,
            env_params=None,
        )
        assert result is not None
        jac_s_fn, jac_a_fn = result

        states = np.array([[1.0, 2.0], [3.0, 4.0]])
        actions = np.array([[0.5], [1.0]])

        jac_s = jac_s_fn(states, actions)
        assert jac_s.shape == (2, 2, 2)
        # Expected: [[1, dt], [0, 1]]
        expected_js = np.array([
            [[1, dt], [0, 1]],
            [[1, dt], [0, 1]],
        ])
        np.testing.assert_allclose(jac_s, expected_js)

        jac_a = jac_a_fn(states, actions)
        assert jac_a.shape == (2, 2, 1)
        # Expected: [[0], [dt]]
        expected_ja = np.array([
            [[0], [dt]],
            [[0], [dt]],
        ])
        np.testing.assert_allclose(jac_a, expected_ja)

    def test_with_calibration(self) -> None:
        """Calibration coefficients scale Jacobian elements."""
        import sympy

        class MockExpr:
            def __init__(self, expr: sympy.Expr) -> None:
                self.sympy_expr = expr

        x, u = sympy.symbols("x u")
        exprs = {0: MockExpr(x + u)}

        cal = {0: (2.0, 0.5)}  # alpha=2.0, beta=0.5

        result = build_batched_jacobian_fns(
            dynamics_expressions=exprs,
            state_names=["x"],
            action_names=["u"],
            state_dim=1,
            env_params=None,
            calibration_coefficients=cal,
        )
        assert result is not None
        jac_s_fn, jac_a_fn = result

        states = np.array([[1.0]])
        actions = np.array([[0.5]])

        # d(delta)/dx = 1, with calibration alpha=2.0: jac = I + 2.0*1 = 3.0
        jac_s = jac_s_fn(states, actions)
        assert jac_s.shape == (1, 1, 1)
        np.testing.assert_allclose(jac_s[0, 0, 0], 3.0)

        # d(delta)/du = 1, with calibration alpha=2.0: jac = 2.0
        jac_a = jac_a_fn(states, actions)
        np.testing.assert_allclose(jac_a[0, 0, 0], 2.0)
