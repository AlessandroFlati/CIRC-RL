# ruff: noqa: ANN001 ANN201

"""Unit tests for the iLQR solver."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.ilqr_solver import (
    ILQRConfig,
    ILQRSolution,
    ILQRSolver,
    _finite_diff_jac_action,
    _finite_diff_jac_state,
)


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------

def _linear_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Simple 2D linear system: x' = A x + B u."""
    a_mat = np.array([[1.0, 0.1], [0.0, 1.0]])
    b_mat = np.array([[0.0], [0.1]])
    return a_mat @ state + b_mat @ action


def _quadratic_reward(state: np.ndarray, action: np.ndarray) -> float:
    """Quadratic reward: -(x^T Q x + u^T R u)."""
    return -float(state @ state + 0.1 * action @ action)


def _nonlinear_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """1D nonlinear system: x' = x + 0.1 * sin(x) + 0.1 * u."""
    x = state[0]
    u = action[0]
    return np.array([x + 0.1 * np.sin(x) + 0.1 * u])


def _nonlinear_reward(state: np.ndarray, action: np.ndarray) -> float:
    """1D quadratic reward."""
    return -float(state[0] ** 2 + 0.1 * action[0] ** 2)


# ---------------------------------------------------------------------------
# Finite difference Jacobian tests
# ---------------------------------------------------------------------------

class TestFiniteDiffJacobians:
    """Test the finite difference Jacobian functions."""

    def test_jac_state_linear(self):
        """For a linear system, the Jacobian should match A."""
        state = np.array([1.0, 2.0])
        action = np.array([0.5])
        jac = _finite_diff_jac_state(
            _linear_dynamics, state, action, state_dim=2,
        )
        expected = np.array([[1.0, 0.1], [0.0, 1.0]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)

    def test_jac_action_linear(self):
        """For a linear system, the Jacobian should match B."""
        state = np.array([1.0, 2.0])
        action = np.array([0.5])
        jac = _finite_diff_jac_action(
            _linear_dynamics, state, action,
            state_dim=2, action_dim=1,
        )
        expected = np.array([[0.0], [0.1]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)

    def test_jac_state_nonlinear(self):
        """For x' = x + 0.1*sin(x) + 0.1*u, df/dx = 1 + 0.1*cos(x)."""
        x0 = 1.0
        state = np.array([x0])
        action = np.array([0.0])
        jac = _finite_diff_jac_state(
            _nonlinear_dynamics, state, action, state_dim=1,
        )
        expected = np.array([[1.0 + 0.1 * np.cos(x0)]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)

    def test_jac_action_nonlinear(self):
        """For x' = x + 0.1*sin(x) + 0.1*u, df/du = 0.1."""
        state = np.array([1.0])
        action = np.array([0.0])
        jac = _finite_diff_jac_action(
            _nonlinear_dynamics, state, action,
            state_dim=1, action_dim=1,
        )
        expected = np.array([[0.1]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# ILQRSolver tests
# ---------------------------------------------------------------------------

class TestILQRSolver:
    """Test the ILQRSolver class."""

    def test_linear_system_converges(self):
        """iLQR on a linear system should converge."""
        config = ILQRConfig(
            horizon=50,
            max_iterations=30,
            gamma=0.99,
            max_action=10.0,
            convergence_tol=1e-6,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        initial_state = np.array([1.0, 0.0])
        sol = solver.plan(initial_state, action_dim=1)

        assert isinstance(sol, ILQRSolution)
        assert sol.converged
        assert sol.n_iterations <= 30
        assert sol.total_reward > -100  # Should find a decent policy
        assert sol.nominal_states.shape == (51, 2)
        assert sol.nominal_actions.shape == (50, 1)
        assert len(sol.feedback_gains) == 50
        assert len(sol.feedforward_gains) == 50

    def test_linear_system_improves_reward(self):
        """iLQR should improve over the zero-action baseline."""
        config = ILQRConfig(
            horizon=50,
            max_iterations=20,
            gamma=0.99,
            max_action=10.0,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        initial_state = np.array([1.0, 0.0])

        # Zero-action baseline reward
        states_zero = np.zeros((51, 2))
        states_zero[0] = initial_state
        for t in range(50):
            states_zero[t + 1] = _linear_dynamics(
                states_zero[t], np.zeros(1),
            )
        zero_reward = sum(
            0.99 ** t * _quadratic_reward(states_zero[t], np.zeros(1))
            for t in range(50)
        )

        sol = solver.plan(initial_state, action_dim=1)
        assert sol.total_reward > zero_reward

    def test_nonlinear_system_converges(self):
        """iLQR on a simple nonlinear system should converge."""
        config = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            convergence_tol=1e-6,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([2.0])
        sol = solver.plan(initial_state, action_dim=1)

        assert sol.converged
        assert sol.total_reward > -50
        # The policy should drive state toward zero
        assert abs(sol.nominal_states[-1, 0]) < abs(initial_state[0])

    def test_action_bounds_respected(self):
        """Actions should be clipped to [-max_action, max_action]."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            gamma=0.99,
            max_action=0.5,  # Tight bound
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([5.0])  # Large initial state
        sol = solver.plan(initial_state, action_dim=1)

        assert np.all(sol.nominal_actions >= -0.5 - 1e-10)
        assert np.all(sol.nominal_actions <= 0.5 + 1e-10)

    def test_warm_start(self):
        """Warm-starting with a good initial guess should converge faster."""
        config = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            convergence_tol=1e-6,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([2.0])

        # First solve from scratch
        sol1 = solver.plan(initial_state, action_dim=1)

        # Warm-start with the solution from the first solve
        sol2 = solver.plan(
            initial_state, action_dim=1,
            warm_start_actions=sol1.nominal_actions,
        )

        # Warm-started should converge in fewer iterations
        assert sol2.n_iterations <= sol1.n_iterations

    def test_analytic_jacobians_match_finite_diff(self):
        """When analytic Jacobians are provided, the result should
        be similar to (or better than) finite-difference Jacobians."""
        def jac_state(state, action):
            x = state[0]
            return np.array([[1.0 + 0.1 * np.cos(x)]])

        def jac_action(state, action):
            return np.array([[0.1]])

        config = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            convergence_tol=1e-6,
        )

        # With analytic Jacobians
        solver_analytic = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
            dynamics_jac_state_fn=jac_state,
            dynamics_jac_action_fn=jac_action,
        )

        # Without (finite diff)
        solver_fd = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([2.0])
        sol_analytic = solver_analytic.plan(initial_state, action_dim=1)
        sol_fd = solver_fd.plan(initial_state, action_dim=1)

        # Both should converge
        assert sol_analytic.converged
        assert sol_fd.converged

        # Rewards should be similar
        assert abs(sol_analytic.total_reward - sol_fd.total_reward) < 1.0

    def test_feedback_gains_shape(self):
        """Feedback gains should have correct shapes."""
        config = ILQRConfig(
            horizon=10,
            max_iterations=10,
            gamma=0.99,
            max_action=10.0,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        initial_state = np.array([1.0, 0.0])
        sol = solver.plan(initial_state, action_dim=1)

        for t in range(10):
            assert sol.feedback_gains[t].shape == (1, 2), (
                f"K[{t}] shape: {sol.feedback_gains[t].shape}"
            )
            assert sol.feedforward_gains[t].shape == (1,), (
                f"k[{t}] shape: {sol.feedforward_gains[t].shape}"
            )

    def test_feedback_correction(self):
        """Closed-loop policy u = u* + K @ dx should stay near nominal."""
        config = ILQRConfig(
            horizon=30,
            max_iterations=20,
            gamma=0.99,
            max_action=10.0,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        initial_state = np.array([1.0, 0.0])
        sol = solver.plan(initial_state, action_dim=1)

        # Simulate with feedback correction from a perturbed start
        perturbed_state = initial_state + np.array([0.1, -0.05])
        states = np.zeros((31, 2))
        states[0] = perturbed_state

        for t in range(30):
            dx = states[t] - sol.nominal_states[t]
            action = sol.nominal_actions[t] + sol.feedback_gains[t] @ dx
            action = np.clip(action, -10.0, 10.0)
            states[t + 1] = _linear_dynamics(states[t], action)

        # Perturbed trajectory should converge toward nominal
        final_deviation = np.linalg.norm(
            states[-1] - sol.nominal_states[-1],
        )
        initial_deviation = np.linalg.norm(
            perturbed_state - initial_state,
        )
        assert final_deviation < initial_deviation

    def test_zero_initial_state(self):
        """Starting from the origin should converge immediately."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            gamma=0.99,
            max_action=5.0,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([0.0])
        sol = solver.plan(initial_state, action_dim=1)

        # At the origin, the system is at equilibrium
        # The optimal action should be near zero
        assert abs(sol.nominal_actions[0, 0]) < 0.5
        assert sol.total_reward > -1.0
