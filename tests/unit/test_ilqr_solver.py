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
    make_quadratic_terminal_cost,
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
            use_tanh_squash=False,
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

    def test_random_restarts_improve_or_match(self):
        """Multi-start iLQR should match or beat zero-init."""
        config_zero = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=0,
        )
        config_multi = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=3,
        )

        solver_zero = ILQRSolver(
            config=config_zero,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )
        solver_multi = ILQRSolver(
            config=config_multi,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )

        initial_state = np.array([2.0])
        sol_zero = solver_zero.plan(initial_state, action_dim=1)
        sol_multi = solver_multi.plan(initial_state, action_dim=1)

        # Multi-start should be at least as good as zero-init
        assert sol_multi.total_reward >= sol_zero.total_reward - 1e-6

        # Solution shapes should be correct
        assert sol_multi.nominal_states.shape == (31, 1)
        assert sol_multi.nominal_actions.shape == (30, 1)
        assert len(sol_multi.feedback_gains) == 30
        assert len(sol_multi.feedforward_gains) == 30

    def test_terminal_cost_improves_horizon_aware_planning(self):
        """Terminal cost should penalize ending in a bad state."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=30,
            gamma=0.99,
            max_action=10.0,
        )

        # Without terminal cost
        solver_no_tc = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        # With terminal cost from running reward
        tc_fn = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
        )
        solver_tc = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
            terminal_cost_fn=tc_fn,
        )

        initial_state = np.array([3.0, 1.0])
        sol_no_tc = solver_no_tc.plan(initial_state, action_dim=1)
        sol_tc = solver_tc.plan(initial_state, action_dim=1)

        # With terminal cost, the planner should drive the state
        # closer to zero at the end of the horizon
        final_norm_no_tc = float(np.linalg.norm(sol_no_tc.nominal_states[-1]))
        final_norm_tc = float(np.linalg.norm(sol_tc.nominal_states[-1]))
        assert final_norm_tc <= final_norm_no_tc + 0.1

    def test_terminal_cost_fn_outputs_correct_shapes(self):
        """make_quadratic_terminal_cost should return correct shapes."""
        tc_fn = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
        )

        state = np.array([1.0, 2.0])
        cost, grad, hess = tc_fn(state)

        assert isinstance(cost, float)
        assert grad.shape == (2,)
        assert hess.shape == (2, 2)

        # Cost should be positive (state is nonzero, reward is negative)
        assert cost > 0

        # Hessian should be symmetric
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)

    def test_analytic_reward_derivatives_match_fd(self):
        """When analytic reward derivatives are provided, the result
        should match finite-difference cost derivatives."""
        def reward_derivs(state, action):
            """Exact derivatives of _quadratic_reward for 2D state."""
            # r = -(x0^2 + x1^2 + 0.1*u0^2)
            r_x = -2 * state                                # (2,)
            r_u = -0.2 * action                              # (1,)
            r_xx = -2 * np.eye(len(state))                   # (2, 2)
            r_uu = -0.2 * np.eye(len(action))                # (1, 1)
            r_ux = np.zeros((len(action), len(state)))       # (1, 2)
            return r_x, r_u, r_xx, r_uu, r_ux

        config = ILQRConfig(
            horizon=30,
            max_iterations=20,
            gamma=0.99,
            max_action=10.0,
        )

        solver_analytic = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
            reward_derivatives_fn=reward_derivs,
        )
        solver_fd = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        initial_state = np.array([1.0, 0.5])
        sol_analytic = solver_analytic.plan(initial_state, action_dim=1)
        sol_fd = solver_fd.plan(initial_state, action_dim=1)

        # Both should converge
        assert sol_analytic.converged
        assert sol_fd.converged

        # Rewards should be nearly identical
        np.testing.assert_allclose(
            sol_analytic.total_reward, sol_fd.total_reward, atol=0.5,
        )

        # Actions should be very similar
        np.testing.assert_allclose(
            sol_analytic.nominal_actions,
            sol_fd.nominal_actions,
            atol=0.1,
        )

    def test_restart_scale_config(self):
        """restart_scale should control random restart action magnitude."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            gamma=0.99,
            max_action=10.0,
            n_random_restarts=3,
            restart_scale=0.1,
        )
        assert config.restart_scale == 0.1

        # Solver should accept the config without error
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )
        initial_state = np.array([1.0])
        sol = solver.plan(initial_state, action_dim=1)
        assert sol.nominal_states.shape == (21, 1)


# ---------------------------------------------------------------------------
# replan_interval validation
# ---------------------------------------------------------------------------

class TestReplanInterval:
    """Tests for the MPC-style replan_interval config field."""

    def test_replan_interval_default_is_none(self):
        """Default replan_interval is None (same as horizon)."""
        cfg = ILQRConfig(horizon=100)
        assert cfg.replan_interval is None

    def test_replan_interval_valid(self):
        """replan_interval <= horizon is accepted."""
        cfg = ILQRConfig(horizon=200, replan_interval=25)
        assert cfg.replan_interval == 25

    def test_replan_interval_equals_horizon(self):
        """replan_interval == horizon is valid (single-plan mode)."""
        cfg = ILQRConfig(horizon=50, replan_interval=50)
        assert cfg.replan_interval == 50

    def test_replan_interval_exceeds_horizon_raises(self):
        """replan_interval > horizon should raise."""
        with pytest.raises(ValueError, match="replan_interval"):
            ILQRConfig(horizon=50, replan_interval=100)

    def test_replan_interval_zero_raises(self):
        """replan_interval < 1 should raise."""
        with pytest.raises(ValueError, match="replan_interval"):
            ILQRConfig(replan_interval=0)

    def test_replan_interval_negative_raises(self):
        """Negative replan_interval should raise."""
        with pytest.raises(ValueError, match="replan_interval"):
            ILQRConfig(replan_interval=-5)

    def test_adaptive_replan_threshold_default_is_none(self):
        """Default adaptive_replan_threshold is None (disabled)."""
        cfg = ILQRConfig()
        assert cfg.adaptive_replan_threshold is None

    def test_adaptive_replan_threshold_valid(self):
        """Positive threshold is accepted."""
        cfg = ILQRConfig(adaptive_replan_threshold=0.5)
        assert cfg.adaptive_replan_threshold == 0.5

    def test_adaptive_replan_threshold_zero_raises(self):
        """Zero threshold should raise."""
        with pytest.raises(ValueError, match="adaptive_replan_threshold"):
            ILQRConfig(adaptive_replan_threshold=0.0)

    def test_adaptive_replan_threshold_negative_raises(self):
        """Negative threshold should raise."""
        with pytest.raises(ValueError, match="adaptive_replan_threshold"):
            ILQRConfig(adaptive_replan_threshold=-1.0)

    def test_min_replan_interval_default_is_three(self):
        """Default min_replan_interval is 3."""
        cfg = ILQRConfig()
        assert cfg.min_replan_interval == 3

    def test_min_replan_interval_valid(self):
        """min_replan_interval >= 1 is accepted."""
        cfg = ILQRConfig(min_replan_interval=1)
        assert cfg.min_replan_interval == 1

    def test_min_replan_interval_zero_raises(self):
        """min_replan_interval < 1 should raise."""
        with pytest.raises(ValueError, match="min_replan_interval"):
            ILQRConfig(min_replan_interval=0)


# ---------------------------------------------------------------------------
# Tanh squashing tests
# ---------------------------------------------------------------------------

class TestTanhSquashing:
    """Tests for smooth tanh action squashing."""

    def test_tanh_squash_bounds_respected(self):
        """With tanh squashing, actions stay strictly within bounds."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=15,
            gamma=0.99,
            max_action=0.5,
            use_tanh_squash=True,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )
        initial_state = np.array([5.0])
        sol = solver.plan(initial_state, action_dim=1)
        # tanh output is bounded to [-max_action, max_action]
        # (exactly +/-max_action is possible when tanh saturates in float64)
        assert np.all(sol.nominal_actions >= -0.5)
        assert np.all(sol.nominal_actions <= 0.5)

    def test_tanh_squash_converges(self):
        """Tanh squashing should converge to a reasonable solution."""
        config = ILQRConfig(
            horizon=30,
            max_iterations=30,
            gamma=0.99,
            max_action=5.0,
            convergence_tol=1e-6,
            use_tanh_squash=True,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )
        initial_state = np.array([2.0])
        sol = solver.plan(initial_state, action_dim=1)
        assert sol.converged

    def test_clip_mode_still_works(self):
        """Setting use_tanh_squash=False uses hard clipping."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=15,
            gamma=0.99,
            max_action=0.5,
            use_tanh_squash=False,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_nonlinear_dynamics,
            reward_fn=_nonlinear_reward,
        )
        initial_state = np.array([5.0])
        sol = solver.plan(initial_state, action_dim=1)
        # Hard clip allows exactly +/- max_action
        assert np.all(sol.nominal_actions >= -0.5)
        assert np.all(sol.nominal_actions <= 0.5)


# ---------------------------------------------------------------------------
# Terminal cost scale override and eigenvalue clamping
# ---------------------------------------------------------------------------

class TestTerminalCostScale:
    """Tests for terminal cost scale_override and Hessian clamping."""

    def test_scale_override_reduces_cost(self):
        """scale_override=10 should produce ~10x smaller cost than
        the default gamma/(1-gamma)=99."""
        tc_default = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
        )
        tc_reduced = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
            scale_override=10.0,
        )
        state = np.array([1.0, 2.0])
        cost_default, _, _ = tc_default(state)
        cost_reduced, _, _ = tc_reduced(state)
        # Approximately 10/99 ratio
        ratio = cost_reduced / cost_default
        np.testing.assert_allclose(ratio, 10.0 / 99.0, rtol=0.05)

    def test_hessian_eigenvalue_clamping(self):
        """max_hessian_eigval should clamp large eigenvalues."""
        tc_fn = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
            max_hessian_eigval=50.0,
        )
        state = np.array([1.0, 2.0])
        _, _, hess = tc_fn(state)
        eigvals = np.linalg.eigvalsh(hess)
        assert np.all(eigvals <= 50.0 + 1e-10)
        assert np.all(eigvals >= -50.0 - 1e-10)

    def test_hessian_still_symmetric_after_clamping(self):
        """Eigenvalue clamping should preserve symmetry."""
        tc_fn = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
            max_hessian_eigval=10.0,
        )
        state = np.array([1.0, 2.0])
        _, _, hess = tc_fn(state)
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)

    def test_scale_override_none_uses_default(self):
        """When scale_override is None, use gamma/(1-gamma)."""
        tc_a = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
            scale_override=None,
        )
        tc_b = make_quadratic_terminal_cost(
            reward_fn=_quadratic_reward,
            action_dim=1,
            gamma=0.99,
            state_dim=2,
        )
        state = np.array([1.0, 2.0])
        cost_a, _, _ = tc_a(state)
        cost_b, _, _ = tc_b(state)
        np.testing.assert_allclose(cost_a, cost_b)


# ---------------------------------------------------------------------------
# Parallel restarts tests
# ---------------------------------------------------------------------------

class TestParallelRestarts:
    """Test that parallel restarts produce valid solutions."""

    def test_parallel_restarts_returns_valid_solution(self):
        """Parallel restarts should produce a valid ILQRSolution."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=3,
            parallel_restarts=True,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )
        state = np.array([1.0, 0.5])
        sol = solver.plan(state, action_dim=1)

        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_states.shape == (21, 2)
        assert sol.nominal_actions.shape == (20, 1)
        assert np.isfinite(sol.total_reward)

    def test_parallel_matches_sequential_quality(self):
        """Parallel mode should produce at least as good reward as sequential."""
        config_par = ILQRConfig(
            horizon=20,
            max_iterations=15,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=4,
            parallel_restarts=True,
        )
        config_seq = ILQRConfig(
            horizon=20,
            max_iterations=15,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=0,
            parallel_restarts=False,
        )

        solver_par = ILQRSolver(
            config=config_par,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )
        solver_seq = ILQRSolver(
            config=config_seq,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )

        state = np.array([2.0, -1.0])
        sol_par = solver_par.plan(state, action_dim=1)
        sol_seq = solver_seq.plan(state, action_dim=1)

        # Multi-start should be at least as good
        assert sol_par.total_reward >= sol_seq.total_reward - 1.0

    def test_parallel_false_runs_sequentially(self):
        """With parallel_restarts=False, should still work (sequential)."""
        config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            gamma=0.99,
            max_action=5.0,
            n_random_restarts=2,
            parallel_restarts=False,
        )
        solver = ILQRSolver(
            config=config,
            dynamics_fn=_linear_dynamics,
            reward_fn=_quadratic_reward,
        )
        state = np.array([1.0, 0.5])
        sol = solver.plan(state, action_dim=1)
        assert np.isfinite(sol.total_reward)
