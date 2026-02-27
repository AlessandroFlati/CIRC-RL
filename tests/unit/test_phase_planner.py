# ruff: noqa: ANN001 ANN201

"""Unit tests for the PhasePlanner."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.ilqr_solver import (
    ILQRConfig,
    ILQRSolution,
    ILQRSolver,
    make_quadratic_terminal_cost,
)
from circ_rl.analytic_policy.mppi_solver import MPPIConfig, MPPISolver
from circ_rl.analytic_policy.phase_planner import PhasePlanner


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


def _jac_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Jacobian of linear dynamics w.r.t. state."""
    return np.array([[1.0, 0.1], [0.0, 1.0]])


def _jac_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Jacobian of linear dynamics w.r.t. action."""
    return np.array([[0.0], [0.1]])


def _make_mppi_solver(horizon: int = 20) -> MPPISolver:
    cfg = MPPIConfig(
        horizon=horizon,
        n_samples=32,
        n_iterations=1,
        temperature=1.0,
        max_action=5.0,
        replan_interval=5,
    )
    return MPPISolver(cfg, _linear_dynamics, _quadratic_reward)


def _make_ilqr_solver(horizon: int = 10) -> ILQRSolver:
    cfg = ILQRConfig(
        horizon=horizon,
        gamma=0.99,
        max_action=5.0,
        n_random_restarts=0,
        replan_interval=5,
    )
    terminal = make_quadratic_terminal_cost(
        reward_fn=_quadratic_reward,
        action_dim=1,
        gamma=0.99,
        state_dim=2,
    )
    return ILQRSolver(
        cfg, _linear_dynamics, _quadratic_reward,
        _jac_state, _jac_action, terminal,
    )


# ---------------------------------------------------------------------------
# PhasePlanner tests
# ---------------------------------------------------------------------------


class TestPhasePlanner:
    """Test PhasePlanner solver switching."""

    def test_uses_global_when_far(self):
        """Should use global solver when use_local_fn returns False."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        # Always far from goal
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)
        sol = planner.plan(np.array([5.0, 5.0]), action_dim=1)

        assert isinstance(sol, ILQRSolution)
        assert planner.last_phase == "global"
        # Should have MPPI horizon (20)
        assert sol.nominal_actions.shape == (20, 1)

    def test_uses_local_when_near(self):
        """Should use local solver when use_local_fn returns True."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        # Always near goal
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: True)
        sol = planner.plan(np.array([0.1, 0.0]), action_dim=1)

        assert isinstance(sol, ILQRSolution)
        assert planner.last_phase == "local"
        # Should have iLQR horizon (10)
        assert sol.nominal_actions.shape == (10, 1)

    def test_state_dependent_switching(self):
        """Should switch based on state magnitude."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        # Use local when state norm < 1.0
        planner = PhasePlanner(
            mppi, ilqr,
            use_local_fn=lambda s: float(np.linalg.norm(s)) < 1.0,
        )

        # Far from goal -> global
        sol_far = planner.plan(np.array([5.0, 5.0]), action_dim=1)
        assert planner.last_phase == "global"
        assert sol_far.nominal_actions.shape[0] == 20

        # Near goal -> local
        sol_near = planner.plan(np.array([0.1, 0.0]), action_dim=1)
        assert planner.last_phase == "local"
        assert sol_near.nominal_actions.shape[0] == 10

    def test_config_returns_global(self):
        """config property should return global solver's config."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        assert planner.config.horizon == 20
        assert planner.config.replan_interval == 5

    def test_warm_start_resize_truncate(self):
        """Warm start larger than target should be truncated."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        # Near goal -> iLQR (horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: True)

        # Warm start with MPPI horizon (20)
        warm = np.ones((20, 1)) * 0.1
        sol = planner.plan(np.array([0.1, 0.0]), action_dim=1, warm_start_actions=warm)

        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_actions.shape == (10, 1)

    def test_warm_start_resize_pad(self):
        """Warm start smaller than target should be zero-padded."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        # Far from goal -> MPPI (horizon=20)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        # Warm start with iLQR horizon (10)
        warm = np.ones((10, 1)) * 0.1
        sol = planner.plan(np.array([5.0, 5.0]), action_dim=1, warm_start_actions=warm)

        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_actions.shape == (20, 1)

    def test_warm_start_none_passes_through(self):
        """None warm start should pass through cleanly."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)
        sol = planner.plan(np.array([1.0, 0.0]), action_dim=1, warm_start_actions=None)
        assert isinstance(sol, ILQRSolution)

    def test_returns_ilqr_solution(self):
        """Both phases should return ILQRSolution for compatibility."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: True)
        sol = planner.plan(np.array([0.1, 0.0]), action_dim=1)

        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_states is not None
        assert sol.feedback_gains is not None
        assert sol.feedforward_gains is not None

    def test_ilqr_feedback_gains_nonzero(self):
        """iLQR phase should produce nonzero feedback gains."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: True)
        sol = planner.plan(np.array([1.0, 0.5]), action_dim=1)

        # iLQR should have nonzero feedback gains (unlike MPPI)
        gains_norm = sum(
            float(np.linalg.norm(K)) for K in sol.feedback_gains
        )
        assert gains_norm > 0.0, "iLQR should produce nonzero feedback gains"

    def test_mppi_feedback_gains_zero(self):
        """MPPI phase should produce zero feedback gains."""
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)

        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)
        sol = planner.plan(np.array([1.0, 0.5]), action_dim=1)

        # MPPI has zero feedback gains (open-loop)
        for K in sol.feedback_gains:
            np.testing.assert_array_equal(K, np.zeros((1, 2)))


class TestResizeWarmStart:
    """Test the warm-start resizing logic."""

    def test_same_size_unchanged(self):
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        warm = np.ones((20, 1))
        result = planner._resize_warm_start(warm, 20, 1)
        assert result is not None
        np.testing.assert_array_equal(result, warm)

    def test_truncate(self):
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        warm = np.ones((20, 1))
        result = planner._resize_warm_start(warm, 10, 1)
        assert result is not None
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result, np.ones((10, 1)))

    def test_pad(self):
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        warm = np.ones((5, 1))
        result = planner._resize_warm_start(warm, 10, 1)
        assert result is not None
        assert result.shape == (10, 1)
        np.testing.assert_array_equal(result[:5], np.ones((5, 1)))
        np.testing.assert_array_equal(result[5:], np.zeros((5, 1)))

    def test_none_returns_none(self):
        mppi = _make_mppi_solver(horizon=20)
        ilqr = _make_ilqr_solver(horizon=10)
        planner = PhasePlanner(mppi, ilqr, use_local_fn=lambda s: False)

        result = planner._resize_warm_start(None, 10, 1)
        assert result is None
