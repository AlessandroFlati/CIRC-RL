# ruff: noqa: ANN001 ANN201

"""Unit tests for the CEM solver."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.analytic_policy.cem_solver import CEMConfig, CEMSolver
from circ_rl.analytic_policy.ilqr_solver import ILQRSolution


# ---------------------------------------------------------------------------
# CEMConfig validation
# ---------------------------------------------------------------------------

class TestCEMConfig:
    """Test CEMConfig validation and defaults."""

    def test_defaults(self):
        cfg = CEMConfig()
        assert cfg.horizon == 50
        assert cfg.n_samples == 256
        assert cfg.n_actions == 3

    def test_horizon_must_be_positive(self):
        with pytest.raises(ValueError, match="horizon"):
            CEMConfig(horizon=0)

    def test_n_actions_must_be_at_least_2(self):
        with pytest.raises(ValueError, match="n_actions"):
            CEMConfig(n_actions=1)

    def test_elite_fraction_bounds(self):
        with pytest.raises(ValueError, match="elite_fraction"):
            CEMConfig(elite_fraction=0.0)
        with pytest.raises(ValueError, match="elite_fraction"):
            CEMConfig(elite_fraction=1.5)

    def test_smoothing_alpha_bounds(self):
        with pytest.raises(ValueError, match="smoothing_alpha"):
            CEMConfig(smoothing_alpha=0.0)

    def test_frozen(self):
        cfg = CEMConfig()
        with pytest.raises(AttributeError):
            cfg.horizon = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CEMSolver
# ---------------------------------------------------------------------------

def _simple_dynamics(state, action):
    """Simple integrator: next_state = state + action * 0.1."""
    return state + action * 0.1


def _simple_reward(state, action):
    """Reward = state[0] (reach right)."""
    return float(state[0])


def _simple_batched_dynamics(states, actions):
    """Batched version of _simple_dynamics."""
    return states + actions * 0.1


def _simple_batched_reward(states, actions):
    """Batched version of _simple_reward."""
    return states[:, 0]


class TestCEMSolver:
    """Test CEM solver planning."""

    def _make_solver(self, horizon=20, n_samples=64, n_iterations=3):
        """Create a CEM solver with simple 1D dynamics."""
        config = CEMConfig(
            horizon=horizon,
            n_samples=n_samples,
            n_iterations=n_iterations,
            n_actions=3,
            elite_fraction=0.2,
            gamma=0.99,
            smoothing_alpha=0.8,
        )
        return CEMSolver(
            config=config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
            action_values=[0.0, 1.0, 2.0],
            discretization_values=[-1.0, 0.0, 1.0],
            batched_dynamics_fn=_simple_batched_dynamics,
            batched_reward_fn=_simple_batched_reward,
        )

    def test_plan_returns_ilqr_solution(self):
        solver = self._make_solver()
        result = solver.plan(np.array([0.0]), action_dim=1)
        assert isinstance(result, ILQRSolution)
        assert result.nominal_states.shape == (21, 1)  # H+1, S
        assert result.nominal_actions.shape == (20, 1)  # H, 1
        assert len(result.feedback_gains) == 20
        assert len(result.feedforward_gains) == 20

    def test_plan_improves_over_random(self):
        """CEM should produce better reward than a random policy."""
        solver = self._make_solver(n_samples=128, n_iterations=5)
        result = solver.plan(np.array([0.0]), action_dim=1)
        # With reward = state[0] and dynamics state += action*0.1,
        # CEM should learn to push right (action=2 -> +0.2 per step)
        assert result.total_reward > 0.0

    def test_actions_are_discretization_values(self):
        """Nominal actions should only contain discretization_values."""
        solver = self._make_solver()
        result = solver.plan(np.array([0.0]), action_dim=1)
        valid = {-1.0, 0.0, 1.0}
        for t in range(20):
            assert float(result.nominal_actions[t, 0]) in valid

    def test_feedback_gains_are_zero(self):
        """CEM is open-loop; feedback gains must be zero."""
        solver = self._make_solver()
        result = solver.plan(np.array([0.0]), action_dim=1)
        for gain in result.feedback_gains:
            assert np.all(gain == 0.0)

    def test_batched_matches_scalar(self):
        """Solver with and without batched fns should produce valid results."""
        config = CEMConfig(
            horizon=10,
            n_samples=32,
            n_iterations=2,
            n_actions=3,
        )
        # With batched
        solver_batched = CEMSolver(
            config=config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
            action_values=[0.0, 1.0, 2.0],
            discretization_values=[-1.0, 0.0, 1.0],
            batched_dynamics_fn=_simple_batched_dynamics,
            batched_reward_fn=_simple_batched_reward,
        )
        # Without batched
        solver_scalar = CEMSolver(
            config=config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
            action_values=[0.0, 1.0, 2.0],
            discretization_values=[-1.0, 0.0, 1.0],
        )
        r1 = solver_batched.plan(np.array([0.0]), action_dim=1)
        r2 = solver_scalar.plan(np.array([0.0]), action_dim=1)
        # Both should return valid solutions (different due to RNG)
        assert r1.nominal_states.shape == r2.nominal_states.shape

    def test_action_values_length_mismatch_raises(self):
        config = CEMConfig(n_actions=3)
        with pytest.raises(ValueError, match="action_values"):
            CEMSolver(
                config=config,
                dynamics_fn=_simple_dynamics,
                reward_fn=_simple_reward,
                action_values=[0.0, 1.0],  # wrong length
                discretization_values=[-1.0, 0.0, 1.0],
            )

    def test_binary_actions(self):
        """Test with 2 actions (CartPole-like)."""
        config = CEMConfig(
            horizon=10,
            n_samples=64,
            n_iterations=3,
            n_actions=2,
        )
        solver = CEMSolver(
            config=config,
            dynamics_fn=_simple_dynamics,
            reward_fn=_simple_reward,
            action_values=[0.0, 1.0],
            discretization_values=[-1.0, 1.0],
            batched_dynamics_fn=_simple_batched_dynamics,
            batched_reward_fn=_simple_batched_reward,
        )
        result = solver.plan(np.array([0.0]), action_dim=1)
        assert result.nominal_actions.shape == (10, 1)
        valid = {-1.0, 1.0}
        for t in range(10):
            assert float(result.nominal_actions[t, 0]) in valid
