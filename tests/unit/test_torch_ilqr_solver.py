# ruff: noqa: ANN001 ANN201

"""Unit tests for the GPU-batched PyTorch iLQR solver."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from circ_rl.analytic_policy.ilqr_solver import ILQRConfig
from circ_rl.analytic_policy.torch_ilqr_solver import (
    TorchILQRConfig,
    TorchILQRSolver,
    _resolve_device,
)


# ---------------------------------------------------------------------------
# Helper dynamics/reward as torch callables
# ---------------------------------------------------------------------------

def _torch_linear_dyn_dim0(s0, s1, action):
    """delta_s0 = 0.1 * s1."""
    return 0.1 * s1


def _torch_linear_dyn_dim1(s0, s1, action):
    """delta_s1 = 0.1 * action."""
    return 0.1 * action


def _torch_reward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Quadratic reward: -(s^T s + 0.1 * a^T a)."""
    s_cost = (state ** 2).sum(dim=-1)  # (B,)
    a_cost = (action ** 2).sum(dim=-1)  # (B,)
    return -(s_cost + 0.1 * a_cost)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolveDevice:
    """Test device resolution."""

    def test_auto_returns_device(self):
        dev = _resolve_device("auto")
        assert isinstance(dev, torch.device)

    def test_cpu_explicit(self):
        dev = _resolve_device("cpu")
        assert dev == torch.device("cpu")


class TestTorchILQRSolver:
    """Test the batched PyTorch iLQR solver."""

    @pytest.fixture()
    def solver(self):
        """Create a solver for a simple 2D linear system."""
        np_config = ILQRConfig(
            horizon=20,
            max_iterations=10,
            convergence_tol=1e-3,
            gamma=0.99,
            max_action=2.0,
            n_random_restarts=3,
            restart_scale=0.3,
        )
        t_config = TorchILQRConfig(
            n_batch=4,
            device="cpu",
            dtype=torch.float64,
            n_alpha=4,
        )

        dynamics_fns = {
            0: _torch_linear_dyn_dim0,
            1: _torch_linear_dyn_dim1,
        }

        return TorchILQRSolver(
            numpy_config=np_config,
            torch_config=t_config,
            dynamics_fns=dynamics_fns,
            reward_fn=_torch_reward,
            state_dim=2,
            action_dim=1,
        )

    def test_plan_returns_solution(self, solver):
        """Plan should return an ILQRSolution with correct shapes."""
        from circ_rl.analytic_policy.ilqr_solver import ILQRSolution

        state = np.array([1.0, 0.5])
        sol = solver.plan(state)

        assert isinstance(sol, ILQRSolution)
        assert sol.nominal_states.shape == (21, 2)  # horizon+1, state_dim
        assert sol.nominal_actions.shape == (20, 1)  # horizon, action_dim
        assert len(sol.feedback_gains) == 20
        assert len(sol.feedforward_gains) == 20
        assert isinstance(sol.total_reward, float)

    def test_plan_with_warm_start(self, solver):
        """Plan should accept warm-start actions."""
        state = np.array([1.0, 0.5])
        warm = np.zeros((20, 1))
        sol = solver.plan(state, warm_start_actions=warm)

        assert sol.nominal_actions.shape == (20, 1)

    def test_plan_improves_reward(self, solver):
        """Optimized plan should have positive reward contribution.

        Starting from state [1.0, 0.5], optimal actions should drive
        state towards zero, achieving a less negative (better) reward
        than doing nothing.
        """
        state = np.array([1.0, 0.5])
        sol = solver.plan(state)

        # Zero-action reward for reference
        zero_reward = 0.0
        s = state.copy()
        for t in range(20):
            a = np.zeros(1)
            zero_reward += 0.99 ** t * float(-(s @ s + 0.1 * a @ a))
            s = s + np.array([0.1 * s[1], 0.0])

        # Optimized should be at least as good as zero-action
        assert sol.total_reward >= zero_reward - 1.0  # small tolerance

    def test_feedback_gains_have_correct_shapes(self, solver):
        """Feedback gains should be (action_dim, state_dim) per step."""
        state = np.array([0.5, -0.3])
        sol = solver.plan(state)

        for K in sol.feedback_gains:
            assert K.shape == (1, 2)  # (action_dim, state_dim)
        for k in sol.feedforward_gains:
            assert k.shape == (1,)  # (action_dim,)

    def test_dynamics_step_batched(self, solver):
        """Internal dynamics step should work on batched states."""
        B = 4
        state = torch.tensor(
            [[1.0, 0.5], [0.0, 0.0], [-1.0, 0.3], [0.5, -0.5]],
            dtype=torch.float64,
        )
        action = torch.tensor(
            [[0.1], [0.0], [-0.1], [0.5]],
            dtype=torch.float64,
        )

        next_state = solver._dynamics_step(state, action)
        assert next_state.shape == (B, 2)

        # Check first batch: s0' = s0 + 0.1*s1, s1' = s1 + 0.1*action
        np.testing.assert_allclose(
            next_state[0, 0].item(), 1.0 + 0.1 * 0.5, atol=1e-10,
        )
        np.testing.assert_allclose(
            next_state[0, 1].item(), 0.5 + 0.1 * 0.1, atol=1e-10,
        )

    def test_rollout_shape(self, solver):
        """Rollout should produce correct shape."""
        B = 4
        H = 20
        x0 = torch.zeros(B, 2, dtype=torch.float64)
        actions = torch.zeros(B, H, 1, dtype=torch.float64)

        states = solver._rollout(x0, actions)
        assert states.shape == (B, H + 1, 2)

    def test_total_cost_batch(self, solver):
        """Total cost should return (B,) shape."""
        B = 4
        H = 20
        states = torch.zeros(B, H + 1, 2, dtype=torch.float64)
        actions = torch.zeros(B, H, 1, dtype=torch.float64)

        costs = solver._total_cost(states, actions)
        assert costs.shape == (B,)

        # Zero state and action -> zero cost
        np.testing.assert_allclose(costs.numpy(), 0.0, atol=1e-10)


class TestTorchILQRSolverAngular:
    """Test angular dimension wrapping in the torch solver."""

    def test_angular_wrapping(self):
        """Angular dims should be wrapped to [-pi, pi]."""

        def dyn_dim0(s0, action):
            """Large delta that pushes past pi."""
            return action * 2.0

        np_config = ILQRConfig(
            horizon=5,
            max_iterations=3,
            gamma=0.99,
            max_action=5.0,
        )
        t_config = TorchILQRConfig(
            n_batch=1,
            device="cpu",
            dtype=torch.float64,
        )

        solver = TorchILQRSolver(
            numpy_config=np_config,
            torch_config=t_config,
            dynamics_fns={0: dyn_dim0},
            reward_fn=lambda s, a: -(s ** 2).sum(dim=-1),
            state_dim=1,
            action_dim=1,
            angular_dims=(0,),
        )

        # Start at pi - 0.1, apply large positive action
        x0 = torch.tensor([[3.0]], dtype=torch.float64)
        actions = torch.tensor([[[4.0]]], dtype=torch.float64)  # (1, 1, 1)

        states = solver._rollout(x0, actions)
        # After adding delta = 4*2 = 8, raw = 3+8=11
        # atan2(sin(11), cos(11)) should be wrapped
        wrapped = states[0, 1, 0].item()
        assert -np.pi <= wrapped <= np.pi
