"""Tests for residual and composite policy (M9)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch

from circ_rl.analytic_policy.analytic_policy import AnalyticPolicy
from circ_rl.analytic_policy.lqr_solver import LQRSolution
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.hypothesis_register import HypothesisEntry, HypothesisStatus
from circ_rl.policy.composite_policy import CompositePolicy
from circ_rl.policy.residual_policy import ResidualPolicy


def _make_lqr_policy(k_gain: np.ndarray) -> AnalyticPolicy:
    """Create a simple LQR-based AnalyticPolicy."""
    state_dim = k_gain.shape[1]
    action_dim = k_gain.shape[0]

    sol = LQRSolution(
        k_gain=k_gain,
        p_matrix=np.eye(state_dim),
        is_stable=True,
    )

    x = sympy.Symbol("x")
    expr = SymbolicExpression.from_sympy(x)
    entry = HypothesisEntry(
        hypothesis_id="test",
        target_variable="delta_s0",
        expression=expr,
        complexity=1,
        training_r2=0.99,
        training_mse=0.01,
        status=HypothesisStatus.VALIDATED,
    )

    return AnalyticPolicy(
        dynamics_hypothesis=entry,
        reward_hypothesis=None,
        solver_type="lqr",
        state_dim=state_dim,
        action_dim=action_dim,
        n_envs=1,
        lqr_solutions={0: sol},
    )


class TestResidualPolicy:
    def test_magnitude_bounded(self):
        """Residual correction is bounded by eta_max * |a_analytic|."""
        eta_max = 0.1
        residual = ResidualPolicy(
            state_dim=2, action_dim=1, hidden_dims=(32,), eta_max=eta_max,
        )

        state = torch.randn(100, 2)
        analytic_action = torch.ones(100, 1) * 5.0  # Large analytic action

        with torch.no_grad():
            output = residual(state, analytic_action)

        # |delta_a| <= eta_max * |a_analytic| = 0.1 * 5.0 = 0.5
        max_correction = eta_max * analytic_action.abs()
        assert torch.all(output.delta_action.abs() <= max_correction + 1e-6)

    def test_no_env_param_access(self):
        """ResidualPolicy takes only state, not env params."""
        residual = ResidualPolicy(state_dim=3, action_dim=1)

        # The input is state only -- no context_dim argument
        state = torch.randn(10, 3)
        analytic = torch.ones(10, 1)

        output = residual(state, analytic)
        assert output.delta_action.shape == (10, 1)
        assert output.value.shape == (10,)

    def test_evaluate_actions_matches(self):
        """evaluate_actions computes valid log-probs for taken actions."""
        residual = ResidualPolicy(state_dim=2, action_dim=1)

        state = torch.randn(10, 2)
        analytic = torch.ones(10, 1)

        with torch.no_grad():
            fwd = residual(state, analytic)
            eval_out = residual.evaluate_actions(
                state, analytic, fwd.raw_output,
            )

        # Log-probs should match for the same raw actions
        np.testing.assert_allclose(
            fwd.log_prob.numpy(), eval_out.log_prob.numpy(), atol=1e-5,
        )


class TestCompositePolicy:
    def test_combines_correctly(self):
        """Composite = analytic + residual."""
        k_gain = np.array([[0.5, 0.3]])
        analytic = _make_lqr_policy(k_gain)
        residual = ResidualPolicy(
            state_dim=2, action_dim=1, eta_max=0.05,
        )

        composite = CompositePolicy(
            analytic_policy=analytic,
            residual_policy=residual,
            explained_variance=0.95,
        )

        state = np.array([1.0, 0.0])
        action = composite.get_action(state, env_idx=0)

        # Should be close to the analytic action (residual is small)
        analytic_action = analytic.get_action(state, env_idx=0)
        assert action.shape == analytic_action.shape
        # Residual is bounded by eta_max * |analytic|, so composite
        # should be within that distance
        max_delta = 0.05 * np.abs(analytic_action)
        np.testing.assert_array_less(
            np.abs(action - analytic_action), max_delta + 1e-3,
        )

    def test_purely_analytic(self):
        """Without residual, composite = analytic."""
        k_gain = np.array([[0.5, 0.3]])
        analytic = _make_lqr_policy(k_gain)

        composite = CompositePolicy(
            analytic_policy=analytic,
            residual_policy=None,
            explained_variance=0.99,
        )

        assert not composite.has_residual

        state = np.array([1.0, 0.5])
        c_action = composite.get_action(state, env_idx=0)
        a_action = analytic.get_action(state, env_idx=0)
        np.testing.assert_allclose(c_action, a_action)

    def test_skip_when_eta2_high(self):
        """When explained_variance > threshold, residual should not be needed."""
        composite = CompositePolicy(
            analytic_policy=_make_lqr_policy(np.array([[1.0, 0.0]])),
            residual_policy=None,
            explained_variance=0.99,
        )
        assert composite.explained_variance > 0.98
        assert not composite.has_residual
