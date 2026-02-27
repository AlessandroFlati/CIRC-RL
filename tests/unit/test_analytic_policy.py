"""Tests for circ_rl.analytic_policy module (M8: Analytic Policy Derivation)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from circ_rl.analytic_policy.action_normalizer import ActionNormalizer
from circ_rl.analytic_policy.analytic_policy import (
    AnalyticPolicy,
    extract_linear_dynamics,
    extract_quadratic_cost,
)
from circ_rl.analytic_policy.hypothesis_classifier import (
    HypothesisClassifier,
    is_linear_in,
)
from circ_rl.analytic_policy.lqr_solver import (
    LinearDynamics,
    LQRSolver,
    QuadraticCost,
)
from circ_rl.analytic_policy.mpc_solver import MPCConfig, MPCSolver
from circ_rl.hypothesis.expression import SymbolicExpression
from circ_rl.hypothesis.hypothesis_register import (
    HypothesisEntry,
    HypothesisStatus,
)


# ------------------------------------------------------------------
# LQR Solver tests
# ------------------------------------------------------------------


class TestLQRSolver:
    def test_recovers_known_dare_solution(self):
        """DARE solution matches known result for a simple 1D system."""
        # Simple double integrator: s' = s + a
        a_mat = np.array([[1.0]])
        b_mat = np.array([[1.0]])
        q_mat = np.array([[1.0]])
        r_mat = np.array([[1.0]])

        dynamics = LinearDynamics(a_matrix=a_mat, b_matrix=b_mat)
        cost = QuadraticCost(q_matrix=q_mat, r_matrix=r_mat)

        solver = LQRSolver()
        solution = solver.solve(dynamics, cost, gamma=0.99)

        # K should be positive (negative feedback)
        assert solution.k_gain.shape == (1, 1)
        assert solution.k_gain[0, 0] > 0  # Gain should stabilize
        assert solution.is_stable
        assert solution.p_matrix.shape == (1, 1)
        assert solution.p_matrix[0, 0] > 0  # P should be positive

    def test_stabilizes_simple_system(self):
        """LQR should stabilize a marginally stable 2D system."""
        # Double integrator: position and velocity
        dt = 0.1
        a_mat = np.array([[1.0, dt], [0.0, 1.0]])
        b_mat = np.array([[0.0], [dt]])
        q_mat = np.eye(2)
        r_mat = np.array([[0.1]])

        dynamics = LinearDynamics(a_matrix=a_mat, b_matrix=b_mat)
        cost = QuadraticCost(q_matrix=q_mat, r_matrix=r_mat)

        solver = LQRSolver()
        solution = solver.solve(dynamics, cost, gamma=0.99)

        assert solution.is_stable
        assert solution.k_gain.shape == (1, 2)

        # Simulate: starting from [1, 0], should converge to origin
        state = np.array([1.0, 0.0])
        for _ in range(100):
            action = -solution.k_gain @ state
            state = a_mat @ state + b_mat @ action.ravel()

        assert np.linalg.norm(state) < 0.01

    def test_per_env_correct_gains(self):
        """Different dynamics per env should produce different gains."""
        solver = LQRSolver()
        cost = QuadraticCost(q_matrix=np.eye(1), r_matrix=np.eye(1))

        dynamics_per_env = {
            0: LinearDynamics(
                a_matrix=np.array([[1.0]]),
                b_matrix=np.array([[0.5]]),
            ),
            1: LinearDynamics(
                a_matrix=np.array([[1.0]]),
                b_matrix=np.array([[2.0]]),
            ),
        }

        solutions = solver.solve_per_env(dynamics_per_env, cost, gamma=0.99)

        assert 0 in solutions
        assert 1 in solutions
        # Different B should produce different K
        assert not np.allclose(
            solutions[0].k_gain, solutions[1].k_gain
        )


# ------------------------------------------------------------------
# Action Normalizer tests
# ------------------------------------------------------------------


class TestActionNormalizer:
    def test_scales_correctly(self):
        """Scale ratio should correctly normalize/denormalize actions."""
        scales = np.array([1.0, 2.0, 0.5])
        ref = 1.0
        normalizer = ActionNormalizer(dynamics_scales=scales, reference_scale=ref)

        action = np.array([1.0])

        # Env 0: D_e/D_ref = 1.0 -> no change
        np.testing.assert_allclose(
            normalizer.normalize_action(action, 0), [1.0]
        )

        # Env 1: D_e/D_ref = 2.0 -> doubled
        np.testing.assert_allclose(
            normalizer.normalize_action(action, 1), [2.0]
        )

        # Env 2: D_e/D_ref = 0.5 -> halved
        np.testing.assert_allclose(
            normalizer.normalize_action(action, 2), [0.5]
        )

        # Round-trip: normalize then denormalize
        for env_idx in range(3):
            normalized = normalizer.normalize_action(action, env_idx)
            recovered = normalizer.denormalize_action(normalized, env_idx)
            np.testing.assert_allclose(recovered, action, atol=1e-10)


# ------------------------------------------------------------------
# Hypothesis Classifier tests
# ------------------------------------------------------------------


class TestHypothesisClassifier:
    def test_detects_linear(self):
        """Linear expressions are classified as LQR-eligible."""
        s0, s1, action = sympy.symbols("s0 s1 action")

        # Simple linear: 2*s0 + 3*action + 1
        expr = 2 * s0 + 3 * action + 1
        result = HypothesisClassifier.classify(
            expr, ["s0", "s1"], ["action"]
        )
        assert result == "lqr"

        # Multi-variable linear: s0 + 0.5*s1 - action
        expr = s0 + 0.5 * s1 - action
        result = HypothesisClassifier.classify(
            expr, ["s0", "s1"], ["action"]
        )
        assert result == "lqr"

    def test_detects_nonlinear(self):
        """Nonlinear expressions are classified as MPC-required."""
        s0, s1, action = sympy.symbols("s0 s1 action")

        # Quadratic: s0^2
        expr = s0**2 + action
        result = HypothesisClassifier.classify(
            expr, ["s0", "s1"], ["action"]
        )
        assert result == "mpc"

        # Cross term: s0 * action
        expr = s0 * action
        result = HypothesisClassifier.classify(
            expr, ["s0", "s1"], ["action"]
        )
        assert result == "mpc"

        # Transcendental: sin(s0)
        expr = sympy.sin(s0) + action
        result = HypothesisClassifier.classify(
            expr, ["s0", "s1"], ["action"]
        )
        assert result == "mpc"


# ------------------------------------------------------------------
# MPC Solver tests
# ------------------------------------------------------------------


class TestMPCSolver:
    def test_produces_bounded_actions(self):
        """MPC actions should respect bounds."""
        max_action = 1.5

        def dynamics_fn(s: np.ndarray, a: np.ndarray) -> np.ndarray:
            return s + 0.1 * a

        config = MPCConfig(
            horizon=5, gamma=0.99, max_action=max_action,
            max_iter=50, tol=1e-4,
        )
        solver = MPCSolver(config, dynamics_fn)

        state = np.array([5.0])  # Far from origin
        action = solver.solve(state, action_dim=1)

        assert action.shape == (1,)
        assert np.all(np.abs(action) <= max_action + 1e-6)


# ------------------------------------------------------------------
# AnalyticPolicy integration tests
# ------------------------------------------------------------------


class TestAnalyticPolicy:
    def test_matches_lqr(self):
        """AnalyticPolicy with LQR produces the correct gain-based action."""
        from circ_rl.analytic_policy.lqr_solver import LQRSolution

        k_gain = np.array([[0.5, 0.3]])
        p_mat = np.eye(2)

        sol = LQRSolution(k_gain=k_gain, p_matrix=p_mat, is_stable=True)

        # Create a minimal hypothesis entry
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

        policy = AnalyticPolicy(
            dynamics_hypothesis=entry,
            reward_hypothesis=None,
            solver_type="lqr",
            state_dim=2,
            action_dim=1,
            n_envs=1,
            lqr_solutions={0: sol},
        )

        state = np.array([1.0, 2.0])
        action = policy.get_action(state, env_idx=0)

        expected = -k_gain @ state
        np.testing.assert_allclose(action, expected.ravel())


# ------------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------------


class TestExtractLinearDynamics:
    def test_extract_linear_dynamics(self):
        """Extract A row and B row from linear expression."""
        s0, s1, action = sympy.symbols("s0 s1 action")
        expr = 0.5 * s0 + 0.3 * s1 + 2.0 * action + 0.1
        se = SymbolicExpression.from_sympy(expr)

        ld = extract_linear_dynamics(
            se, ["s0", "s1"], ["action"],
        )

        np.testing.assert_allclose(ld.a_matrix, [[0.5, 0.3]])
        np.testing.assert_allclose(ld.b_matrix, [[2.0]])
        np.testing.assert_allclose(ld.c_vector, [0.1])


class TestExtractQuadraticCost:
    def test_extract_quadratic_cost(self):
        """Extract Q and R from -s0^2 - 0.1*action^2."""
        s0, action = sympy.symbols("s0 action")
        reward_expr = -s0**2 - 0.1 * action**2

        result = extract_quadratic_cost(
            reward_expr, ["s0"], ["action"]
        )

        assert result is not None
        np.testing.assert_allclose(result.q_matrix, [[1.0]])
        np.testing.assert_allclose(result.r_matrix, [[0.1]])

    def test_returns_none_for_nonquadratic(self):
        """Non-quadratic reward returns None."""
        s0 = sympy.Symbol("s0")
        result = extract_quadratic_cost(
            sympy.sin(s0), ["s0"], ["action"]
        )
        assert result is None
