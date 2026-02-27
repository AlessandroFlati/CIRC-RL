# ruff: noqa: ANN001 ANN201

"""Unit tests for Numba JIT and PyTorch dynamics compilation."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from circ_rl.analytic_policy.fast_dynamics import (
    _has_numba,
    _sympy_to_python_source,
    build_fast_dynamics_fn,
    build_fast_jacobian_fns,
    build_fast_reward_fn,
    compile_numba_scalar_fn,
    compile_torch_fn,
)
from circ_rl.hypothesis.expression import SymbolicExpression


# ---------------------------------------------------------------------------
# Helper: make a SymbolicExpression with a .sympy_expr attribute
# ---------------------------------------------------------------------------

def _make_expr(sympy_expr: sympy.Expr) -> SymbolicExpression:
    """Wrap a sympy expression in a SymbolicExpression object."""
    return SymbolicExpression.from_sympy(sympy_expr)


# ---------------------------------------------------------------------------
# Sympy to Python source
# ---------------------------------------------------------------------------

class TestSympyToPythonSource:
    """Test the sympy-to-Python source conversion."""

    def test_basic_arithmetic(self):
        x, y = sympy.symbols("x y")
        code = _sympy_to_python_source(x + y * 2, ["x", "y"])
        assert "x" in code
        assert "y" in code

    def test_math_functions(self):
        x = sympy.Symbol("x")
        code = _sympy_to_python_source(sympy.sin(x), ["x"])
        assert "sin" in code

    def test_returns_string(self):
        x = sympy.Symbol("x")
        result = _sympy_to_python_source(x**2, ["x"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Numba compilation (if available)
# ---------------------------------------------------------------------------

class TestCompileNumbaScalarFn:
    """Test Numba JIT compilation of scalar functions."""

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_simple_expression(self):
        x, y = sympy.symbols("x y")
        expr = x**2 + y * 3
        fn = compile_numba_scalar_fn(expr, ["x", "y"])
        assert fn is not None
        result = fn(2.0, 1.0)
        assert abs(result - 7.0) < 1e-10

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_trig_expression(self):
        theta = sympy.Symbol("theta")
        expr = sympy.sin(theta)
        fn = compile_numba_scalar_fn(expr, ["theta"])
        assert fn is not None
        result = fn(np.pi / 2)
        assert abs(result - 1.0) < 1e-10

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_constant_expression(self):
        x = sympy.Symbol("x")
        expr = sympy.Integer(42)
        fn = compile_numba_scalar_fn(expr, ["x"])
        assert fn is not None
        result = fn(0.0)
        assert abs(result - 42.0) < 1e-10

    def test_returns_none_without_numba_or_on_failure(self):
        """If numba is unavailable, should return None gracefully."""
        if _has_numba():
            # Can't easily test "no numba" case, but test invalid expr
            # This should still succeed for valid expressions
            x = sympy.Symbol("x")
            fn = compile_numba_scalar_fn(x, ["x"])
            assert fn is not None
        else:
            x = sympy.Symbol("x")
            fn = compile_numba_scalar_fn(x, ["x"])
            assert fn is None


# ---------------------------------------------------------------------------
# Build fast dynamics
# ---------------------------------------------------------------------------

class TestBuildFastDynamicsFn:
    """Test the Numba-accelerated dynamics function builder."""

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_pendulum_like_dynamics(self):
        """Test dynamics for a simple pendulum-like system."""
        # delta_s0 = s1 * dt
        # delta_s1 = g * sin(s0) + action
        s0, s1, action = sympy.symbols("s0 s1 action")
        dt = 0.05
        expr0 = _make_expr(s1 * dt)
        expr1 = _make_expr(9.81 * sympy.sin(s0) + action)

        dynamics = {0: expr0, 1: expr1}
        fn = build_fast_dynamics_fn(
            dynamics,
            state_names=["s0", "s1"],
            action_names=["action"],
            state_dim=2,
            env_params=None,
        )
        assert fn is not None

        state = np.array([0.1, 0.5])
        action_val = np.array([0.2])
        result = fn(state, action_val)

        expected_s0 = 0.1 + 0.5 * dt
        expected_s1 = 0.5 + 9.81 * np.sin(0.1) + 0.2
        np.testing.assert_allclose(result[0], expected_s0, atol=1e-8)
        np.testing.assert_allclose(result[1], expected_s1, atol=1e-8)

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_env_param_substitution(self):
        """Test that env params are substituted before compilation."""
        g, s0, action = sympy.symbols("g s0 action")
        expr0 = _make_expr(g * s0 + action)

        dynamics = {0: expr0}
        fn = build_fast_dynamics_fn(
            dynamics,
            state_names=["s0"],
            action_names=["action"],
            state_dim=1,
            env_params={"g": 9.81},
        )
        assert fn is not None

        state = np.array([1.0])
        action_val = np.array([0.5])
        result = fn(state, action_val)
        expected = 1.0 + 9.81 * 1.0 + 0.5
        np.testing.assert_allclose(result[0], expected, atol=1e-8)

    def test_returns_none_without_numba(self):
        """Without numba, should return None."""
        if _has_numba():
            pytest.skip("Numba is installed, can't test fallback")
        s0 = sympy.Symbol("s0")
        expr = _make_expr(s0)
        fn = build_fast_dynamics_fn(
            {0: expr}, ["s0"], ["action"], 1, None,
        )
        assert fn is None


# ---------------------------------------------------------------------------
# Build fast Jacobians
# ---------------------------------------------------------------------------

class TestBuildFastJacobianFns:
    """Test Numba-accelerated Jacobian functions."""

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_linear_system_jacobian(self):
        """For delta = 0.1*s1 + 0.2*action, df/ds = [1, 0.1], df/du = [0.2]."""
        s0, s1, action = sympy.symbols("s0 s1 action")
        # delta_s0 = 0 (identity), delta_s1 = 0.1*s0 + 0.2*action
        expr0 = _make_expr(sympy.Float(0.0))
        expr1 = _make_expr(0.1 * s0 + 0.2 * action)

        jac_s, jac_a = build_fast_jacobian_fns(
            {0: expr0, 1: expr1},
            state_names=["s0", "s1"],
            action_names=["action"],
            state_dim=2,
            env_params=None,
        )
        assert jac_s is not None
        assert jac_a is not None

        state = np.array([1.0, 2.0])
        act = np.array([0.5])

        # State Jacobian: I + d(delta)/ds
        # delta_0 = 0 -> row0 = [1, 0]
        # delta_1 = 0.1*s0 + 0.2*action -> row1 = [0.1, 1]
        j_s = jac_s(state, act)
        np.testing.assert_allclose(j_s[0, 0], 1.0, atol=1e-8)
        np.testing.assert_allclose(j_s[0, 1], 0.0, atol=1e-8)
        np.testing.assert_allclose(j_s[1, 0], 0.1, atol=1e-8)
        np.testing.assert_allclose(j_s[1, 1], 1.0, atol=1e-8)

        # Action Jacobian
        j_a = jac_a(state, act)
        np.testing.assert_allclose(j_a[0, 0], 0.0, atol=1e-8)
        np.testing.assert_allclose(j_a[1, 0], 0.2, atol=1e-8)


# ---------------------------------------------------------------------------
# Build fast reward
# ---------------------------------------------------------------------------

class TestBuildFastRewardFn:
    """Test Numba-accelerated reward function."""

    @pytest.mark.skipif(not _has_numba(), reason="Numba not installed")
    def test_quadratic_reward(self):
        s0, action = sympy.symbols("s0 action")
        expr = _make_expr(-(s0**2) - 0.1 * action**2)

        fn = build_fast_reward_fn(
            expr,
            state_names=["s0"],
            action_names=["action"],
            env_params=None,
        )
        assert fn is not None

        state = np.array([2.0])
        act = np.array([1.0])
        result = fn(state, act)
        expected = -(4.0) - 0.1
        assert abs(result - expected) < 1e-8

    def test_returns_none_with_derived_features(self):
        """Reward with derived features should return None."""
        s0, action = sympy.symbols("s0 action")
        expr = _make_expr(-(s0**2))

        fn = build_fast_reward_fn(
            expr,
            state_names=["s0"],
            action_names=["action"],
            env_params=None,
            derived_feature_specs=[object()],  # non-None triggers fallback
        )
        assert fn is None


# ---------------------------------------------------------------------------
# PyTorch compilation
# ---------------------------------------------------------------------------

class TestCompileTorchFn:
    """Test PyTorch-compatible function compilation."""

    def test_simple_expression(self):
        x, y = sympy.symbols("x y")
        expr = x**2 + y * 3
        fn = compile_torch_fn(expr, ["x", "y"])
        assert fn is not None

        import torch
        result = fn(torch.tensor(2.0), torch.tensor(1.0))
        assert abs(float(result) - 7.0) < 1e-6

    def test_trig_expression(self):
        theta = sympy.Symbol("theta")
        expr = sympy.sin(theta)
        fn = compile_torch_fn(expr, ["theta"])
        assert fn is not None

        import torch
        result = fn(torch.tensor(np.pi / 2))
        assert abs(float(result) - 1.0) < 1e-6

    def test_batched_evaluation(self):
        """Torch functions should work with batched inputs."""
        x = sympy.Symbol("x")
        expr = x**2
        fn = compile_torch_fn(expr, ["x"])
        assert fn is not None

        import torch
        batch = torch.tensor([1.0, 2.0, 3.0])
        result = fn(batch)
        expected = torch.tensor([1.0, 4.0, 9.0])
        assert torch.allclose(result, expected, atol=1e-6)
