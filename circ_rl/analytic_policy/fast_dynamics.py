"""Fast dynamics compilation for iLQR inner loops.

Provides Numba JIT-compiled and PyTorch-compatible callables for
symbolic dynamics, Jacobian, and reward functions. After environment
parameter substitution, sympy expressions reduce to pure numeric
math on scalars. This module compiles them to eliminate Python
function-call overhead in the iLQR inner loop (called millions of
times per evaluation run).

Falls back gracefully to numpy lambdify when Numba is not installed.

Usage::

    from circ_rl.analytic_policy.fast_dynamics import (
        build_fast_dynamics_fn,
        build_fast_jacobian_fns,
    )

    # Returns Numba-compiled callable, or None if numba unavailable
    fast_fn = build_fast_dynamics_fn(expressions, ...)
    if fast_fn is None:
        fast_fn = _build_dynamics_fn(...)  # numpy fallback
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import sympy


def _has_numba() -> bool:
    """Check if Numba is available."""
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _sympy_to_python_source(
    expr: sympy.Expr,
    var_names: list[str],
) -> str:
    """Convert a sympy expression to a Python source string.

    Uses sympy's Python code printer to generate code that uses
    ``math`` module functions (not numpy). This produces code
    compatible with Numba's ``@njit`` decorator.

    :param expr: Sympy expression (env params already substituted).
    :param var_names: Names of the input variables in order.
    :returns: Python source string for the function body.
    """
    from sympy.printing.pycode import PythonCodePrinter

    printer = PythonCodePrinter({"fully_qualified_modules": False})
    code = printer.doprint(expr)
    # PythonCodePrinter uses math.* prefixes; ensure consistency
    code = code.replace("math.", "math.")
    return code


def compile_numba_scalar_fn(
    expr: sympy.Expr,
    var_names: list[str],
    fn_name: str = "_numba_fn",
) -> Callable[..., float] | None:
    """Compile a sympy expression to a Numba-JIT scalar function.

    The generated function takes individual float arguments (one per
    variable in ``var_names``) and returns a float. It is compiled
    with ``@numba.njit`` for near-C-speed execution.

    :param expr: Sympy expression with env params already substituted.
    :param var_names: Variable names in call order.
    :param fn_name: Name for the generated function.
    :returns: Compiled callable, or None if Numba is not available
        or compilation fails.
    """
    if not _has_numba():
        return None

    try:
        import numba

        body = _sympy_to_python_source(expr, var_names)
        args_str = ", ".join(var_names)

        source = (
            f"def {fn_name}({args_str}):\n"
            f"    return float({body})\n"
        )

        local_ns: dict[str, Any] = {"math": math}
        exec(source, local_ns)  # noqa: S102
        raw_fn = local_ns[fn_name]

        compiled: Any = numba.njit(cache=True)(raw_fn)

        # Warm up the JIT with dummy values to trigger compilation
        dummy_args = [0.1] * len(var_names)
        compiled(*dummy_args)

        return compiled  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug(
            "Numba compilation failed for {}: {}",
            fn_name,
            exc,
        )
        return None


def build_fast_dynamics_fn(
    dynamics_expressions: dict[int, Any],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
    angular_dims: tuple[int, ...] = (),
) -> Callable[[np.ndarray, np.ndarray], np.ndarray] | None:
    """Build a Numba-accelerated dynamics function.

    Compiles each per-dimension delta expression to a Numba scalar
    function. The returned callable has the same signature as
    ``_build_dynamics_fn`` but with significantly lower per-call
    overhead.

    :returns: Compiled dynamics function, or None if Numba is
        unavailable or compilation fails for any dimension.
    """
    import sympy

    var_names = list(state_names) + list(action_names)
    dim_fns: dict[int, Callable[..., float]] = {}

    for dim_idx, expr_obj in dynamics_expressions.items():
        sympy_expr = expr_obj.sympy_expr

        if env_params:
            subs = {sympy.Symbol(k): v for k, v in env_params.items()}
            sympy_expr = sympy_expr.subs(subs)

        # Simplify to reduce expression complexity
        sympy_expr = sympy.nsimplify(sympy_expr, rational=False)

        fn = compile_numba_scalar_fn(
            sympy_expr, var_names, fn_name=f"_dyn_{dim_idx}",
        )
        if fn is None:
            logger.debug(
                "Numba compilation failed for dim {}, "
                "falling back to numpy",
                dim_idx,
            )
            return None
        dim_fns[dim_idx] = fn

    def fast_dynamics_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        next_state = state.copy()
        # Build flat args: [s0, s1, ..., a0, a1, ...]
        args = [float(v) for v in state] + [float(v) for v in action]

        for dim_idx, fn in dim_fns.items():
            delta = fn(*args)
            next_state[dim_idx] += delta

        for d in angular_dims:
            next_state[d] = math.atan2(
                math.sin(next_state[d]),
                math.cos(next_state[d]),
            )
        return next_state

    return fast_dynamics_fn


def build_fast_jacobian_fns(
    dynamics_expressions: dict[int, Any],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
]:
    """Build Numba-accelerated Jacobian functions.

    Symbolically differentiates each dynamics expression, then
    compiles the partial derivatives to Numba scalar functions.

    :returns: Tuple of (jac_state_fn, jac_action_fn), or (None, None)
        if compilation fails.
    """
    import sympy

    var_names = list(state_names) + list(action_names)
    all_symbols = [sympy.Symbol(n) for n in var_names]
    state_symbols = [sympy.Symbol(n) for n in state_names]
    action_symbols = [sympy.Symbol(n) for n in action_names]

    a_fns: dict[tuple[int, int], Callable[..., float]] = {}
    b_fns: dict[tuple[int, int], Callable[..., float]] = {}

    for dim_idx, expr_obj in dynamics_expressions.items():
        sympy_expr = expr_obj.sympy_expr

        if env_params:
            subs = {sympy.Symbol(k): v for k, v in env_params.items()}
            sympy_expr = sympy_expr.subs(subs)

        # Replace Abs with smooth approximation for differentiability
        eps_smooth = 1e-8
        sympy_expr = sympy_expr.replace(
            lambda e: isinstance(e, sympy.Abs),
            lambda e: sympy.sqrt(e.args[0] ** 2 + eps_smooth),
        )

        for j, s_sym in enumerate(state_symbols):
            deriv = sympy.diff(sympy_expr, s_sym)
            deriv = sympy.nsimplify(deriv, rational=False)
            fn = compile_numba_scalar_fn(
                deriv, var_names,
                fn_name=f"_dAdx_{dim_idx}_{j}",
            )
            if fn is None:
                return None, None
            a_fns[(dim_idx, j)] = fn

        for j, a_sym in enumerate(action_symbols):
            deriv = sympy.diff(sympy_expr, a_sym)
            deriv = sympy.nsimplify(deriv, rational=False)
            fn = compile_numba_scalar_fn(
                deriv, var_names,
                fn_name=f"_dBdu_{dim_idx}_{j}",
            )
            if fn is None:
                return None, None
            b_fns[(dim_idx, j)] = fn

    action_dim = len(action_names)

    def fast_jac_state_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        jac = np.eye(state_dim)
        args = [float(v) for v in state] + [float(v) for v in action]
        for (di, j), fn in a_fns.items():
            jac[di, j] += fn(*args)
        return jac

    def fast_jac_action_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> np.ndarray:
        jac = np.zeros((state_dim, action_dim))
        args = [float(v) for v in state] + [float(v) for v in action]
        for (di, j), fn in b_fns.items():
            jac[di, j] = fn(*args)
        return jac

    return fast_jac_state_fn, fast_jac_action_fn


def build_fast_reward_fn(
    reward_expression: Any,
    state_names: list[str],
    action_names: list[str],
    env_params: dict[str, float] | None,
    derived_feature_specs: list[Any] | None = None,
    canonical_to_obs_fn: Callable[..., Any] | None = None,
    obs_state_names: list[str] | None = None,
) -> Callable[[np.ndarray, np.ndarray], float] | None:
    """Build a Numba-accelerated reward function.

    :returns: Compiled reward function, or None if compilation fails
        or the reward has complex derived features that resist
        Numba compilation.
    """
    import sympy

    # Reward with derived features or canonical mapping is too
    # complex for straightforward Numba compilation; fall back
    if derived_feature_specs or canonical_to_obs_fn:
        return None

    sympy_expr = reward_expression.sympy_expr

    if env_params:
        subs = {sympy.Symbol(k): v for k, v in env_params.items()}
        sympy_expr = sympy_expr.subs(subs)

    var_names = list(state_names) + list(action_names)
    fn = compile_numba_scalar_fn(sympy_expr, var_names, fn_name="_reward")
    if fn is None:
        return None

    def fast_reward_fn(
        state: np.ndarray, action: np.ndarray,
    ) -> float:
        args = [float(v) for v in state] + [float(v) for v in action]
        return fn(*args)

    return fast_reward_fn


# -- PyTorch compilation for GPU-batched iLQR --


def compile_torch_fn(
    expr: sympy.Expr,
    var_names: list[str],
) -> Callable[..., Any] | None:
    """Compile a sympy expression to a PyTorch-compatible callable.

    The generated function operates on torch tensors and supports
    batched evaluation (inputs can be any shape as long as the last
    dimension matches ``len(var_names)``).

    :param expr: Sympy expression with env params already substituted.
    :param var_names: Variable names in order.
    :returns: PyTorch-compatible callable, or None on failure.
    """
    try:
        import torch

        import sympy as sp

        # Build torch-compatible math module mapping
        torch_module = {
            "sin": torch.sin,
            "cos": torch.cos,
            "tan": torch.tan,
            "exp": torch.exp,
            "log": torch.log,
            "sqrt": torch.sqrt,
            "Abs": torch.abs,
            "abs": torch.abs,
            "sign": torch.sign,
            "asin": torch.asin,
            "acos": torch.acos,
            "atan": torch.atan,
            "atan2": torch.atan2,
        }

        symbols = [sp.Symbol(n) for n in var_names]
        fn = sp.lambdify(symbols, expr, modules=[torch_module, "torch"])
        return fn  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug(
            "PyTorch compilation failed: {}",
            exc,
        )
        return None


def build_torch_dynamics_fns(
    dynamics_expressions: dict[int, Any],
    state_names: list[str],
    action_names: list[str],
    state_dim: int,
    env_params: dict[str, float] | None,
) -> dict[int, Callable[..., Any]] | None:
    """Build per-dimension PyTorch-compiled dynamics delta functions.

    Each function takes individual tensor arguments (one per state/
    action variable) and returns a tensor of deltas. Suitable for
    batched evaluation in ``TorchILQRSolver``.

    :returns: Dict mapping dim_idx to torch callable, or None if
        any compilation fails.
    """
    import sympy

    var_names = list(state_names) + list(action_names)
    dim_fns: dict[int, Callable[..., Any]] = {}

    for dim_idx, expr_obj in dynamics_expressions.items():
        sympy_expr = expr_obj.sympy_expr

        if env_params:
            subs = {sympy.Symbol(k): v for k, v in env_params.items()}
            sympy_expr = sympy_expr.subs(subs)

        fn = compile_torch_fn(sympy_expr, var_names)
        if fn is None:
            return None
        dim_fns[dim_idx] = fn

    return dim_fns
