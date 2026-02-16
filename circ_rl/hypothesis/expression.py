"""Symbolic expression representation for hypothesis generation.

Wraps sympy expressions with complexity scoring and callable evaluation.

See ``CIRC-RL_Framework.md`` Section 2.3 (Symbolic Complexity) and
Section 3.4.3 (The Hypothesis Register).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sympy

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class SymbolicExpression:
    r"""A symbolic mathematical expression with complexity metadata.

    Wraps a sympy expression with its string representation, complexity
    score, and a compiled callable for fast numeric evaluation.

    The symbolic complexity is defined as the number of nodes in the
    expression tree:

    .. math::

        C_{\text{sym}}(h) = |h|_{\text{nodes}}

    See ``CIRC-RL_Framework.md`` Section 2.3, Definition 2.4.

    :param expression_str: Human-readable string form of the expression.
    :param sympy_expr: The sympy expression object.
    :param complexity: Number of nodes in the expression tree
        (:math:`C_{\text{sym}}`).
    :param free_symbols: Names of all free symbolic variables.
    :param n_constants: Number of fitted numeric constants in the expression.
    """

    expression_str: str
    sympy_expr: sympy.Expr
    complexity: int
    free_symbols: frozenset[str]
    n_constants: int

    @staticmethod
    def count_tree_nodes(expr: sympy.Expr) -> int:
        """Count the number of nodes in a sympy expression tree.

        Each symbol, number, and operator counts as one node.

        :param expr: A sympy expression.
        :returns: The number of nodes in the expression tree.
        """
        count = 1  # Count the current node
        for arg in expr.args:
            count += SymbolicExpression.count_tree_nodes(arg)
        return count

    @staticmethod
    def from_sympy(
        expr: sympy.Expr,
        constant_symbols: frozenset[str] | None = None,
    ) -> SymbolicExpression:
        """Create a SymbolicExpression from a sympy expression.

        :param expr: The sympy expression.
        :param constant_symbols: Names of symbols that represent fitted
            numeric constants (not input variables). If None, all
            ``sympy.Number`` nodes are counted as constants.
        :returns: A new SymbolicExpression.
        """
        free = frozenset(str(s) for s in expr.free_symbols)
        complexity = SymbolicExpression.count_tree_nodes(expr)

        if constant_symbols is not None:
            n_constants = len(constant_symbols & free)
        else:
            # Count numeric atoms (fitted constants embedded in expression)
            n_constants = sum(
                1 for atom in expr.atoms() if isinstance(atom, sympy.Number)
            )

        return SymbolicExpression(
            expression_str=str(expr),
            sympy_expr=expr,
            complexity=complexity,
            free_symbols=free,
            n_constants=n_constants,
        )

    def to_callable(
        self,
        variable_names: list[str],
    ) -> Callable[..., np.ndarray]:
        """Compile the expression into a fast numpy-callable function.

        :param variable_names: Ordered list of variable names matching
            the columns of the input array. Symbols in the expression
            that are not in this list are treated as missing (will raise
            at call time if the expression depends on them).
        :returns: A callable ``f(X) -> y`` where ``X`` has shape
            ``(n_samples, len(variable_names))`` and ``y`` has shape
            ``(n_samples,)``.
        :raises ValueError: If any free symbol in the expression is not
            found in variable_names.
        """
        sym_vars = [sympy.Symbol(name) for name in variable_names]

        # Validate that all free symbols are provided
        missing = self.free_symbols - frozenset(variable_names)
        if missing:
            raise ValueError(
                f"Expression has free symbols {missing} not found in "
                f"variable_names {variable_names}"
            )

        func = sympy.lambdify(
            sym_vars,
            self.sympy_expr,
            modules=["numpy"],
        )

        def evaluate(x: np.ndarray) -> np.ndarray:
            """Evaluate the expression on input data.

            :param x: Input array of shape ``(n_samples, n_variables)``.
            :returns: Output array of shape ``(n_samples,)``.
            """
            assert x.shape[1] == len(variable_names), (
                f"Expected {len(variable_names)} columns, got {x.shape[1]}"
            )
            cols: list[Any] = [x[:, i] for i in range(x.shape[1])]
            result = func(*cols)
            return np.asarray(result, dtype=np.float64).ravel()

        return evaluate

    def __repr__(self) -> str:
        return (
            f"SymbolicExpression({self.expression_str!r}, "
            f"complexity={self.complexity})"
        )
