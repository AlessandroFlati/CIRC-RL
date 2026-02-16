"""PySR-based symbolic regression wrapper for hypothesis generation.

Wraps PySR to discover analytic expressions from numerical data,
producing Pareto fronts of candidate hypotheses ordered by
complexity and accuracy.

See ``CIRC-RL_Framework.md`` Section 3.4 (Phase 3: Structural
Hypothesis Generation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from circ_rl.hypothesis.expression import SymbolicExpression

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class SymbolicRegressionConfig:
    """Configuration for symbolic regression.

    :param max_complexity: Maximum expression tree complexity allowed.
    :param n_iterations: Number of PySR evolutionary iterations.
    :param populations: Number of evolutionary populations.
    :param binary_operators: Allowed binary operators.
    :param unary_operators: Allowed unary operators.
    :param parsimony: Parsimony coefficient (penalty per complexity unit).
    :param timeout_seconds: Maximum wall-clock time for the search.
    :param deterministic: If True, use fixed random state for reproducibility.
    :param random_state: Random seed (used only when deterministic=True).
    """

    max_complexity: int = 30
    n_iterations: int = 40
    populations: int = 15
    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("sin", "cos", "sqrt", "abs")
    parsimony: float = 0.0032
    timeout_seconds: int = 300
    deterministic: bool = True
    random_state: int = 42


class SymbolicRegressor:
    """PySR wrapper for discovering analytic expressions.

    Fits symbolic regression models using PySR and returns a list
    of candidate expressions forming a Pareto front of accuracy
    vs. complexity.

    See ``CIRC-RL_Framework.md`` Section 3.4.

    :param config: Symbolic regression configuration.
    """

    def __init__(self, config: SymbolicRegressionConfig | None = None) -> None:
        self._config = config or SymbolicRegressionConfig()

    @property
    def config(self) -> SymbolicRegressionConfig:
        """The regression configuration."""
        return self._config

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        variable_names: list[str],
    ) -> list[SymbolicExpression]:
        """Run symbolic regression to discover analytic expressions.

        :param x: Input data of shape ``(n_samples, n_features)``.
        :param y: Target values of shape ``(n_samples,)``.
        :param variable_names: Names of input features (columns of x).
        :returns: List of SymbolicExpressions forming the Pareto front,
            sorted by complexity (ascending).
        :raises ImportError: If PySR is not installed.
        :raises ValueError: If input dimensions are inconsistent.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x has {x.shape[0]} samples but y has {y.shape[0]}"
            )
        if x.shape[1] != len(variable_names):
            raise ValueError(
                f"x has {x.shape[1]} features but {len(variable_names)} "
                f"variable names provided"
            )

        try:
            from pysr import PySRRegressor
        except ImportError as exc:
            raise ImportError(
                "PySR is required for symbolic regression. "
                "Install with: pip install 'circ-rl[symbolic]'"
            ) from exc

        import sympy

        cfg = self._config

        logger.info(
            "Starting symbolic regression: {} samples, {} features, "
            "max_complexity={}, n_iterations={}",
            x.shape[0],
            x.shape[1],
            cfg.max_complexity,
            cfg.n_iterations,
        )

        model = PySRRegressor(
            niterations=cfg.n_iterations,
            maxsize=cfg.max_complexity,
            populations=cfg.populations,
            binary_operators=list(cfg.binary_operators),
            unary_operators=list(cfg.unary_operators),
            parsimony=cfg.parsimony,
            timeout_in_seconds=cfg.timeout_seconds,
            deterministic=cfg.deterministic,
            random_state=cfg.random_state if cfg.deterministic else None,
            procs=0 if cfg.deterministic else 1,
            multithreading=not cfg.deterministic,
            verbosity=0,
        )

        model.fit(x, y, variable_names=variable_names)

        # Extract Pareto front from PySR results
        expressions: list[SymbolicExpression] = []
        equations = model.equations_

        if equations is None or len(equations) == 0:
            logger.warning("Symbolic regression produced no equations")
            return expressions

        for _, row in equations.iterrows():
            sympy_expr = row["sympy_format"]
            if not isinstance(sympy_expr, sympy.Basic):
                continue

            expr = SymbolicExpression.from_sympy(sympy.sympify(sympy_expr))
            expressions.append(expr)

        # Sort by complexity ascending
        expressions.sort(key=lambda e: e.complexity)

        # Deduplicate by expression string
        seen: set[str] = set()
        unique: list[SymbolicExpression] = []
        for expr in expressions:
            if expr.expression_str not in seen:
                seen.add(expr.expression_str)
                unique.append(expr)

        logger.info(
            "Symbolic regression complete: {} unique expressions on Pareto front",
            len(unique),
        )

        return unique
