"""Classify dynamics hypotheses as linear or nonlinear.

Inspects sympy expressions to determine whether LQR (linear) or
MPC (nonlinear) is appropriate.

See ``CIRC-RL_Framework.md`` Section 3.6.
"""

from __future__ import annotations

from typing import Literal

import sympy
from loguru import logger


class HypothesisClassifier:
    """Classify a dynamics hypothesis to select the derivation method.

    Inspects the sympy expression to determine linearity in state
    and action variables.
    """

    @staticmethod
    def classify(
        expr: sympy.Expr,
        state_symbols: list[str],
        action_symbols: list[str],
    ) -> Literal["lqr", "mpc"]:
        """Classify an expression as linear or nonlinear.

        An expression is classified as linear (LQR-eligible) if it is
        a polynomial of degree <= 1 in all state and action variables
        combined.

        :param expr: The sympy expression.
        :param state_symbols: Names of state variables.
        :param action_symbols: Names of action variables.
        :returns: ``"lqr"`` if linear, ``"mpc"`` if nonlinear.
        """
        all_vars = [sympy.Symbol(s) for s in state_symbols + action_symbols]

        if is_linear_in(expr, all_vars):
            logger.debug(
                "Expression '{}' classified as LINEAR (LQR eligible)",
                expr,
            )
            return "lqr"

        logger.debug(
            "Expression '{}' classified as NONLINEAR (MPC required)",
            expr,
        )
        return "mpc"


def is_linear_in(
    expr: sympy.Expr,
    variables: list[sympy.Symbol],
) -> bool:
    """Check if an expression is linear (affine) in the given variables.

    An expression is linear if the total polynomial degree in the
    given variables collectively is at most 1. That is, no quadratic
    terms (x_i^2) and no cross-terms (x_i * x_j).

    :param expr: The sympy expression.
    :param variables: The variables to check linearity with respect to.
    :returns: True if linear (affine) in all variables.
    """
    expr = sympy.expand(expr)

    if not variables:
        return True

    try:
        poly = sympy.Poly(expr, *variables, domain="EX")
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        # Expression contains non-polynomial functions of the variables
        # (e.g., sin, cos) -> nonlinear
        return False

    return poly.total_degree() <= 1
