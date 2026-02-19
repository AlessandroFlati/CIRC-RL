"""Physics-informed template library for fast system identification.

Provides a library of known physics templates (rigid-body dynamics,
spring-mass-damper, gravitational pendulum, etc.) with coefficient
fitting via least-squares. When a template matches the problem
structure, this is orders of magnitude faster than full symbolic
regression.

Usage as a first-pass filter before PySR::

    identifier = TemplateBasedIdentifier()
    results = identifier.identify(x, y, var_names, min_r2=0.99)
    if results:
        # Skip PySR -- template matched with high accuracy
        best_expr, best_r2 = results[0]
    else:
        # Fall through to PySR
        ...

See ``CIRC-RL_Framework.md`` Section 3.4.1 (Dynamics Hypotheses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from loguru import logger

from circ_rl.hypothesis.expression import SymbolicExpression

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PhysicsTemplate:
    """A parametric physics template with free coefficients.

    :param name: Human-readable template name.
    :param expression: Sympy expression with free coefficient symbols.
    :param free_coefficients: List of sympy symbols to be fitted.
    :param required_variables: Set of variable names that must be present
        in the dataset for this template to be applicable.
    :param domain: Physics domain label (e.g., ``"mechanics"``,
        ``"electrodynamics"``).
    :param initial_guesses: Starting point for coefficient fitting.
        Defaults to ones if not provided.
    """

    name: str
    expression: sp.Expr
    free_coefficients: list[sp.Symbol]
    required_variables: frozenset[str]
    domain: str = "mechanics"
    initial_guesses: tuple[float, ...] | None = None


class PhysicsTemplateLibrary:
    """Library of known physics templates for system identification.

    Built-in templates cover common dynamics patterns:

    - **rigid_body_rotation**: ``c1 * sin(theta) / l + c2 * action / (m * l^2)``
    - **gravity_pendulum**: ``c1 * g * sin(theta) / l + c2 * action / (m * l^2)``
    - **damped_pendulum**: ``c1 * g * sin(theta) / l + c2 * action / (m * l^2)
      + c3 * omega`` (with velocity damping)
    - **spring_mass**: ``c1 * x + c2 * action / m``
    - **spring_mass_damper**: ``c1 * x + c2 * v + c3 * action / m``
    - **linear_drag**: ``c1 * v + c2 * action / m``
    - **simple_harmonic**: ``c1 * x``

    Custom templates can be added via ``add_template()``.
    """

    def __init__(self) -> None:
        self._templates: list[PhysicsTemplate] = []
        self._build_defaults()

    def _build_defaults(self) -> None:
        """Populate the library with common physics templates."""
        # Free coefficient symbols
        c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4")

        # Variable symbols
        theta = sp.Symbol("phi_0")  # canonical angular coordinate
        omega = sp.Symbol("s2")  # angular velocity (pendulum)
        g_sym = sp.Symbol("g")
        m_sym = sp.Symbol("m")
        l_sym = sp.Symbol("l")
        action = sp.Symbol("action")

        # --- Pendulum templates ---

        # Gravity pendulum with torque (parametric)
        self._templates.append(PhysicsTemplate(
            name="gravity_pendulum",
            expression=c1 * g_sym * sp.sin(theta) / l_sym + c2 * action / (m_sym * l_sym ** 2),
            free_coefficients=[c1, c2],
            required_variables=frozenset({"phi_0", "action", "g", "m", "l"}),
            domain="mechanics",
            initial_guesses=(0.075, 0.15),
        ))

        # Gravity pendulum with velocity damping
        self._templates.append(PhysicsTemplate(
            name="damped_pendulum",
            expression=(
                c1 * g_sym * sp.sin(theta) / l_sym
                + c2 * action / (m_sym * l_sym ** 2)
                + c3 * omega
            ),
            free_coefficients=[c1, c2, c3],
            required_variables=frozenset({"phi_0", "s2", "action", "g", "m", "l"}),
            domain="mechanics",
            initial_guesses=(0.075, 0.15, -0.01),
        ))

        # Rigid body rotation (no gravity parameter needed)
        s1 = sp.Symbol("s1")
        self._templates.append(PhysicsTemplate(
            name="rigid_body_rotation",
            expression=c1 * s1 / l_sym + c2 * action / (m_sym * l_sym ** 2),
            free_coefficients=[c1, c2],
            required_variables=frozenset({"s1", "action", "m", "l"}),
            domain="mechanics",
            initial_guesses=(0.075, 0.15),
        ))

        # --- Spring-mass templates ---

        x_sym = sp.Symbol("s0")
        v_sym = sp.Symbol("s1")

        self._templates.append(PhysicsTemplate(
            name="spring_mass",
            expression=c1 * x_sym + c2 * action / m_sym,
            free_coefficients=[c1, c2],
            required_variables=frozenset({"s0", "action", "m"}),
            domain="mechanics",
            initial_guesses=(-1.0, 1.0),
        ))

        self._templates.append(PhysicsTemplate(
            name="spring_mass_damper",
            expression=c1 * x_sym + c2 * v_sym + c3 * action / m_sym,
            free_coefficients=[c1, c2, c3],
            required_variables=frozenset({"s0", "s1", "action", "m"}),
            domain="mechanics",
            initial_guesses=(-1.0, -0.1, 1.0),
        ))

        # --- Generic templates ---

        self._templates.append(PhysicsTemplate(
            name="linear_drag",
            expression=c1 * v_sym + c2 * action / m_sym,
            free_coefficients=[c1, c2],
            required_variables=frozenset({"s1", "action", "m"}),
            domain="mechanics",
            initial_guesses=(-0.1, 1.0),
        ))

        self._templates.append(PhysicsTemplate(
            name="simple_harmonic",
            expression=c1 * x_sym,
            free_coefficients=[c1],
            required_variables=frozenset({"s0"}),
            domain="mechanics",
            initial_guesses=(-1.0,),
        ))

        # Linear velocity (identity-ish dynamics for velocity dim)
        self._templates.append(PhysicsTemplate(
            name="linear_velocity",
            expression=c1 * omega,
            free_coefficients=[c1],
            required_variables=frozenset({"s2"}),
            domain="mechanics",
            initial_guesses=(0.05,),
        ))

    @property
    def templates(self) -> list[PhysicsTemplate]:
        """All registered templates."""
        return list(self._templates)

    def add_template(self, template: PhysicsTemplate) -> None:
        """Register a custom physics template."""
        self._templates.append(template)

    def match(
        self,
        variable_names: list[str],
    ) -> list[PhysicsTemplate]:
        """Find templates whose required variables are a subset of available ones.

        :param variable_names: Available variable names in the dataset.
        :returns: List of matching templates, sorted by specificity
            (most required variables first).
        """
        available = frozenset(variable_names)
        matched = [
            t for t in self._templates
            if t.required_variables <= available
        ]
        # Sort by specificity (more required vars = more specific = prefer)
        matched.sort(key=lambda t: len(t.required_variables), reverse=True)
        return matched

    def fit(
        self,
        template: PhysicsTemplate,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
    ) -> tuple[SymbolicExpression, float]:
        """Fit a template's coefficients to data via least-squares.

        Uses ``scipy.optimize.curve_fit`` for nonlinear least-squares
        fitting of the free coefficients.

        :param template: The physics template to fit.
        :param x: Input data, shape ``(n_samples, n_features)``.
        :param y: Target values, shape ``(n_samples,)``.
        :param var_names: Variable names for columns of ``x``.
        :returns: Tuple of ``(fitted_expression, r_squared)``.
        :raises RuntimeError: If fitting fails.
        """
        from scipy.optimize import curve_fit

        # Build a numpy callable from the template expression
        all_symbols = [sp.Symbol(n) for n in var_names]
        coeff_symbols = template.free_coefficients

        # lambdify: f(vars..., coeffs...) -> value
        func_symbols = all_symbols + coeff_symbols
        func = sp.lambdify(func_symbols, template.expression, modules=["numpy"])

        def model(x_data: np.ndarray, *coeffs: float) -> np.ndarray:
            """Model function for curve_fit: x_data (N, F) -> y_pred (N,)."""
            args = [x_data[:, i] for i in range(x_data.shape[1])]
            args.extend(coeffs)
            return np.asarray(func(*args), dtype=np.float64)

        # Initial guesses
        n_coeffs = len(coeff_symbols)
        if template.initial_guesses is not None:
            p0 = list(template.initial_guesses)
        else:
            p0 = [1.0] * n_coeffs

        try:
            popt, _pcov = curve_fit(
                model, x, y, p0=p0, maxfev=10000,
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Template '{template.name}' fitting failed: {exc}"
            ) from exc

        # Substitute fitted coefficients back into the expression
        subs = {
            coeff: float(val)
            for coeff, val in zip(coeff_symbols, popt)
        }
        fitted_expr = template.expression.subs(subs)

        # Compute R2
        y_pred = model(x, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        symbolic_expr = SymbolicExpression.from_sympy(sp.sympify(fitted_expr))

        logger.info(
            "Template '{}': R2={:.6f}, coefficients={}",
            template.name,
            r2,
            {str(k): f"{v:.6f}" for k, v in subs.items()},
        )

        return symbolic_expr, r2


class TemplateBasedIdentifier:
    """Fast system identification via physics template matching.

    Tries all matching templates against the data. Returns templates
    that exceed a minimum R2 threshold, sorted by fit quality.

    :param library: Template library to use. If None, uses the
        default built-in library.
    :param min_r2: Minimum R2 threshold for a template to be accepted.
    """

    def __init__(
        self,
        library: PhysicsTemplateLibrary | None = None,
        min_r2: float = 0.99,
    ) -> None:
        self._library = library or PhysicsTemplateLibrary()
        self._min_r2 = min_r2

    def identify(
        self,
        x: np.ndarray,
        y: np.ndarray,
        var_names: list[str],
        min_r2: float | None = None,
    ) -> list[tuple[SymbolicExpression, float]]:
        """Try all matching templates and return those exceeding min_r2.

        :param x: Input data, shape ``(n_samples, n_features)``.
        :param y: Target values, shape ``(n_samples,)``.
        :param var_names: Variable names.
        :param min_r2: Override the default min_r2 threshold.
        :returns: List of ``(expression, r2)`` tuples, sorted by R2
            descending. Empty if no template matches above threshold.
        """
        threshold = min_r2 if min_r2 is not None else self._min_r2
        matched_templates = self._library.match(var_names)

        if not matched_templates:
            logger.debug("No templates match variables: {}", var_names)
            return []

        logger.info(
            "Trying {} templates against {} variables",
            len(matched_templates),
            len(var_names),
        )

        results: list[tuple[SymbolicExpression, float]] = []

        for template in matched_templates:
            try:
                expr, r2 = self._library.fit(template, x, y, var_names)
                if r2 >= threshold:
                    results.append((expr, r2))
                    logger.info(
                        "Template '{}' accepted: R2={:.6f} >= {:.6f}",
                        template.name, r2, threshold,
                    )
                else:
                    logger.debug(
                        "Template '{}' rejected: R2={:.6f} < {:.6f}",
                        template.name, r2, threshold,
                    )
            except RuntimeError as exc:
                logger.debug(
                    "Template '{}' fitting failed: {}",
                    template.name, exc,
                )

        # Sort by R2 descending
        results.sort(key=lambda r: r[1], reverse=True)

        return results
