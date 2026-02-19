# ruff: noqa: ANN001 ANN201

"""Unit tests for the physics template library."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from circ_rl.hypothesis.physics_templates import (
    PhysicsTemplate,
    PhysicsTemplateLibrary,
    TemplateBasedIdentifier,
)


# ---------------------------------------------------------------------------
# PhysicsTemplateLibrary
# ---------------------------------------------------------------------------

class TestPhysicsTemplateLibrary:
    """Test the template library and matching."""

    def test_default_library_has_templates(self):
        lib = PhysicsTemplateLibrary()
        assert len(lib.templates) > 0

    def test_match_pendulum_variables(self):
        """With pendulum variables, should match gravity_pendulum."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["phi_0", "s2", "action", "g", "m", "l"])
        names = [t.name for t in matched]
        assert "gravity_pendulum" in names
        assert "damped_pendulum" in names

    def test_match_spring_variables(self):
        """With spring-mass variables, should match spring templates."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "m"])
        names = [t.name for t in matched]
        assert "spring_mass" in names
        assert "spring_mass_damper" in names

    def test_match_returns_empty_for_unknown_vars(self):
        """With unrecognized variables, should return empty."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["xyz", "abc"])
        assert matched == []

    def test_match_sorted_by_specificity(self):
        """More specific templates (more required vars) should come first."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["phi_0", "s2", "action", "g", "m", "l"])
        if len(matched) >= 2:
            assert len(matched[0].required_variables) >= len(
                matched[1].required_variables
            )

    def test_add_custom_template(self):
        """Custom templates should be findable after addition."""
        lib = PhysicsTemplateLibrary()
        c1 = sp.Symbol("c1")
        custom = PhysicsTemplate(
            name="custom_test",
            expression=c1 * sp.Symbol("q"),
            free_coefficients=[c1],
            required_variables=frozenset({"q"}),
        )
        lib.add_template(custom)
        matched = lib.match(["q", "r"])
        names = [t.name for t in matched]
        assert "custom_test" in names


# ---------------------------------------------------------------------------
# Template fitting
# ---------------------------------------------------------------------------

class TestTemplateFitting:
    """Test coefficient fitting on synthetic data."""

    def test_fit_linear_system(self):
        """Fit c1*x to data y = 3*x."""
        lib = PhysicsTemplateLibrary()
        c1 = sp.Symbol("c1")
        template = PhysicsTemplate(
            name="test_linear",
            expression=c1 * sp.Symbol("s0"),
            free_coefficients=[c1],
            required_variables=frozenset({"s0"}),
            initial_guesses=(1.0,),
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 1))
        y = 3.0 * x[:, 0]

        expr, r2 = lib.fit(template, x, y, ["s0"])
        assert r2 > 0.999
        assert "3.0" in str(expr.sympy_expr) or abs(
            float(expr.sympy_expr.subs(sp.Symbol("s0"), 1.0)) - 3.0
        ) < 0.01

    def test_fit_pendulum_dynamics(self):
        """Fit c1*g*sin(phi)/l + c2*action/(m*l^2) to pendulum data."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["phi_0", "s2", "action", "g", "m", "l"])
        pendulum_template = next(
            t for t in matched if t.name == "gravity_pendulum"
        )

        rng = np.random.default_rng(42)
        n = 2000

        # Generate pendulum-like data with known coefficients
        phi = rng.uniform(-np.pi, np.pi, n)
        s2 = rng.uniform(-5, 5, n)
        action = rng.uniform(-2, 2, n)
        g = rng.uniform(8, 12, n)
        m = rng.uniform(0.5, 2, n)
        l = rng.uniform(0.5, 1.5, n)

        # True dynamics: 0.075 * g * sin(phi) / l + 0.15 * action / (m * l^2)
        y = 0.075 * g * np.sin(phi) / l + 0.15 * action / (m * l ** 2)

        x = np.column_stack([phi, s2, action, g, m, l])
        var_names = ["phi_0", "s2", "action", "g", "m", "l"]

        expr, r2 = lib.fit(pendulum_template, x, y, var_names)
        assert r2 > 0.999

    def test_fit_bad_template_has_low_r2(self):
        """A mismatched template should have low R2 even if fitting succeeds."""
        lib = PhysicsTemplateLibrary()
        c1 = sp.Symbol("c1")
        template = PhysicsTemplate(
            name="test_bad_fit",
            expression=c1 * sp.Symbol("s0"),
            free_coefficients=[c1],
            required_variables=frozenset({"s0"}),
            initial_guesses=(1.0,),
        )

        # Data: y = sin(x), template: y = c1*x -> poor fit
        rng = np.random.default_rng(42)
        x = rng.uniform(-3, 3, size=(500, 1))
        y = np.sin(x[:, 0]) ** 3 + 0.5 * np.cos(x[:, 0] * 3)

        expr, r2 = lib.fit(template, x, y, ["s0"])
        # Linear model should not fit this well
        assert r2 < 0.5


# ---------------------------------------------------------------------------
# TemplateBasedIdentifier
# ---------------------------------------------------------------------------

class TestTemplateBasedIdentifier:
    """Test the end-to-end template identifier."""

    def test_identifies_simple_linear(self):
        """Should identify y = c1*x when data is perfectly linear."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 1))
        y = -2.5 * x[:, 0]

        identifier = TemplateBasedIdentifier(min_r2=0.99)
        results = identifier.identify(x, y, ["s0"])

        assert len(results) > 0
        best_expr, best_r2 = results[0]
        assert best_r2 > 0.99

    def test_returns_empty_for_noise(self):
        """Pure noise should not match any template."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 1))
        y = rng.normal(0, 1, 500)

        identifier = TemplateBasedIdentifier(min_r2=0.99)
        results = identifier.identify(x, y, ["s0"])

        # No template should fit noise with R2 > 0.99
        assert len(results) == 0

    def test_returns_empty_for_unknown_vars(self):
        """With no matching variables, should return empty."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 1))
        y = 3.0 * x[:, 0]

        identifier = TemplateBasedIdentifier(min_r2=0.99)
        results = identifier.identify(x, y, ["zzz"])

        assert len(results) == 0

    def test_results_sorted_by_r2(self):
        """Results should be sorted by R2 descending."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 2))
        y = -2.5 * x[:, 0]  # only depends on s0

        identifier = TemplateBasedIdentifier(min_r2=0.9)
        results = identifier.identify(x, y, ["s0", "s1"])

        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_min_r2_override(self):
        """The min_r2 parameter in identify() should override default."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, size=(500, 1))
        y = -2.5 * x[:, 0] + rng.normal(0, 0.1, 500)

        identifier = TemplateBasedIdentifier(min_r2=1.0)  # impossibly high

        # With override, should find results
        results = identifier.identify(x, y, ["s0"], min_r2=0.9)
        assert len(results) > 0

        # Without override (uses default 1.0), should find none
        results_strict = identifier.identify(x, y, ["s0"])
        assert len(results_strict) == 0
