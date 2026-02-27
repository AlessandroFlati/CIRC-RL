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


# ---------------------------------------------------------------------------
# Cosine terrain templates (MountainCar family)
# ---------------------------------------------------------------------------

class TestCosineTerrainTemplates:
    """Test the cosine terrain velocity templates for MountainCar."""

    def test_match_mountaincar_discrete_variables(self):
        """With MountainCar-v0 variables, should match cosine_terrain_velocity."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "gravity", "force"])
        names = [t.name for t in matched]
        assert "cosine_terrain_velocity" in names

    def test_match_mountaincar_continuous_variables(self):
        """With MCContinuous variables, should match powered_cosine_terrain."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "power"])
        names = [t.name for t in matched]
        assert "powered_cosine_terrain" in names

    def test_match_position_integration(self):
        """position_integration should match when s1 is available."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1"])
        names = [t.name for t in matched]
        assert "position_integration" in names

    def test_fit_mountaincar_discrete_dynamics(self):
        """Fit cosine terrain velocity to MountainCar-v0 dynamics.

        True dynamics: velocity += (action - 1) * force - cos(3 * pos) * gravity
        with force=0.001, gravity=0.0025.
        """
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "gravity", "force"])
        template = next(t for t in matched if t.name == "cosine_terrain_velocity")

        rng = np.random.default_rng(42)
        n = 2000
        position = rng.uniform(-1.2, 0.6, n)
        velocity = rng.uniform(-0.07, 0.07, n)
        action = rng.choice([0.0, 1.0, 2.0], n)
        gravity = np.full(n, 0.0025)
        force = np.full(n, 0.001)

        y = (action - 1) * force - np.cos(3 * position) * gravity

        x = np.column_stack([position, velocity, action, gravity, force])
        var_names = ["s0", "s1", "action", "gravity", "force"]

        expr, r2 = lib.fit(template, x, y, var_names)
        assert r2 > 0.999, f"R2={r2:.6f} too low for exact MountainCar dynamics"

    def test_fit_mountaincar_continuous_dynamics(self):
        """Fit powered cosine terrain to MountainCarContinuous-v0 dynamics.

        True dynamics: velocity += action * power - 0.0025 * cos(3 * pos)
        with power=0.0015.
        """
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "power"])
        template = next(t for t in matched if t.name == "powered_cosine_terrain")

        rng = np.random.default_rng(42)
        n = 2000
        position = rng.uniform(-1.2, 0.6, n)
        velocity = rng.uniform(-0.07, 0.07, n)
        action = rng.uniform(-1, 1, n)
        power = np.full(n, 0.0015)

        y = action * power - 0.0025 * np.cos(3 * position)

        x = np.column_stack([position, velocity, action, power])
        var_names = ["s0", "s1", "action", "power"]

        expr, r2 = lib.fit(template, x, y, var_names)
        assert r2 > 0.999, f"R2={r2:.6f} too low for exact MCCont dynamics"

    def test_fit_position_integration(self):
        """Fit position_integration to delta_pos = velocity data."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1"])
        template = next(t for t in matched if t.name == "position_integration")

        rng = np.random.default_rng(42)
        n = 1000
        position = rng.uniform(-1.2, 0.6, n)
        velocity = rng.uniform(-0.07, 0.07, n)

        y = velocity  # delta_pos = velocity

        x = np.column_stack([position, velocity])
        var_names = ["s0", "s1"]

        expr, r2 = lib.fit(template, x, y, var_names)
        assert r2 > 0.999, f"R2={r2:.6f} too low for velocity integration"

    def test_fit_mountaincar_with_varied_params(self):
        """Fit cosine terrain with varied gravity/force across envs."""
        lib = PhysicsTemplateLibrary()
        matched = lib.match(["s0", "s1", "action", "gravity", "force"])
        template = next(t for t in matched if t.name == "cosine_terrain_velocity")

        rng = np.random.default_rng(42)
        n = 3000
        position = rng.uniform(-1.2, 0.6, n)
        velocity = rng.uniform(-0.07, 0.07, n)
        action = rng.choice([0.0, 1.0, 2.0], n)
        # Varied env params
        gravity = rng.uniform(0.001, 0.005, n)
        force = rng.uniform(0.0005, 0.002, n)

        y = (action - 1) * force - np.cos(3 * position) * gravity

        x = np.column_stack([position, velocity, action, gravity, force])
        var_names = ["s0", "s1", "action", "gravity", "force"]

        expr, r2 = lib.fit(template, x, y, var_names)
        assert r2 > 0.999, f"R2={r2:.6f} too low for varied-param MountainCar"

    def test_identifier_finds_cosine_terrain(self):
        """TemplateBasedIdentifier should find cosine terrain for MC data."""
        rng = np.random.default_rng(42)
        n = 2000
        position = rng.uniform(-1.2, 0.6, n)
        velocity = rng.uniform(-0.07, 0.07, n)
        action = rng.choice([0.0, 1.0, 2.0], n)
        gravity = np.full(n, 0.0025)
        force = np.full(n, 0.001)

        y = (action - 1) * force - np.cos(3 * position) * gravity

        x = np.column_stack([position, velocity, action, gravity, force])
        var_names = ["s0", "s1", "action", "gravity", "force"]

        identifier = TemplateBasedIdentifier(min_r2=0.99)
        results = identifier.identify(x, y, var_names)
        assert len(results) > 0, "Should find at least one template match"
        assert results[0][1] > 0.999
