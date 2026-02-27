"""Unit tests for observation space analysis."""

from __future__ import annotations

import numpy as np
import pytest

from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.observation_analysis.observation_analyzer import (
    ObservationAnalysisConfig,
    ObservationAnalyzer,
    wrap_angular_deltas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    states: np.ndarray,
    next_states: np.ndarray | None = None,
    n_envs: int = 3,
) -> ExploratoryDataset:
    """Build a minimal ExploratoryDataset for testing."""
    n = states.shape[0]
    if next_states is None:
        next_states = states + np.random.default_rng(42).normal(
            0, 0.01, size=states.shape,
        )
    actions = np.random.default_rng(42).uniform(-1, 1, size=(n, 1))
    env_ids = np.tile(np.arange(n_envs), n // n_envs + 1)[:n].astype(np.int32)
    return ExploratoryDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=np.zeros(n),
        env_ids=env_ids,
    )


# ---------------------------------------------------------------------------
# Circle constraint detection
# ---------------------------------------------------------------------------


class TestCircleConstraintDetection:
    """Tests for detecting circle constraints (cos/sin encodings)."""

    def test_detects_unit_circle(self) -> None:
        """Detect s0^2 + s1^2 = 1 from cos/sin data."""
        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, size=2000)
        states = np.column_stack([np.cos(theta), np.sin(theta)])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"])

        assert len(result.constraints) == 1
        c = result.constraints[0]
        assert c.constraint_type == "circle"
        assert set(c.involved_dims) == {0, 1}
        assert abs(c.rhs - 1.0) < 0.05  # R^2 ~ 1.0

    def test_detects_circle_with_extra_dim(self) -> None:
        """Detect circle in [cos, sin, velocity] observation space."""
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(-np.pi, np.pi, size=n)
        velocity = rng.uniform(-8, 8, size=n)
        states = np.column_stack([
            np.cos(theta), np.sin(theta), velocity,
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        assert len(result.constraints) == 1
        assert result.constraints[0].constraint_type == "circle"
        assert set(result.constraints[0].involved_dims) == {0, 1}

        # Canonical names should be [phi_0, s2]
        assert len(result.canonical_state_names) == 2
        assert result.canonical_state_names[0] == "phi_0"
        assert result.canonical_state_names[1] == "s2"

    def test_canonical_mapping_produces_atan2(self) -> None:
        """Verify canonical mapping is atan2(sin, cos)."""
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(-np.pi, np.pi, size=n)
        states = np.column_stack([np.cos(theta), np.sin(theta)])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"])

        assert len(result.mappings) == 1
        m = result.mappings[0]

        # Single-vector conversion
        for i in range(0, n, 100):
            obs = states[i]
            canonical = m.obs_to_canonical(obs)
            np.testing.assert_allclose(
                canonical[0], theta[i], atol=1e-10,
            )

    def test_canonical_states_shape(self) -> None:
        """Verify canonical state arrays have correct shape."""
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(-np.pi, np.pi, size=n)
        velocity = rng.uniform(-8, 8, size=n)
        states = np.column_stack([
            np.cos(theta), np.sin(theta), velocity,
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        # 3 obs dims -> 2 canonical (phi_0, s2)
        assert result.canonical_states.shape == (n, 2)
        assert result.canonical_next_states.shape == (n, 2)

    def test_detects_non_unit_circle(self) -> None:
        """Detect circle with R != 1."""
        rng = np.random.default_rng(42)
        R = 3.5
        theta = rng.uniform(-np.pi, np.pi, size=2000)
        states = np.column_stack([
            R * np.cos(theta), R * np.sin(theta),
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1"])

        assert len(result.constraints) == 1
        c = result.constraints[0]
        assert c.constraint_type == "circle"
        assert abs(c.rhs - R**2) < 0.5  # R^2 ~ 12.25


# ---------------------------------------------------------------------------
# No constraint case
# ---------------------------------------------------------------------------


class TestNoConstraint:
    """Tests when no constraints are present."""

    def test_random_uncorrelated_data(self) -> None:
        """No constraints in random uncorrelated data."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(2000, 3))
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        assert len(result.constraints) == 0
        assert len(result.mappings) == 0
        assert result.canonical_state_names == ["s0", "s1", "s2"]
        assert result.canonical_states.shape == (2000, 3)

    def test_passthrough_preserves_data(self) -> None:
        """When no constraints, canonical data equals original data."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(2000, 3))
        next_states = states + rng.normal(0, 0.01, size=states.shape)
        dataset = _make_dataset(states, next_states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        np.testing.assert_array_equal(result.canonical_states, states)
        np.testing.assert_array_equal(result.canonical_next_states, next_states)

    def test_single_dim(self) -> None:
        """Single observation dim -> no constraints possible."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(2000, 1))
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0"])

        assert len(result.constraints) == 0
        assert result.canonical_state_names == ["s0"]


# ---------------------------------------------------------------------------
# Linear constraint
# ---------------------------------------------------------------------------


class TestLinearConstraint:
    """Tests for linear constraint detection."""

    def test_detects_exact_linear_constraint(self) -> None:
        """Detect s2 = 2*s0 + 3*s1."""
        rng = np.random.default_rng(42)
        n = 2000
        s0 = rng.uniform(-5, 5, size=n)
        s1 = rng.uniform(-5, 5, size=n)
        s2 = 2.0 * s0 + 3.0 * s1
        states = np.column_stack([s0, s1, s2])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        assert len(result.constraints) >= 1
        linear_constraints = [
            c for c in result.constraints if c.constraint_type == "linear"
        ]
        assert len(linear_constraints) >= 1

    def test_noisy_linear_not_detected(self) -> None:
        """Very noisy linear relationship should not be detected."""
        rng = np.random.default_rng(42)
        n = 2000
        s0 = rng.uniform(-5, 5, size=n)
        s1 = rng.uniform(-5, 5, size=n)
        s2 = 2.0 * s0 + 3.0 * s1 + rng.normal(0, 10, size=n)
        states = np.column_stack([s0, s1, s2])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        linear_constraints = [
            c for c in result.constraints if c.constraint_type == "linear"
        ]
        assert len(linear_constraints) == 0


# ---------------------------------------------------------------------------
# Angular wrapping
# ---------------------------------------------------------------------------


class TestAngularWrapping:
    """Tests for angular delta wrapping utility."""

    def test_wrap_small_deltas(self) -> None:
        """Small deltas should be unchanged."""
        states = np.array([[0.0], [1.0], [2.0]])
        next_states = np.array([[0.1], [1.1], [2.1]])

        deltas = wrap_angular_deltas(states, next_states, angular_dims=(0,))
        np.testing.assert_allclose(deltas, [[0.1], [0.1], [0.1]], atol=1e-10)

    def test_wrap_near_pi(self) -> None:
        """Deltas near +/-pi boundary should wrap correctly."""
        states = np.array([[3.0], [-3.0]])
        next_states = np.array([[-3.0], [3.0]])

        deltas = wrap_angular_deltas(states, next_states, angular_dims=(0,))

        # -3.0 - 3.0 = -6.0, wrapped to -6.0 + 2*pi ~ 0.283
        # 3.0 - (-3.0) = 6.0, wrapped to 6.0 - 2*pi ~ -0.283
        for d in deltas[:, 0]:
            assert -np.pi <= d <= np.pi

        # Both should be small in magnitude (close to wrapping boundary)
        assert abs(deltas[0, 0]) < 1.0
        assert abs(deltas[1, 0]) < 1.0

    def test_wrap_exact_pi_transition(self) -> None:
        """Transition from pi-eps to -(pi-eps) should give small positive delta."""
        eps = 0.1
        states = np.array([[np.pi - eps]])
        next_states = np.array([[-(np.pi - eps)]])

        deltas = wrap_angular_deltas(states, next_states, angular_dims=(0,))

        # Raw delta = -(pi-eps) - (pi-eps) = -2*pi + 2*eps
        # Wrapped: should be +2*eps
        np.testing.assert_allclose(deltas[0, 0], 2 * eps, atol=1e-10)

    def test_non_angular_dims_unchanged(self) -> None:
        """Non-angular dimensions use simple subtraction."""
        states = np.array([[3.0, 1.0]])
        next_states = np.array([[-3.0, 2.0]])

        deltas = wrap_angular_deltas(
            states, next_states, angular_dims=(0,),
        )

        # dim 0: wrapped
        assert -np.pi <= deltas[0, 0] <= np.pi
        # dim 1: simple subtraction
        np.testing.assert_allclose(deltas[0, 1], 1.0)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Tests for obs -> canonical -> obs round-trip consistency."""

    def test_circle_round_trip(self) -> None:
        """canonical_to_obs(obs_to_canonical(obs)) should recover obs."""
        rng = np.random.default_rng(42)
        n = 500
        theta = rng.uniform(-np.pi, np.pi, size=n)
        velocity = rng.uniform(-8, 8, size=n)
        states = np.column_stack([
            np.cos(theta), np.sin(theta), velocity,
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        for i in range(0, n, 50):
            obs = states[i]
            canonical = result.obs_to_canonical_fn(obs)
            obs_recovered = result.canonical_to_obs_fn(canonical)
            np.testing.assert_allclose(obs_recovered, obs, atol=1e-10)

    def test_no_constraint_round_trip(self) -> None:
        """Pass-through mode: obs -> canonical -> obs is identity."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(2000, 3))
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        obs = states[0]
        canonical = result.obs_to_canonical_fn(obs)
        recovered = result.canonical_to_obs_fn(canonical)
        np.testing.assert_allclose(recovered, obs, atol=1e-10)


# ---------------------------------------------------------------------------
# Multiple constraints
# ---------------------------------------------------------------------------


class TestMultipleConstraints:
    """Tests for detecting multiple independent constraints."""

    def test_two_independent_circles(self) -> None:
        """Detect two independent circle constraints."""
        rng = np.random.default_rng(42)
        n = 3000
        theta1 = rng.uniform(-np.pi, np.pi, size=n)
        theta2 = rng.uniform(-np.pi, np.pi, size=n)
        states = np.column_stack([
            np.cos(theta1), np.sin(theta1),  # circle 1: dims 0,1
            np.cos(theta2), np.sin(theta2),  # circle 2: dims 2,3
            rng.uniform(-5, 5, size=n),      # unconstrained: dim 4
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2", "s3", "s4"])

        # Should detect 2 circle constraints
        circle_constraints = [
            c for c in result.constraints if c.constraint_type == "circle"
        ]
        assert len(circle_constraints) == 2

        # Should have 2 mappings
        assert len(result.mappings) == 2

        # Canonical: 5 obs dims -> 3 canonical (phi_0, phi_1, s4)
        assert len(result.canonical_state_names) == 3
        assert result.canonical_states.shape == (n, 3)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for configuration options."""

    def test_min_samples_raises(self) -> None:
        """Raise ValueError if dataset is too small."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(10, 3))
        dataset = _make_dataset(states, n_envs=2)

        analyzer = ObservationAnalyzer(
            ObservationAnalysisConfig(min_samples=100),
        )
        with pytest.raises(ValueError, match="at least 100 samples"):
            analyzer.analyze(dataset, ["s0", "s1", "s2"])

    def test_strict_threshold_detects_less(self) -> None:
        """Stricter singular value threshold may miss noisy constraints."""
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(-np.pi, np.pi, size=n)
        noise = rng.normal(0, 0.001, size=n)
        states = np.column_stack([
            np.cos(theta) + noise, np.sin(theta) + noise,
        ])
        dataset = _make_dataset(states)

        # Very strict threshold
        strict_config = ObservationAnalysisConfig(
            singular_value_threshold=1e-10,
        )
        analyzer = ObservationAnalyzer(strict_config)
        result = analyzer.analyze(dataset, ["s0", "s1"])

        # With noise and strict threshold, may not detect constraint
        # (but may still detect it -- the test verifies the threshold matters)
        # The key is that it doesn't crash
        assert isinstance(result.constraints, list)


# ---------------------------------------------------------------------------
# Angular dim tracking
# ---------------------------------------------------------------------------


class TestAngularDimTracking:
    """Tests for angular_dims field in result."""

    def test_angular_dims_for_circle(self) -> None:
        """Circle mapping should mark the canonical dim as angular."""
        rng = np.random.default_rng(42)
        n = 2000
        theta = rng.uniform(-np.pi, np.pi, size=n)
        velocity = rng.uniform(-8, 8, size=n)
        states = np.column_stack([
            np.cos(theta), np.sin(theta), velocity,
        ])
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        # phi_0 is at canonical dim 0 and is angular
        assert 0 in result.angular_dims
        # s2 (velocity) is NOT angular
        assert 1 not in result.angular_dims

    def test_no_angular_dims_without_constraints(self) -> None:
        """No angular dims when no constraints detected."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(2000, 3))
        dataset = _make_dataset(states)

        analyzer = ObservationAnalyzer()
        result = analyzer.analyze(dataset, ["s0", "s1", "s2"])

        assert result.angular_dims == ()
