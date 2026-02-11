"""Tests for circ_rl.causal_discovery.ci_tests."""

import numpy as np
import pytest

from circ_rl.causal_discovery.ci_tests import (
    causal_ci_test_fisher_z,
    causal_ci_test_kernel,
)


@pytest.fixture
def correlated_data() -> np.ndarray:
    """X causes Y: Y = 2*X + noise."""
    rng = np.random.RandomState(42)
    n = 500
    x = rng.randn(n)
    y = 2.0 * x + 0.3 * rng.randn(n)
    return np.column_stack([x, y])


@pytest.fixture
def independent_data() -> np.ndarray:
    """X and Y are independent."""
    rng = np.random.RandomState(42)
    n = 500
    x = rng.randn(n)
    y = rng.randn(n)
    return np.column_stack([x, y])


@pytest.fixture
def confounded_data() -> np.ndarray:
    """X <- Z -> Y: X and Y are dependent but conditionally independent given Z."""
    rng = np.random.RandomState(42)
    n = 500
    z = rng.randn(n)
    x = z + 0.3 * rng.randn(n)
    y = z + 0.3 * rng.randn(n)
    return np.column_stack([x, y, z])


class TestFisherZ:
    def test_detects_dependence(self, correlated_data: np.ndarray) -> None:
        result = causal_ci_test_fisher_z(correlated_data, 0, 1, [])
        assert not result.independent
        assert result.p_value < 0.01

    def test_detects_independence(self, independent_data: np.ndarray) -> None:
        result = causal_ci_test_fisher_z(independent_data, 0, 1, [])
        assert result.independent
        assert result.p_value > 0.05

    def test_conditional_independence(
        self, confounded_data: np.ndarray
    ) -> None:
        # X and Y are dependent marginally
        marginal = causal_ci_test_fisher_z(confounded_data, 0, 1, [])
        assert not marginal.independent

        # X and Y are independent given Z
        conditional = causal_ci_test_fisher_z(confounded_data, 0, 1, [2])
        assert conditional.independent

    def test_insufficient_samples_raises(self) -> None:
        data = np.random.randn(3, 5)
        with pytest.raises(ValueError, match="Insufficient samples"):
            causal_ci_test_fisher_z(data, 0, 1, [2, 3, 4])

    def test_result_has_correct_conditioning_set(
        self, confounded_data: np.ndarray
    ) -> None:
        result = causal_ci_test_fisher_z(confounded_data, 0, 1, [2])
        assert result.conditioning_set == frozenset({2})

    def test_p_value_in_valid_range(
        self, correlated_data: np.ndarray
    ) -> None:
        result = causal_ci_test_fisher_z(correlated_data, 0, 1, [])
        assert 0.0 <= result.p_value <= 1.0


class TestKernelCI:
    def test_detects_dependence(self, correlated_data: np.ndarray) -> None:
        result = causal_ci_test_kernel(
            correlated_data, 0, 1, [], n_permutations=100
        )
        assert not result.independent

    def test_detects_independence(self, independent_data: np.ndarray) -> None:
        result = causal_ci_test_kernel(
            independent_data, 0, 1, [], n_permutations=100
        )
        assert result.independent

    def test_conditional_independence(self) -> None:
        # Kernel CI needs more samples than Fisher-z to condition reliably
        rng = np.random.RandomState(42)
        n = 2000
        z = rng.randn(n)
        x = z + 0.5 * rng.randn(n)
        y = z + 0.5 * rng.randn(n)
        data = np.column_stack([x, y, z])
        result = causal_ci_test_kernel(
            data, 0, 1, [2], n_permutations=200
        )
        assert result.independent
