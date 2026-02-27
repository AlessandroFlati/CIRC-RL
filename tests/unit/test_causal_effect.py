"""Tests for circ_rl.feature_selection.causal_effect."""

import numpy as np
import pytest

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.feature_selection.causal_effect import (
    CausalEffectEstimator,
)


def _make_chain_graph() -> CausalGraph:
    """A -> B -> reward."""
    return CausalGraph.from_domain_knowledge(
        [("A", "B"), ("B", "reward")], reward_node="reward"
    )


def _make_confounded_graph() -> CausalGraph:
    """C -> A, C -> B, A -> reward, B -> reward."""
    return CausalGraph.from_domain_knowledge(
        [("C", "A"), ("C", "B"), ("A", "reward"), ("B", "reward")],
        reward_node="reward",
    )


def _generate_chain_data(
    n_samples: int = 2000, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate data from A -> B -> reward.

    A ~ N(0, 1)
    B = 1.5 * A + noise
    reward = 2.0 * B + noise

    True ATE(A -> reward) = 1.5 * 2.0 = 3.0 (total effect)
    True ATE(B -> reward) = 2.0 (direct effect)
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(n_samples)
    b = 1.5 * a + 0.2 * rng.randn(n_samples)
    r = 2.0 * b + 0.2 * rng.randn(n_samples)
    data = np.column_stack([a, b, r])
    return data, ["A", "B", "reward"]


def _generate_confounded_data(
    n_samples: int = 2000, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate data from C -> A, C -> B, A -> reward, B -> reward.

    C ~ N(0, 1)
    A = 1.0 * C + noise
    B = 0.5 * C + noise
    reward = 2.0 * A + 1.0 * B + noise

    True ATE(A -> reward) = 2.0 (adjusting for C and B)
    """
    rng = np.random.RandomState(seed)
    c = rng.randn(n_samples)
    a = 1.0 * c + 0.2 * rng.randn(n_samples)
    b = 0.5 * c + 0.2 * rng.randn(n_samples)
    r = 2.0 * a + 1.0 * b + 0.2 * rng.randn(n_samples)
    data = np.column_stack([a, b, c, r])
    return data, ["A", "B", "C", "reward"]


class TestAdjustmentSet:
    def test_chain_adjustment_set(self) -> None:
        graph = _make_chain_graph()
        adj = CausalEffectEstimator.find_adjustment_set(graph, "A", "reward")
        # Parents of A is empty (A is a root)
        assert adj == frozenset()

    def test_chain_b_adjustment_set(self) -> None:
        graph = _make_chain_graph()
        adj = CausalEffectEstimator.find_adjustment_set(graph, "B", "reward")
        # Parents of B is {A}
        assert adj == frozenset({"A"})

    def test_confounded_adjustment_set(self) -> None:
        graph = _make_confounded_graph()
        adj = CausalEffectEstimator.find_adjustment_set(graph, "A", "reward")
        # Parents of A is {C}
        assert adj == frozenset({"C"})


class TestATEEstimation:
    def test_direct_effect_in_chain(self) -> None:
        """ATE(B -> reward) should be close to 2.0."""
        data, names = _generate_chain_data(n_samples=5000)
        graph = _make_chain_graph()
        estimator = CausalEffectEstimator()
        result = estimator.estimate(data, names, graph, "B", "reward")

        assert result.cause == "B"
        assert result.effect == "reward"
        assert abs(result.ate - 2.0) < 0.1

    def test_total_effect_in_chain(self) -> None:
        """ATE(A -> reward) without adjustment captures total effect ~3.0."""
        data, names = _generate_chain_data(n_samples=5000)
        graph = _make_chain_graph()
        estimator = CausalEffectEstimator()
        result = estimator.estimate(data, names, graph, "A", "reward")

        # A -> B -> reward total effect = 1.5 * 2.0 = 3.0
        assert abs(result.ate - 3.0) < 0.2

    def test_confounded_ate(self) -> None:
        """ATE(A -> reward) with proper adjustment should be ~2.0."""
        data, names = _generate_confounded_data(n_samples=5000)
        graph = _make_confounded_graph()
        estimator = CausalEffectEstimator()
        result = estimator.estimate(data, names, graph, "A", "reward")

        # Adjusting for C (parents of A), ATE should be ~2.0
        assert abs(result.ate - 2.0) < 0.2

    def test_unknown_variable_raises(self) -> None:
        data, names = _generate_chain_data(n_samples=100)
        with pytest.raises(ValueError, match="not found"):
            CausalEffectEstimator.estimate_ate(
                data, names, "UNKNOWN", "reward", frozenset()
            )

    def test_result_has_correct_fields(self) -> None:
        data, names = _generate_chain_data(n_samples=500)
        graph = _make_chain_graph()
        estimator = CausalEffectEstimator()
        result = estimator.estimate(data, names, graph, "B", "reward")

        assert result.cause == "B"
        assert result.effect == "reward"
        assert isinstance(result.ate, float)
        assert isinstance(result.adjustment_set, frozenset)
        assert result.coefficients.shape[0] >= 2  # intercept + cause


class TestMechanismInvariance:
    """Tests for CausalEffectEstimator.test_mechanism_invariance()."""

    def test_stable_linear_mechanism(self) -> None:
        """Same linear mechanism in both envs -> high p-value."""
        rng = np.random.RandomState(42)
        n = 500

        # Env A: X ~ N(0, 1), R = 2*X + noise
        x_a = rng.randn(n)
        r_a = 2.0 * x_a + 0.1 * rng.randn(n)
        data_a = np.column_stack([x_a, r_a])

        # Env B: X ~ N(3, 1) (different distribution), R = 2*X + noise (same mechanism)
        x_b = rng.randn(n) + 3.0
        r_b = 2.0 * x_b + 0.1 * rng.randn(n)
        data_b = np.column_stack([x_b, r_b])

        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=1, predictor_idxs=[0],
        )
        assert p_value > 0.05, f"Expected invariant, got p={p_value:.4f}"

    def test_unstable_linear_mechanism(self) -> None:
        """Different linear coefficients -> low p-value."""
        rng = np.random.RandomState(42)
        n = 500

        # Env A: R = 2*X + noise
        x_a = rng.randn(n)
        r_a = 2.0 * x_a + 0.1 * rng.randn(n)
        data_a = np.column_stack([x_a, r_a])

        # Env B: R = -1*X + noise (different mechanism)
        x_b = rng.randn(n)
        r_b = -1.0 * x_b + 0.1 * rng.randn(n)
        data_b = np.column_stack([x_b, r_b])

        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=1, predictor_idxs=[0],
        )
        assert p_value < 0.01, f"Expected non-invariant, got p={p_value:.4f}"

    def test_stable_nonlinear_mechanism(self) -> None:
        """Same quadratic mechanism, different X distributions -> high p-value.

        This is the key test: the polynomial Chow test should correctly
        identify mechanism invariance even when the feature distributions
        differ across environments.
        """
        rng = np.random.RandomState(42)
        n = 500

        # Env A: X ~ N(0, 1), R = X^2 + 0.5*X + noise
        x_a = rng.randn(n)
        r_a = x_a**2 + 0.5 * x_a + 0.1 * rng.randn(n)
        data_a = np.column_stack([x_a, r_a])

        # Env B: X ~ N(2, 0.5) (different distribution), same mechanism
        x_b = rng.randn(n) * 0.5 + 2.0
        r_b = x_b**2 + 0.5 * x_b + 0.1 * rng.randn(n)
        data_b = np.column_stack([x_b, r_b])

        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=1, predictor_idxs=[0],
            poly_degree=2,
        )
        assert p_value > 0.05, f"Expected invariant (quadratic), got p={p_value:.4f}"

    def test_poly_degree_fallback(self) -> None:
        """When n_features > n_samples/5, degree should reduce automatically."""
        rng = np.random.RandomState(42)
        n = 20  # Very few samples

        # 5 predictors with degree 2 -> many poly features, should fall back
        x_a = rng.randn(n, 5)
        r_a = x_a[:, 0] + 0.1 * rng.randn(n)
        data_a = np.column_stack([x_a, r_a])

        x_b = rng.randn(n, 5)
        r_b = x_b[:, 0] + 0.1 * rng.randn(n)
        data_b = np.column_stack([x_b, r_b])

        # Should not crash even with many features and few samples
        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=5, predictor_idxs=[0, 1, 2, 3, 4],
            poly_degree=2,
        )
        assert 0.0 <= p_value <= 1.0

    def test_multivariate_stable_mechanism(self) -> None:
        """Two predictors with interaction, same mechanism across envs."""
        rng = np.random.RandomState(42)
        n = 500

        # Env A: R = X1*X2 + X1^2 + noise (same mechanism)
        x1_a = rng.randn(n)
        x2_a = rng.randn(n)
        r_a = x1_a * x2_a + x1_a**2 + 0.1 * rng.randn(n)
        data_a = np.column_stack([x1_a, x2_a, r_a])

        # Env B: different distributions, same mechanism
        x1_b = rng.randn(n) + 1.0
        x2_b = rng.randn(n) * 2.0
        r_b = x1_b * x2_b + x1_b**2 + 0.1 * rng.randn(n)
        data_b = np.column_stack([x1_b, x2_b, r_b])

        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=2, predictor_idxs=[0, 1],
            poly_degree=2,
        )
        assert p_value > 0.05, f"Expected invariant, got p={p_value:.4f}"

    def test_empty_predictors_returns_one(self) -> None:
        """No predictors -> p-value 1.0."""
        rng = np.random.RandomState(42)
        data_a = rng.randn(100, 2)
        data_b = rng.randn(100, 2)
        p_value = CausalEffectEstimator.test_mechanism_invariance(
            data_a, data_b, target_idx=0, predictor_idxs=[],
        )
        assert p_value == 1.0
