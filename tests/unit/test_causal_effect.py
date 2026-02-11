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
