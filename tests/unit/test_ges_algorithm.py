"""Tests for circ_rl.causal_discovery.ges_algorithm."""

import numpy as np
import pytest

from circ_rl.causal_discovery.ges_algorithm import GESAlgorithm


def _generate_simple_data(
    n_samples: int = 500, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate data from A -> B -> reward."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n_samples)
    b = 1.5 * a + 0.3 * rng.randn(n_samples)
    r = 1.5 * b + 0.3 * rng.randn(n_samples)
    data = np.column_stack([a, b, r])
    return data, ["A", "B", "reward"]


class TestGESAlgorithm:
    def test_finds_edges_in_simple_chain(self) -> None:
        data, names = _generate_simple_data(n_samples=1000)
        ges = GESAlgorithm(score_fn="bic")
        graph = ges.fit(data, names)

        # Should have edges (not necessarily in the right direction,
        # but at least the skeleton should be correct)
        assert len(graph.edges) > 0
        assert len(graph.nodes) == 3

    def test_bic_score_prefers_correct_edges(self) -> None:
        """GES with BIC should find that B is a parent of reward."""
        data, names = _generate_simple_data(n_samples=1000)
        ges = GESAlgorithm(score_fn="bic")
        graph = ges.fit(data, names)

        # B should be connected to reward
        parents = graph.causal_parents_of_reward()
        ancestors = graph.ancestors_of_reward()
        assert "B" in parents or "B" in ancestors

    def test_max_parents_is_respected(self) -> None:
        data, names = _generate_simple_data(n_samples=500)
        ges = GESAlgorithm(score_fn="bic", max_parents=1)
        graph = ges.fit(data, names)

        for node in graph.nodes:
            assert len(graph.parents(node)) <= 1

    def test_rejects_invalid_score_fn(self) -> None:
        with pytest.raises(ValueError, match="score_fn"):
            GESAlgorithm(score_fn="invalid")

    def test_produces_valid_dag(self) -> None:
        data, names = _generate_simple_data(n_samples=500)
        ges = GESAlgorithm(score_fn="bic")
        graph = ges.fit(data, names)

        # CausalGraph constructor validates DAG
        assert len(graph.nodes) == 3
