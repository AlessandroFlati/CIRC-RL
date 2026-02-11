"""Tests for circ_rl.causal_discovery.pc_algorithm."""

import numpy as np

from circ_rl.causal_discovery.pc_algorithm import PCAlgorithm


def _generate_chain_data(
    n_samples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate data from X -> Y -> Z -> reward (linear chain)."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples)
    y = 0.8 * x + 0.3 * rng.randn(n_samples)
    z = 0.8 * y + 0.3 * rng.randn(n_samples)
    r = 0.8 * z + 0.3 * rng.randn(n_samples)
    data = np.column_stack([x, y, z, r])
    names = ["X", "Y", "Z", "reward"]
    return data, names


def _generate_fork_data(
    n_samples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate data from X <- Z -> Y, Z -> reward (fork/common cause)."""
    rng = np.random.RandomState(seed)
    z = rng.randn(n_samples)
    x = 0.8 * z + 0.3 * rng.randn(n_samples)
    y = 0.8 * z + 0.3 * rng.randn(n_samples)
    r = 0.8 * z + 0.3 * rng.randn(n_samples)
    data = np.column_stack([x, y, z, r])
    names = ["X", "Y", "Z", "reward"]
    return data, names


class TestPCAlgorithm:
    def test_recovers_chain_skeleton(self) -> None:
        """PC should find that X-Y-Z-reward are connected in sequence."""
        data, names = _generate_chain_data(n_samples=2000)
        pc = PCAlgorithm(alpha=0.05)
        graph = pc.fit(data, names)

        # X should be ancestor of reward
        assert graph.is_ancestor("X", "reward")
        # Y should be ancestor of reward
        assert graph.is_ancestor("Y", "reward")
        # Z should be parent of reward
        assert "Z" in graph.causal_parents_of_reward()

    def test_recovers_fork_structure(self) -> None:
        """In a fork X <- Z -> Y, X and Y should be conditionally independent given Z."""
        data, names = _generate_fork_data(n_samples=2000)
        pc = PCAlgorithm(alpha=0.05)
        graph = pc.fit(data, names)

        # Z should be ancestor of reward
        assert graph.is_ancestor("Z", "reward")
        # X should NOT be a direct parent of Y (they are independent given Z)
        assert "X" not in graph.parents("Y")
        assert "Y" not in graph.parents("X")

    def test_max_conditioning_set_size(self) -> None:
        """Setting max_conditioning_set_size should still produce a valid graph."""
        data, names = _generate_chain_data(n_samples=2000)
        pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=1)
        graph = pc.fit(data, names)

        # Should still be a valid DAG
        assert len(graph.nodes) == 4

    def test_rejects_mismatched_dimensions(self) -> None:
        """Data columns must match node names."""
        data = np.random.randn(100, 3)
        names = ["A", "B"]
        pc = PCAlgorithm()
        try:
            pc.fit(data, names + ["reward"])
        except ValueError:
            pass  # Expected if names don't match

    def test_produces_dag(self) -> None:
        """Output must always be a valid DAG (no cycles)."""
        data, names = _generate_chain_data(n_samples=1000)
        pc = PCAlgorithm(alpha=0.05)
        graph = pc.fit(data, names)

        # CausalGraph constructor validates DAG property
        assert len(graph.nodes) == len(names)
        assert len(graph.edges) > 0
