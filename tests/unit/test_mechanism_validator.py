"""Tests for circ_rl.causal_discovery.mechanism_validator."""

import numpy as np

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.causal_discovery.mechanism_validator import MechanismValidator
from circ_rl.environments.data_collector import ExploratoryDataset


def _make_invariant_dataset() -> tuple[ExploratoryDataset, list[str]]:
    """Two environments with the SAME causal mechanism X -> reward.

    reward = 2 * X + noise in both environments.
    X distribution differs (shifted mean), but mechanism is the same.
    """
    rng = np.random.RandomState(42)
    n = 300

    # Environment 0: X ~ N(0, 1)
    x0 = rng.randn(n)
    r0 = 2.0 * x0 + 0.3 * rng.randn(n)

    # Environment 1: X ~ N(2, 1) -- different distribution, same mechanism
    x1 = rng.randn(n) + 2.0
    r1 = 2.0 * x1 + 0.3 * rng.randn(n)

    states = np.concatenate([x0[:, None], x1[:, None]])
    rewards = np.concatenate([r0, r1])
    env_ids = np.array([0] * n + [1] * n, dtype=np.int32)

    dataset = ExploratoryDataset(
        states=states,
        actions=np.zeros(2 * n, dtype=np.int32),
        next_states=states,
        rewards=rewards,
        env_ids=env_ids,
    )
    # Flat array: [X, action, reward, X_next]
    node_names = ["X", "action", "reward", "X_next"]
    return dataset, node_names


def _make_variant_dataset() -> tuple[ExploratoryDataset, list[str]]:
    """Two environments with DIFFERENT causal mechanisms X -> reward.

    Env 0: reward = 2 * X + noise
    Env 1: reward = -1 * X + noise (different coefficient)
    """
    rng = np.random.RandomState(42)
    n = 300

    x0 = rng.randn(n)
    r0 = 2.0 * x0 + 0.3 * rng.randn(n)

    x1 = rng.randn(n)
    r1 = -1.0 * x1 + 0.3 * rng.randn(n)

    states = np.concatenate([x0[:, None], x1[:, None]])
    rewards = np.concatenate([r0, r1])
    env_ids = np.array([0] * n + [1] * n, dtype=np.int32)

    dataset = ExploratoryDataset(
        states=states,
        actions=np.zeros(2 * n, dtype=np.int32),
        next_states=states,
        rewards=rewards,
        env_ids=env_ids,
    )
    node_names = ["X", "action", "reward", "X_next"]
    return dataset, node_names


class TestMechanismValidator:
    def test_invariant_mechanism_passes(self) -> None:
        dataset, node_names = _make_invariant_dataset()
        graph = CausalGraph.from_domain_knowledge(
            [("X", "reward")], reward_node="reward"
        )
        # Add remaining nodes
        import networkx as nx

        g = graph.graph
        for name in node_names:
            if name not in g.nodes:
                g.add_node(name)
        graph = CausalGraph(g, reward_node="reward")

        validator = MechanismValidator(alpha=0.05)
        result = validator.validate_invariance(
            dataset, graph, node_names, target_node="reward"
        )

        assert result.is_invariant
        assert len(result.unstable_mechanisms) == 0

    def test_variant_mechanism_detected(self) -> None:
        dataset, node_names = _make_variant_dataset()
        graph = CausalGraph.from_domain_knowledge(
            [("X", "reward")], reward_node="reward"
        )
        import networkx as nx

        g = graph.graph
        for name in node_names:
            if name not in g.nodes:
                g.add_node(name)
        graph = CausalGraph(g, reward_node="reward")

        validator = MechanismValidator(alpha=0.05)
        result = validator.validate_invariance(
            dataset, graph, node_names, target_node="reward"
        )

        assert not result.is_invariant
        assert len(result.unstable_mechanisms) > 0

    def test_single_env_returns_invariant(self) -> None:
        """With only one environment, invariance is trivially true."""
        rng = np.random.RandomState(42)
        n = 100
        dataset = ExploratoryDataset(
            states=rng.randn(n, 1).astype(np.float32),
            actions=np.zeros(n, dtype=np.int32),
            next_states=rng.randn(n, 1).astype(np.float32),
            rewards=rng.randn(n).astype(np.float32),
            env_ids=np.zeros(n, dtype=np.int32),
        )
        graph = CausalGraph.from_domain_knowledge(
            [("X", "reward")], reward_node="reward"
        )

        validator = MechanismValidator(alpha=0.05)
        result = validator.validate_invariance(
            dataset, graph, ["X", "action", "reward", "X_next"], target_node="reward"
        )
        assert result.is_invariant
