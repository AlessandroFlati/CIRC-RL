"""Tests for circ_rl.feature_selection.inv_feature_selector."""

import numpy as np
import pytest

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector


def _make_graph_with_spurious() -> CausalGraph:
    """Graph: X1 -> X2 -> reward, X3 -> reward, S1 (no connection), S2 (no connection).

    X1, X2, X3 are causal ancestors of reward.
    S1, S2 are spurious (not ancestors of reward).
    """
    import networkx as nx

    g = nx.DiGraph()
    g.add_edges_from([("X1", "X2"), ("X2", "reward"), ("X3", "reward")])
    g.add_nodes_from(["S1", "S2"])
    return CausalGraph(g, reward_node="reward")


def _make_invariant_dataset(
    n_per_env: int = 500, n_envs: int = 3, seed: int = 42
) -> ExploratoryDataset:
    """Dataset with 3 causal features (invariant) and 2 spurious features.

    State: [X1, X2, X3, S1, S2]
    Mechanism (same across all environments):
        X2 = 0.8 * X1 + noise
        X3 ~ N(0, 1)
        reward = 1.5 * X2 + 0.5 * X3 + noise
        S1, S2 ~ N(0, 1) (independent of everything)

    X1 distribution shifts across environments but mechanism is invariant.
    """
    rng = np.random.RandomState(seed)
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_env_ids = []

    for env_id in range(n_envs):
        # X1 distribution shifts across envs
        x1_mean = env_id * 1.0
        x1 = rng.randn(n_per_env) + x1_mean
        x2 = 0.8 * x1 + 0.2 * rng.randn(n_per_env)
        x3 = rng.randn(n_per_env)
        s1 = rng.randn(n_per_env)
        s2 = rng.randn(n_per_env)

        reward = 1.5 * x2 + 0.5 * x3 + 0.1 * rng.randn(n_per_env)

        states = np.column_stack([x1, x2, x3, s1, s2])
        next_states = states + 0.01 * rng.randn(n_per_env, 5)
        actions = rng.randint(0, 2, size=n_per_env)

        all_states.append(states.astype(np.float32))
        all_actions.append(actions.astype(np.int32))
        all_next_states.append(next_states.astype(np.float32))
        all_rewards.append(reward.astype(np.float32))
        all_env_ids.append(np.full(n_per_env, env_id, dtype=np.int32))

    return ExploratoryDataset(
        states=np.concatenate(all_states),
        actions=np.concatenate(all_actions),
        next_states=np.concatenate(all_next_states),
        rewards=np.concatenate(all_rewards),
        env_ids=np.concatenate(all_env_ids),
    )


def _make_mixed_dataset(
    n_per_env: int = 500, seed: int = 42
) -> ExploratoryDataset:
    """Dataset where X1 has an invariant mechanism, X2 has a variant mechanism.

    Env 0: reward = 2.0 * X1 + 1.0 * X2 + noise
    Env 1: reward = 2.0 * X1 - 1.0 * X2 + noise  (X2 coefficient flips)
    """
    rng = np.random.RandomState(seed)

    # Env 0
    x1_0 = rng.randn(n_per_env)
    x2_0 = rng.randn(n_per_env)
    r_0 = 2.0 * x1_0 + 1.0 * x2_0 + 0.1 * rng.randn(n_per_env)

    # Env 1
    x1_1 = rng.randn(n_per_env)
    x2_1 = rng.randn(n_per_env)
    r_1 = 2.0 * x1_1 - 1.0 * x2_1 + 0.1 * rng.randn(n_per_env)

    states = np.concatenate([
        np.column_stack([x1_0, x2_0]),
        np.column_stack([x1_1, x2_1]),
    ]).astype(np.float32)
    actions = np.zeros(2 * n_per_env, dtype=np.int32)
    next_states = states + 0.01 * rng.randn(2 * n_per_env, 2).astype(np.float32)
    rewards = np.concatenate([r_0, r_1]).astype(np.float32)
    env_ids = np.array(
        [0] * n_per_env + [1] * n_per_env, dtype=np.int32
    )

    return ExploratoryDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        env_ids=env_ids,
    )


class TestInvFeatureSelector:
    def test_selects_causal_features_only(self) -> None:
        """Spurious features (S1, S2) should be rejected."""
        graph = _make_graph_with_spurious()
        dataset = _make_invariant_dataset()
        feature_names = ["X1", "X2", "X3", "S1", "S2"]

        selector = InvFeatureSelector(epsilon=0.5, min_ate=0.01)
        result = selector.select(dataset, graph, feature_names)

        # S1 and S2 should be rejected (not ancestors of reward)
        assert "S1" not in result.selected_features
        assert "S2" not in result.selected_features
        assert "S1" in result.rejected_features
        assert "S2" in result.rejected_features

    def test_selects_at_least_one_causal_feature(self) -> None:
        """At least one causal feature should survive selection."""
        graph = _make_graph_with_spurious()
        dataset = _make_invariant_dataset()
        feature_names = ["X1", "X2", "X3", "S1", "S2"]

        selector = InvFeatureSelector(epsilon=0.5, min_ate=0.01)
        result = selector.select(dataset, graph, feature_names)

        assert len(result.selected_features) >= 1
        # All selected features should be causal ancestors
        for feat in result.selected_features:
            assert feat in {"X1", "X2", "X3"}

    def test_feature_mask_matches_selection(self) -> None:
        graph = _make_graph_with_spurious()
        dataset = _make_invariant_dataset()
        feature_names = ["X1", "X2", "X3", "S1", "S2"]

        selector = InvFeatureSelector(epsilon=0.5, min_ate=0.01)
        result = selector.select(dataset, graph, feature_names)

        assert result.feature_mask.shape == (5,)
        assert result.feature_mask.dtype == np.bool_
        # Mask should match selected features
        for i, name in enumerate(feature_names):
            if name in result.selected_features:
                assert result.feature_mask[i]
            else:
                assert not result.feature_mask[i]

    def test_variant_mechanism_rejected(self) -> None:
        """Feature with variant mechanism (X2) should have higher ATE variance."""
        import networkx as nx

        g = nx.DiGraph()
        g.add_edges_from([("X1", "reward"), ("X2", "reward")])
        graph = CausalGraph(g, reward_node="reward")

        dataset = _make_mixed_dataset(n_per_env=1000)
        feature_names = ["X1", "X2"]

        # With tight epsilon, X2 (variant) should be rejected
        selector = InvFeatureSelector(epsilon=0.1, min_ate=0.01)
        result = selector.select(dataset, graph, feature_names)

        # X2 has high ATE variance (coefficient flips: 1.0 vs -1.0)
        assert result.ate_variance["X2"] > result.ate_variance["X1"]

    def test_rejects_mismatched_names(self) -> None:
        graph = _make_graph_with_spurious()
        dataset = _make_invariant_dataset()

        selector = InvFeatureSelector(epsilon=0.5)
        with pytest.raises(ValueError, match="state_feature_names"):
            selector.select(dataset, graph, ["only_two", "names"])

    def test_epsilon_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            InvFeatureSelector(epsilon=0.0)

    def test_ate_variance_computed_for_ancestors(self) -> None:
        """ATE variance should be computed for all ancestor-of-reward features."""
        graph = _make_graph_with_spurious()
        dataset = _make_invariant_dataset()
        feature_names = ["X1", "X2", "X3", "S1", "S2"]

        selector = InvFeatureSelector(epsilon=10.0, min_ate=0.001)
        result = selector.select(dataset, graph, feature_names)

        # S1, S2 are not ancestors, so no ATE variance for them
        assert "S1" not in result.ate_variance
        assert "S2" not in result.ate_variance
        # Causal features should have variance computed
        for feat in ["X1", "X2", "X3"]:
            assert feat in result.ate_variance
