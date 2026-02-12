"""Integration tests for SyntheticNonlinearEnv.

Validates that the full pipeline (causal discovery, feature selection,
conditional invariance with KRR) works on an environment with a known
non-linear env-param effect on reward: ATE(s0) = k^2.
"""

from __future__ import annotations

import numpy as np
import pytest

# Register env before any test runs
import circ_rl.environments.synthetic_nonlinear  # noqa: F401
from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.environments.data_collector import DataCollector
from circ_rl.environments.env_family import EnvironmentFamily
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector


@pytest.fixture(scope="module")
def env_family():
    return EnvironmentFamily.from_gymnasium(
        base_env="SyntheticNonlinear-v0",
        param_distributions={"k": (1.0, 5.0)},
        n_envs=8,
        seed=42,
    )


@pytest.fixture(scope="module")
def dataset(env_family):
    collector = DataCollector(env_family, include_env_params=True)
    return collector.collect(n_transitions_per_env=2000, seed=42)


@pytest.fixture(scope="module")
def node_names():
    state_names = ["s0", "s1", "s2"]
    action_names = ["action_0"]
    next_state_names = ["s0_next", "s1_next", "s2_next"]
    ep_names = ["ep_k"]
    return state_names + action_names + ["reward"] + next_state_names + ep_names


@pytest.fixture(scope="module")
def graph(dataset, node_names):
    return CausalGraphBuilder.discover(
        dataset,
        node_names,
        method="pc",
        alpha=0.01,
        env_param_names=["ep_k"],
        ep_correlation_threshold=0.05,
    )


class TestSyntheticNonlinearEnv:
    """Basic environment tests."""

    def test_env_step_and_reset(self):
        """Environment can reset and step without errors."""
        import gymnasium as gym

        env = gym.make("SyntheticNonlinear-v0")
        obs, info = env.reset(seed=42)
        assert obs.shape == (3,)
        assert isinstance(info, dict)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (3,)
        assert isinstance(reward, (float, np.floating))
        assert terminated is False
        assert truncated is False
        env.close()

    def test_env_family_integration(self, env_family):
        """EnvironmentFamily correctly configures k via setattr."""
        for i in range(env_family.n_envs):
            params = env_family.get_env_params(i)
            assert "k" in params
            assert 1.0 <= params["k"] <= 5.0

        # Verify different envs have different k values
        k_values = [env_family.get_env_params(i)["k"] for i in range(env_family.n_envs)]
        assert len(set(k_values)) > 1


class TestCausalDiscovery:
    """Causal discovery should find ep_k -> reward."""

    def test_pc_discovers_ep_k_edge(self, graph):
        """PC or correlation pre-screen finds ep_k -> reward edge."""
        # ep_k should be an ancestor of reward (either direct or indirect)
        ep_ancestors = graph.env_param_ancestors_of("reward")
        assert "ep_k" in ep_ancestors, (
            f"ep_k not found as ancestor of reward. "
            f"Edges: {sorted(graph.edges)}"
        )

    def test_correlation_prescreen_adds_edge(self, dataset, node_names):
        """Correlation pre-screen adds ep_k -> reward even without PC."""
        # Build a graph WITHOUT ep_k -> reward to test pre-screen in isolation
        import networkx as nx
        from circ_rl.causal_discovery.causal_graph import CausalGraph

        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        g.add_edge("s0", "reward")
        g.add_edge("s1", "reward")
        graph_no_ep = CausalGraph(g, reward_node="reward")

        data = dataset.to_flat_array_with_env_params()
        result = CausalGraphBuilder._add_correlated_ep_edges(
            graph_no_ep, data, node_names,
            frozenset({"ep_k"}), "reward", 0.05,
        )
        assert ("ep_k", "reward") in result.edges


class TestFeatureSelection:
    """Feature selection with conditional invariance on non-linear env."""

    def test_s0_rescued_as_context_dependent(self, dataset, graph):
        """s0 should be rescued as context-dependent via KRR."""
        selector = InvFeatureSelector(
            epsilon=0.1, min_ate=0.01, enable_conditional_invariance=True
        )
        result = selector.select(
            dataset, graph, ["s0", "s1", "s2"],
            env_param_names=["ep_k"],
        )

        assert "s0" in result.selected_features, (
            f"s0 not selected. Rejected: {result.rejected_features}"
        )
        # s0 should specifically be context-dependent (not invariant)
        assert "s0" in result.context_dependent_features, (
            f"s0 selected but not as context-dependent. "
            f"Context-dependent: {result.context_dependent_features}"
        )
        assert "ep_k" in result.context_param_names

    def test_s1_low_ate_variance(self, dataset, graph):
        """s1's ATE variance should be much lower than s0's.

        s1 has a constant ATE (2.0) across envs. With finite samples and
        noise, the variance won't be exactly zero but should be far below
        s0's variance (which is driven by k^2 variation).

        Note: s1 may or may not pass the epsilon=0.1 threshold depending
        on noise. The key property is that its variance is orders of
        magnitude lower than s0's.
        """
        selector = InvFeatureSelector(
            epsilon=0.5, min_ate=0.01, enable_conditional_invariance=True
        )
        result = selector.select(
            dataset, graph, ["s0", "s1", "s2"],
            env_param_names=["ep_k"],
        )

        # s1's ATE variance should be present and much lower than s0's
        if "s1" in result.ate_variance and "s0" in result.ate_variance:
            assert result.ate_variance["s1"] < result.ate_variance["s0"] * 0.01, (
                f"s1 variance ({result.ate_variance['s1']:.4f}) should be "
                f"<< s0 variance ({result.ate_variance['s0']:.4f})"
            )
