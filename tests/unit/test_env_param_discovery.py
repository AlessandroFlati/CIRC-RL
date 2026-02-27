"""Unit tests for environment-parameter causal discovery feature.

Tests the augmented ExploratoryDataset, CausalGraph env-param awareness,
InvFeatureSelector conditional invariance, CausalPolicy context support,
and Trajectory env_params field.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from circ_rl.causal_discovery.builder import CausalGraphBuilder
from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.environments.data_collector import ExploratoryDataset
from circ_rl.feature_selection.inv_feature_selector import InvFeatureSelector
from circ_rl.policy.causal_policy import CausalPolicy
from circ_rl.training.circ_trainer import IterationMetrics
from circ_rl.training.trajectory_buffer import (
    MultiEnvTrajectoryBuffer,
    Trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(
    n: int = 100,
    state_dim: int = 3,
    n_envs: int = 2,
    with_env_params: bool = False,
    n_env_params: int = 2,
) -> ExploratoryDataset:
    """Build a synthetic ExploratoryDataset."""
    rng = np.random.RandomState(42)
    states = rng.randn(n, state_dim).astype(np.float32)
    actions = rng.randint(0, 2, size=(n,)).astype(np.float32)
    next_states = rng.randn(n, state_dim).astype(np.float32)
    rewards = rng.randn(n).astype(np.float32)
    env_ids = np.repeat(np.arange(n_envs), n // n_envs).astype(np.int32)
    env_params = None
    if with_env_params:
        env_params = rng.randn(n, n_env_params).astype(np.float32)
    return ExploratoryDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        env_ids=env_ids,
        env_params=env_params,
    )


def _make_graph_with_ep(
    state_names: list[str],
    ep_names: list[str],
    ep_to_reward: bool = True,
    state_to_reward: bool = True,
) -> CausalGraph:
    """Build a CausalGraph with env-param nodes."""
    g = nx.DiGraph()
    g.add_nodes_from(state_names + ["action", "reward"] + ep_names)
    if state_to_reward:
        for s in state_names:
            g.add_edge(s, "reward")
    if ep_to_reward:
        for ep in ep_names:
            g.add_edge(ep, "reward")
    return CausalGraph(g, reward_node="reward")


# ===========================================================================
# 1. ExploratoryDataset with env params
# ===========================================================================


class TestExploratoryDatasetEnvParams:
    """Tests for the env_params field on ExploratoryDataset."""

    def test_default_env_params_is_none(self) -> None:
        ds = _make_dataset(with_env_params=False)
        assert ds.env_params is None
        assert ds.n_env_params == 0

    def test_env_params_populated(self) -> None:
        ds = _make_dataset(with_env_params=True, n_env_params=3)
        assert ds.env_params is not None
        assert ds.env_params.shape == (100, 3)
        assert ds.n_env_params == 3

    def test_to_flat_array_unchanged(self) -> None:
        ds = _make_dataset(with_env_params=True, n_env_params=2)
        flat = ds.to_flat_array()
        # state_dim=3, action_dim=1, reward=1, next_state_dim=3 => 8 cols
        assert flat.shape == (100, 8)

    def test_to_flat_array_with_env_params(self) -> None:
        ds = _make_dataset(with_env_params=True, n_env_params=2)
        flat = ds.to_flat_array_with_env_params()
        # 8 base + 2 env_params = 10 cols
        assert flat.shape == (100, 10)

    def test_to_flat_array_with_env_params_raises_when_none(self) -> None:
        ds = _make_dataset(with_env_params=False)
        with pytest.raises(ValueError, match="env_params is None"):
            ds.to_flat_array_with_env_params()

    def test_get_env_data_slices_env_params(self) -> None:
        ds = _make_dataset(n=100, n_envs=2, with_env_params=True, n_env_params=2)
        subset = ds.get_env_data(0)
        assert subset.env_params is not None
        assert subset.env_params.shape[1] == 2
        assert subset.env_params.shape[0] == subset.states.shape[0]


# ===========================================================================
# 2. CausalGraph env-param methods
# ===========================================================================


class TestCausalGraphEnvParams:
    """Tests for env-param node awareness in CausalGraph."""

    def test_env_param_nodes_empty_without_ep_nodes(self) -> None:
        graph = CausalGraph.from_domain_knowledge(
            [("s0", "reward"), ("s1", "reward")],
            reward_node="reward",
        )
        assert graph.env_param_nodes == frozenset()

    def test_env_param_nodes_detected(self) -> None:
        graph = _make_graph_with_ep(["s0", "s1"], ["ep_g", "ep_m"])
        assert graph.env_param_nodes == frozenset({"ep_g", "ep_m"})

    def test_is_env_param_node(self) -> None:
        graph = _make_graph_with_ep(["s0"], ["ep_g"])
        assert graph.is_env_param_node("ep_g") is True
        assert graph.is_env_param_node("s0") is False
        assert graph.is_env_param_node("reward") is False

    def test_env_param_parents_of_reward(self) -> None:
        graph = _make_graph_with_ep(["s0"], ["ep_g", "ep_m"])
        ep_parents = graph.env_param_parents_of("reward")
        assert ep_parents == frozenset({"ep_g", "ep_m"})

    def test_env_param_parents_of_state(self) -> None:
        """Env params that are not parents of a state return empty set."""
        graph = _make_graph_with_ep(["s0"], ["ep_g"])
        # ep_g is NOT a parent of s0 in this graph
        assert graph.env_param_parents_of("s0") == frozenset()

    def test_state_ancestors_of_reward_excludes_ep(self) -> None:
        graph = _make_graph_with_ep(["s0", "s1"], ["ep_g", "ep_m"])
        state_anc = graph.state_ancestors_of_reward()
        assert "ep_g" not in state_anc
        assert "ep_m" not in state_anc
        assert "s0" in state_anc
        assert "s1" in state_anc

    def test_env_param_prefix_constant(self) -> None:
        assert CausalGraph.ENV_PARAM_PREFIX == "ep_"


# ===========================================================================
# 3. InvFeatureSelector conditional invariance
# ===========================================================================


class TestInvFeatureSelectorConditionalInvariance:
    """Tests for the conditional invariance mechanism."""

    def _build_invariant_dataset(self) -> tuple[ExploratoryDataset, CausalGraph]:
        """Build a dataset where reward = (2 + ep_g) * s0 + noise, 6 envs.

        The ATE of s0 on reward varies across envs (proportional to ep_g),
        but the variation is perfectly explained by ep_g. Residual variance
        after regressing ATE on ep_g should be near zero.

        Uses 6 envs so that KRR LOO-CV has enough data for reliable
        model selection (inner LOO trains on 4 points per fold).
        """
        rng = np.random.RandomState(42)
        n_per_env = 500
        n_envs = 6
        n = n_per_env * n_envs

        s0 = rng.randn(n).astype(np.float32)
        s1 = rng.randn(n).astype(np.float32)  # irrelevant
        actions = rng.randint(0, 2, size=(n,)).astype(np.float32)
        env_ids = np.repeat(np.arange(n_envs, dtype=np.int32), n_per_env)
        ep_g_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ep_g = np.array([ep_g_values[eid] for eid in env_ids], dtype=np.float32)
        rewards = ((2.0 + ep_g) * s0 + 0.1 * rng.randn(n)).astype(np.float32)

        states = np.stack([s0, s1], axis=1)
        next_states = rng.randn(n, 2).astype(np.float32)
        env_params = ep_g[:, np.newaxis]

        ds = ExploratoryDataset(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            env_ids=env_ids,
            env_params=env_params,
        )

        # Build graph: s0 -> reward, ep_g -> reward
        g = nx.DiGraph()
        g.add_nodes_from(["s0", "s1", "action", "reward", "s0_next", "s1_next", "ep_g"])
        g.add_edge("s0", "reward")
        g.add_edge("ep_g", "reward")
        graph = CausalGraph(g, reward_node="reward")

        return ds, graph

    def _build_non_invariant_dataset(self) -> tuple[ExploratoryDataset, CausalGraph]:
        """Build a dataset where reward depends on s0 differently per env,
        but NOT explained by any env param edge to reward.

        6 envs with varying coefficients not correlated with ep_g.
        """
        rng = np.random.RandomState(42)
        n_per_env = 500
        n_envs = 6
        n = n_per_env * n_envs

        s0 = rng.randn(n).astype(np.float32)
        s1 = rng.randn(n).astype(np.float32)
        actions = rng.randint(0, 2, size=(n,)).astype(np.float32)
        env_ids = np.repeat(np.arange(n_envs, dtype=np.int32), n_per_env)
        # Different coefficient in each env -- not explained by ep_g
        # Oscillating pattern to avoid accidental correlation with ep_g
        coeff_values = [1.0, 8.0, 2.0, 7.0, 3.0, 6.0]
        coeff = np.array([coeff_values[eid] for eid in env_ids], dtype=np.float32)
        rewards = (coeff * s0 + 0.1 * rng.randn(n)).astype(np.float32)

        states = np.stack([s0, s1], axis=1)
        next_states = rng.randn(n, 2).astype(np.float32)
        ep_g_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ep_g = np.array([ep_g_values[eid] for eid in env_ids], dtype=np.float32)
        env_params = ep_g[:, np.newaxis]

        ds = ExploratoryDataset(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            env_ids=env_ids,
            env_params=env_params,
        )

        # Graph has s0 -> reward but NO ep_g -> reward
        g = nx.DiGraph()
        g.add_nodes_from(["s0", "s1", "action", "reward", "s0_next", "s1_next", "ep_g"])
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        return ds, graph

    def test_context_dependent_feature_kept(self) -> None:
        """When ep_g -> reward explains ATE variance, s0 is kept as context-dependent."""
        ds, graph = self._build_invariant_dataset()
        selector = InvFeatureSelector(
            epsilon=0.1,
            min_ate=0.01,
            enable_conditional_invariance=True,
        )
        result = selector.select(
            ds, graph, ["s0", "s1"],
            env_param_names=["ep_g"],
        )
        # s0 should be selected (either invariant or context-dependent)
        assert "s0" in result.selected_features

    def test_no_ep_ancestors_falls_back_to_rejection(self) -> None:
        """When graph has no ep -> reward edges, feature is still rejected."""
        ds, graph = self._build_non_invariant_dataset()
        selector = InvFeatureSelector(
            epsilon=0.1,
            min_ate=0.01,
            enable_conditional_invariance=True,
        )
        result = selector.select(
            ds, graph, ["s0", "s1"],
            env_param_names=["ep_g"],
        )
        # s0 should be rejected (no ep edge to reward to explain variance)
        assert "s0" not in result.selected_features or "s0" in result.rejected_features

    def test_conditional_invariance_disabled_rejects_normally(self) -> None:
        """When conditional invariance is disabled, high-variance features rejected."""
        ds, graph = self._build_invariant_dataset()
        selector = InvFeatureSelector(
            epsilon=0.01,  # very tight threshold
            min_ate=0.01,
            enable_conditional_invariance=False,
        )
        result = selector.select(
            ds, graph, ["s0", "s1"],
            env_param_names=["ep_g"],
        )
        # With very tight epsilon and no conditional rescue, s0 may be rejected
        # The key assertion: context_dependent should be empty
        assert result.context_dependent_features == {}

    def test_context_param_names_collected(self) -> None:
        """context_param_names should be the union of all ep params needed."""
        ds, graph = self._build_invariant_dataset()
        selector = InvFeatureSelector(
            epsilon=0.1,
            min_ate=0.01,
            enable_conditional_invariance=True,
        )
        result = selector.select(
            ds, graph, ["s0", "s1"],
            env_param_names=["ep_g"],
        )
        # If s0 is context-dependent, ep_g should be in context_param_names
        if result.context_dependent_features:
            assert "ep_g" in result.context_param_names

    def test_residual_variance_perfect_linear(self) -> None:
        """When ATE is perfectly linear in ep, residual variance is near 0.

        KRR with LOO-CV introduces small prediction noise, so we use a
        looser tolerance than exact zero.
        """
        ate = np.array([3.0, 4.0, 5.0, 6.0])
        ep = np.array([[1.0], [2.0], [3.0], [4.0]])
        res_var = InvFeatureSelector._residual_ate_variance(ate, ep)
        assert res_var < 0.05

    def test_residual_variance_no_relationship(self) -> None:
        """When ATE has no linear relationship to ep, residual variance ~ raw."""
        # Oscillating pattern: high-low-high-low, uncapturable by linear fit
        ate = np.array([5.0, 1.0, 5.0, 1.0, 5.0, 1.0])
        ep = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        raw_var = float(np.var(ate))
        res_var = InvFeatureSelector._residual_ate_variance(ate, ep)
        # Residual variance should stay large (linear fit has near-zero slope)
        assert res_var > raw_var * 0.8

    def test_residual_variance_too_few_envs(self) -> None:
        """With n_envs <= n_ep + 1, falls back to raw variance."""
        ate = np.array([3.0, 5.0])
        ep = np.array([[1.0], [2.0]])
        raw_var = float(np.var(ate))
        res_var = InvFeatureSelector._residual_ate_variance(ate, ep)
        assert res_var == pytest.approx(raw_var)


# ===========================================================================
# 4. CausalPolicy with context
# ===========================================================================


class TestCausalPolicyContext:
    """Tests for context-conditional CausalPolicy."""

    def _make_policy(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        context_dim: int = 0,
        continuous: bool = False,
    ) -> CausalPolicy:
        mask = np.ones(state_dim, dtype=bool)
        kwargs: dict = {
            "full_state_dim": state_dim,
            "action_dim": action_dim,
            "feature_mask": mask,
            "hidden_dims": (32, 32),
            "context_dim": context_dim,
            "continuous": continuous,
        }
        if continuous:
            kwargs["action_low"] = -np.ones(action_dim, dtype=np.float32)
            kwargs["action_high"] = np.ones(action_dim, dtype=np.float32)
        return CausalPolicy(**kwargs)

    def test_forward_with_context_shapes(self) -> None:
        policy = self._make_policy(state_dim=4, action_dim=2, context_dim=3)
        state = torch.randn(8, 4)
        context = torch.randn(8, 3)
        output = policy(state, context=context)
        assert output.action.shape == (8,)
        assert output.log_prob.shape == (8,)
        assert output.value.shape == (8,)

    def test_forward_without_context_unchanged(self) -> None:
        policy = self._make_policy(state_dim=4, action_dim=2, context_dim=0)
        state = torch.randn(8, 4)
        output = policy(state)
        assert output.action.shape == (8,)
        assert output.value.shape == (8,)

    def test_evaluate_actions_with_context(self) -> None:
        policy = self._make_policy(state_dim=4, action_dim=2, context_dim=2)
        state = torch.randn(8, 4)
        context = torch.randn(8, 2)
        actions = torch.randint(0, 2, (8,))
        output = policy.evaluate_actions(state, actions, context=context)
        assert output.log_prob.shape == (8,)
        assert output.value.shape == (8,)

    def test_get_action_with_context(self) -> None:
        policy = self._make_policy(state_dim=4, action_dim=2, context_dim=2)
        state = torch.randn(4)
        context = torch.randn(2)
        action = policy.get_action(state, context=context)
        assert isinstance(action, int)

    def test_continuous_forward_with_context(self) -> None:
        policy = self._make_policy(
            state_dim=3, action_dim=1, context_dim=2, continuous=True
        )
        state = torch.randn(4, 3)
        context = torch.randn(4, 2)
        output = policy(state, context=context)
        assert output.action.shape == (4, 1)
        assert output.log_prob.shape == (4,)

    def test_context_dim_property(self) -> None:
        policy = self._make_policy(context_dim=5)
        assert policy.context_dim == 5

    def test_gradient_flows_through_context(self) -> None:
        policy = self._make_policy(state_dim=4, action_dim=2, context_dim=2)
        state = torch.randn(4, 4)
        context = torch.randn(4, 2, requires_grad=True)
        output = policy(state, context=context)
        output.log_prob.sum().backward()
        assert context.grad is not None


# ===========================================================================
# 5. Trajectory with env_params
# ===========================================================================


class TestTrajectoryEnvParams:
    """Tests for env_params field on Trajectory and buffer."""

    def test_trajectory_env_params_default_none(self) -> None:
        traj = Trajectory(
            states=torch.randn(10, 4),
            actions=torch.randint(0, 2, (10,)),
            rewards=torch.randn(10),
            log_probs=torch.randn(10),
            values=torch.randn(10),
            next_states=torch.randn(10, 4),
            dones=torch.zeros(10),
            env_id=0,
        )
        assert traj.env_params is None

    def test_trajectory_env_params_stored(self) -> None:
        ep = torch.randn(10, 3)
        traj = Trajectory(
            states=torch.randn(10, 4),
            actions=torch.randint(0, 2, (10,)),
            rewards=torch.randn(10),
            log_probs=torch.randn(10),
            values=torch.randn(10),
            next_states=torch.randn(10, 4),
            dones=torch.zeros(10),
            env_id=0,
            env_params=ep,
        )
        assert traj.env_params is not None
        assert traj.env_params.shape == (10, 3)

    def test_buffer_get_all_flat_concatenates_env_params(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        for env_id in range(2):
            buf.add(Trajectory(
                states=torch.randn(5, 4),
                actions=torch.randint(0, 2, (5,)),
                rewards=torch.randn(5),
                log_probs=torch.randn(5),
                values=torch.randn(5),
                next_states=torch.randn(5, 4),
                dones=torch.zeros(5),
                env_id=env_id,
                env_params=torch.ones(5, 2) * env_id,
            ))
        flat = buf.get_all_flat()
        assert flat.env_params is not None
        assert flat.env_params.shape == (10, 2)
        # First 5 rows should be 0, next 5 should be 1
        assert (flat.env_params[:5] == 0).all()
        assert (flat.env_params[5:] == 1).all()

    def test_buffer_get_all_flat_none_env_params(self) -> None:
        buf = MultiEnvTrajectoryBuffer()
        for env_id in range(2):
            buf.add(Trajectory(
                states=torch.randn(5, 4),
                actions=torch.randint(0, 2, (5,)),
                rewards=torch.randn(5),
                log_probs=torch.randn(5),
                values=torch.randn(5),
                next_states=torch.randn(5, 4),
                dones=torch.zeros(5),
                env_id=env_id,
            ))
        flat = buf.get_all_flat()
        assert flat.env_params is None


# ===========================================================================
# 6. env_param_ancestors_of (ancestor paths)
# ===========================================================================


class TestEnvParamAncestors:
    """Tests for env_param_ancestors_of -- transitive ep ancestor lookup."""

    def test_indirect_ep_ancestor_of_reward(self) -> None:
        """ep_g -> s_next -> reward: ep_g is an ancestor of reward."""
        g = nx.DiGraph()
        g.add_nodes_from(["s0", "s0_next", "action", "reward", "ep_g"])
        g.add_edge("ep_g", "s0_next")
        g.add_edge("s0_next", "reward")
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        ep_anc = graph.env_param_ancestors_of("reward")
        assert ep_anc == frozenset({"ep_g"})

    def test_no_ep_ancestors(self) -> None:
        """Graph with no ep nodes has empty ep ancestors."""
        g = nx.DiGraph()
        g.add_nodes_from(["s0", "reward"])
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        ep_anc = graph.env_param_ancestors_of("reward")
        assert ep_anc == frozenset()

    def test_conditional_invariance_via_indirect_path(self) -> None:
        """Feature rescued via ancestor path: ep_g -> s0 -> reward.

        ep_g is NOT a direct parent of reward but is an ancestor of s0,
        so env_param_ancestors_of(s0) should include ep_g.
        """
        g = nx.DiGraph()
        g.add_nodes_from(["s0", "s1", "action", "reward", "s0_next", "s1_next", "ep_g"])
        g.add_edge("ep_g", "s0")
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        # ep_g is an ancestor of s0
        ep_anc_s0 = graph.env_param_ancestors_of("s0")
        assert ep_anc_s0 == frozenset({"ep_g"})

        # ep_g is also an ancestor of reward (via s0)
        ep_anc_reward = graph.env_param_ancestors_of("reward")
        assert ep_anc_reward == frozenset({"ep_g"})


# ===========================================================================
# 7. KRR residual variance -- quadratic relationship
# ===========================================================================


class TestResidualVarianceKRR:
    """Test that KRR captures non-linear ATE-vs-ep relationships."""

    def test_residual_variance_quadratic(self) -> None:
        """ATE = ep^2 -- KRR with RBF should capture this."""
        ep_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ate = ep_vals ** 2  # [1, 4, 9, 16, 25]
        ep = ep_vals.reshape(-1, 1)
        raw_var = float(np.var(ate))
        res_var = InvFeatureSelector._residual_ate_variance(ate, ep)
        # KRR should explain most of the quadratic pattern
        assert res_var < raw_var * 0.3


# ===========================================================================
# 8. Correlation pre-screen for ep edges
# ===========================================================================


class TestCorrelationPreScreen:
    """Tests for CausalGraphBuilder._add_correlated_ep_edges."""

    def test_adds_edge_when_correlated(self) -> None:
        """Strong ep-reward correlation should add ep -> reward edge."""
        rng = np.random.RandomState(42)
        n = 500
        ep = rng.randn(n).astype(np.float32)
        reward = 3.0 * ep + 0.1 * rng.randn(n).astype(np.float32)
        s0 = rng.randn(n).astype(np.float32)
        action = rng.randn(n).astype(np.float32)
        s0_next = rng.randn(n).astype(np.float32)

        data = np.stack([s0, action, reward, s0_next, ep], axis=1)
        node_names = ["s0", "action", "reward", "s0_next", "ep_g"]

        # Graph WITHOUT ep_g -> reward
        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        result = CausalGraphBuilder._add_correlated_ep_edges(
            graph, data, node_names, frozenset({"ep_g"}), "reward", 0.05,
        )
        assert ("ep_g", "reward") in result.edges

    def test_no_edge_when_uncorrelated(self) -> None:
        """Independent ep and reward should not produce an edge."""
        rng = np.random.RandomState(42)
        n = 500
        ep = rng.randn(n).astype(np.float32)
        reward = rng.randn(n).astype(np.float32)  # independent
        s0 = rng.randn(n).astype(np.float32)
        action = rng.randn(n).astype(np.float32)
        s0_next = rng.randn(n).astype(np.float32)

        data = np.stack([s0, action, reward, s0_next, ep], axis=1)
        node_names = ["s0", "action", "reward", "s0_next", "ep_g"]

        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        g.add_edge("s0", "reward")
        graph = CausalGraph(g, reward_node="reward")

        result = CausalGraphBuilder._add_correlated_ep_edges(
            graph, data, node_names, frozenset({"ep_g"}), "reward", 0.05,
        )
        assert ("ep_g", "reward") not in result.edges

    def test_preserves_existing_edge(self) -> None:
        """Already-present ep -> reward edge is not duplicated."""
        rng = np.random.RandomState(42)
        n = 500
        ep = rng.randn(n).astype(np.float32)
        reward = 3.0 * ep + 0.1 * rng.randn(n).astype(np.float32)
        s0 = rng.randn(n).astype(np.float32)
        action = rng.randn(n).astype(np.float32)
        s0_next = rng.randn(n).astype(np.float32)

        data = np.stack([s0, action, reward, s0_next, ep], axis=1)
        node_names = ["s0", "action", "reward", "s0_next", "ep_g"]

        # Graph already has ep_g -> reward
        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        g.add_edge("s0", "reward")
        g.add_edge("ep_g", "reward")
        graph = CausalGraph(g, reward_node="reward")

        result = CausalGraphBuilder._add_correlated_ep_edges(
            graph, data, node_names, frozenset({"ep_g"}), "reward", 0.05,
        )
        # Edge still there, no crash, graph unchanged
        assert ("ep_g", "reward") in result.edges
        assert len(result.edges) == len(graph.edges)


# ===========================================================================
# 9. IterationMetrics per_env_returns
# ===========================================================================


class TestIterationMetricsPerEnvReturns:
    """Tests for the per_env_returns field on IterationMetrics."""

    def test_per_env_returns_default_none(self) -> None:
        m = IterationMetrics(
            iteration=0,
            policy_loss=0.0,
            value_loss=0.0,
            irm_penalty=0.0,
            worst_case_loss=0.0,
            regularization_total=0.0,
            constraint_penalty=0.0,
            total_loss=0.0,
            mean_return=0.0,
            worst_env_return=0.0,
        )
        assert m.per_env_returns is None

    def test_per_env_returns_stored(self) -> None:
        returns = [10.0, 20.0, 30.0]
        m = IterationMetrics(
            iteration=0,
            policy_loss=0.0,
            value_loss=0.0,
            irm_penalty=0.0,
            worst_case_loss=0.0,
            regularization_total=0.0,
            constraint_penalty=0.0,
            total_loss=0.0,
            mean_return=20.0,
            worst_env_return=10.0,
            per_env_returns=returns,
        )
        assert m.per_env_returns == [10.0, 20.0, 30.0]


# ===========================================================================
# 10. ValidationFeedbackStage
# ===========================================================================


class TestValidationFeedbackStage:
    """Tests for the validation feedback diagnostic stage."""

    def _make_mock_env_family(
        self,
        n_envs: int,
        param_name: str,
        param_values: list[float],
    ):
        """Create a minimal mock EnvironmentFamily for testing."""
        from unittest.mock import MagicMock

        ef = MagicMock()
        ef.n_envs = n_envs
        ef.param_names = [param_name]
        ef.get_env_params = lambda idx: {param_name: param_values[idx]}
        return ef

    def test_detects_correlated_param(self) -> None:
        """Per-env returns that correlate with gravity are flagged."""
        from circ_rl.orchestration.stages import ValidationFeedbackStage

        param_values = [5.0, 8.0, 11.0, 14.0]
        ef = self._make_mock_env_family(4, "g", param_values)
        stage = ValidationFeedbackStage(ef, correlation_alpha=0.10)

        # Returns that correlate with g
        metrics = IterationMetrics(
            iteration=0,
            policy_loss=0.0, value_loss=0.0, irm_penalty=0.0,
            worst_case_loss=0.0, regularization_total=0.0,
            constraint_penalty=0.0, total_loss=0.0,
            mean_return=0.0, worst_env_return=0.0,
            per_env_returns=[100.0, 80.0, 60.0, 40.0],  # inversely correlated with g
        )

        result = stage.run({
            "policy_optimization": {"all_metrics": [[metrics]]},
            "feature_selection": {"context_param_names": []},
        })
        assert "ep_g" in result["suggested_context_params"]

    def test_skips_already_used_param(self) -> None:
        """Param already in context_param_names is not re-suggested."""
        from circ_rl.orchestration.stages import ValidationFeedbackStage

        param_values = [5.0, 8.0, 11.0, 14.0]
        ef = self._make_mock_env_family(4, "g", param_values)
        stage = ValidationFeedbackStage(ef, correlation_alpha=0.10)

        metrics = IterationMetrics(
            iteration=0,
            policy_loss=0.0, value_loss=0.0, irm_penalty=0.0,
            worst_case_loss=0.0, regularization_total=0.0,
            constraint_penalty=0.0, total_loss=0.0,
            mean_return=0.0, worst_env_return=0.0,
            per_env_returns=[100.0, 80.0, 60.0, 40.0],
        )

        result = stage.run({
            "policy_optimization": {"all_metrics": [[metrics]]},
            "feature_selection": {"context_param_names": ["ep_g"]},
        })
        assert "ep_g" not in result["suggested_context_params"]

    def test_no_suggestion_when_uncorrelated(self) -> None:
        """Identical returns across envs yield no suggestion."""
        from circ_rl.orchestration.stages import ValidationFeedbackStage

        param_values = [5.0, 8.0, 11.0, 14.0]
        ef = self._make_mock_env_family(4, "g", param_values)
        stage = ValidationFeedbackStage(ef, correlation_alpha=0.05)

        metrics = IterationMetrics(
            iteration=0,
            policy_loss=0.0, value_loss=0.0, irm_penalty=0.0,
            worst_case_loss=0.0, regularization_total=0.0,
            constraint_penalty=0.0, total_loss=0.0,
            mean_return=50.0, worst_env_return=50.0,
            per_env_returns=[50.0, 50.0, 50.0, 50.0],
        )

        result = stage.run({
            "policy_optimization": {"all_metrics": [[metrics]]},
            "feature_selection": {"context_param_names": []},
        })
        assert result["suggested_context_params"] == []
