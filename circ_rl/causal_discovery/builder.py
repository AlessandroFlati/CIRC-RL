"""CausalGraphBuilder: high-level facade for causal graph construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.causal_discovery.ci_tests import CITestMethod
from circ_rl.causal_discovery.fci_algorithm import FCIAlgorithm
from circ_rl.causal_discovery.ges_algorithm import GESAlgorithm
from circ_rl.causal_discovery.pc_algorithm import PCAlgorithm

if TYPE_CHECKING:
    from circ_rl.environments.data_collector import ExploratoryDataset


def _distance_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 200,
    alpha: float = 0.05,
) -> bool:
    """Test for non-linear association using distance covariance.

    Distance covariance (dCov) equals zero iff X and Y are independent.
    It detects arbitrary non-linear relationships that Pearson
    correlation misses (e.g., Y = X^2 when X is symmetric).

    Uses a permutation test: permute Y, recompute dCov, count how
    often the permuted statistic exceeds the observed one.

    :param x: First variable, shape ``(n,)``.
    :param y: Second variable, shape ``(n,)``.
    :param n_permutations: Number of permutations.
    :param alpha: Significance level.
    :returns: True if association is significant (reject independence).
    """
    n = len(x)
    # Pairwise distance matrices
    a = np.abs(x[:, None] - x[None, :])  # (n, n)
    b = np.abs(y[:, None] - y[None, :])  # (n, n)

    # Double center A
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()

    # Double center B and compute observed dCov^2
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    observed = float((A * B).sum())  # proportional to dCov^2 * n^2

    # Permutation test
    rng = np.random.RandomState(42)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        b_perm = b[np.ix_(perm, perm)]
        B_perm = (
            b_perm
            - b_perm.mean(axis=0, keepdims=True)
            - b_perm.mean(axis=1, keepdims=True)
            + b_perm.mean()
        )
        perm_stat = float((A * B_perm).sum())
        if perm_stat >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return p_value < alpha


class CausalGraphBuilder:
    """High-level API for causal graph construction.

    Supports data-driven discovery, domain knowledge, and hybrid approaches.
    This is the primary entry point for Phase 1 of the CIRC-RL pipeline.

    See ``CIRC-RL_Framework.md`` Section 3.6, Phase 1.
    """

    @staticmethod
    def discover(
        dataset: ExploratoryDataset,
        node_names: list[str],
        method: str = "pc",
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
        env_param_names: list[str] | None = None,
        state_feature_names: list[str] | None = None,
        nonlinear_state_reward_screen: bool = False,
        **kwargs: Any,
    ) -> CausalGraph:
        """Discover causal structure from observational data.

        :param dataset: Exploratory data collected from environment family.
        :param node_names: Names for each variable column in the data.
            When ``env_param_names`` is provided, ``node_names`` must already
            include the ``ep_``-prefixed env-param node names at the end.
        :param method: Discovery algorithm: ``"pc"``, ``"ges"``, or ``"fci"``.
        :param reward_node: Name of the reward node.
        :param env_param_names: Names of env-param nodes (``ep_``-prefixed).
            When provided, the augmented flat array (with env-param columns)
            is used, and structural constraints are enforced: no non-ep node
            can be a parent of an ep node (env params are exogenous).
        :param state_feature_names: Names of state feature nodes. Required
            when ``nonlinear_state_reward_screen`` is True.
        :param nonlinear_state_reward_screen: When True, adds state -> reward
            edges for features with significant non-linear association with
            reward (using distance correlation). This catches relationships
            like ``s^2 -> reward`` that the Fisher-z CI test misses.
        :param kwargs: Additional arguments passed to the algorithm constructor.
        :returns: Discovered CausalGraph.
        :raises ValueError: If method is not recognized.
        """
        # Pop ep-specific kwargs before they reach the algorithm constructor
        ep_correlation_threshold: float = kwargs.pop(
            "ep_correlation_threshold", 0.05
        )

        if env_param_names:
            data = dataset.to_flat_array_with_env_params()
        else:
            data = dataset.to_flat_array()

        if method == "pc":
            ci_test_str = kwargs.pop("ci_test", "fisher_z")
            ci_test = CITestMethod(ci_test_str)
            algorithm = PCAlgorithm(ci_test=ci_test, **kwargs)
        elif method == "ges":
            algorithm = GESAlgorithm(**kwargs)  # type: ignore[assignment]
        elif method == "fci":
            ci_test_str = kwargs.pop("ci_test", "fisher_z")
            ci_test = CITestMethod(ci_test_str)
            algorithm = FCIAlgorithm(ci_test=ci_test, **kwargs)  # type: ignore[assignment]
        else:
            raise ValueError(
                f"Unknown discovery method '{method}'. "
                f"Supported: 'pc', 'ges', 'fci'"
            )

        graph = algorithm.fit(data, node_names, reward_node=reward_node)

        # Enforce structural constraint: env params are exogenous.
        # Flip any discovered edges from non-ep nodes into ep nodes.
        if env_param_names:
            graph = CausalGraphBuilder._enforce_exogenous_env_params(
                graph, frozenset(env_param_names), reward_node
            )
            # Add ep -> reward edges for significantly correlated pairs
            # that PC may have missed.
            graph = CausalGraphBuilder._add_correlated_ep_edges(
                graph, data, node_names, frozenset(env_param_names),
                reward_node, ep_correlation_threshold,
            )

        # Add state -> reward edges for non-linearly associated features
        if nonlinear_state_reward_screen and state_feature_names:
            graph = CausalGraphBuilder._add_nonlinear_state_reward_edges(
                graph, data, node_names, state_feature_names, reward_node,
            )

        return graph

    @staticmethod
    def _enforce_exogenous_env_params(
        graph: CausalGraph,
        env_param_names: frozenset[str],
        reward_node: str,
    ) -> CausalGraph:
        """Flip edges from non-env-param nodes into env-param nodes.

        Environment parameters are exogenous: they cannot be caused by
        state, action, or reward variables within an episode. When the
        discovery algorithm orients a skeleton edge as ``non_ep -> ep``,
        we know the direction is wrong. We flip it to ``ep -> non_ep``
        (the only valid direction for that skeleton edge).

        :param graph: Discovered causal graph.
        :param env_param_names: Set of env-param node names (``ep_``-prefixed).
        :param reward_node: Name of the reward node.
        :returns: A new CausalGraph with forbidden edges flipped.
        """
        g = graph.graph  # already a copy
        edges_to_flip = [
            (src, dst)
            for src, dst in g.edges
            if dst in env_param_names and src not in env_param_names
        ]
        if edges_to_flip:
            from loguru import logger

            logger.info(
                "Flipping {} forbidden edges into env-param nodes: {}",
                len(edges_to_flip),
                edges_to_flip,
            )
            for src, dst in edges_to_flip:
                g.remove_edge(src, dst)
                # Add the reverse (correct) direction if not already present
                if not g.has_edge(dst, src):
                    g.add_edge(dst, src)

        return CausalGraph(g, reward_node=reward_node)

    @staticmethod
    def _add_correlated_ep_edges(
        graph: CausalGraph,
        data: np.ndarray,
        node_names: list[str],
        env_param_names: frozenset[str],
        reward_node: str,
        correlation_threshold: float = 0.05,
    ) -> CausalGraph:
        """Add ep -> reward edges for significantly correlated ep-reward pairs.

        PC may miss ep -> reward edges due to alpha sensitivity, sample size,
        or hash-dependent orientation. This post-hoc screen uses Pearson
        correlation on the pooled data as a safety net: if an ep param is
        significantly correlated with reward, we add the edge (since the
        exogenous constraint guarantees the only valid direction is ep -> reward).

        :param graph: Causal graph after exogenous constraint enforcement.
        :param data: Pooled flat array, shape ``(n_samples, n_vars)``.
        :param node_names: Variable names matching data columns.
        :param env_param_names: Set of env-param node names (``ep_``-prefixed).
        :param reward_node: Name of the reward node.
        :param correlation_threshold: p-value threshold for Pearson test.
        :returns: A new CausalGraph with added ep -> reward edges.
        """
        from scipy import stats

        name_to_idx = {name: i for i, name in enumerate(node_names)}
        reward_idx = name_to_idx.get(reward_node)
        if reward_idx is None:
            return graph

        reward_col = data[:, reward_idx]

        g = graph.graph  # already a copy
        edges_added: list[tuple[str, str]] = []

        for ep_name in sorted(env_param_names):
            ep_idx = name_to_idx.get(ep_name)
            if ep_idx is None:
                continue

            # Already have ep -> reward edge
            if g.has_edge(ep_name, reward_node):
                continue

            ep_col = data[:, ep_idx]

            # Guard: skip constant columns (zero variance breaks pearsonr)
            if np.std(ep_col) < 1e-12:
                continue

            _corr, p_value = stats.pearsonr(ep_col, reward_col)

            if p_value < correlation_threshold:
                g.add_edge(ep_name, reward_node)
                edges_added.append((ep_name, reward_node))

        if edges_added:
            from loguru import logger

            logger.info(
                "Correlation pre-screen added {} ep->reward edges: {}",
                len(edges_added),
                edges_added,
            )

        return CausalGraph(g, reward_node=reward_node)

    @staticmethod
    def _add_nonlinear_state_reward_edges(
        graph: CausalGraph,
        data: np.ndarray,
        node_names: list[str],
        state_feature_names: list[str],
        reward_node: str,
        alpha: float = 0.05,
        max_samples: int = 2000,
        n_permutations: int = 200,
    ) -> CausalGraph:
        """Add state -> reward edges for non-linearly associated features.

        Uses distance correlation with a permutation test to detect
        non-linear associations that the Fisher-z (linear) CI test misses.
        For example, if reward depends on ``s^2``, the Pearson correlation
        with ``s`` is zero (when ``s`` is symmetric around 0), but the
        distance correlation is non-zero.

        :param graph: Causal graph (possibly missing state -> reward edges).
        :param data: Pooled flat array, shape ``(n_samples, n_vars)``.
        :param node_names: Variable names matching data columns.
        :param state_feature_names: Names of state feature nodes.
        :param reward_node: Name of the reward node.
        :param alpha: Significance level for permutation test.
        :param max_samples: Maximum samples for distance correlation.
        :param n_permutations: Number of permutations.
        :returns: A new CausalGraph with added state -> reward edges.
        """
        import networkx as nx
        from loguru import logger

        name_to_idx = {name: i for i, name in enumerate(node_names)}
        reward_idx = name_to_idx.get(reward_node)
        if reward_idx is None:
            return graph

        n_samples = data.shape[0]
        # Subsample for O(n^2) distance computation
        if n_samples > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples, max_samples, replace=False)
            data_sub = data[idx]
        else:
            data_sub = data

        reward_col = data_sub[:, reward_idx]

        g = graph.graph  # copy
        edges_added: list[tuple[str, str]] = []

        for feat_name in sorted(state_feature_names):
            feat_idx = name_to_idx.get(feat_name)
            if feat_idx is None:
                continue
            # Already has edge
            if g.has_edge(feat_name, reward_node):
                continue

            feat_col = data_sub[:, feat_idx]
            significant = _distance_correlation_test(
                feat_col, reward_col, n_permutations, alpha,
            )
            if significant:
                g.add_edge(feat_name, reward_node)
                if not nx.is_directed_acyclic_graph(g):
                    # Adding this edge would create a cycle -- skip it
                    g.remove_edge(feat_name, reward_node)
                    logger.debug(
                        "Non-linear pre-screen: skipping {} -> {} (would create cycle)",
                        feat_name, reward_node,
                    )
                else:
                    edges_added.append((feat_name, reward_node))

        if edges_added:
            logger.info(
                "Non-linear pre-screen added {} state->reward edges: {}",
                len(edges_added),
                edges_added,
            )

        return CausalGraph(g, reward_node=reward_node)

    @staticmethod
    def from_domain_knowledge(
        edges: list[tuple[str, str]],
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Construct a causal graph from expert-specified edges.

        :param edges: List of (parent, child) tuples.
        :param reward_node: Name of the reward node.
        :returns: CausalGraph built from the specified edges.
        """
        return CausalGraph.from_domain_knowledge(edges, reward_node=reward_node)

    @staticmethod
    def hybrid(
        dataset: ExploratoryDataset,
        node_names: list[str],
        required_edges: list[tuple[str, str]],
        forbidden_edges: list[tuple[str, str]],
        method: str = "pc",
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
        **kwargs: Any,
    ) -> CausalGraph:
        """Data-driven discovery constrained by domain knowledge.

        First discovers the graph from data, then enforces hard constraints:
        required edges are added if missing, forbidden edges are removed.

        :param dataset: Exploratory data.
        :param node_names: Variable names.
        :param required_edges: Edges that must be present (domain knowledge).
        :param forbidden_edges: Edges that must not be present.
        :param method: Discovery algorithm.
        :param reward_node: Name of the reward node.
        :param kwargs: Additional algorithm arguments.
        :returns: Constrained CausalGraph.
        :raises ValueError: If enforcing constraints creates a cycle.
        """
        graph = CausalGraphBuilder.discover(
            dataset, node_names, method=method, reward_node=reward_node, **kwargs
        )


        g = graph.graph

        # Remove forbidden edges
        for src, dst in forbidden_edges:
            if g.has_edge(src, dst):
                g.remove_edge(src, dst)

        # Add required edges
        for src, dst in required_edges:
            if not g.has_edge(src, dst):
                g.add_edge(src, dst)

        return CausalGraph(g, reward_node=reward_node)
