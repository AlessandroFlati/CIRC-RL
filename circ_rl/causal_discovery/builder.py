"""CausalGraphBuilder: high-level facade for causal graph construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.causal_discovery.ci_tests import CITestMethod
from circ_rl.causal_discovery.fci_algorithm import FCIAlgorithm
from circ_rl.causal_discovery.ges_algorithm import GESAlgorithm
from circ_rl.causal_discovery.pc_algorithm import PCAlgorithm
from circ_rl.environments.data_collector import ExploratoryDataset


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

        import networkx as nx

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
