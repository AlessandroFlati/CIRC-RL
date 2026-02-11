"""CausalGraphBuilder: high-level facade for causal graph construction."""

from __future__ import annotations

from typing import Any

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
        **kwargs: Any,
    ) -> CausalGraph:
        """Discover causal structure from observational data.

        :param dataset: Exploratory data collected from environment family.
        :param node_names: Names for each variable column in the data.
        :param method: Discovery algorithm: ``"pc"``, ``"ges"``, or ``"fci"``.
        :param reward_node: Name of the reward node.
        :param kwargs: Additional arguments passed to the algorithm constructor.
        :returns: Discovered CausalGraph.
        :raises ValueError: If method is not recognized.
        """
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

        return algorithm.fit(data, node_names, reward_node=reward_node)

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
