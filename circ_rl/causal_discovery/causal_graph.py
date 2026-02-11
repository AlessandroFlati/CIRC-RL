"""CausalGraph: wrapper around networkx DiGraph for causal DAG operations."""

from __future__ import annotations

import networkx as nx
import numpy as np


class CausalGraph:
    """Wrapper around a networkx DiGraph representing a causal DAG.

    Provides causal-specific queries: parents, ancestors, Markov blanket,
    d-separation, and reward-specific accessors. The graph may be partially
    directed (CPDAG) when produced by constraint-based discovery algorithms.

    :param graph: A networkx DiGraph with string-named nodes.
    :param reward_node: Name of the reward node in the graph.
    :raises ValueError: If the graph contains cycles.

    See ``CIRC-RL_Framework.md`` Section 2.1 for the causal graph formalization.
    """

    REWARD_NODE_DEFAULT = "reward"

    def __init__(
        self,
        graph: nx.DiGraph,
        reward_node: str = REWARD_NODE_DEFAULT,
    ) -> None:
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Causal graph must be a DAG (directed acyclic graph), "
                "but the provided graph contains cycles."
            )
        if reward_node not in graph.nodes:
            raise ValueError(
                f"Reward node '{reward_node}' not found in graph. "
                f"Available nodes: {list(graph.nodes)}"
            )

        self._graph = graph.copy()
        self._reward_node = reward_node

    @property
    def graph(self) -> nx.DiGraph:
        """Return a copy of the underlying networkx DiGraph."""
        return self._graph.copy()

    @property
    def nodes(self) -> frozenset[str]:
        """Return all node names."""
        return frozenset(self._graph.nodes)

    @property
    def edges(self) -> frozenset[tuple[str, str]]:
        """Return all directed edges as (parent, child) tuples."""
        return frozenset(self._graph.edges)

    @property
    def reward_node(self) -> str:
        """Return the name of the reward node."""
        return self._reward_node

    def parents(self, node: str) -> frozenset[str]:
        """Return the direct parents (predecessors) of a node.

        :param node: Target node name.
        :returns: Set of parent node names.
        :raises KeyError: If node is not in the graph.
        """
        self._check_node(node)
        return frozenset(self._graph.predecessors(node))

    def children(self, node: str) -> frozenset[str]:
        """Return the direct children (successors) of a node.

        :param node: Target node name.
        :returns: Set of child node names.
        :raises KeyError: If node is not in the graph.
        """
        self._check_node(node)
        return frozenset(self._graph.successors(node))

    def ancestors(self, node: str) -> frozenset[str]:
        """Return all ancestors of a node (transitive predecessors).

        :param node: Target node name.
        :returns: Set of ancestor node names (excludes the node itself).
        :raises KeyError: If node is not in the graph.
        """
        self._check_node(node)
        return frozenset(nx.ancestors(self._graph, node))

    def descendants(self, node: str) -> frozenset[str]:
        """Return all descendants of a node (transitive successors).

        :param node: Target node name.
        :returns: Set of descendant node names (excludes the node itself).
        :raises KeyError: If node is not in the graph.
        """
        self._check_node(node)
        return frozenset(nx.descendants(self._graph, node))

    def is_ancestor(self, node: str, target: str) -> bool:
        """Check if ``node`` is an ancestor of ``target``.

        :param node: Potential ancestor.
        :param target: Target node.
        :returns: True if there is a directed path from node to target.
        :raises KeyError: If either node is not in the graph.
        """
        self._check_node(node)
        self._check_node(target)
        return nx.has_path(self._graph, node, target)

    def get_markov_blanket(self, node: str) -> frozenset[str]:
        """Return the Markov blanket of a node.

        The Markov blanket consists of: parents, children, and parents of
        children (co-parents / spouses).

        :param node: Target node name.
        :returns: Set of nodes in the Markov blanket.
        :raises KeyError: If node is not in the graph.
        """
        self._check_node(node)
        blanket: set[str] = set()
        blanket.update(self._graph.predecessors(node))
        for child in self._graph.successors(node):
            blanket.add(child)
            blanket.update(self._graph.predecessors(child))
        blanket.discard(node)
        return frozenset(blanket)

    def causal_parents_of_reward(self) -> frozenset[str]:
        r"""Return :math:`\text{Pa}_{\mathcal{G}}(R)` -- direct parents of reward.

        Implements the reward parent identification from
        ``CIRC-RL_Framework.md`` Section 3.6, Phase 1, step 3.

        :returns: Set of node names that are direct parents of the reward node.
        """
        return self.parents(self._reward_node)

    def ancestors_of_reward(self) -> frozenset[str]:
        r"""Return :math:`\text{Anc}_{\mathcal{G}}(R)` -- all ancestors of reward.

        Used in Phase 2 (feature selection) to identify candidate features.
        See ``CIRC-RL_Framework.md`` Section 3.6, Phase 2, step 1.

        :returns: Set of node names that are ancestors of the reward node.
        """
        return self.ancestors(self._reward_node)

    def to_adjacency_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Convert the graph to an adjacency matrix.

        :returns: Tuple of (adjacency_matrix, node_names) where
            ``adjacency_matrix[i, j] = 1`` means node i -> node j.
        """
        node_names = sorted(self._graph.nodes)
        adj = nx.to_numpy_array(self._graph, nodelist=node_names)
        return adj.astype(np.int32), node_names

    @classmethod
    def from_adjacency_matrix(
        cls,
        adj: np.ndarray,
        node_names: list[str],
        reward_node: str = REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Construct a CausalGraph from an adjacency matrix.

        :param adj: Square adjacency matrix where ``adj[i, j] = 1`` means
            node_names[i] -> node_names[j].
        :param node_names: Names for each node (must match matrix dimensions).
        :param reward_node: Name of the reward node.
        :returns: A new CausalGraph instance.
        :raises ValueError: If matrix dimensions don't match node_names length.
        """
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"Adjacency matrix must be square, got shape {adj.shape}"
            )
        if adj.shape[0] != len(node_names):
            raise ValueError(
                f"Adjacency matrix size {adj.shape[0]} does not match "
                f"number of node names {len(node_names)}"
            )

        graph = nx.DiGraph()
        graph.add_nodes_from(node_names)
        for i, src in enumerate(node_names):
            for j, dst in enumerate(node_names):
                if adj[i, j] != 0:
                    graph.add_edge(src, dst)

        return cls(graph, reward_node=reward_node)

    @classmethod
    def from_domain_knowledge(
        cls,
        edges: list[tuple[str, str]],
        reward_node: str = REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Construct a CausalGraph from a list of known causal edges.

        :param edges: List of (parent, child) tuples representing causal edges.
        :param reward_node: Name of the reward node.
        :returns: A new CausalGraph instance.
        :raises ValueError: If the resulting graph has cycles.
        """
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        return cls(graph, reward_node=reward_node)

    def _check_node(self, node: str) -> None:
        """Raise KeyError if node is not in the graph."""
        if node not in self._graph.nodes:
            raise KeyError(
                f"Node '{node}' not found in graph. "
                f"Available nodes: {sorted(self._graph.nodes)}"
            )

    def __repr__(self) -> str:
        return (
            f"CausalGraph(nodes={len(self._graph.nodes)}, "
            f"edges={len(self._graph.edges)}, "
            f"reward_node='{self._reward_node}')"
        )
