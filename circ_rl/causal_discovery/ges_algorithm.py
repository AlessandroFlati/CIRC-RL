"""GES algorithm for score-based causal discovery.

Implements the Greedy Equivalence Search algorithm for learning causal
structure by optimizing a scoring function (BIC or MDL) over DAG space.

Reference: Chickering (2002). *Optimal Structure Identification With
Greedy Search*. JMLR.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from loguru import logger

from circ_rl.causal_discovery.causal_graph import CausalGraph


class GESAlgorithm:
    """Greedy Equivalence Search for score-based causal discovery.

    The algorithm proceeds in two phases:

    1. **Forward phase**: starting from the empty graph, greedily add
       single edges that most improve the BIC score.
    2. **Backward phase**: greedily remove single edges that most improve
       the BIC score.

    :param score_fn: Scoring function to use (``"bic"`` or ``"mdl"``).
    :param max_parents: Maximum number of parents per node. ``None`` means
        no limit.

    See ``CIRC-RL_Framework.md`` Section 7.1.
    """

    def __init__(
        self,
        score_fn: str = "bic",
        max_parents: int | None = None,
    ) -> None:
        if score_fn not in ("bic", "mdl"):
            raise ValueError(f"score_fn must be 'bic' or 'mdl', got '{score_fn}'")
        self._score_fn = score_fn
        self._max_parents = max_parents

    def fit(
        self,
        data: np.ndarray,
        node_names: list[str],
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Run the GES algorithm on observational data.

        :param data: Data matrix of shape ``(n_samples, n_variables)``.
        :param node_names: Names for each variable (column).
        :param reward_node: Name of the reward node.
        :returns: A CausalGraph representing the learned structure.
        :raises ValueError: If data dimensions don't match node_names.
        """
        if data.shape[1] != len(node_names):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(node_names)} "
                f"node names provided"
            )

        n_vars = len(node_names)
        logger.info(
            "Running GES algorithm: {} samples, {} variables, score={}",
            data.shape[0],
            n_vars,
            self._score_fn,
        )

        graph = nx.DiGraph()
        graph.add_nodes_from(node_names)
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        # Forward phase: greedily add edges
        graph = self._forward_phase(data, graph, node_names, name_to_idx)
        logger.info(
            "Forward phase complete: {} edges",
            graph.number_of_edges(),
        )

        # Backward phase: greedily remove edges
        graph = self._backward_phase(data, graph, node_names, name_to_idx)
        logger.info(
            "Backward phase complete: {} edges",
            graph.number_of_edges(),
        )

        return CausalGraph(graph, reward_node=reward_node)

    def _forward_phase(
        self,
        data: np.ndarray,
        graph: nx.DiGraph,
        node_names: list[str],
        name_to_idx: dict[str, int],
    ) -> nx.DiGraph:
        """Greedily add edges that improve the score."""
        current_score = self._graph_score(data, graph, name_to_idx)

        while True:
            best_edge: tuple[str, str] | None = None
            best_score = current_score

            for src, dst in combinations(node_names, 2):
                for u, v in [(src, dst), (dst, src)]:
                    if graph.has_edge(u, v):
                        continue

                    if self._max_parents is not None:
                        n_parents = len(list(graph.predecessors(v)))
                        if n_parents >= self._max_parents:
                            continue

                    graph.add_edge(u, v)
                    if nx.is_directed_acyclic_graph(graph):
                        score = self._graph_score(data, graph, name_to_idx)
                        if score > best_score:
                            best_score = score
                            best_edge = (u, v)
                    graph.remove_edge(u, v)

            if best_edge is None:
                break

            graph.add_edge(best_edge[0], best_edge[1])
            current_score = best_score

        return graph

    def _backward_phase(
        self,
        data: np.ndarray,
        graph: nx.DiGraph,
        node_names: list[str],
        name_to_idx: dict[str, int],
    ) -> nx.DiGraph:
        """Greedily remove edges that improve the score."""
        current_score = self._graph_score(data, graph, name_to_idx)

        while True:
            best_edge: tuple[str, str] | None = None
            best_score = current_score

            for u, v in list(graph.edges()):
                graph.remove_edge(u, v)
                score = self._graph_score(data, graph, name_to_idx)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
                graph.add_edge(u, v)

            if best_edge is None:
                break

            graph.remove_edge(best_edge[0], best_edge[1])
            current_score = best_score

        return graph

    def _graph_score(
        self,
        data: np.ndarray,
        graph: nx.DiGraph,
        name_to_idx: dict[str, int],
    ) -> float:
        """Compute the total graph score (sum of local BIC scores per node).

        Higher is better.

        :returns: Total BIC score (negative BIC, so higher = better fit with penalty).
        """
        total = 0.0
        for node in graph.nodes():
            node_idx = name_to_idx[node]
            parent_idxs = [name_to_idx[p] for p in graph.predecessors(node)]
            total += self._local_bic_score(data, node_idx, parent_idxs)
        return total

    @staticmethod
    def _local_bic_score(
        data: np.ndarray,
        node_idx: int,
        parent_idxs: list[int],
    ) -> float:
        """Compute local BIC score for a node given its parents.

        .. math::

            \\text{BIC}_\\text{local}(X_i \\mid \\text{Pa}_i) =
            -\\frac{n}{2} \\ln(\\hat{\\sigma}^2) - \\frac{k}{2} \\ln(n)

        where :math:`\\hat{\\sigma}^2` is the residual variance from
        regressing :math:`X_i` on its parents, k is the number of
        parameters, and n is the sample size.

        :returns: Local BIC score (higher is better).
        """
        n_samples = data.shape[0]
        target = data[:, node_idx]  # (n_samples,)

        if len(parent_idxs) == 0:
            residual_var = float(np.var(target))
        else:
            parents = data[:, parent_idxs]  # (n_samples, n_parents)
            parents_aug = np.column_stack(
                [np.ones(n_samples), parents]
            )  # (n_samples, n_parents+1)
            coeffs, _, _, _ = np.linalg.lstsq(parents_aug, target, rcond=None)
            residuals = target - parents_aug @ coeffs
            residual_var = float(np.var(residuals))

        residual_var = max(residual_var, 1e-10)
        k = len(parent_idxs) + 1  # parameters: parents + intercept
        bic = -0.5 * n_samples * np.log(residual_var) - 0.5 * k * np.log(n_samples)
        return float(bic)
