"""FCI algorithm for causal discovery with latent confounders.

Implements the Fast Causal Inference algorithm, which extends the PC
algorithm to handle unobserved (latent) confounders by producing a
PAG (Partial Ancestral Graph).

Reference: Spirtes, Meek, Richardson (1999). *An Algorithm for Causal
Inference in the Presence of Latent Variables and Selection Bias*.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from loguru import logger

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.causal_discovery.ci_tests import (
    CITestMethod,
    causal_ci_test_fisher_z,
    causal_ci_test_kernel,
)


class FCIAlgorithm:
    """Fast Causal Inference algorithm for causal discovery with latent confounders.

    FCI extends the PC algorithm to handle settings where not all relevant
    variables are observed. Instead of a DAG or CPDAG, FCI produces a PAG
    (Partial Ancestral Graph) that represents an equivalence class of
    MAGs (Maximal Ancestral Graphs).

    The current implementation uses a simplified approach:
    1. Run PC-like skeleton discovery
    2. Orient v-structures
    3. Apply FCI-specific orientation rules (discriminating paths)

    .. note::
        This implementation produces an approximate PAG. For a fully
        correct FCI, discriminating path rules and additional orientation
        rules (Zhang 2008) would need to be added. This is documented as
        a known limitation.

    :param ci_test: Which conditional independence test to use.
    :param alpha: Significance level for the CI tests.

    See ``CIRC-RL_Framework.md`` Section 5.2 for limitations on
    causal discovery with latent confounders.
    """

    def __init__(
        self,
        ci_test: CITestMethod = CITestMethod.FISHER_Z,
        alpha: float = 0.05,
    ) -> None:
        self._ci_test = ci_test
        self._alpha = alpha

    def fit(
        self,
        data: np.ndarray,
        node_names: list[str],
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Run the FCI algorithm on observational data.

        :param data: Data matrix of shape ``(n_samples, n_variables)``.
        :param node_names: Names for each variable (column).
        :param reward_node: Name of the reward node.
        :returns: A CausalGraph representing the learned structure.
            Note: this is an approximation of the full PAG.
        :raises ValueError: If data dimensions don't match node_names.
        """
        if data.shape[1] != len(node_names):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(node_names)} "
                f"node names provided"
            )

        logger.info(
            "Running FCI algorithm: {} samples, {} variables, alpha={}",
            data.shape[0],
            len(node_names),
            self._alpha,
        )

        name_to_idx = {name: i for i, name in enumerate(node_names)}

        # Phase 1: skeleton discovery (same as PC)
        skeleton, sep_sets = self._skeleton_phase(data, node_names, name_to_idx)
        logger.info(
            "FCI skeleton: {} edges remaining",
            skeleton.number_of_edges(),
        )

        # Phase 2: orient v-structures
        pdag = self._orient_v_structures(skeleton, sep_sets, node_names)

        # Phase 3: additional FCI orientation (possible ancestors)
        # Simplified: re-run CI tests with larger conditioning sets
        pdag = self._possible_d_sep_phase(data, pdag, node_names, name_to_idx, sep_sets)

        # Resolve remaining undirected edges
        pdag = self._resolve_to_dag(pdag)

        logger.info(
            "FCI complete: {} directed edges",
            pdag.number_of_edges(),
        )

        return CausalGraph(pdag, reward_node=reward_node)

    def _skeleton_phase(
        self,
        data: np.ndarray,
        node_names: list[str],
        name_to_idx: dict[str, int],
    ) -> tuple[nx.Graph, dict[tuple[str, str], frozenset[str]]]:
        """Phase 1: learn undirected skeleton via CI tests.

        Same as PC skeleton phase.
        """
        skeleton = nx.complete_graph(node_names)
        sep_sets: dict[tuple[str, str], frozenset[str]] = {}
        n_vars = len(node_names)

        for cond_size in range(n_vars - 1):
            edges_to_remove: list[tuple[str, str]] = []

            for x_name, y_name in list(skeleton.edges()):
                x_idx = name_to_idx[x_name]
                y_idx = name_to_idx[y_name]
                neighbors = set(skeleton.neighbors(x_name)) - {y_name}

                if len(neighbors) < cond_size:
                    continue

                for cond_set_names in combinations(sorted(neighbors), cond_size):
                    cond_idxs = [name_to_idx[n] for n in cond_set_names]
                    result = self._run_ci_test(data, x_idx, y_idx, cond_idxs)

                    if result.independent:
                        edges_to_remove.append((x_name, y_name))
                        sep = frozenset(cond_set_names)
                        sep_sets[(x_name, y_name)] = sep
                        sep_sets[(y_name, x_name)] = sep
                        break

            for x_name, y_name in edges_to_remove:
                if skeleton.has_edge(x_name, y_name):
                    skeleton.remove_edge(x_name, y_name)

        return skeleton, sep_sets

    @staticmethod
    def _orient_v_structures(
        skeleton: nx.Graph,
        sep_sets: dict[tuple[str, str], frozenset[str]],
        node_names: list[str],
    ) -> nx.DiGraph:
        """Phase 2: orient v-structures (same as PC)."""
        pdag = nx.DiGraph()
        pdag.add_nodes_from(node_names)

        for u, v in skeleton.edges():
            pdag.add_edge(u, v)
            pdag.add_edge(v, u)

        for z_name in node_names:
            neighbors = list(skeleton.neighbors(z_name))
            for i, x_name in enumerate(neighbors):
                for y_name in neighbors[i + 1 :]:
                    if not skeleton.has_edge(x_name, y_name):
                        sep = sep_sets.get((x_name, y_name), frozenset())
                        if z_name not in sep:
                            if pdag.has_edge(z_name, x_name):
                                pdag.remove_edge(z_name, x_name)
                            if pdag.has_edge(z_name, y_name):
                                pdag.remove_edge(z_name, y_name)

        return pdag

    def _possible_d_sep_phase(
        self,
        data: np.ndarray,
        pdag: nx.DiGraph,
        node_names: list[str],
        name_to_idx: dict[str, int],
        sep_sets: dict[tuple[str, str], frozenset[str]],
    ) -> nx.DiGraph:
        """FCI-specific phase: test with possible-d-sep sets.

        In FCI, after initial orientation, we re-test edges using
        possible-d-sep(X, Y) as conditioning sets, which may be larger
        than the adjacency set used in skeleton discovery.

        This simplified implementation re-tests remaining undirected edges
        with all possible conditioning sets from the current neighbors.
        """
        edges_to_remove: list[tuple[str, str]] = []

        for u, v in list(pdag.edges()):
            if not pdag.has_edge(v, u):
                continue  # Already directed

            u_idx = name_to_idx[u]
            v_idx = name_to_idx[v]

            # possible-d-sep: union of neighbors in the partially directed graph
            pds = set()
            for node in pdag.predecessors(u):
                pds.add(node)
            for node in pdag.successors(u):
                pds.add(node)
            pds.discard(v)

            # Test with increasing conditioning set sizes from pds
            for cond_size in range(min(len(pds) + 1, 4)):
                found_independent = False
                for cond_names in combinations(sorted(pds), cond_size):
                    cond_idxs = [name_to_idx[n] for n in cond_names]
                    result = self._run_ci_test(data, u_idx, v_idx, cond_idxs)
                    if result.independent:
                        edges_to_remove.append((u, v))
                        edges_to_remove.append((v, u))
                        sep_sets[(u, v)] = frozenset(cond_names)
                        sep_sets[(v, u)] = frozenset(cond_names)
                        found_independent = True
                        break
                if found_independent:
                    break

        for u, v in edges_to_remove:
            if pdag.has_edge(u, v):
                pdag.remove_edge(u, v)

        return pdag

    @staticmethod
    def _resolve_to_dag(pdag: nx.DiGraph) -> nx.DiGraph:
        """Resolve remaining undirected edges to produce a DAG."""
        edges_to_resolve: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for u, v in list(pdag.edges()):
            if pdag.has_edge(v, u) and (v, u) not in seen:
                edges_to_resolve.append((u, v))
                seen.add((u, v))

        for u, v in edges_to_resolve:
            pdag.remove_edge(v, u)
            if not nx.is_directed_acyclic_graph(pdag):
                pdag.add_edge(v, u)
                pdag.remove_edge(u, v)
                if not nx.is_directed_acyclic_graph(pdag):
                    pdag.add_edge(u, v)

        return pdag

    def _run_ci_test(self, data: np.ndarray, x_idx: int, y_idx: int, cond_idxs: list[int]):  # type: ignore[no-untyped-def]
        """Dispatch to the CI test method."""
        if self._ci_test == CITestMethod.FISHER_Z:
            return causal_ci_test_fisher_z(data, x_idx, y_idx, cond_idxs, self._alpha)
        if self._ci_test == CITestMethod.KERNEL_CI:
            return causal_ci_test_kernel(data, x_idx, y_idx, cond_idxs, self._alpha)
        raise ValueError(f"Unsupported CI test: {self._ci_test}")
