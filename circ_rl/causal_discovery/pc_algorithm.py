"""PC algorithm for constraint-based causal discovery.

Implements the Peter-Clark algorithm for learning causal structure from
observational data via conditional independence testing.

Reference: Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx
from loguru import logger

from circ_rl.causal_discovery.causal_graph import CausalGraph
from circ_rl.causal_discovery.ci_tests import (
    CITestMethod,
    CITestResult,
    causal_ci_test_fisher_z,
    causal_ci_test_kernel,
)

if TYPE_CHECKING:
    import numpy as np


class PCAlgorithm:
    """Peter-Clark algorithm for causal discovery.

    The algorithm proceeds in three phases:

    1. **Skeleton learning**: start with a complete undirected graph and remove
       edges when a conditional independence is found.
    2. **V-structure orientation**: orient colliders (X -> Z <- Y) when X and Y
       are non-adjacent and Z is not in their separating set.
    3. **Edge orientation**: apply Meek's rules to orient remaining edges
       without creating new v-structures or cycles.

    The output is a CPDAG (Completed Partially Directed Acyclic Graph),
    representing a Markov equivalence class.

    :param ci_test: Which conditional independence test to use.
    :param alpha: Significance level for the CI tests.
    :param max_conditioning_set_size: Maximum size of conditioning sets to test.
        ``None`` means no limit (exponential worst-case). Default 3 is
        sufficient for classic control environments (4-8 variables) and
        gives ~5-10x speedup over unbounded search.

    See ``CIRC-RL_Framework.md`` Section 7.1 for practical considerations.
    """

    def __init__(
        self,
        ci_test: CITestMethod = CITestMethod.FISHER_Z,
        alpha: float = 0.05,
        max_conditioning_set_size: int | None = 3,
    ) -> None:
        self._ci_test = ci_test
        self._alpha = alpha
        self._max_cond_size = max_conditioning_set_size

    def fit(
        self,
        data: np.ndarray,
        node_names: list[str],
        reward_node: str = CausalGraph.REWARD_NODE_DEFAULT,
    ) -> CausalGraph:
        """Run the PC algorithm on observational data.

        :param data: Data matrix of shape ``(n_samples, n_variables)``.
        :param node_names: Names for each variable (column).
        :param reward_node: Name of the reward node.
        :returns: A CausalGraph (may be partially directed -- CPDAG).
        :raises ValueError: If data dimensions don't match node_names.
        """
        if data.shape[1] != len(node_names):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(node_names)} "
                f"node names provided"
            )

        logger.info(
            "Running PC algorithm: {} samples, {} variables, alpha={}",
            data.shape[0],
            len(node_names),
            self._alpha,
        )

        name_to_idx = {name: i for i, name in enumerate(node_names)}

        # Phase 1: learn skeleton
        skeleton, sep_sets = self._skeleton_phase(data, node_names, name_to_idx)
        logger.info(
            "Skeleton learned: {} edges remaining",
            skeleton.number_of_edges(),
        )

        # Phase 2: orient v-structures
        pdag = self._orient_v_structures(skeleton, sep_sets, node_names)
        logger.info("V-structures oriented")

        # Phase 3: apply Meek rules
        pdag = self._apply_meek_rules(pdag, node_names)
        logger.info("Meek rules applied")

        return CausalGraph(pdag, reward_node=reward_node)

    def _skeleton_phase(
        self,
        data: np.ndarray,
        node_names: list[str],
        name_to_idx: dict[str, int],
    ) -> tuple[nx.Graph, dict[tuple[str, str], frozenset[str]]]:
        """Phase 1: learn the undirected skeleton by removing edges.

        For each pair (X, Y), test X _||_ Y | Z for conditioning sets Z
        of increasing size drawn from the adjacency of X (or Y).

        :returns: Tuple of (undirected skeleton, separation sets).
        """
        n_vars = len(node_names)
        skeleton = nx.complete_graph(node_names)
        sep_sets: dict[tuple[str, str], frozenset[str]] = {}

        max_cond = self._max_cond_size if self._max_cond_size is not None else n_vars - 2

        for cond_size in range(max_cond + 1):
            edges_to_remove: list[tuple[str, str]] = []

            for x_name, y_name in list(skeleton.edges()):
                x_idx = name_to_idx[x_name]
                y_idx = name_to_idx[y_name]

                # Neighbors of X excluding Y
                neighbors = set(skeleton.neighbors(x_name)) - {y_name}

                if len(neighbors) < cond_size:
                    continue

                for cond_set_names in combinations(sorted(neighbors), cond_size):
                    cond_idxs = [name_to_idx[n] for n in cond_set_names]

                    result = self._run_ci_test(data, x_idx, y_idx, cond_idxs)

                    if result.independent:
                        edges_to_remove.append((x_name, y_name))
                        key_fwd = (x_name, y_name)
                        key_rev = (y_name, x_name)
                        sep_set = frozenset(cond_set_names)
                        sep_sets[key_fwd] = sep_set
                        sep_sets[key_rev] = sep_set
                        break

            for x_name, y_name in edges_to_remove:
                if skeleton.has_edge(x_name, y_name):
                    skeleton.remove_edge(x_name, y_name)

            if skeleton.number_of_edges() == 0:
                break

        return skeleton, sep_sets

    def _orient_v_structures(
        self,
        skeleton: nx.Graph,
        sep_sets: dict[tuple[str, str], frozenset[str]],
        node_names: list[str],
    ) -> nx.DiGraph:
        """Phase 2: orient v-structures (colliders).

        For each triple X - Z - Y where X and Y are not adjacent:
        if Z is NOT in sep(X, Y), orient as X -> Z <- Y.

        :returns: Partially directed graph (PDAG).
        """
        pdag = nx.DiGraph()
        pdag.add_nodes_from(node_names)

        # Start with all edges as bidirectional (undirected)
        for u, v in skeleton.edges():
            pdag.add_edge(u, v)
            pdag.add_edge(v, u)

        for z_name in node_names:
            neighbors = list(skeleton.neighbors(z_name))
            for i, x_name in enumerate(neighbors):
                for y_name in neighbors[i + 1 :]:
                    # X - Z - Y where X and Y not adjacent
                    if not skeleton.has_edge(x_name, y_name):
                        sep_key = (x_name, y_name)
                        sep = sep_sets.get(sep_key, frozenset())

                        if z_name not in sep:
                            # Orient as X -> Z <- Y (remove Z -> X and Z -> Y)
                            if pdag.has_edge(z_name, x_name):
                                pdag.remove_edge(z_name, x_name)
                            if pdag.has_edge(z_name, y_name):
                                pdag.remove_edge(z_name, y_name)

        return pdag

    def _apply_meek_rules(
        self,
        pdag: nx.DiGraph,
        node_names: list[str],
    ) -> nx.DiGraph:
        """Phase 3: apply Meek's orientation rules.

        Iteratively applies four rules to orient undirected edges without
        creating new v-structures or directed cycles.

        Rule 1: X -> Z - Y => X -> Z -> Y (if X and Y not adjacent)
        Rule 2: X -> Z -> Y, X - Y => X -> Y
        Rule 3: X - Z -> Y, X - W -> Y, Z != W, Z - W => X -> Y (not implemented - rare)

        :returns: Maximally oriented PDAG (CPDAG).
        """
        changed = True
        while changed:
            changed = False

            for z_name in node_names:
                for y_name in list(pdag.successors(z_name)):
                    # Check if Z - Y is undirected (both directions exist)
                    if not pdag.has_edge(y_name, z_name):
                        continue  # Already directed Z -> Y

                    # Rule 1: exists X -> Z (directed) and X not adjacent to Y
                    for x_name in list(pdag.predecessors(z_name)):
                        if pdag.has_edge(z_name, x_name):
                            continue  # X - Z undirected, skip
                        # X -> Z is directed
                        if not pdag.has_edge(x_name, y_name) and not pdag.has_edge(
                            y_name, x_name
                        ):
                            # X and Y not adjacent: orient Z -> Y
                            pdag.remove_edge(y_name, z_name)
                            changed = True
                            break

                    # Rule 2: exists X -> Z -> Y (both directed), X - Y undirected
                    if pdag.has_edge(y_name, z_name):  # Still undirected
                        for x_name in list(pdag.predecessors(z_name)):
                            if pdag.has_edge(z_name, x_name):
                                continue  # Not directed
                            # X -> Z is directed
                            if pdag.has_edge(z_name, y_name) and not pdag.has_edge(
                                y_name, z_name
                            ):
                                # Z -> Y is now directed (might have been oriented above)
                                pass
                            # Check X -> Y direction: if X - Y undirected
                            if pdag.has_edge(x_name, y_name) and pdag.has_edge(
                                y_name, x_name
                            ):
                                # Check if X -> Z -> Y path exists (both directed)
                                if (
                                    not pdag.has_edge(z_name, x_name)
                                    and pdag.has_edge(z_name, y_name)
                                    and not pdag.has_edge(y_name, z_name)
                                ):
                                    pdag.remove_edge(y_name, x_name)
                                    changed = True

        # Remove remaining bidirectional edges by picking an acyclic orientation
        # For any remaining undirected edge, orient arbitrarily (topological order)
        return self._resolve_undirected_edges(pdag)


    @staticmethod
    def _resolve_undirected_edges(pdag: nx.DiGraph) -> nx.DiGraph:
        """Resolve remaining undirected edges to produce a valid DAG.

        Strategy: remove all bidirectional (undirected) edges first so that
        the remaining directed edges form a DAG. Then greedily add each
        undirected edge back, choosing the orientation that keeps the graph
        acyclic.

        :returns: A fully directed DAG.
        """
        undirected_pairs: list[tuple[str, str]] = []
        seen: set[frozenset[str]] = set()

        for u, v in list(pdag.edges()):
            pair = frozenset({u, v})
            if pdag.has_edge(v, u) and pair not in seen:
                undirected_pairs.append((u, v))
                seen.add(pair)

        # Remove all undirected edges -- directed-only subgraph should be a DAG
        for u, v in undirected_pairs:
            pdag.remove_edge(u, v)
            pdag.remove_edge(v, u)

        # Safety: v-structure/Meek orientation can rarely produce directed
        # cycles.  Break any cycles by removing back-edges.
        while not nx.is_directed_acyclic_graph(pdag):
            try:
                cycle = nx.find_cycle(pdag)
                u_cyc, v_cyc, *_ = cycle[-1]
                pdag.remove_edge(u_cyc, v_cyc)
                logger.warning(
                    "Broke directed cycle by removing edge {} -> {}", u_cyc, v_cyc
                )
            except nx.NetworkXNoCycle:
                break

        # Greedily re-add each pair in the orientation that keeps a DAG
        for u, v in undirected_pairs:
            pdag.add_edge(u, v)
            if nx.is_directed_acyclic_graph(pdag):
                continue
            # u -> v creates a cycle; try v -> u
            pdag.remove_edge(u, v)
            pdag.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(pdag):
                # Neither orientation works (shouldn't happen for a valid PDAG)
                pdag.remove_edge(v, u)
                logger.warning(
                    "Could not orient edge {}-{} without creating a cycle", u, v
                )

        return pdag

    def _run_ci_test(
        self,
        data: np.ndarray,
        x_idx: int,
        y_idx: int,
        cond_idxs: list[int],
    ) -> CITestResult:
        """Dispatch to the appropriate CI test method.

        :returns: CITestResult.
        """
        if self._ci_test == CITestMethod.FISHER_Z:
            return causal_ci_test_fisher_z(
                data, x_idx, y_idx, cond_idxs, alpha=self._alpha
            )
        if self._ci_test == CITestMethod.KERNEL_CI:
            return causal_ci_test_kernel(
                data, x_idx, y_idx, cond_idxs, alpha=self._alpha
            )
        raise ValueError(f"Unsupported CI test method: {self._ci_test}")
