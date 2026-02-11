"""Tests for circ_rl.causal_discovery.causal_graph."""

import networkx as nx
import numpy as np
import pytest

from circ_rl.causal_discovery.causal_graph import CausalGraph


@pytest.fixture
def chain_graph() -> CausalGraph:
    """A -> B -> C -> reward."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "reward")])
    return CausalGraph(g)


@pytest.fixture
def diamond_graph() -> CausalGraph:
    """A -> B, A -> C, B -> reward, C -> reward."""
    g = nx.DiGraph()
    g.add_edges_from([
        ("A", "B"),
        ("A", "C"),
        ("B", "reward"),
        ("C", "reward"),
    ])
    return CausalGraph(g)


@pytest.fixture
def complex_graph() -> CausalGraph:
    """
    X1 -> X2 -> X4 -> reward
    X1 -> X3 -> X4
    X5 (isolated node, not ancestor of reward)
    """
    g = nx.DiGraph()
    g.add_edges_from([
        ("X1", "X2"),
        ("X1", "X3"),
        ("X2", "X4"),
        ("X3", "X4"),
        ("X4", "reward"),
    ])
    g.add_node("X5")
    return CausalGraph(g)


class TestCausalGraphBasicQueries:
    def test_parents_of_reward_in_chain(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.parents("reward") == frozenset({"C"})

    def test_parents_of_root(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.parents("A") == frozenset()

    def test_children_of_a(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.children("A") == frozenset({"B"})

    def test_ancestors_of_reward_in_chain(
        self, chain_graph: CausalGraph
    ) -> None:
        assert chain_graph.ancestors("reward") == frozenset({"A", "B", "C"})

    def test_descendants_of_a_in_chain(
        self, chain_graph: CausalGraph
    ) -> None:
        assert chain_graph.descendants("A") == frozenset(
            {"B", "C", "reward"}
        )

    def test_is_ancestor(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.is_ancestor("A", "reward")
        assert not chain_graph.is_ancestor("reward", "A")


class TestCausalGraphRewardMethods:
    def test_causal_parents_of_reward_chain(
        self, chain_graph: CausalGraph
    ) -> None:
        assert chain_graph.causal_parents_of_reward() == frozenset({"C"})

    def test_causal_parents_of_reward_diamond(
        self, diamond_graph: CausalGraph
    ) -> None:
        assert diamond_graph.causal_parents_of_reward() == frozenset(
            {"B", "C"}
        )

    def test_ancestors_of_reward_chain(
        self, chain_graph: CausalGraph
    ) -> None:
        assert chain_graph.ancestors_of_reward() == frozenset({"A", "B", "C"})

    def test_ancestors_of_reward_complex(
        self, complex_graph: CausalGraph
    ) -> None:
        ancestors = complex_graph.ancestors_of_reward()
        assert ancestors == frozenset({"X1", "X2", "X3", "X4"})
        assert "X5" not in ancestors


class TestMarkovBlanket:
    def test_markov_blanket_middle_node(
        self, chain_graph: CausalGraph
    ) -> None:
        # B's blanket: parent=A, child=C, no co-parents
        assert chain_graph.get_markov_blanket("B") == frozenset({"A", "C"})

    def test_markov_blanket_with_co_parents(
        self, diamond_graph: CausalGraph
    ) -> None:
        # B's blanket: parent=A, child=reward, co-parent of reward=C
        blanket = diamond_graph.get_markov_blanket("B")
        assert blanket == frozenset({"A", "reward", "C"})


class TestCausalGraphConstruction:
    def test_from_domain_knowledge(self) -> None:
        edges = [("X", "Y"), ("Y", "reward")]
        g = CausalGraph.from_domain_knowledge(edges)
        assert g.parents("reward") == frozenset({"Y"})
        assert g.ancestors("reward") == frozenset({"X", "Y"})

    def test_from_adjacency_matrix_roundtrip(
        self, chain_graph: CausalGraph
    ) -> None:
        adj, names = chain_graph.to_adjacency_matrix()
        reconstructed = CausalGraph.from_adjacency_matrix(adj, names)

        assert reconstructed.nodes == chain_graph.nodes
        assert reconstructed.edges == chain_graph.edges

    def test_from_adjacency_matrix_rejects_dimension_mismatch(self) -> None:
        adj = np.zeros((3, 3), dtype=np.int32)
        with pytest.raises(ValueError, match="does not match"):
            CausalGraph.from_adjacency_matrix(adj, ["A", "B"])

    def test_from_adjacency_matrix_rejects_non_square(self) -> None:
        adj = np.zeros((3, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="square"):
            CausalGraph.from_adjacency_matrix(adj, ["A", "B", "C"])

    def test_rejects_cyclic_graph(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "reward")])
        with pytest.raises(ValueError, match="cycles"):
            CausalGraph(g)

    def test_rejects_missing_reward_node(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        with pytest.raises(ValueError, match="not found"):
            CausalGraph(g)

    def test_nonexistent_node_raises_key_error(
        self, chain_graph: CausalGraph
    ) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            chain_graph.parents("nonexistent")


class TestCausalGraphProperties:
    def test_nodes(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.nodes == frozenset({"A", "B", "C", "reward"})

    def test_edges(self, chain_graph: CausalGraph) -> None:
        assert chain_graph.edges == frozenset({
            ("A", "B"),
            ("B", "C"),
            ("C", "reward"),
        })

    def test_repr(self, chain_graph: CausalGraph) -> None:
        r = repr(chain_graph)
        assert "nodes=4" in r
        assert "edges=3" in r
        assert "reward" in r
