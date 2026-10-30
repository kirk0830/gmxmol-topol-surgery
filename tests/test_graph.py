"""Graph-theoretic invariants used by topology surgery."""

from __future__ import annotations

from collections import deque
from itertools import permutations
from random import Random

import pytest

from topolsurgery import AtomicGraph


def test_graph_core_invariants() -> None:
    """The public graph has one immutable representation."""
    graph = AtomicGraph.from_edges(
        5,
        ((1, 2), (2, 1), (2, 3), (3, 4), (1, 4), (4, 5)),
        bond_orders={(2, 1): 2, (4, 5): 1},
    )

    assert graph.nodes == (1, 2, 3, 4, 5)
    assert graph.edges == ((1, 2), (2, 3), (3, 4), (1, 4), (4, 5))
    assert graph.components == ((1, 2, 3, 4, 5),)
    assert graph.connected
    assert graph.cycle_rank == 1
    assert graph.bond_order(1, 2) == 2
    assert graph.distance(1, 3) == 2
    assert graph.shortest_path(2, 5) in ((2, 1, 4, 5), (2, 3, 4, 5))
    assert graph.pairs_at_distance(3) == {(2, 5)}


def test_edit_operations_are_immutable_and_preserve_bond_orders() -> None:
    """Adding and removing graph edges never mutates the source graph."""
    graph = AtomicGraph.from_edges(
        4,
        ((1, 2), (2, 3)),
        bond_orders={(1, 2): 1},
    )

    joined = graph.with_edge(3, 4, bond_order=3)
    cut = graph.without_edge(1, 2)
    assert graph.edges == ((1, 2), (2, 3))
    assert joined.bond_order(3, 4) == 3
    assert cut.bond_order(1, 2) is None
    assert cut.edges == ((2, 3),)
    assert joined.components == ((1, 2, 3, 4),)


def test_graph_generates_exact_edge_local_angles_and_dihedrals() -> None:
    """The added edge owns exactly the graph paths that contain it."""
    graph = AtomicGraph.from_edges(5, ((1, 2), (2, 3), (4, 5)))
    joined = graph.with_edge(3, 4)

    assert joined.angles_through_edge((3, 4)) == {
        (2, 3, 4),
        (3, 4, 5),
    }
    assert joined.proper_dihedrals_through_edge((3, 4)) == {
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    }
    assert joined.proper_dihedrals() == {
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    }


def test_graph_remap_subgraph_and_errors() -> None:
    """Compaction and induced subgraphs retain only valid edges."""
    graph = AtomicGraph.from_edges(
        5,
        ((1, 2), (2, 3), (3, 4), (4, 5)),
        bond_orders={(2, 3): 2},
    )
    remapped = graph.remap({1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    subgraph = graph.subgraph((2, 3, 4))

    assert remapped.bond_order(2, 3) == 2
    assert subgraph.edges == ((2, 3), (3, 4))
    assert subgraph.bond_order(2, 3) == 2
    with pytest.raises(ValueError, match="self-loop"):
        AtomicGraph(2, ((1, 1),))
    with pytest.raises(ValueError, match="unknown"):
        AtomicGraph(2, ((1, 3),))
    with pytest.raises(ValueError, match="conflicting"):
        AtomicGraph(2, ((1, 2),), bond_orders={(1, 2): 1, (2, 1): 2})


def test_graph_query_and_metadata_endpoints_are_strict_integers() -> None:
    """Boolean and string endpoint coercion cannot alias graph node 1."""
    graph = AtomicGraph.from_edges(2, ((1, 2),))

    with pytest.raises(TypeError, match="nodes must be integers"):
        graph.has_node(True)
    with pytest.raises(TypeError, match="nodes must be integers"):
        graph.has_edge("1", 2)
    with pytest.raises(TypeError, match="endpoints must be integers"):
        AtomicGraph.from_edges(2, (), bond_orders={("1", 2): 1})


def test_random_graph_matches_independent_brute_force_graph() -> None:
    """Seeded random graphs agree with direct BFS and path enumeration."""
    rng = Random(20261030)
    for _case in range(30):
        nodes = tuple(range(1, 9))
        edges = tuple(
            (first, second)
            for first in nodes
            for second in range(first + 1, 9)
            if rng.random() < 0.3
        )
        graph = AtomicGraph.from_edges(nodes, edges)
        adjacency = {node: set() for node in nodes}
        for first, second in edges:
            adjacency[first].add(second)
            adjacency[second].add(first)

        reference_components = []
        visited = set()
        for start in nodes:
            if start in visited:
                continue
            queue = deque([start])
            visited.add(start)
            component = []
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbour in sorted(adjacency[node]):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
            reference_components.append(tuple(sorted(component)))
        assert graph.components == tuple(reference_components)

        for first in nodes:
            for second in nodes:
                reference_distance = _reference_distance(adjacency, first, second)
                assert graph.distance(first, second) == reference_distance
        assert graph.pairs_at_distance(3) == {
            (first, second)
            for first in nodes
            for second in range(first + 1, 9)
            if _reference_distance(adjacency, first, second) == 3
        }

        for edge in graph.edges:
            expected = set()
            for sequence in permutations(nodes, 4):
                if edge not in {
                    tuple(sorted(pair)) for pair in zip(sequence[:-1], sequence[1:])
                }:
                    continue
                if all(
                    second in adjacency[first]
                    for first, second in zip(sequence[:-1], sequence[1:])
                ):
                    expected.add(min(sequence, tuple(reversed(sequence))))
            assert graph.proper_dihedrals_through_edge(edge) == expected


def _reference_distance(
    adjacency: dict[int, set[int]],
    first: int,
    second: int,
) -> int | None:
    """Return an independently computed unweighted shortest distance."""
    if first == second:
        return 0
    visited = {first}
    queue = deque([(first, 0)])
    while queue:
        node, depth = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour == second:
                return depth + 1
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))
    return None
