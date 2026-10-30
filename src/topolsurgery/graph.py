"""Immutable graph-theoretic view of topology connectivity."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, order=True)
class AtomEdge:
    """One undirected atom connectivity edge.

    Parameters
    ----------
    first, second : int
        Distinct endpoint atom identifiers. They are stored in ascending
        order.
    order : int, optional
        Caller-owned chemical bond order from 1 to 4. This is graph
        metadata and does not change a GROMACS interaction function.
    """

    first: int
    second: int
    order: int | None = None

    def __post_init__(self) -> None:
        """Validate endpoints and canonicalize their order."""
        if (
            not isinstance(self.first, int)
            or isinstance(self.first, bool)
            or not isinstance(self.second, int)
            or isinstance(self.second, bool)
        ):
            raise TypeError("atom edge endpoints must be integers")
        if self.first == self.second:
            raise ValueError("atom edge endpoints must be distinct")
        if self.order is not None and (
            not isinstance(self.order, int)
            or isinstance(self.order, bool)
            or not 1 <= self.order <= 4
        ):
            raise ValueError("atom edge order must be an integer from 1 to 4")
        first, second = sorted((self.first, self.second))
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)

    @property
    def pair(self) -> tuple[int, int]:
        """Return the canonical endpoint pair."""
        return (self.first, self.second)


@dataclass(frozen=True, init=False)
class AtomicGraph:
    """Immutable undirected graph whose nodes are atoms.

    Edges come from ``[ bonds ]``, ``[ constraints ]``, and the implicit O-H
    dependencies of ``[ settles ]``. A pair that appears in multiple
    sections is one edge. Virtual-site ownership is deliberately kept out of
    this chemical graph and handled separately by component grouping.

    Parameters
    ----------
    nodes : Sequence[int]
        Unique atom identifiers. Topology sessions use contiguous 1-based
        numbers.
    edges : Iterable[tuple[int, int] | AtomEdge]
        Undirected edges. Endpoint orientation and duplicate edges are
        normalized while first-appearance order is retained.
    bond_orders : Mapping[tuple[int, int], int], optional
        Chemical bond orders keyed by any endpoint orientation. Every key
        must identify an edge in *edges*.
    """

    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    _adjacency: Mapping[int, frozenset[int]]
    _component_of: Mapping[int, int]
    _bond_orders: Mapping[tuple[int, int], int]
    _components: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        nodes: int | Sequence[int],
        edges: Iterable[tuple[int, int] | AtomEdge] = (),
        *,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
    ) -> None:
        normalized_nodes = _normalize_nodes(nodes)
        node_set = set(normalized_nodes)
        normalized_edges, orders = _normalize_edges(edges, node_set)
        _merge_bond_orders(normalized_edges, orders, bond_orders)

        adjacency = {node: set() for node in normalized_nodes}
        for first, second in normalized_edges:
            adjacency[first].add(second)
            adjacency[second].add(first)
        components = _breadth_first_components(normalized_nodes, adjacency)
        component_of = {
            node: index
            for index, component in enumerate(components)
            for node in component
        }

        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", tuple(normalized_edges))
        object.__setattr__(
            self,
            "_adjacency",
            MappingProxyType(
                {node: frozenset(neighbors) for node, neighbors in adjacency.items()}
            ),
        )
        object.__setattr__(self, "_component_of", MappingProxyType(component_of))
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_bond_orders", MappingProxyType(orders))

    @classmethod
    def from_edges(
        cls,
        nodes: int | Sequence[int],
        edges: Iterable[tuple[int, int] | AtomEdge] = (),
        *,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
    ) -> AtomicGraph:
        """Build a graph from an atom count or explicit node sequence."""
        if isinstance(nodes, int) and not isinstance(nodes, bool):
            if nodes < 1:
                raise ValueError("atomic graph node count must be positive")
            nodes = tuple(range(1, nodes + 1))
        return cls(nodes, edges, bond_orders=bond_orders)

    @classmethod
    def from_gromologist(
        cls,
        molecule: object,
        *,
        include_constraints: bool = True,
    ) -> AtomicGraph:
        """Build connectivity from a ``gromologist.SectionMol``-like object.

        Parameters
        ----------
        molecule : object
            Object exposing ``atoms`` and ``get_subsection``.
        include_constraints : bool
            Treat two-atom ``[ constraints ]`` rows as graph edges.
        """
        nodes = tuple(int(atom.num) for atom in molecule.atoms)
        headers = ["bonds"]
        if include_constraints:
            headers.append("constraints")
        edges: list[tuple[int, int]] = []
        for header in headers:
            try:
                subsection = molecule.get_subsection(header)
            except KeyError:
                continue
            edges.extend(
                _canonical_pair(tuple(entry.atom_numbers))
                for entry in subsection.entries_bonded
                if len(entry.atom_numbers) == 2
            )
        try:
            settles = molecule.get_subsection("settles")
        except KeyError:
            settles = None
        if settles is not None:
            for entry in settles.entries_bonded:
                if len(entry.atom_numbers) != 1:
                    raise ValueError(
                        "each [ settles ] row must name exactly one oxygen"
                    )
                oxygen = int(entry.atom_numbers[0])
                if oxygen < 1 or oxygen + 1 not in nodes or oxygen + 2 not in nodes:
                    raise ValueError(
                        f"SETTLE oxygen {oxygen} requires two following atoms"
                    )
                edges.extend(((oxygen, oxygen + 1), (oxygen, oxygen + 2)))
        return cls(nodes, edges)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AtomicGraph:
        """Restore a graph from :meth:`to_dict` output."""
        return cls(
            tuple(int(node) for node in data["nodes"]),
            (
                AtomEdge(
                    int(edge[0]),
                    int(edge[1]),
                    None if len(edge) < 3 or edge[2] is None else int(edge[2]),
                )
                for edge in data["edges"]
            ),
        )

    @property
    def adjacency(self) -> Mapping[int, frozenset[int]]:
        """Read-only node-to-neighbour mapping."""
        return self._adjacency

    @property
    def atoms(self) -> tuple[int, ...]:
        """Alias for :attr:`nodes`."""
        return self.nodes

    @property
    def component_of(self) -> Mapping[int, int]:
        """Read-only node-to-zero-based-component mapping."""
        return self._component_of

    @property
    def components(self) -> tuple[tuple[int, ...], ...]:
        """Connected components in deterministic node traversal order."""
        return self._components

    @property
    def edge_table(self) -> tuple[AtomEdge, ...]:
        """Return every edge with any attached chemical order."""
        return tuple(
            AtomEdge(first, second, self._bond_orders.get((first, second)))
            for first, second in self.edges
        )

    @property
    def bond_orders(self) -> Mapping[tuple[int, int], int]:
        """Read-only chemical bond-order metadata."""
        return self._bond_orders

    @property
    def edge_count(self) -> int:
        """Number of unique undirected edges."""
        return len(self.edges)

    @property
    def component_count(self) -> int:
        """Number of connected components."""
        return len(self.components)

    @property
    def cycle_rank(self) -> int:
        """Number of independent cycles, ``E - V + C``."""
        return self.edge_count - len(self.nodes) + len(self.components)

    @property
    def connected(self) -> bool:
        """Whether all nodes belong to one connected component."""
        return len(self.components) <= 1

    def bond_order(self, first: int, second: int) -> int | None:
        """Return optional chemical order metadata, or ``None``."""
        self._require_node(first)
        self._require_node(second)
        edge = _canonical_pair((first, second))
        return self._bond_orders.get(edge)

    def neighbors(self, node: int) -> tuple[int, ...]:
        """Return the sorted neighbors of one node."""
        self._require_node(node)
        return tuple(sorted(self._adjacency[node]))

    def degree(self, node: int) -> int:
        """Return the number of incident edges."""
        return len(self.neighbors(node))

    def has_node(self, node: int) -> bool:
        """Return whether a node belongs to this graph."""
        if not isinstance(node, int) or isinstance(node, bool):
            raise TypeError("atomic graph nodes must be integers")
        return node in self._adjacency

    def signature(
        self,
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[tuple[int, int], int], ...],
    ]:
        """Return deterministic node, edge, and bond-order identity."""
        return (
            self.nodes,
            self.edges,
            tuple(sorted(self._bond_orders.items())),
        )

    def __hash__(self) -> int:
        """Hash the immutable graph identity."""
        return hash(self.signature())

    def has_edge(self, first: int, second: int) -> bool:
        """Return whether two nodes are directly connected."""
        self._require_node(first)
        self._require_node(second)
        return second in self._adjacency[first]

    def shortest_path(
        self,
        first: int,
        second: int,
        *,
        max_depth: int | None = None,
    ) -> tuple[int, ...] | None:
        """Return one deterministic shortest path, or ``None`` if disconnected."""
        self._require_node(first)
        self._require_node(second)
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if first == second:
            return (first,)
        previous: dict[int, int | None] = {first: None}
        queue = deque([(first, 0)])
        while queue:
            node, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbour in sorted(self._adjacency[node]):
                if neighbour in previous:
                    continue
                previous[neighbour] = node
                if neighbour == second:
                    path = [second]
                    cursor = node
                    while cursor is not None:
                        path.append(cursor)
                        cursor = previous[cursor]
                    return tuple(reversed(path))
                queue.append((neighbour, depth + 1))
        return None

    def distance(
        self,
        first: int,
        second: int,
        *,
        max_depth: int | None = None,
    ) -> int | None:
        """Return unweighted shortest-path length, or ``None`` if disconnected."""
        path = self.shortest_path(first, second, max_depth=max_depth)
        return None if path is None else len(path) - 1

    def vertices_at_distance(
        self,
        source: int,
        distance: int,
    ) -> tuple[int, ...]:
        """Return nodes whose shortest graph distance is exactly *distance*."""
        self._require_node(source)
        if distance < 0:
            raise ValueError("graph distance cannot be negative")
        if distance == 0:
            return (source,)
        visited = {source}
        frontier = {source}
        for _ in range(distance):
            reached: set[int] = set()
            for node in frontier:
                reached.update(self._adjacency[node] - visited)
            visited.update(reached)
            frontier = reached
            if not frontier:
                break
        return tuple(sorted(frontier))

    def pairs_at_distance(self, distance: int = 3) -> set[tuple[int, int]]:
        """Return pairs separated by exact shortest graph distance."""
        if distance < 1:
            raise ValueError("pair distance must be positive")
        pairs: set[tuple[int, int]] = set()
        for source in self.nodes:
            pairs.update(
                _canonical_pair((source, target))
                for target in self.vertices_at_distance(source, distance)
            )
        return pairs

    def distance_three_pairs(self) -> set[tuple[int, int]]:
        """Return graph-distance-three pairs used by ``[ pairs ]``."""
        return self.pairs_at_distance(3)

    def angles_through_edge(
        self,
        edge: tuple[int, int],
    ) -> set[tuple[int, int, int]]:
        """Return every angle whose two bonds include *edge*."""
        first, second = self._require_edge(edge)
        angles = {
            _canonical_angle((neighbour, first, second))
            for neighbour in self._adjacency[first] - {second}
        }
        angles.update(
            _canonical_angle((first, second, neighbour))
            for neighbour in self._adjacency[second] - {first}
        )
        return angles

    def proper_dihedrals_through_edge(
        self,
        edge: tuple[int, int],
    ) -> set[tuple[int, int, int, int]]:
        """Return every simple three-bond path containing *edge*."""
        first, second = self._require_edge(edge)
        dihedrals: set[tuple[int, int, int, int]] = set()
        for neighbour_first in self._adjacency[first] - {second}:
            for neighbour_second in self._adjacency[second] - {first}:
                if neighbour_first != neighbour_second:
                    dihedrals.add(
                        _canonical_dihedral(
                            (neighbour_first, first, second, neighbour_second)
                        )
                    )
            for outer in self._adjacency[neighbour_first] - {first, second}:
                dihedrals.add(
                    _canonical_dihedral((outer, neighbour_first, first, second))
                )
        for neighbour_second in self._adjacency[second] - {first}:
            for outer in self._adjacency[neighbour_second] - {first, second}:
                dihedrals.add(
                    _canonical_dihedral((first, second, neighbour_second, outer))
                )
        return dihedrals

    def angles(self) -> set[tuple[int, int, int]]:
        """Return every angle implied by the edge graph."""
        angles: set[tuple[int, int, int]] = set()
        for center in self.nodes:
            neighbors = sorted(self._adjacency[center])
            for index, first in enumerate(neighbors):
                for second in neighbors[index + 1 :]:
                    angles.add(_canonical_angle((first, center, second)))
        return angles

    def proper_dihedrals(self) -> set[tuple[int, int, int, int]]:
        """Return every simple three-bond path as a proper torsion."""
        dihedrals: set[tuple[int, int, int, int]] = set()
        for edge in self.edges:
            dihedrals.update(self.proper_dihedrals_through_edge(edge))
        return dihedrals

    def has_path(self, nodes: Sequence[int]) -> bool:
        """Whether every consecutive pair in *nodes* is an edge."""
        values = tuple(nodes)
        for node in values:
            self._require_node(node)
        return bool(values) and all(
            self.has_edge(first, second) for first, second in pairwise(values)
        )

    def with_edges(
        self,
        edges: Iterable[tuple[int, int] | AtomEdge],
        *,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
    ) -> AtomicGraph:
        """Return a copy with edges added or their order metadata updated."""
        orders = dict(self._bond_orders)
        orders.update(bond_orders or {})
        return AtomicGraph(
            self.nodes,
            (*self.edges, *tuple(edges)),
            bond_orders=orders,
        )

    def with_edge(
        self,
        first: int,
        second: int,
        *,
        bond_order: int | None = None,
    ) -> AtomicGraph:
        """Return a copy with one edge added or its order metadata updated."""
        pair = _canonical_pair((first, second))
        return self.with_edges(
            (pair,),
            bond_orders={} if bond_order is None else {pair: bond_order},
        )

    def without_edges(
        self,
        edges: Iterable[tuple[int, int]],
    ) -> AtomicGraph:
        """Return a copy with connectivity edges removed when present."""
        removed = {_canonical_pair(edge) for edge in edges}
        for first, second in removed:
            if first == second:
                raise ValueError(f"atomic graph cannot contain self-loop {first}")
            self._require_node(first)
            self._require_node(second)
        return AtomicGraph(
            self.nodes,
            (edge for edge in self.edges if edge not in removed),
            bond_orders={
                pair: order
                for pair, order in self._bond_orders.items()
                if pair not in removed
            },
        )

    def without_edge(self, first: int, second: int) -> AtomicGraph:
        """Return a copy with one connectivity edge removed."""
        return self.without_edges(((first, second),))

    def subgraph(self, nodes: Iterable[int]) -> AtomicGraph:
        """Return the node-induced subgraph."""
        selected = tuple(nodes)
        selected_set = set(selected)
        unknown = selected_set - set(self.nodes)
        if unknown:
            raise ValueError(f"subgraph nodes are not in the graph: {sorted(unknown)}")
        return AtomicGraph(
            selected,
            (
                edge
                for edge in self.edges
                if edge[0] in selected_set and edge[1] in selected_set
            ),
            bond_orders={
                pair: order
                for pair, order in self._bond_orders.items()
                if pair[0] in selected_set and pair[1] in selected_set
            },
        )

    def remap(self, node_mapping: Mapping[int, int]) -> AtomicGraph:
        """Return the graph in a renamed or compacted node space."""
        missing = [node for node in self.nodes if node not in node_mapping]
        if missing:
            raise KeyError(f"node mapping is missing atoms {missing}")
        return AtomicGraph(
            tuple(node_mapping[node] for node in self.nodes),
            (
                (node_mapping[first], node_mapping[second])
                for first, second in self.edges
            ),
            bond_orders={
                (
                    node_mapping[first],
                    node_mapping[second],
                ): order
                for (first, second), order in self._bond_orders.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready deterministic graph representation."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                [edge.first, edge.second, edge.order] for edge in self.edge_table
            ],
        }

    def _require_node(self, node: int) -> None:
        if not isinstance(node, int) or isinstance(node, bool):
            raise TypeError("atomic graph nodes must be integers")
        if node not in self._adjacency:
            raise KeyError(f"unknown atomic graph node {node}")

    def _require_edge(self, edge: tuple[int, int]) -> tuple[int, int]:
        first, second = edge
        self._require_node(first)
        self._require_node(second)
        normalized = _canonical_pair((first, second))
        if normalized[1] not in self._adjacency[normalized[0]]:
            raise ValueError(
                f"atomic graph has no edge {normalized[0]}-{normalized[1]}"
            )
        return normalized


def _breadth_first_components(
    nodes: Sequence[int],
    adjacency: Mapping[int, set[int]],
) -> tuple[tuple[int, ...], ...]:
    """Partition nodes with deterministic breadth-first traversal."""
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in nodes:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _normalize_nodes(nodes: int | Sequence[int]) -> tuple[int, ...]:
    """Validate graph nodes and accept a count shorthand."""
    if isinstance(nodes, int) and not isinstance(nodes, bool):
        if nodes < 1:
            raise ValueError("atomic graph node count must be positive")
        normalized = tuple(range(1, nodes + 1))
    else:
        normalized = tuple(nodes)
    if not normalized:
        raise ValueError("AtomicGraph requires at least one node")
    if any(not isinstance(node, int) or isinstance(node, bool) for node in normalized):
        raise TypeError("atomic graph nodes must be integers")
    if len(normalized) != len(set(normalized)):
        raise ValueError("atomic graph nodes must be unique")
    return normalized


def _normalize_edges(
    edges: Iterable[tuple[int, int] | AtomEdge],
    node_set: set[int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """Validate, canonicalize, and de-duplicate graph edges."""
    normalized_edges: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    orders: dict[tuple[int, int], int] = {}
    for raw_edge in edges:
        if isinstance(raw_edge, AtomEdge):
            first, second = raw_edge.pair
            embedded_order = raw_edge.order
        else:
            try:
                first, second = raw_edge
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "each atomic graph edge must contain exactly two endpoints"
                ) from exc
            embedded_order = None
        if (
            not isinstance(first, int)
            or isinstance(first, bool)
            or not isinstance(second, int)
            or isinstance(second, bool)
        ):
            raise TypeError("atomic graph edge endpoints must be integers")
        if first == second:
            raise ValueError(f"atomic graph cannot contain self-loop {first}")
        unknown = {first, second} - node_set
        if unknown:
            raise ValueError(
                f"atomic graph edge references unknown nodes {sorted(unknown)}"
            )
        edge = _canonical_pair((first, second))
        if edge not in seen_edges:
            seen_edges.add(edge)
            normalized_edges.append(edge)
        if embedded_order is not None:
            _merge_order(orders, edge, embedded_order)
    return normalized_edges, orders


def _merge_bond_orders(
    edges: Sequence[tuple[int, int]],
    orders: dict[tuple[int, int], int],
    bond_orders: Mapping[tuple[int, int], int] | None,
) -> None:
    """Validate and merge caller-provided bond-order metadata."""
    edge_set = set(edges)
    for raw_pair, raw_order in (bond_orders or {}).items():
        pair = _canonical_pair(raw_pair)
        if pair not in edge_set:
            raise ValueError(f"bond_orders references non-edge pairs: {[pair]}")
        if (
            not isinstance(raw_order, int)
            or isinstance(raw_order, bool)
            or not 1 <= raw_order <= 4
        ):
            raise ValueError("bond orders must be integers from 1 to 4")
        _merge_order(orders, pair, raw_order)


def _canonical_pair(pair: Sequence[int]) -> tuple[int, int]:
    """Return an unordered edge key."""
    first, second = tuple(pair)
    if (
        not isinstance(first, int)
        or isinstance(first, bool)
        or not isinstance(second, int)
        or isinstance(second, bool)
    ):
        raise TypeError("atomic graph edge endpoints must be integers")
    return (first, second) if first < second else (second, first)


def _canonical_angle(angle: Sequence[int]) -> tuple[int, int, int]:
    """Return an angle key while preserving its centre."""
    first, center, last = (int(value) for value in angle)
    return (min(first, last), center, max(first, last))


def _canonical_dihedral(
    dihedral: Sequence[int],
) -> tuple[int, int, int, int]:
    """Return a proper-torsion key with global reversal removed."""
    values = tuple(int(value) for value in dihedral)
    return min(values, tuple(reversed(values)))


def _merge_order(
    orders: dict[tuple[int, int], int],
    edge: tuple[int, int],
    order: int,
) -> None:
    """Insert one bond order and reject contradictory duplicates."""
    previous = orders.get(edge)
    if previous is not None and previous != order:
        raise ValueError(
            f"conflicting bond orders {previous} and {order} for edge {edge}"
        )
    orders[edge] = order


__all__ = ["AtomEdge", "AtomicGraph"]
