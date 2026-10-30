"""Public API contracts and adversarial graph/topology interactions."""

from __future__ import annotations

from pathlib import Path

import gromologist as gml
import pytest

from topolsurgery import (
    AtomEdge,
    AtomicGraph,
    AtomSpec,
    BondedTerm,
    InteractionSpec,
    ParameterRequest,
    TopologySurgery,
)


def _atoms(count: int) -> tuple[AtomSpec, ...]:
    """Return a contiguous carbon chain in the surgery atom space."""
    return tuple(
        AtomSpec(
            number=index,
            atom_type="c3",
            atom_name=f"C{index}",
            residue_number=1,
            residue_name="MOL",
            charge_group=index,
            charge=0.0,
            mass=12.011,
        )
        for index in range(1, count + 1)
    )


def test_atom_edge_and_graph_round_trip_preserve_order() -> None:
    """Graph serialization retains node, edge, and bond-order metadata."""
    graph = AtomicGraph.from_edges(
        5,
        (
            AtomEdge(4, 3, 2),
            (1, 2),
            (4, 5),
        ),
    )
    restored = AtomicGraph.from_dict(graph.to_dict())

    assert AtomEdge(4, 3, 2).pair == (3, 4)
    assert graph.edges == ((3, 4), (1, 2), (4, 5))
    assert graph.bond_order(3, 4) == 2
    assert restored == graph


def test_from_records_accepts_single_pass_iterables() -> None:
    """Generator-built records are materialized before topology rendering."""
    surgery = TopologySurgery.from_records(
        (atom for atom in _atoms(3)),
        (
            interaction
            for interaction in (
                InteractionSpec("bonds", (1, 2), 1),
                InteractionSpec("bonds", (2, 3), 1),
            )
        ),
    )

    assert surgery.graph.edges == ((1, 2), (2, 3))
    assert surgery.components == ((1, 2, 3),)


def test_noncontiguous_duplicate_system_rows_are_rejected(
    tmp_path: Path,
) -> None:
    """Grouping duplicate molecule rows would silently reorder coordinates."""
    topology = tmp_path / "ambiguous.top"
    topology.write_text(
        """\
[ moleculetype ]
CHAIN 3
[ atoms ]
1 c3 1 MOL C1 1 0.0 12.011
[ moleculetype ]
SOL 3
[ atoms ]
1 c3 1 SOL O1 1 0.0 12.011
[ system ]
Ambiguous
[ molecules ]
CHAIN 1
SOL 1
CHAIN 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be contiguous"):
        TopologySurgery.from_files(topology)


def test_ring_cut_keeps_one_component_but_repairs_terms() -> None:
    """A non-bridge cut preserves the component and drops broken paths."""
    interactions = (
        InteractionSpec("bonds", (1, 2), 1),
        InteractionSpec("bonds", (2, 3), 1),
        InteractionSpec("bonds", (3, 4), 1),
        InteractionSpec("bonds", (4, 1), 1),
        InteractionSpec("angles", (1, 2, 3), 1),
        InteractionSpec("angles", (2, 3, 4), 1),
        InteractionSpec("angles", (3, 4, 1), 1),
        InteractionSpec("dihedrals", (1, 2, 3, 4), 9),
    )
    surgery = TopologySurgery.from_records(_atoms(4), interactions)

    result = surgery.cut_bond(4, 1)

    assert result.graph.connected
    assert result.graph.components == ((1, 2, 3, 4),)
    assert result.edit is not None
    assert {item.atoms for item in surgery.interactions("dihedrals")} == {(1, 2, 3, 4)}
    assert {item.atoms for item in surgery.interactions("angles")} == {
        (1, 2, 3),
        (2, 3, 4),
    }


def test_connection_in_ring_adds_all_graph_paths_once() -> None:
    """Closing a ring generates every torsion path containing the new edge."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    result = surgery.connect_bond(
        4,
        1,
        update_pairs=False,
        resolve_parameters=False,
    )

    actual = {
        min(item.atoms, tuple(reversed(item.atoms)))
        for item in result.topology.interactions_of("dihedrals")
    }
    assert actual == result.graph.proper_dihedrals_through_edge((1, 4))
    assert len(actual) == 3


def test_improper_is_preserved_when_endpoints_stay_connected() -> None:
    """A non-bridge cut keeps impropers inside the surviving component."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
            InteractionSpec("bonds", (1, 5), 1),
            InteractionSpec("dihedrals", (2, 1, 3, 4), 4),
            InteractionSpec("dihedrals", (1, 2, 4, 5), 2),
        ),
    )

    result = surgery.cut_bond(4, 5)

    assert result.graph.connected
    assert {
        (item.atoms, item.function)
        for item in result.topology.interactions_of("dihedrals")
    } == {
        ((2, 1, 3, 4), 4),
        ((1, 2, 4, 5), 2),
    }


def test_failed_resolver_rolls_back_the_whole_operation() -> None:
    """A resolver exception after partial edits leaves the session unchanged."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    before = surgery.snapshot()

    def failing_resolver(
        request: ParameterRequest,
    ) -> list[BondedTerm] | None:
        if request.kind == "angles":
            raise RuntimeError("resolver failed")
        return [BondedTerm(request.default_function)]

    with pytest.raises(RuntimeError, match="resolver failed"):
        surgery.connect_bond(
            2,
            3,
            resolve_parameters=False,
            parameter_resolver=failing_resolver,
        )

    assert surgery.snapshot() == before
    assert surgery.graph.edges == ((1, 2), (3, 4))


def test_results_expose_snapshot_graph_and_independent_session() -> None:
    """Result topology/graph are public and preserve graph bond orders."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
        ),
        coordinates=tuple((index, 0.0, 0.0) for index in range(5)),
        bond_orders={(1, 2): 2, (4, 5): 3},
    )

    removal = surgery.remove_atoms((2,))
    assert removal.edit is not None
    assert removal.edit.operation == "remove_atoms"
    assert removal.topology.atom_count == 4
    assert removal.graph.bond_order(3, 4) == 3
    assert removal.topology.graph.bond_order(3, 4) == 3
    assert {(item.atoms, item.function) for item in removal.edit.removed} == {
        ((1, 2), 1),
        ((2, 3), 1),
    }

    branch = TopologySurgery.from_result(removal)
    branch.connect_bond(1, 2, resolve_parameters=False)
    assert surgery.atom_count == 4
    assert branch.atom_count == 4
    assert removal.graph.edges == ((2, 3), (3, 4))
    assert branch.graph.has_edge(1, 2)


def test_from_result_preserves_removed_parameter_templates() -> None:
    """A branch keeps explicit templates produced by a now-removed row."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("angles", (2, 3, 4), 1, (109.2, 410.2)),
            InteractionSpec(
                "dihedrals",
                (1, 2, 3, 4),
                9,
                (0.0, 0.8, 2),
            ),
        ),
    )
    cut = surgery.cut_bond(3, 4)

    branch = TopologySurgery.from_result(cut)
    joined = branch.connect_bond(3, 4, resolve_parameters=False)

    angle = next(
        item
        for item in joined.topology.interactions_of("angles")
        if item.atoms == (2, 3, 4)
    )
    dihedral = next(
        item
        for item in joined.topology.interactions_of("dihedrals")
        if item.atoms == (1, 2, 3, 4)
    )
    assert angle.parameters == pytest.approx((109.2, 410.2))
    assert dihedral.parameters == pytest.approx((0.0, 0.8, 2))


def test_remove_atoms_rejects_empty_molecule_and_invalid_numbers() -> None:
    """Destructive input is validated before a topology is changed."""
    surgery = TopologySurgery.from_records(_atoms(2))
    before = surgery.snapshot()

    with pytest.raises(ValueError, match="cannot remove every atom"):
        surgery.remove_atoms((1, 2))
    with pytest.raises(ValueError, match="at least one"):
        surgery.remove_atoms(())
    with pytest.raises(ValueError, match="duplicates"):
        surgery.remove_atoms((1, 1))
    with pytest.raises(TypeError, match="integers"):
        surgery.remove_atoms(("1",))
    assert surgery.snapshot() == before


def test_removing_bridge_atom_reaudits_surviving_pairs_and_impropers() -> None:
    """Rows bypassing a removed bridge atom cannot span new components."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
            InteractionSpec("pairs", (2, 5), 1),
            InteractionSpec("pairs", (1, 5), 1),
            InteractionSpec("dihedrals", (1, 2, 4, 5), 4),
        ),
    )

    removal = surgery.remove_atoms((3,))

    assert tuple(component.atom_numbers for component in removal.components) == (
        (1, 2),
        (3, 4),
    )
    assert not surgery.interactions("pairs")
    assert not surgery.interactions("dihedrals")


def test_removal_and_export_round_trip_topology_and_coordinates(
    tmp_path: Path,
) -> None:
    """Post-removal component files preserve atom counts and local indices."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (4, 5), 1),
        ),
        coordinates=((0.0, 0.0, 0.0),) * 5,
    )
    removal = surgery.remove_atoms((2,))
    exported = removal.export(tmp_path / "parts", format="top")

    assert len(exported.components) == 3
    for component in exported.components:
        topology = gml.Top(str(component.topology_path), suppress=True)
        gro_lines = component.gro_path.read_text(encoding="utf-8").splitlines()
        assert topology.molecules[0].natoms == int(gro_lines[1])
        assert topology.molecules[0].natoms == len(component.atom_numbers)


def test_sequential_operations_remain_graph_consistent() -> None:
    """Cut/connect/cut cycles do not leave stale interactions or edges."""
    surgery = TopologySurgery.from_records(
        _atoms(6),
        tuple(InteractionSpec("bonds", (index, index + 1), 1) for index in range(1, 6)),
    )
    first = surgery.cut_bond(3, 4)
    second = surgery.connect_bond(4, 3, resolve_parameters=False)
    third = surgery.cut_bond(3, 4)

    assert tuple(item.atom_numbers for item in first.components) == (
        (1, 2, 3),
        (4, 5, 6),
    )
    assert second.graph.connected
    assert tuple(item.atom_numbers for item in third.components) == (
        (1, 2, 3),
        (4, 5, 6),
    )
    assert surgery.graph.edges == (
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 6),
    )


def test_malformed_override_does_not_break_other_interaction_kinds() -> None:
    """A wrong-sized override key is skipped instead of crashing angle lookup."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )

    result = surgery.connect_bond(
        2,
        3,
        resolve_parameters=False,
        parameter_overrides={(1, 2): BondedTerm(1, (0.2, 1000.0))},
    )

    assert result.graph.connected
    assert result.topology.interactions_of("angles")


def test_empty_resolver_result_falls_back_without_suppressing_terms() -> None:
    """An empty resolver result is not interpreted as a graph-term veto."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )

    result = surgery.connect_bond(
        2,
        3,
        resolve_parameters=False,
        parameter_resolver=lambda request: (),
    )

    assert result.graph.has_edge(2, 3)
    assert (2, 3) in {item.atoms for item in result.topology.interactions_of("bonds")}
    assert result.topology.interactions_of("angles")
