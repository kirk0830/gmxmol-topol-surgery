"""Additional public API contracts for in-memory and file-backed callers."""

from __future__ import annotations

import math
from pathlib import Path

import gromologist as gml
import numpy as np
import pytest

import topolsurgery
import topolsurgery.api as public_api
from topolsurgery import (
    AtomSpec,
    BondedTerm,
    Component,
    ExportedComponent,
    ExportResult,
    GroAtom,
    GroFile,
    IndexMap,
    InteractionSpec,
    SurgeryResult,
    SurgeryWarning,
    TopologyEdit,
    TopologySnapshot,
    TopologySurgery,
)


def _atoms(count: int) -> tuple[AtomSpec, ...]:
    return tuple(
        AtomSpec(
            number=index,
            atom_type="c3",
            atom_name=f"C{index}",
            residue_number=index,
            residue_name="MOL",
            charge_group=index,
            mass=12.011,
        )
        for index in range(1, count + 1)
    )


def _chain_interactions(count: int) -> tuple[InteractionSpec, ...]:
    return tuple(
        InteractionSpec("bonds", (index, index + 1), 1) for index in range(1, count)
    )


def test_root_and_api_module_export_the_same_public_names() -> None:
    """The documented stable import surfaces cannot drift silently."""
    assert set(topolsurgery.__all__) == set(public_api.__all__)
    assert all(hasattr(topolsurgery, name) for name in topolsurgery.__all__)
    assert all(hasattr(public_api, name) for name in public_api.__all__)


def test_duplicate_system_rows_are_consolidated_in_system_order(
    tmp_path: Path,
) -> None:
    """Repeated [ molecules ] names keep their combined copy count."""
    topology_text = """\
[ moleculetype ]
CHAIN 3
[ atoms ]
1 c3 1 MOL C1 1 0.0 12.011
2 c3 1 MOL C2 2 0.0 12.011
3 c3 1 MOL C3 3 0.0 12.011
[ bonds ]
1 2 1
2 3 1
[ system ]
Repeated rows
[ molecules ]
CHAIN 2
CHAIN 1
"""
    topology_path = tmp_path / "repeated.top"
    topology_path.write_text(topology_text, encoding="utf-8")
    lines = ["Repeated rows", "9"]
    for index in range(1, 10):
        residue = (index - 1) // 3 + 1
        atom = (index - 1) % 3 + 1
        lines.append(
            f"{residue:5d}{'MOL':<5s}{f'C{atom}':>5s}{index:5d}"
            f"{0.15 * index:8.3f}{0.0:8.3f}{0.0:8.3f}"
        )
    lines.append(f"{3.0:10.5f}{3.0:10.5f}{3.0:10.5f}")
    structure_path = tmp_path / "repeated.gro"
    structure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    surgery = TopologySurgery.from_files(topology_path, structure_path)

    assert surgery.atom_count == 9
    assert surgery.components == ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    result = surgery.cut_bond(2, 3)
    assert tuple(component.atom_numbers for component in result.components) == (
        (1, 2),
        (3,),
        (4, 5, 6),
        (7, 8, 9),
    )


def test_in_memory_topology_and_clones_are_isolated(tmp_path: Path) -> None:
    """The caller's Top, the session, and session clones never alias."""
    topology_text = """\
[ moleculetype ]
CHAIN 3
[ atoms ]
1 c3 1 MOL C1 1 0.0 12.011
2 c3 1 MOL C2 2 0.0 12.011
3 c3 1 MOL C3 3 0.0 12.011
[ bonds ]
1 2 1
2 3 1
[ system ]
Chain
[ molecules ]
CHAIN 1
"""
    topology_path = tmp_path / "chain.top"
    topology_path.write_text(topology_text, encoding="utf-8")
    original = gml.Top(str(topology_path), suppress=True)
    surgery = TopologySurgery.from_gromologist(original)
    branch = surgery.clone()

    surgery.cut_bond(2, 3)
    branch.connect_bond(1, 3, resolve_parameters=False)

    assert len(original.get_molecule("CHAIN").bonds_section.entries_bonded) == 2
    assert surgery.graph.edges == ((1, 2),)
    assert branch.graph.edges == ((1, 2), (2, 3), (1, 3))
    defensive = surgery.to_gromologist()
    defensive.get_molecule("CHAIN").rename("MUTATED")
    assert surgery.name == "CHAIN"


def test_snapshot_can_be_used_as_a_new_session() -> None:
    """Public records round-trip after an operation without private objects."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        _chain_interactions(4),
        name="SNAP",
        nrexcl=2,
    )
    cut = surgery.cut_bond(2, 3)
    rebuilt = TopologySurgery.from_snapshot(cut.topology)

    assert rebuilt.name == "SNAP"
    assert rebuilt.nrexcl == 2
    assert rebuilt.graph.edges == ((1, 2), (3, 4))
    rebuilt.connect_bond(2, 3, resolve_parameters=False)
    assert rebuilt.graph.connected


def test_snapshot_preserves_bond_orders_and_clone_parameter_history() -> None:
    """Snapshots round-trip graph metadata; clones also retain removed rows."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("angles", (2, 3, 4), 1, (109.2, 410.2)),
            InteractionSpec("dihedrals", (1, 2, 3, 4), 9, (0.0, 0.8, 2)),
        ),
        bond_orders={(1, 2): 2},
    )
    cut = surgery.cut_bond(3, 4)
    rebuilt = TopologySurgery.from_snapshot(cut.topology)
    cloned = surgery.clone()

    assert cut.topology.bond_orders == {(1, 2): 2}
    assert rebuilt.graph.bond_order(1, 2) == 2
    rebuilt.connect_bond(3, 4, resolve_parameters=False)
    joined = cloned.connect_bond(3, 4, resolve_parameters=False)
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
    assert angle.parameters == (109.2, 410.2)
    assert dihedral.parameters == (0.0, 0.8, 2)


def test_bond_and_constraint_on_one_pair_are_one_cuttable_edge() -> None:
    """Cutting one pair removes both row families and their graph metadata."""
    surgery = TopologySurgery.from_records(
        _atoms(2),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("constraints", (1, 2), 1, (0.1,)),
        ),
        bond_orders={(1, 2): 2},
    )

    cut = surgery.cut_bond(2, 1)

    assert cut.graph.edges == ()
    assert cut.graph.bond_order(1, 2) is None
    assert not surgery.interactions("bonds")
    assert not surgery.interactions("constraints")
    joined = surgery.connect_bond(
        1,
        2,
        bond_order=3,
        resolve_parameters=False,
    )
    assert joined.graph.bond_order(1, 2) == 3
    assert len(surgery.interactions("bonds")) == 1
    assert not surgery.interactions("constraints")


def test_invalid_record_topology_is_rejected_before_mutation() -> None:
    """Record validation catches bad names, arity, and atom references."""
    with pytest.raises(ValueError, match="molecule name"):
        TopologySurgery.from_records(_atoms(2), name="bad name")
    with pytest.raises(ValueError, match="requires 2 atoms"):
        TopologySurgery.from_records(
            _atoms(3),
            (InteractionSpec("bonds", (1, 2, 3), 1),),
        )
    with pytest.raises(ValueError, match="outside"):
        TopologySurgery.from_records(
            _atoms(2),
            (InteractionSpec("bonds", (1, 3), 1),),
        )
    with pytest.raises(ValueError, match="coordinates must be finite"):
        TopologySurgery.from_records(
            _atoms(2),
            coordinates=((math.nan, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )


def test_settle_water_is_one_component_and_is_not_a_cuttable_bond() -> None:
    """SETTLE is protected connectivity and disappears with any member atom."""
    atoms = (
        AtomSpec(1, "OW", "OW", 1, "SOL", 1, -0.834, 15.999),
        AtomSpec(2, "HW", "HW1", 1, "SOL", 1, 0.417, 1.008),
        AtomSpec(3, "HW", "HW2", 1, "SOL", 1, 0.417, 1.008),
    )
    surgery = TopologySurgery.from_records(
        atoms,
        (InteractionSpec("settles", (1,), 1, (0.09572, 0.15139)),),
        name="SOL",
        nrexcl=2,
    )
    assert surgery.components == ((1, 2, 3),)
    assert surgery.snapshot().graph.components == ((1, 2, 3),)
    with pytest.raises(ValueError, match="implicit SETTLE dependency"):
        surgery.cut_bond(1, 2)

    surgery.remove_atoms((2,))
    assert not surgery.interactions("settles")


def test_nonbridge_cut_removes_cmap_without_continuous_path() -> None:
    """CMAP validity requires the five consecutive edges, not only a component."""
    atoms = _atoms(5)
    bonds = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 1))
    surgery = TopologySurgery.from_records(
        atoms,
        tuple(InteractionSpec("bonds", edge, 1) for edge in bonds)
        + (InteractionSpec("cmap", (1, 2, 3, 4, 5), 1, (1.0,)),),
    )
    surgery.cut_bond(1, 2)

    assert surgery.graph.connected
    assert not surgery.graph.has_path((1, 2, 3, 4, 5))
    assert not surgery.interactions("cmap")


def test_virtual_site_stays_with_its_host_component(tmp_path: Path) -> None:
    """TIP4P virtual-site ownership does not become chemical connectivity."""
    atoms = (
        AtomSpec(1, "OW", "OW", 1, "SOL", 1, 0.0, 15.999),
        AtomSpec(2, "HW", "HW1", 1, "SOL", 1, 0.52, 1.008),
        AtomSpec(3, "HW", "HW2", 1, "SOL", 1, 0.52, 1.008),
        AtomSpec(4, "MW", "MW", 1, "SOL", 1, -1.04, 0.0),
        AtomSpec(5, "c3", "C1", 2, "M", 5, 0.0, 12.011),
        AtomSpec(6, "c3", "C2", 2, "M", 6, 0.0, 12.011),
    )
    surgery = TopologySurgery.from_records(
        atoms,
        (
            InteractionSpec("settles", (1,), 1, (0.09572, 0.15139)),
            InteractionSpec(
                "virtual_sites3",
                (4, 1, 2, 3),
                1,
                (0.128012065, 0.128012065),
            ),
            InteractionSpec("bonds", (5, 6), 1),
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(6)),
    )

    assert surgery.graph.components == ((1, 2, 3), (4,), (5, 6))
    assert surgery.components == ((1, 2, 3, 4), (5, 6))
    assert surgery.snapshot().component_groups == ((1, 2, 3, 4), (5, 6))

    result = surgery.cut_bond(5, 6)
    water = next(
        component
        for component in result.component_snapshots()
        if component.atom_count == 4
    )
    assert water.interactions_of("virtual_sites3")[0].atoms == (4, 1, 2, 3)


def test_mixed_nrexcl_is_rejected_before_flattening(tmp_path: Path) -> None:
    """Different exclusion depths cannot be silently unified."""
    topology = tmp_path / "mixed.top"
    structure = tmp_path / "mixed.gro"
    topology.write_text(
        """\
[ moleculetype ]
PROT 3
[ atoms ]
1 c3 1 P C1 1 0.0 12.011
2 c3 1 P C2 2 0.0 12.011
[ bonds ]
1 2 1
[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 S OW 1 0.0 15.999
2 HW 1 S HW1 1 0.0 1.008
[ bonds ]
1 2 1
[ system ]
Mixed
[ molecules ]
PROT 1
SOL 1
""",
        encoding="utf-8",
    )
    structure.write_text(
        "Mixed\n"
        "4\n"
        f"{1:5d}{'P':<5s}{'C1':>5s}{1:5d}"
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2:5d}{'P':<5s}{'C2':>5s}{2:5d}"
        f"{0.15:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{1:5d}{'S':<5s}{'OW':>5s}{3:5d}"
        f"{1.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2:5d}{'S':<5s}{'HW1':>5s}{4:5d}"
        f"{1.1:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different nrexcl"):
        TopologySurgery.from_files(topology, structure)


def test_top_level_intermolecular_and_conditional_system_are_rejected(
    tmp_path: Path,
) -> None:
    """Unsupported system composition cannot be flattened silently."""
    intermolecular = tmp_path / "intermolecular.top"
    intermolecular.write_text(
        """\
[ moleculetype ]
A 3
[ atoms ]
1 c3 1 A C1 1 0.0 12.011
[ moleculetype ]
B 3
[ atoms ]
1 c3 1 B C1 1 0.0 12.011
[ system ]
Mixed
[ molecules ]
A 1
B 1
[ intermolecular_interactions ]
[ bonds ]
1 2 1 0.1 100.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="system-level"):
        TopologySurgery.from_files(intermolecular)

    conditional = tmp_path / "conditional.top"
    conditional.write_text(
        """\
[ moleculetype ]
A 3
[ atoms ]
1 c3 1 A C1 1 0.0 12.011
[ system ]
Conditional
[ molecules ]
#ifdef FOO
A 1
#endif
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conditional"):
        TopologySurgery.from_files(conditional)


def test_out_of_order_atom_rows_are_rejected(tmp_path: Path) -> None:
    """Atom-list position and atom number cannot silently disagree."""
    topology = tmp_path / "out-of-order.itp"
    topology.write_text(
        """\
[ moleculetype ]
MOL 1
[ atoms ]
2 cB 1 M C2 2 0.0 12.011
1 cA 1 M C1 1 0.0 12.011
[ bonds ]
1 2 1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordered and contiguously"):
        TopologySurgery.from_files(topology)


def test_position_restraints_roundtrip_and_unsupported_section_is_explicit(
    tmp_path: Path,
) -> None:
    """Supported restrained rows survive snapshots; unknown rows fail fast."""
    surgery = TopologySurgery.from_records(
        _atoms(2),
        (
            InteractionSpec(
                "position_restraints",
                (1,),
                1,
                (100.0, 100.0, 100.0),
            ),
        ),
    )
    rebuilt = TopologySurgery.from_snapshot(surgery.snapshot())
    assert rebuilt.interactions("position_restraints")[0].parameters == (
        100.0,
        100.0,
        100.0,
    )

    topology = tmp_path / "unsupported.itp"
    topology.write_text(
        """\
[ moleculetype ]
UNK 1
[ atoms ]
1 c3 1 M C1 1 0.0 12.011
2 c3 1 M C2 2 0.0 12.011
[ bonds ]
1 2 1
[ orientation_restraints ]
1 1 0 0 0 1 1 10 1
""",
        encoding="utf-8",
    )
    structure = tmp_path / "unsupported.gro"
    structure.write_text(
        "UNK\n2\n"
        f"{1:5d}{'M':<5s}{'C1':>5s}{1:5d}"
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{1:5d}{'M':<5s}{'C2':>5s}{2:5d}"
        f"{0.15:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}\n",
        encoding="utf-8",
    )
    parsed = TopologySurgery.from_files(topology, structure)
    assert parsed.snapshot().unsupported_sections == ("orientation_restraints",)
    with pytest.raises(ValueError, match="cannot represent"):
        TopologySurgery.from_snapshot(parsed.snapshot())
    with pytest.raises(ValueError, match="unsupported or lossy"):
        parsed.cut_bond(1, 2)
    with pytest.raises(ValueError, match="unsupported or lossy"):
        parsed.connect_bond(1, 2)
    with pytest.raises(ValueError, match="unsupported or lossy"):
        parsed.remove_atoms((2,))


def test_structure_identity_and_finite_value_validation(tmp_path: Path) -> None:
    """Mismatched GRO identity and non-finite inputs fail at the boundary."""
    topology = tmp_path / "top.itp"
    topology.write_text(
        """\
[ moleculetype ]
MOL 1
[ atoms ]
1 c3 1 M C1 1 0.0 12.011
""",
        encoding="utf-8",
    )
    gro = tmp_path / "bad.gro"
    gro.write_text(
        "Bad\n1\n"
        f"{1:5d}{'X':<5s}{'Z9':>5s}{1:5d}"
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        TopologySurgery.from_files(topology, gro)
    assert (
        TopologySurgery.from_files(
            topology,
            gro,
            validate_structure=False,
        ).atom_count
        == 1
    )
    with pytest.raises(ValueError, match="finite"):
        BondedTerm(1, (math.nan, 1.0))
    with pytest.raises(ValueError, match="finite"):
        InteractionSpec("bonds", (1, 2), 1, (math.inf, 1.0))
    with pytest.raises(ValueError, match="positive"):
        InteractionSpec("bonds", (1, 2), 0)
    with pytest.raises(ValueError, match="finite"):
        GroAtom(1, "M", "X", 1, (0.0, 0.0, 0.0), (math.nan, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        GroFile(
            "bad", (GroAtom(1, "M", "X", 1, (0.0, 0.0, 0.0)),), (math.nan, 1.0, 1.0)
        )
    with pytest.raises(ValueError, match="5 characters"):
        GroAtom(1, "M", "ATOMNAME", 1, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="require coordinates"):
        TopologySurgery.from_records(
            _atoms(2),
            velocities=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )


def test_alchemical_atom_state_is_not_silently_lost_from_snapshot(
    tmp_path: Path,
) -> None:
    """A/B atom fields are detected as unrepresentable public-record state."""
    topology = tmp_path / "alchemical.itp"
    topology.write_text(
        """\
[ moleculetype ]
MOL 1
[ atoms ]
1 c3 1 M C1 1 0.0 12.011 c3 0.1 12.011
2 c3 1 M C2 2 0.0 12.011 c3 0.2 12.011
[ bonds ]
1 2 1
""",
        encoding="utf-8",
    )
    surgery = TopologySurgery.from_files(topology)
    snapshot = surgery.snapshot()

    assert "alchemical_atom_state" in snapshot.lossy_features
    with pytest.raises(ValueError, match="cannot represent"):
        TopologySurgery.from_snapshot(snapshot)
    with pytest.raises(ValueError, match="unsupported or lossy"):
        surgery.cut_bond(1, 2)
    with pytest.raises(ValueError, match="unsupported or lossy"):
        surgery.connect_bond(1, 2)
    with pytest.raises(ValueError, match="unsupported or lossy"):
        surgery.remove_atoms((2,))
    coordinates = GroFile(
        "alchemical",
        (
            GroAtom(1, "M", "C1", 1, (0.0, 0.0, 0.0)),
            GroAtom(1, "M", "C2", 2, (0.1, 0.0, 0.0)),
        ),
        (2.0, 2.0, 2.0),
    )
    result = SurgeryResult(
        operation="cut",
        atom_pair=(1, 2),
        components=(Component(1, "M", (1, 2)),),
        _top=surgery.to_gromologist(),
        _gro=coordinates,
    )
    with pytest.raises(ValueError, match="unsupported or lossy"):
        result.export(tmp_path / "lossy", format="top")


def test_conditional_molecule_entries_cannot_be_split_or_edited(
    tmp_path: Path,
) -> None:
    """Conditional interaction rows remain lossy outside raw combined save."""
    topology = tmp_path / "conditional-molecule.itp"
    topology.write_text(
        """\
[ moleculetype ]
MOL 1
[ atoms ]
1 c3 1 M C1 1 0.0 12.011
2 c3 1 M C2 2 0.0 12.011
3 c3 1 M C3 3 0.0 12.011
[ bonds ]
1 2 1
#ifdef FLEXIBLE
2 3 1
#endif
""",
        encoding="utf-8",
    )
    surgery = TopologySurgery.from_files(topology)
    snapshot = surgery.snapshot()

    assert "conditional_entries" in snapshot.lossy_features
    with pytest.raises(ValueError, match="unsupported or lossy"):
        surgery.cut_bond(1, 2)
    coordinates = GroFile(
        "conditional",
        (
            GroAtom(1, "M", "C1", 1, (0.0, 0.0, 0.0)),
            GroAtom(1, "M", "C2", 2, (0.1, 0.0, 0.0)),
            GroAtom(1, "M", "C3", 3, (0.2, 0.0, 0.0)),
        ),
        (2.0, 2.0, 2.0),
    )
    result = SurgeryResult(
        operation="cut",
        atom_pair=(1, 2),
        components=(Component(1, "M", (1, 2, 3)),),
        _top=surgery.to_gromologist(),
        _gro=coordinates,
    )
    with pytest.raises(ValueError, match="unsupported or lossy"):
        result.component_snapshots()


def test_public_record_models_reject_inconsistent_state() -> None:
    """Public frozen records cannot hide structurally invalid state."""
    with pytest.raises(TypeError, match="charge must be numeric"):
        AtomSpec(1, "c3", "C1", charge=True)
    with pytest.raises(ValueError, match="nrexcl"):
        TopologySnapshot("M", -1, _atoms(1), ())
    with pytest.raises(ValueError, match="contiguous"):
        IndexMap(forward={1: 2}, backward={2: 1})
    with pytest.raises(ValueError, match="unique"):
        Component(1, "M", (1, 1))
    with pytest.raises(TypeError, match="positive integers"):
        Component(1, "M", (1.9,))
    with pytest.raises(TypeError, match="positive integers"):
        SurgeryWarning("test", "message", atoms=(True,))
    with pytest.raises(TypeError, match="blocking"):
        SurgeryWarning("test", "message", blocking=1)
    with pytest.raises(ValueError, match="unknown warning"):
        SurgeryWarning("test", "message", kind="not-a-section")
    with pytest.raises(ValueError, match="non-edge"):
        TopologySnapshot(
            "M",
            3,
            _atoms(3),
            (InteractionSpec("bonds", (1, 2), 1),),
            bond_orders={(1, 3): 1},
        )
    normalized = TopologySnapshot(
        "M",
        3,
        _atoms(3),
        (InteractionSpec("bonds", (1, 2), 1),),
        bond_orders={(2, 1): 2},
    )
    assert normalized.bond_orders == {(1, 2): 2}
    with pytest.raises(ValueError, match="unknown topology edit"):
        TopologyEdit("replace", (1, 2), (), ())
    with pytest.raises(ValueError, match="exactly two"):
        TopologyEdit("cut", (1,), (), ())
    with pytest.raises(TypeError, match="pathlib.Path"):
        ExportedComponent(
            1,
            "M",
            (1,),
            "component.gro",
            Path("component.itp"),
        )
    with pytest.raises(ValueError, match=".gro suffix"):
        ExportedComponent(
            1,
            "M",
            (1,),
            Path("component.xyz"),
            Path("component.itp"),
        )
    with pytest.raises(ValueError, match="at least one component"):
        ExportResult(format="itp", components=())
    exported = ExportedComponent(
        1,
        "M",
        (1,),
        Path("component.gro"),
        Path("component.itp"),
    )
    with pytest.raises(ValueError, match="ordered 1..N"):
        ExportResult(
            format="itp",
            components=(
                exported,
                ExportedComponent(
                    3,
                    "M2",
                    (2,),
                    Path("component_003.gro"),
                    Path("component_003.itp"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="require first-state"):
        InteractionSpec("bonds", (1, 2), 1, (), (0.2, 200.0))
    with pytest.raises(ValueError, match="first-state parameters"):
        TopologySurgery.from_records(
            _atoms(2),
            (
                InteractionSpec(
                    "bonds",
                    (1, 2),
                    1,
                    (0.1, 100.0, 0.2, 200.0),
                ),
            ),
        )
    with pytest.raises(ValueError, match="bond orders"):
        TopologySurgery.from_records(
            _atoms(2),
            (InteractionSpec("bonds", (1, 2), 1),),
            bond_orders={(1, 2): 1.9},
        )


def test_numpy_parameters_are_normalized_to_native_scalars() -> None:
    """NumPy dtype cannot change the public parameter representation."""
    interaction = InteractionSpec(
        "bonds",
        (1, 2),
        1,
        (np.int64(3), np.float32(1.25)),
    )
    assert interaction.parameters == (3, 1.25)
    assert type(interaction.parameters[0]) is int
    assert type(interaction.parameters[1]) is float
