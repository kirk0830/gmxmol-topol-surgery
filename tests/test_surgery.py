"""End-to-end tests for one-bond surgery and component export."""

from __future__ import annotations

from pathlib import Path

import gromologist as gml
import pytest

from topolsurgery import TopologySurgery


def canonical_pair(pair: tuple[int, int]) -> tuple[int, int]:
    """Return an unordered pair in ascending order."""
    return tuple(sorted(pair))


def canonical_angle(angle: tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize angle terminal ordering while preserving its centre."""
    first, center, last = angle
    return (min(first, last), center, max(first, last))


def canonical_dihedral(
    dihedral: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Remove global reversal from a proper-torsion key."""
    return min(dihedral, tuple(reversed(dihedral)))


CHAIN_TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
c3 c3 12.011 0.0 A 0.339967 0.45773

[ bondtypes ]
c3 c3 1 0.1530 251040.0

[ angletypes ]
c3 c3 c3 1 109.5 460.24

[ dihedraltypes ]
c3 c3 c3 c3 9 0.0 0.0 3

[ moleculetype ]
CHAIN 3

[ atoms ]
1 c3 1 MOL C1 1 0.0 12.011
2 c3 1 MOL C2 2 0.0 12.011
3 c3 1 MOL C3 3 0.0 12.011
4 c3 1 MOL C4 4 0.0 12.011
5 c3 1 MOL C5 5 0.0 12.011

[ bonds ]
1 2 1
2 3 1
3 4 1
4 5 1

[ angles ]
1 2 3 1
2 3 4 1
3 4 5 1

[ dihedrals ]
1 2 3 4 9
2 3 4 5 9

[ pairs ]
1 4 1
2 5 1

[ system ]
Chain

[ molecules ]
CHAIN 1
"""


def chain_gro() -> str:
    """Return five fixed-column atoms in chain order."""
    coordinates = (
        (0.000, 0.000, 0.000),
        (0.153, 0.000, 0.000),
        (0.306, 0.000, 0.000),
        (0.459, 0.000, 0.000),
        (0.612, 0.000, 0.000),
    )
    lines = ["Chain", "5"]
    for index, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f"{1:5d}{'MOL':<5s}{f'C{index}':>5s}{index:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def chain_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write the synthetic five-atom chain."""
    topology = tmp_path / "chain.top"
    structure = tmp_path / "chain.gro"
    topology.write_text(CHAIN_TOP, encoding="utf-8")
    structure.write_text(chain_gro(), encoding="utf-8")
    return topology, structure


def test_cut_partitions_and_exports_valid_itp_pairs(
    chain_files: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Cutting one chain bond yields two matching GRO/ITP component pairs."""
    surgery = TopologySurgery.from_files(*chain_files)
    result = surgery.cut_bond(3, 4)

    assert result.operation == "cut"
    assert result.atom_pair == (3, 4)
    assert result.component_count == 2
    assert tuple(component.atom_numbers for component in result.components) == (
        (1, 2, 3),
        (4, 5),
    )

    exported = result.export(tmp_path / "cut", format="itp", prefix="frag")
    assert len(exported.components) == 2
    assert [path.name for path in exported.gro_paths] == [
        "frag_001.gro",
        "frag_002.gro",
    ]
    assert [path.name for path in exported.topology_paths] == [
        "frag_001.itp",
        "frag_002.itp",
    ]

    for component, gro_path, itp_path in zip(
        exported.components,
        exported.gro_paths,
        exported.topology_paths,
        strict=True,
    ):
        gro_lines = gro_path.read_text(encoding="utf-8").splitlines()
        assert gro_lines[1] == str(len(component.atom_numbers))

        parsed = gml.Top(str(itp_path), suppress=True)
        molecule = parsed.molecules[0]
        assert molecule.natoms == len(component.atom_numbers)
        assert molecule.mol_name == itp_path.stem
        for entries in (
            molecule.bonds_section.entries_bonded,
            molecule.angles_section.entries_bonded,
        ):
            assert all(
                1 <= atom <= molecule.natoms
                for entry in entries
                for atom in entry.atom_numbers
            )


def test_cut_removes_terms_that_cross_the_failed_path(
    chain_files: tuple[Path, Path],
) -> None:
    """No angle, torsion, or pair survives across the cut."""
    surgery = TopologySurgery.from_files(*chain_files)
    result = surgery.cut_bond(3, 4)
    topology = result.topology

    assert {
        canonical_pair(entry.atoms) for entry in topology.interactions_of("bonds")
    } == {(1, 2), (2, 3), (4, 5)}
    assert {
        canonical_angle(entry.atoms) for entry in topology.interactions_of("angles")
    } == {(1, 2, 3)}
    assert not topology.interactions_of("dihedrals")
    assert not topology.interactions_of("pairs")


def test_connect_generates_bond_angle_dihedral_and_pairs(
    chain_files: tuple[Path, Path],
) -> None:
    """Reconnecting one edge regenerates every graph-local interaction."""
    surgery = TopologySurgery.from_files(*chain_files)
    surgery.cut_bond(3, 4)
    result = surgery.connect_bond(
        4,
        3,
        bond_parameters=(0.1530, 251040.0),
        update_pairs=True,
        resolve_parameters=False,
    )

    assert result.operation == "connect"
    assert result.atom_pair == (3, 4)
    assert result.component_count == 1
    topology = result.topology

    assert {
        canonical_pair(entry.atoms) for entry in topology.interactions_of("bonds")
    } == {(1, 2), (2, 3), (3, 4), (4, 5)}
    assert {
        canonical_angle(entry.atoms) for entry in topology.interactions_of("angles")
    } == {(1, 2, 3), (2, 3, 4), (3, 4, 5)}
    assert {
        canonical_dihedral(entry.atoms)
        for entry in topology.interactions_of("dihedrals")
    } == {
        canonical_dihedral((1, 2, 3, 4)),
        canonical_dihedral((2, 3, 4, 5)),
    }
    assert {
        canonical_pair(entry.atoms) for entry in topology.interactions_of("pairs")
    } == {(1, 4), (2, 5)}


def test_parameter_resolution_fills_new_records(
    chain_files: tuple[Path, Path],
) -> None:
    """Resolvable new terms receive explicit parameter columns."""
    surgery = TopologySurgery.from_files(*chain_files)
    surgery.cut_bond(3, 4)
    result = surgery.connect_bond(3, 4)
    topology = result.topology

    new_angle = next(
        entry
        for entry in topology.interactions_of("angles")
        if canonical_angle(entry.atoms) == (2, 3, 4)
    )
    new_dihedral = next(
        entry
        for entry in topology.interactions_of("dihedrals")
        if canonical_dihedral(entry.atoms) == canonical_dihedral((1, 2, 3, 4))
    )
    assert new_angle.parameters == pytest.approx((109.5, 460.24))
    assert len(new_dihedral.parameters) == 3
    assert new_dihedral.parameters[2] == 3
    assert not result.warnings


def test_reconnect_restores_explicit_parameter_templates(
    chain_files: tuple[Path, Path],
) -> None:
    """A reconnect reuses explicit angle and torsion values from the input."""
    topology_path, structure_path = chain_files
    text = topology_path.read_text(encoding="utf-8")
    text = text.replace(
        "1 2 3 1\n2 3 4 1\n3 4 5 1",
        ("1 2 3 1 109.1 410.1\n2 3 4 1 109.2 410.2\n3 4 5 1 109.3 410.3"),
    )
    text = text.replace(
        "1 2 3 4 9\n2 3 4 5 9",
        ("1 2 3 4 9 10.0 0.8 2\n1 2 3 4 9 0.5 0.2 3\n2 3 4 5 9 20.0 1.2 3"),
    )
    topology_path.write_text(text, encoding="utf-8")
    surgery = TopologySurgery.from_files(topology_path, structure_path)
    surgery.cut_bond(3, 4)
    result = surgery.connect_bond(3, 4, resolve_parameters=False)
    topology = result.topology

    restored_angle = next(
        entry
        for entry in topology.interactions_of("angles")
        if canonical_angle(entry.atoms) == (2, 3, 4)
    )
    restored_dihedrals = [
        entry
        for entry in topology.interactions_of("dihedrals")
        if canonical_dihedral(entry.atoms) == canonical_dihedral((1, 2, 3, 4))
    ]
    assert restored_angle.parameters == pytest.approx((109.2, 410.2))
    assert len(restored_dihedrals) == 2
    assert sorted(entry.parameters for entry in restored_dihedrals) == [
        pytest.approx((0.5, 0.2, 3)),
        pytest.approx((10.0, 0.8, 2)),
    ]


def test_top_export_round_trips_with_one_system_molecule(
    chain_files: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A TOP component is parseable and lists exactly one molecule copy."""
    surgery = TopologySurgery.from_files(*chain_files)
    result = surgery.cut_bond(3, 4)
    exported = result.export(tmp_path / "tops", format="top", prefix="piece")

    assert len(exported.components) == 2
    for component in exported.components:
        parsed = gml.Top(str(component.topology_path), suppress=True)
        assert parsed.system == [(component.topology_path.stem, 1)]
        assert len(parsed.molecules) == 1
        assert parsed.molecules[0].natoms == len(component.atom_numbers)


def test_repeated_molecule_copies_are_flattened_in_system_order(
    chain_files: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Repeated molecule definitions expose every copy as separate atoms."""
    topology_path, structure_path = chain_files
    text = topology_path.read_text(encoding="utf-8")
    text = text.replace("[ molecules ]\nCHAIN 1", "[ molecules ]\nCHAIN 2")
    topology_path.write_text(text, encoding="utf-8")
    doubled = chain_gro().splitlines()
    atom_lines = doubled[2:7]
    doubled[1] = "10"
    doubled[2:7] = atom_lines + [
        f"{line[:15]}{index:5d}{line[20:]}"
        for index, line in enumerate(atom_lines, start=6)
    ]
    structure_path.write_text("\n".join(doubled) + "\n", encoding="utf-8")

    surgery = TopologySurgery.from_files(topology_path, structure_path)
    assert surgery.atom_count == 10
    result = surgery.cut_bond(3, 4)
    assert tuple(component.atom_numbers for component in result.components) == (
        (1, 2, 3),
        (4, 5),
        (6, 7, 8, 9, 10),
    )


def test_connect_across_two_moleculetypes_merges_them(tmp_path: Path) -> None:
    """Connecting the last atom of one type to the first of another merges both."""
    topology_text = """\
[ moleculetype ]
LEFT 3
[ atoms ]
1 c3 1 L C1 1 0.0 12.011
2 c3 1 L C2 2 0.0 12.011
[ bonds ]
1 2 1
[ moleculetype ]
RIGHT 3
[ atoms ]
1 c3 1 R C1 1 0.0 12.011
2 c3 1 R C2 2 0.0 12.011
[ bonds ]
1 2 1
[ system ]
Two fragments
[ molecules ]
LEFT 1
RIGHT 1
"""
    topology_path = tmp_path / "fragments.top"
    topology_path.write_text(topology_text, encoding="utf-8")
    gro_lines = ["Two fragments", "4"]
    for index, (residue, atom_name) in enumerate(
        (("L", "C1"), ("L", "C2"), ("R", "C1"), ("R", "C2")),
        start=1,
    ):
        gro_lines.append(
            f"{index if residue == 'L' else index - 2:5d}{residue:<5s}"
            f"{atom_name:>5s}{index:5d}"
            f"{0.153 * (index - 1):8.3f}{0.0:8.3f}{0.0:8.3f}"
        )
    gro_lines.append(f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}")
    structure_path = tmp_path / "fragments.gro"
    structure_path.write_text("\n".join(gro_lines) + "\n", encoding="utf-8")

    surgery = TopologySurgery.from_files(topology_path, structure_path)
    result = surgery.connect_bond(
        2,
        3,
        update_pairs=True,
        resolve_parameters=False,
    )
    assert result.component_count == 1
    topology = result.topology
    assert topology.atom_count == 4
    assert {
        canonical_pair(entry.atoms) for entry in topology.interactions_of("bonds")
    } == {(1, 2), (2, 3), (3, 4)}
    assert {
        canonical_angle(entry.atoms) for entry in topology.interactions_of("angles")
    } == {(1, 2, 3), (2, 3, 4)}
    assert {
        canonical_dihedral(entry.atoms)
        for entry in topology.interactions_of("dihedrals")
    } == {canonical_dihedral((1, 2, 3, 4))}
    assert {
        canonical_pair(entry.atoms) for entry in topology.interactions_of("pairs")
    } == {(1, 4)}


def test_constraint_counts_as_connectivity(tmp_path: Path) -> None:
    """Cutting a constraint disconnects otherwise unbonded atoms."""
    topology_path = tmp_path / "constrained.top"
    topology_path.write_text(
        """\
[ moleculetype ]
DIMER 1
[ atoms ]
1 c3 1 A C1 1 0.0 12.011
2 c3 1 B C2 2 0.0 12.011
[ constraints ]
1 2 1 0.1000
[ system ]
Constrained
[ molecules ]
DIMER 1
""",
        encoding="utf-8",
    )
    structure_path = tmp_path / "constrained.gro"
    structure_path.write_text(
        "Constrained\n"
        "2\n"
        f"{1:5d}{'A':<5s}{'C1':>5s}{1:5d}"
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2:5d}{'B':<5s}{'C2':>5s}{2:5d}"
        f"{0.1:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{2.0:10.5f}{2.0:10.5f}{2.0:10.5f}\n",
        encoding="utf-8",
    )

    surgery = TopologySurgery.from_files(topology_path, structure_path)
    assert surgery.components == ((1, 2),)
    result = surgery.cut_bond(1, 2)
    assert tuple(component.atom_numbers for component in result.components) == (
        (1,),
        (2,),
    )
    assert not result.topology.interactions_of("constraints")


def test_pair_update_preserves_custom_pairs_within_a_component(
    chain_files: tuple[Path, Path],
) -> None:
    """Non-1-4 pair records survive while their atoms stay connected."""
    topology_path, structure_path = chain_files
    topology_path.write_text(
        topology_path.read_text(encoding="utf-8").replace(
            "1 4 1\n2 5 1",
            "1 4 1\n2 5 1\n1 3 1",
        ),
        encoding="utf-8",
    )
    surgery = TopologySurgery.from_files(topology_path, structure_path)
    cut = surgery.cut_bond(3, 4)
    left_pairs = {
        canonical_pair(entry.atoms) for entry in cut.topology.interactions_of("pairs")
    }
    assert left_pairs == {(1, 3)}

    joined = surgery.connect_bond(
        3,
        4,
        update_pairs=True,
        resolve_parameters=False,
    )
    joined_pairs = {
        canonical_pair(entry.atoms)
        for entry in joined.topology.interactions_of("pairs")
    }
    assert joined_pairs == {(1, 3), (1, 4), (2, 5)}


def test_itp_input_exports_itp_but_rejects_top(tmp_path: Path) -> None:
    """A molecule-only ITP has enough data for ITP export only."""
    molecule_text = CHAIN_TOP.split("[ system ]", maxsplit=1)[0]
    topology_path = tmp_path / "molecule.itp"
    structure_path = tmp_path / "molecule.gro"
    topology_path.write_text(molecule_text, encoding="utf-8")
    structure_path.write_text(chain_gro(), encoding="utf-8")
    result = TopologySurgery.from_files(topology_path, structure_path).cut_bond(3, 4)

    exported = result.export(tmp_path / "itp", format="itp")
    assert len(exported.topology_paths) == 2
    with pytest.raises(ValueError, match="TOP export requires"):
        result.export(tmp_path / "top", format="top")


def test_invalid_operations_are_rejected(chain_files: tuple[Path, Path]) -> None:
    """Absent, duplicate, and out-of-range operations fail clearly."""
    surgery = TopologySurgery.from_files(*chain_files)
    with pytest.raises(ValueError, match="already connected"):
        surgery.connect_bond(1, 2)
    with pytest.raises(ValueError, match="no \\[ bonds \\] or \\[ constraints \\]"):
        surgery.cut_bond(1, 5)
    with pytest.raises(ValueError, match="outside"):
        surgery.cut_bond(0, 1)
    with pytest.raises(ValueError, match="distinct"):
        surgery.cut_bond(1, 1)
