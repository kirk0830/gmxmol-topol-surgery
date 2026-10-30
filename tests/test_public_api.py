"""Public in-memory API tests for direct external-package integration."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from random import Random

import gromologist as gml
import pytest

from topolsurgery import (
    AtomSpec,
    BondedTerm,
    InteractionSpec,
    ParameterRequest,
    TopologySurgery,
)


def _atoms(count: int = 4) -> tuple[AtomSpec, ...]:
    """Return a short all-carbon chain specification."""
    return tuple(
        AtomSpec(
            number=index,
            atom_type="c3",
            atom_name=f"C{index}",
            residue_number=1,
            residue_name="MOL",
            charge=0.0,
            mass=12.011,
        )
        for index in range(1, count + 1)
    )


def test_from_records_connect_resolver_and_topology_diff(
    tmp_path: Path,
) -> None:
    """Callers can construct records in memory and inject parameters."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        requests.append(request)
        if request.kind == "bonds":
            return [BondedTerm(1, (0.153, 251040.0))]
        if request.kind == "angles":
            return [BondedTerm(1, (109.5, 460.24))]
        if request.kind == "dihedrals":
            return [
                BondedTerm(9, (0.0, 0.2, 2)),
                BondedTerm(9, (180.0, 0.1, 3)),
            ]
        if request.kind == "pairs":
            return [BondedTerm(1)]
        return None

    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        name="PIECES",
    )
    result = surgery.connect_bond(
        2,
        3,
        update_pairs=True,
        resolve_parameters=False,
        parameter_resolver=resolver,
    )

    assert {request.kind for request in requests} == {
        "bonds",
        "angles",
        "dihedrals",
        "pairs",
    }
    assert result.component_count == 1
    assert result.edit is not None
    created = result.edit.created
    assert sum(item.kind == "dihedrals" for item in created) == 2
    assert any(
        item.kind == "bonds"
        and item.atoms == (2, 3)
        and item.parameters == (0.153, 251040.0)
        for item in created
    )
    assert result.edit.removed == ()
    assert surgery.graph.connected
    assert result.graph.connected
    assert result.topology.name == "PIECES"
    assert any(
        interaction.kind == "dihedrals" and len(interaction.parameters) == 3
        for interaction in result.topology.interactions
    )

    output = surgery.save(tmp_path / "combined.top")
    parsed = gml.Top(str(output), suppress=True)
    assert parsed.system == [("PIECES", 1)]
    assert parsed.molecules[0].natoms == 4


def test_atom_spec_accepts_zero_residue_number() -> None:
    """The shared record layer accepts marap-style zero residue indices."""
    surgery = TopologySurgery.from_records(
        (
            AtomSpec(
                1,
                "c3",
                "C1",
                residue_number=0,
                residue_name="MOL",
            ),
        )
    )
    assert surgery.atoms[0].residue_number == 0


def test_from_records_orders_coordinates_by_atom_number(
    tmp_path: Path,
) -> None:
    """Unsorted atom records do not scramble their supplied coordinates."""
    atoms = _atoms(2)
    surgery = TopologySurgery.from_records(
        reversed(atoms),
        coordinates=((2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        box=(2.0, 2.0, 2.0),
    )
    exported = surgery.connect_bond(
        1,
        2,
        generate_angles=False,
        resolve_parameters=False,
    ).export(tmp_path / "ordered", format="itp")
    gro_lines = exported.components[0].gro_path.read_text(encoding="utf-8").splitlines()
    assert float(gro_lines[2][20:28]) == pytest.approx(1.0)
    assert float(gro_lines[3][20:28]) == pytest.approx(2.0)


def test_exact_parameter_override_has_priority_over_resolver() -> None:
    """Atom-sequence overrides bypass the caller resolver for that term."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        requests.append(request)
        return [BondedTerm(request.default_function)]

    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    result = surgery.connect_bond(
        2,
        3,
        resolve_parameters=False,
        parameter_resolver=resolver,
        parameter_overrides={
            (3, 2): BondedTerm(1, (999.0, 888.0)),
        },
    )
    bond = next(item for item in result.edit.created if item.kind == "bonds")
    assert bond.parameters == (999.0, 888.0)
    assert all(request.atoms != (2, 3) for request in requests)


def test_empty_override_does_not_suppress_mandatory_graph_terms() -> None:
    """An empty override falls through instead of deleting the required row."""
    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    result = surgery.connect_bond(
        2,
        3,
        update_pairs=True,
        resolve_parameters=False,
        parameter_overrides={(2, 3): ()},
    )

    assert result.graph.has_edge(2, 3)
    assert {item.atoms for item in result.topology.interactions_of("bonds")} == {
        (1, 2),
        (2, 3),
        (3, 4),
    }
    assert result.topology.interactions_of("angles")


def test_unparameterized_connection_warns_without_table_resolution() -> None:
    """Implicit required rows are diagnosed even when table lookup is off."""
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
        generate_angles=False,
        generate_dihedrals=False,
        update_pairs=False,
        resolve_parameters=False,
    )

    assert result.warnings
    assert result.warnings[0].code == "unresolved_parameter"
    assert result.warnings[0].kind == "bonds"
    assert result.warnings[0].atoms == (2, 3)


def test_ambiguous_atom_type_templates_are_not_guessed() -> None:
    """Two different same-type templates leave a new term unresolved."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1, (0.1, 100.0)),
            InteractionSpec("bonds", (3, 4), 1, (0.2, 200.0)),
        ),
    )
    result = surgery.connect_bond(
        2,
        3,
        generate_angles=False,
        generate_dihedrals=False,
        update_pairs=False,
        resolve_parameters=False,
    )
    bond = next(
        item
        for item in result.topology.interactions_of("bonds")
        if item.atoms == (2, 3)
    )

    assert bond.parameters == ()
    assert result.warnings[0].kind == "bonds"
    assert result.warnings[0].atoms == (2, 3)


def test_parameter_resolver_may_return_one_term_directly() -> None:
    """The resolver protocol accepts a single term as well as a sequence."""

    def resolver(request: ParameterRequest) -> BondedTerm | None:
        if request.kind == "bonds":
            return BondedTerm(1, (0.153, 251040.0))
        if request.kind == "angles":
            return BondedTerm(1, (109.5, 460.24))
        if request.kind == "dihedrals":
            return BondedTerm(9, (0.0, 0.2, 3))
        return None

    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    result = surgery.connect_bond(
        2,
        3,
        parameter_resolver=resolver,
        resolve_parameters=False,
    )

    assert {item.parameters for item in result.topology.interactions_of("angles")} == {
        (109.5, 460.24)
    }


def test_resolver_cannot_emit_duplicate_bond_potentials() -> None:
    """Only multi-term function-9 dihedrals may expand to multiple rows."""

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        if request.kind == "bonds":
            return [
                BondedTerm(1, (0.1, 100.0)),
                BondedTerm(1, (0.2, 200.0)),
            ]
        return None

    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    before = surgery.snapshot()
    with pytest.raises(ValueError, match="accepts one generated"):
        surgery.connect_bond(
            2,
            3,
            resolve_parameters=False,
            parameter_resolver=resolver,
        )
    assert surgery.snapshot() == before


@pytest.mark.parametrize(
    ("terms", "message"),
    (
        (
            [
                BondedTerm(9, (0.0, 0.2, 2), (0.0, 0.3, 2)),
                BondedTerm(9, (180.0, 0.1, 3), (180.0, 0.4, 2)),
            ],
            "state-B.*unique multiplicities",
        ),
        (
            [
                BondedTerm(
                    9,
                    (0.0, 0.2, 2, 180.0, 0.1, 3),
                    (0.0, 0.3, 2, 180.0, 0.4, 2),
                )
            ],
            "state-B.*unique multiplicities",
        ),
        (
            [
                BondedTerm(9, (0.0, 0.2, 2), (0.0, 0.3, 2)),
                BondedTerm(9, (180.0, 0.1, 3)),
            ],
            "all define state-B",
        ),
    ),
)
def test_function_9_state_b_consistency_is_transactional(
    terms: list[BondedTerm],
    message: str,
) -> None:
    """Duplicate, repeated, or partial state-B Fourier rows are rejected."""

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        if request.kind == "dihedrals":
            return terms
        return None

    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    before = surgery.snapshot()

    with pytest.raises(ValueError, match=message):
        surgery.connect_bond(
            2,
            3,
            resolve_parameters=False,
            parameter_resolver=resolver,
        )
    assert surgery.snapshot() == before


def test_atom_removal_compacts_graph_and_exports_components(
    tmp_path: Path,
) -> None:
    """Atom removal returns an index map and keeps topology/coordinates aligned."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
            InteractionSpec("angles", (1, 2, 3), 1),
            InteractionSpec("angles", (2, 3, 4), 1),
            InteractionSpec("angles", (3, 4, 5), 1),
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(5)),
        bond_orders={(1, 2): 1, (3, 4): 2},
    )
    result = surgery.remove_atoms((2,))

    assert result.index_map.forward == {1: 1, 3: 2, 4: 3, 5: 4}
    assert result.removed_atoms[0].number == 2
    with pytest.raises(TypeError):
        result.index_map.forward[1] = 99  # type: ignore[index]
    assert tuple(component.atom_numbers for component in result.components) == (
        (1,),
        (2, 3, 4),
    )
    assert surgery.atom_count == 4
    assert surgery.graph.bond_order(2, 3) == 2
    assert {(item.atoms, item.function) for item in surgery.interactions("bonds")} == {
        ((2, 3), 1),
        ((3, 4), 1),
    }

    exported = result.export(tmp_path / "removed", format="itp")
    assert len(exported.components) == 2
    for component in exported.components:
        parsed = gml.Top(str(component.topology_path), suppress=True)
        assert parsed.molecules[0].natoms == len(component.atom_numbers)


def test_result_components_have_public_local_snapshots() -> None:
    """Components can be consumed in memory without writing or re-reading."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
        ),
        bond_orders={(1, 2): 1, (3, 4): 2},
    )
    result = surgery.cut_bond(3, 4)
    snapshots = result.component_snapshots()

    assert tuple(snapshot.atom_count for snapshot in snapshots) == (3, 2)
    assert tuple(
        tuple(atom.number for atom in snapshot.atoms) for snapshot in snapshots
    ) == ((1, 2, 3), (1, 2))
    assert {item.atoms for item in snapshots[0].interactions_of("bonds")} == {
        (1, 2),
        (2, 3),
    }
    assert snapshots[0].bond_orders == {(1, 2): 1}
    assert snapshots[1].bond_orders == {}


def test_export_and_save_refuse_overwrite_by_default(
    tmp_path: Path,
) -> None:
    """Existing artifacts are protected unless overwrite is explicit."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(4)),
    )
    result = surgery.cut_bond(2, 3)
    exported = result.export(tmp_path / "out", format="itp")
    first_payload = exported.gro_paths[0].read_bytes()
    with pytest.raises(FileExistsError, match="already exist"):
        result.export(tmp_path / "out", format="itp")
    assert exported.gro_paths[0].read_bytes() == first_payload
    result.export(tmp_path / "out", format="itp", overwrite=True)

    target = tmp_path / "combined.itp"
    surgery.save(target)
    previous = target.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        surgery.save(target)
    assert target.read_bytes() == previous
    surgery.save(target, overwrite=True)
    invalid = tmp_path / "bad-name.itp"
    with pytest.raises(ValueError, match="non-empty token"):
        surgery.save(invalid, molecule_name="bad name")
    assert not invalid.exists()


def test_save_no_clobber_is_atomic_after_preexistence_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination created after the early check is never replaced."""
    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    target = tmp_path / "raced.itp"
    target.write_bytes(b"competing writer")
    real_exists = Path.exists

    def hide_competing_target(path: Path) -> bool:
        if path == target:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", hide_competing_target)
    with pytest.raises(FileExistsError):
        surgery.save(target)

    assert target.read_bytes() == b"competing writer"
    assert not list(tmp_path.glob(".raced.itp.*.itp"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits required")
def test_save_applies_normal_umask_permissions(
    tmp_path: Path,
) -> None:
    """Successful saves are not left with NamedTemporaryFile's mode 0600."""
    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    target = tmp_path / "permissions.itp"
    previous_mask = os.umask(0o027)
    try:
        surgery.save(target)
    finally:
        os.umask(previous_mask)

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_export_rolls_back_files_after_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late placement failure cannot leave a half-written component set."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(4)),
    )
    result = surgery.cut_bond(2, 3)
    outdir = tmp_path / "failed"
    import topolsurgery.export as export_module

    real_link = export_module.os.link
    calls = 0

    def failing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated placement failure")
        return real_link(source, target)

    monkeypatch.setattr(export_module.os, "link", failing_link)
    with pytest.raises(OSError, match="simulated"):
        result.export(outdir, format="itp")
    assert not list(outdir.glob("*.gro"))
    assert not list(outdir.glob("*.itp"))


def test_export_rolls_back_many_placed_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback remains complete after dozens of successful hard links."""
    atom_count = 40
    surgery = TopologySurgery.from_records(
        _atoms(atom_count),
        tuple(
            InteractionSpec("bonds", (index, index + 1), 1)
            for index in range(1, atom_count, 2)
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(atom_count)),
    )
    result = surgery.cut_bond(1, 2)
    assert result.component_count == 21
    outdir = tmp_path / "many-components"
    import topolsurgery.export as export_module

    real_link = export_module.os.link
    calls = 0

    def failing_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 31:
            raise OSError("late multi-component placement failure")
        return real_link(source, target)

    monkeypatch.setattr(export_module.os, "link", failing_link)
    with pytest.raises(OSError, match="late multi-component"):
        result.export(outdir, format="itp")
    assert not list(outdir.glob("*.gro"))
    assert not list(outdir.glob("*.itp"))


def test_export_rolls_back_if_staged_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target linked before source-unlink failure is still removed."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(4)),
    )
    result = surgery.cut_bond(2, 3)
    outdir = tmp_path / "unlink-failure"
    real_unlink = Path.unlink
    failed = False

    def failing_unlink(path, *args, **kwargs):
        nonlocal failed
        if (
            not failed
            and path.parent.name.startswith(".component.")
            and path.suffix == ".gro"
        ):
            failed = True
            raise OSError("simulated staged unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="simulated staged"):
        result.export(outdir, format="itp")
    assert not list(outdir.glob("*.gro"))
    assert not list(outdir.glob("*.itp"))


def test_topology_only_cut_uses_no_coordinate_dependency(
    tmp_path: Path,
) -> None:
    """A caller without GRO data can still mutate and save an ITP."""
    surgery = TopologySurgery.from_records(
        _atoms(3),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
        ),
    )
    result = surgery.cut_bond(2, 3)

    assert tuple(component.atom_numbers for component in result.components) == (
        (1, 2),
        (3,),
    )
    with pytest.raises(ValueError, match="in-memory export payload"):
        result.export(tmp_path / "no-coords")


def test_failed_resolver_rolls_back_the_whole_connect_transaction() -> None:
    """A late resolver failure leaves atoms, bonds, and graph metadata intact."""
    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        bond_orders={(1, 2): 1},
    )
    before = surgery.snapshot()
    before_graph = surgery.graph

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        if request.kind == "angles":
            raise RuntimeError("late resolver failure")
        return None

    with pytest.raises(RuntimeError, match="late resolver failure"):
        surgery.connect_bond(
            2,
            3,
            update_pairs=True,
            resolve_parameters=False,
            parameter_resolver=resolver,
        )
    assert surgery.snapshot() == before
    assert surgery.graph.signature() == before_graph.signature()
    assert surgery.graph.bond_order(1, 2) == 1
    assert surgery.graph.bond_order(2, 3) is None


def test_interrupt_during_connect_also_rolls_back() -> None:
    """Interrupt-style BaseExceptions cannot leak partial topology edits."""
    surgery = TopologySurgery.from_records(
        _atoms(4),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    before = surgery.snapshot()

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        if request.kind == "angles":
            raise KeyboardInterrupt
        return None

    with pytest.raises(KeyboardInterrupt):
        surgery.connect_bond(
            2,
            3,
            resolve_parameters=False,
            parameter_resolver=resolver,
        )
    assert surgery.snapshot() == before


def test_invalid_parameter_columns_roll_back_with_actionable_error() -> None:
    """Malformed generated parameter rows fail without partial topology edits."""
    surgery = TopologySurgery.from_records(
        _atoms(),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
    )
    before = surgery.snapshot()
    with pytest.raises(ValueError, match="invalid bonds interaction"):
        surgery.connect_bond(
            2,
            3,
            resolve_parameters=False,
            parameter_overrides={(2, 3): BondedTerm(1, (0.153,))},
        )
    assert surgery.snapshot() == before


def test_removal_rejects_empty_duplicate_or_whole_molecule_requests() -> None:
    """Invalid atom-removal sets fail before mutating the session."""
    surgery = TopologySurgery.from_records(
        _atoms(3),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
        ),
    )
    before = surgery.snapshot()
    for invalid in ((), (1, 1), (1, 2, 3), (0,), (4,)):
        with pytest.raises((TypeError, ValueError)):
            surgery.remove_atoms(invalid)
        assert surgery.snapshot() == before


def test_records_round_trip_constraints_settles_and_exclusions() -> None:
    """Record-only construction preserves all supported non-bond sections."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("constraints", (1, 2), 1, (0.1,)),
            InteractionSpec("settles", (3,), 1, (0.09572, 0.15139)),
            InteractionSpec("exclusions", (1, 2, 3), None),
        ),
    )

    assert surgery.components == ((1, 2), (3, 4, 5))
    assert surgery.interactions("constraints")[0].parameters == (0.1,)
    assert surgery.interactions("settles")[0].atoms == (3,)
    exclusion = surgery.interactions("exclusions")[0]
    assert exclusion.atoms == (1, 2, 3)
    assert exclusion.function is None


def test_atom_removal_remaps_explicit_parameter_templates() -> None:
    """Later edges recover parameters whose original atom numbers were compacted."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
            InteractionSpec("bonds", (4, 5), 1),
            InteractionSpec("angles", (1, 2, 3), 1, (100.0, 10.0)),
            InteractionSpec("angles", (2, 3, 4), 1, (110.0, 20.0)),
            InteractionSpec("angles", (3, 4, 5), 1, (120.0, 30.0)),
            InteractionSpec(
                "dihedrals",
                (1, 2, 3, 4),
                9,
                (0.0, 0.5, 2),
            ),
            InteractionSpec(
                "dihedrals",
                (2, 3, 4, 5),
                9,
                (180.0, 0.7, 3),
            ),
        ),
    )
    surgery.remove_atoms((1,))
    surgery.cut_bond(2, 3)
    result = surgery.connect_bond(2, 3, resolve_parameters=False)

    angle = next(
        item
        for item in result.topology.interactions_of("angles")
        if item.atoms == (1, 2, 3)
    )
    dihedral = next(
        item
        for item in result.topology.interactions_of("dihedrals")
        if item.atoms == (1, 2, 3, 4)
    )
    assert angle.parameters == (110.0, 20.0)
    assert dihedral.parameters == (180.0, 0.7, 3)


def test_reconnect_restores_second_state_parameters() -> None:
    """Cut/connect preserves both A and B parameter columns."""
    surgery = TopologySurgery.from_records(
        _atoms(3),
        (
            InteractionSpec(
                "bonds",
                (1, 2),
                1,
                (0.1, 100.0),
                (0.2, 200.0),
            ),
            InteractionSpec(
                "bonds",
                (2, 3),
                1,
                (0.3, 300.0),
                (0.4, 400.0),
            ),
        ),
    )
    surgery.cut_bond(2, 3)
    result = surgery.connect_bond(2, 3, resolve_parameters=False)
    bond = next(
        item
        for item in result.topology.interactions_of("bonds")
        if item.atoms == (2, 3)
    )

    assert bond.parameters == (0.3, 300.0)
    assert bond.second_parameters == (0.4, 400.0)


def test_random_cut_connect_sequence_keeps_interactions_graph_valid() -> None:
    """Seeded edit churn never leaves duplicate or invalid graph-local terms."""
    rng = Random(7331)
    surgery = TopologySurgery.from_records(
        _atoms(7),
        tuple(InteractionSpec("bonds", (index, index + 1), 1) for index in range(1, 7)),
    )

    pair_tracking = False
    for _step in range(60):
        graph = surgery.graph
        if graph.edges and rng.random() < 0.5:
            surgery.cut_bond(*rng.choice(graph.edges))
        else:
            absent = [
                (first, second)
                for first in range(1, 8)
                for second in range(first + 1, 8)
                if not graph.has_edge(first, second)
            ]
            if absent:
                surgery.connect_bond(
                    *rng.choice(absent),
                    update_pairs=True,
                    resolve_parameters=False,
                )
                pair_tracking = True

        graph = surgery.graph
        topology = surgery.snapshot()
        angles = topology.interactions_of("angles")
        dihedrals = topology.interactions_of("dihedrals")
        pairs = topology.interactions_of("pairs")
        assert all(graph.has_path(item.atoms) for item in angles)
        assert all(
            graph.has_path(item.atoms)
            for item in dihedrals
            if item.function not in (2, 4)
        )
        if pair_tracking:
            assert {
                tuple(sorted(item.atoms)) for item in pairs
            } == graph.pairs_at_distance(3)
        assert len({tuple(item.atoms) for item in angles}) == len(angles)
        assert len(
            {
                min(item.atoms, tuple(reversed(item.atoms)))
                for item in dihedrals
                if item.function not in (2, 4)
            }
        ) == sum(item.function not in (2, 4) for item in dihedrals)
