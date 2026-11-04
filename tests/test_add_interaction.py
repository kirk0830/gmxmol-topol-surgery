"""Public API tests for TopologySurgery.add_interaction."""

from __future__ import annotations

from pathlib import Path

import gromologist as gml
import pytest

from topolsurgery import (
    AtomSpec,
    BondedTerm,
    InteractionSpec,
    ParameterRequest,
    TopologySurgery,
)

_TYPES = ("c3", "n", "ca", "o", "c3", "c3")

_PARAMETER_SECTIONS = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
c3 c3 12.011 0.0 A 0.339967 0.45773

[ angletypes ]
c3 c3 c3 1 109.5 460.24
"""


def _atoms(count: int = 6) -> tuple[AtomSpec, ...]:
    """Return a chain specification with distinguishable atom types."""
    return tuple(
        AtomSpec(
            number=index,
            atom_type=_TYPES[index - 1] if index <= len(_TYPES) else "c3",
            atom_name=f"C{index}",
            residue_number=1,
            residue_name="MOL",
            charge=0.0,
            mass=12.011,
        )
        for index in range(1, count + 1)
    )


def _chain_bonds(count: int) -> tuple[InteractionSpec, ...]:
    """Return the bonds of a linear chain of *count* atoms."""
    return tuple(
        InteractionSpec("bonds", (index, index + 1), 1) for index in range(1, count)
    )


def _chain_surgery(count: int = 6) -> TopologySurgery:
    """Return a connected linear chain session without angled terms."""
    return TopologySurgery.from_records(_atoms(count), _chain_bonds(count))


def _uniform_chain_surgery(count: int = 6) -> TopologySurgery:
    """Return a chain whose atoms share one type."""
    atoms = tuple(
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
    return TopologySurgery.from_records(atoms, _chain_bonds(count))


@pytest.mark.parametrize(
    ("function", "parameters"),
    ((2, (180.0, 4.6)), (4, (180.0, 4.6, 2))),
)
def test_improper_functions_append_with_explicit_terms(
    function: int,
    parameters: tuple[float, ...],
) -> None:
    """Improper rows append without any consecutive-edge requirement."""
    surgery = _chain_surgery()
    before_graph = surgery.graph.signature()

    result = surgery.add_interaction(
        "dihedrals",
        (2, 1, 3, 4),
        terms=BondedTerm(function, parameters),
        resolve_parameters=False,
    )

    assert result.operation == "add"
    assert result.atom_pair == (1, 4)
    assert result.edit is not None
    assert result.edit.operation == "add"
    assert result.edit.atoms == (1, 2, 3, 4)
    assert result.edit.created == (
        InteractionSpec("dihedrals", (2, 1, 3, 4), function, parameters),
    )
    assert result.edit.removed == ()
    assert surgery.interactions("dihedrals") == result.edit.created
    assert surgery.graph.signature() == before_graph
    assert result.component_count == 1
    assert result.warnings == ()


@pytest.mark.parametrize("function", (2, 4))
def test_improper_rejected_across_components(function: int) -> None:
    """Improper requests need all four atoms in one component."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (4, 5), 1),
        ),
    )
    before = surgery.snapshot()

    with pytest.raises(ValueError, match="multiple connected components"):
        surgery.add_interaction("dihedrals", (1, 2, 3, 4), function=function)

    assert surgery.snapshot() == before


def test_proper_dihedral_rejected_when_an_edge_is_missing() -> None:
    """Proper torsions require all three consecutive graph edges."""
    surgery = _chain_surgery()
    before = surgery.snapshot()

    with pytest.raises(ValueError, match="three consecutive"):
        surgery.add_interaction(
            "dihedrals",
            (1, 2, 3, 5),
            terms=BondedTerm(9, (0.0, 0.2, 3)),
            resolve_parameters=False,
        )

    assert surgery.snapshot() == before


def test_angle_rejected_when_an_edge_is_missing() -> None:
    """Angles require both incident graph edges."""
    surgery = _chain_surgery()

    with pytest.raises(ValueError, match="requires graph edges"):
        surgery.add_interaction(
            "angles",
            (1, 2, 4),
            terms=BondedTerm(1, (109.5, 460.24)),
            resolve_parameters=False,
        )


def test_angle_appends_on_graph_valid_triple() -> None:
    """A graph-valid angle appends and leaves connectivity untouched."""
    surgery = _chain_surgery()
    before_graph = surgery.graph.signature()
    before_components = surgery.components

    result = surgery.add_interaction(
        "angles",
        (2, 3, 4),
        terms=BondedTerm(1, (109.5, 460.24)),
        resolve_parameters=False,
    )

    assert surgery.interactions("angles") == (
        InteractionSpec("angles", (2, 3, 4), 1, (109.5, 460.24)),
    )
    assert result.edit is not None
    assert result.edit.created == surgery.interactions("angles")
    assert result.atom_pair == (2, 4)
    assert result.warnings == ()
    assert surgery.graph.signature() == before_graph
    assert surgery.components == before_components
    assert (
        tuple(component.atom_numbers for component in result.components)
        == before_components
    )


def test_pair_rejected_across_components() -> None:
    """Pair rows require both atoms in one connected component."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (4, 5), 1),
        ),
    )

    with pytest.raises(ValueError, match="different connected components"):
        surgery.add_interaction("pairs", (1, 4), terms=BondedTerm(1))


def test_pair_distance_warning_is_recorded_but_row_is_added() -> None:
    """Non-1-4 pairs append with a non-blocking distance warning."""
    surgery = _chain_surgery()

    far = surgery.add_interaction(
        "pairs",
        (1, 3),
        terms=BondedTerm(1),
        resolve_parameters=False,
    )
    assert surgery.interactions("pairs") == (InteractionSpec("pairs", (1, 3), 1),)
    assert len(far.warnings) == 1
    warning = far.warnings[0]
    assert warning.code == "pair_distance"
    assert warning.kind == "pairs"
    assert warning.atoms == (1, 3)
    assert warning.blocking is False

    canonical = surgery.add_interaction(
        "pairs",
        (1, 4),
        terms=BondedTerm(1),
        resolve_parameters=False,
    )
    assert canonical.warnings == ()
    assert surgery.graph.distance(1, 4) == 3


def test_duplicate_policy_per_section() -> None:
    """Angles, pairs, and impropers deduplicate; propers allow extra rows."""
    surgery = _chain_surgery()

    surgery.add_interaction(
        "angles",
        (2, 3, 4),
        terms=BondedTerm(1, (109.5, 460.24)),
        resolve_parameters=False,
    )
    with pytest.raises(ValueError, match="already exists"):
        surgery.add_interaction(
            "angles",
            (4, 3, 2),
            terms=BondedTerm(1, (100.0, 400.0)),
            resolve_parameters=False,
        )

    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        function=2,
        terms=BondedTerm(2, (180.0, 4.6)),
        resolve_parameters=False,
    )
    with pytest.raises(ValueError, match="already exists"):
        surgery.add_interaction(
            "dihedrals",
            (4, 3, 2, 1),
            function=2,
            terms=BondedTerm(2, (0.0, 1.0)),
            resolve_parameters=False,
        )

    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        terms=[
            BondedTerm(9, (0.0, 0.2, 2)),
            BondedTerm(9, (180.0, 0.1, 3)),
        ],
        resolve_parameters=False,
    )
    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        terms=BondedTerm(9, (0.0, 0.5, 1)),
        resolve_parameters=False,
    )
    assert len(surgery.interactions("dihedrals")) == 4
    with pytest.raises(ValueError, match="identical row"):
        surgery.add_interaction(
            "dihedrals",
            (4, 3, 2, 1),
            terms=BondedTerm(9, (0.0, 0.2, 2)),
            resolve_parameters=False,
        )

    surgery.add_interaction(
        "pairs",
        (1, 4),
        terms=BondedTerm(1),
        resolve_parameters=False,
    )
    with pytest.raises(ValueError, match="already exists"):
        surgery.add_interaction(
            "pairs",
            (4, 1),
            terms=BondedTerm(1),
            resolve_parameters=False,
        )


@pytest.mark.parametrize(
    ("existing", "candidate", "shared_state"),
    (
        (
            BondedTerm(9, (0.0, 0.2, 1)),
            BondedTerm(9, (180.0, 0.5, 1)),
            "state-A",
        ),
        (
            BondedTerm(9, (0.0, 0.2, 2), (0.0, 0.3, 1)),
            BondedTerm(9, (180.0, 0.5, 3), (180.0, 0.6, 1)),
            "state-B",
        ),
    ),
)
def test_function9_multiplicities_are_unique_across_calls(
    existing: BondedTerm,
    candidate: BondedTerm,
    shared_state: str,
) -> None:
    """A later add cannot reuse a multiplicity for the same quartet."""
    surgery = _uniform_chain_surgery()
    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        terms=existing,
        resolve_parameters=False,
    )
    before = surgery.snapshot()

    with pytest.raises(
        ValueError,
        match=rf"{shared_state} multiplicities .* must be unique",
    ):
        surgery.add_interaction(
            "dihedrals",
            (4, 3, 2, 1),
            terms=candidate,
            resolve_parameters=False,
        )

    assert surgery.snapshot() == before


def test_improper_request_does_not_reuse_a_proper_template() -> None:
    """Topology templates index proper torsions only."""
    surgery = _uniform_chain_surgery()
    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        terms=BondedTerm(9, (0.0, 0.2, 3)),
        resolve_parameters=False,
    )

    result = surgery.add_interaction(
        "dihedrals",
        (2, 3, 4, 5),
        function=2,
        resolve_parameters=False,
    )

    assert surgery.interactions("dihedrals")[-1] == InteractionSpec(
        "dihedrals",
        (2, 3, 4, 5),
        2,
    )
    assert result.edit is not None
    assert result.edit.created == (surgery.interactions("dihedrals")[-1],)
    assert [(warning.code, warning.kind) for warning in result.warnings] == [
        ("unresolved_parameter", "dihedrals")
    ]


@pytest.mark.parametrize(
    ("function", "parameters", "second_parameters"),
    (
        (2, (180.0, 4.6), (170.0, 5.1)),
        (4, (180.0, 4.6, 2), (170.0, 5.1, 2)),
    ),
)
def test_resolver_receives_improper_request_and_returns_ab_states(
    function: int,
    parameters: tuple[float, ...],
    second_parameters: tuple[float, ...],
) -> None:
    """Improper requests reach the resolver with canonical atoms."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> BondedTerm:
        requests.append(request)
        if request.kind == "dihedrals" and request.default_function in (2, 4):
            return BondedTerm(
                request.default_function,
                parameters,
                second_parameters,
            )
        return None

    surgery = _chain_surgery()
    result = surgery.add_interaction(
        "dihedrals",
        (4, 3, 2, 1),
        function=function,
        parameter_resolver=resolver,
        resolve_parameters=False,
    )

    assert requests == [
        ParameterRequest(
            kind="dihedrals",
            atoms=(1, 2, 3, 4),
            atom_types=("c3", "n", "ca", "o"),
            default_function=function,
        ),
    ]
    assert surgery.interactions("dihedrals") == (
        InteractionSpec(
            "dihedrals",
            (4, 3, 2, 1),
            function,
            parameters,
            second_parameters,
        ),
    )
    assert result.warnings == ()


def test_resolver_multiterm_proper_rows_are_preserved_in_order() -> None:
    """Multi-term function-9 results append in order with A/B states."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
        requests.append(request)
        if request.kind == "dihedrals":
            return [
                BondedTerm(9, (0.0, 0.2, 2), (0.0, 0.3, 2)),
                BondedTerm(9, (180.0, 0.1, 3), (180.0, 0.4, 3)),
            ]
        return None

    surgery = _chain_surgery()
    result = surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        parameter_resolver=resolver,
        resolve_parameters=False,
    )

    assert requests[0].default_function == 9
    rows = surgery.interactions("dihedrals")
    assert [row.parameters for row in rows] == [
        (0.0, 0.2, 2),
        (180.0, 0.1, 3),
    ]
    assert [row.second_parameters for row in rows] == [
        (0.0, 0.3, 2),
        (180.0, 0.4, 3),
    ]
    assert result.edit is not None
    assert len(result.edit.created) == 2


def test_parameter_overrides_take_priority_over_resolver() -> None:
    """Atom-sequence overrides bypass the resolver for that row."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> list[BondedTerm]:
        requests.append(request)
        return [BondedTerm(request.default_function, (1.0,))]

    surgery = _chain_surgery()
    surgery.add_interaction(
        "angles",
        (2, 3, 4),
        parameter_resolver=resolver,
        parameter_overrides={(4, 3, 2): BondedTerm(1, (120.0, 500.0))},
        resolve_parameters=False,
    )

    assert requests == []
    assert surgery.interactions("angles") == (
        InteractionSpec("angles", (2, 3, 4), 1, (120.0, 500.0)),
    )


def test_explicit_terms_bypass_parameter_resolution() -> None:
    """Given terms win over overrides, the resolver, and templates."""
    requests: list[ParameterRequest] = []

    def resolver(request: ParameterRequest) -> list[BondedTerm]:
        requests.append(request)
        return [BondedTerm(request.default_function, (1.0,))]

    surgery = _chain_surgery()
    surgery.add_interaction(
        "angles",
        (2, 3, 4),
        terms=BondedTerm(1, (109.5, 460.24)),
        parameter_resolver=resolver,
        parameter_overrides={(2, 3, 4): BondedTerm(1, (99.0, 99.0))},
        resolve_parameters=False,
    )

    assert requests == []
    assert surgery.interactions("angles") == (
        InteractionSpec("angles", (2, 3, 4), 1, (109.5, 460.24)),
    )


def test_topology_type_templates_resolve_added_angle() -> None:
    """A unique same-type explicit template fills a new angle row."""
    types = ("n", "ca", "o", "n", "ca", "o")
    surgery = TopologySurgery.from_records(
        tuple(
            AtomSpec(
                number=index,
                atom_type=types[index - 1],
                atom_name=f"C{index}",
                residue_number=1,
                residue_name="MOL",
                charge=0.0,
                mass=12.011,
            )
            for index in range(1, 7)
        ),
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (4, 5), 1),
            InteractionSpec("bonds", (5, 6), 1),
            InteractionSpec("angles", (1, 2, 3), 1, (109.5, 460.24)),
        ),
    )

    result = surgery.add_interaction("angles", (4, 5, 6), resolve_parameters=False)

    assert surgery.interactions("angles")[1] == InteractionSpec(
        "angles",
        (4, 5, 6),
        1,
        (109.5, 460.24),
    )
    assert result.warnings == ()


def test_unparameterized_fallback_row_warns_with_table_resolution() -> None:
    """Nothing resolving leaves an unparameterized row plus a warning."""
    surgery = _chain_surgery()

    result = surgery.add_interaction("angles", (2, 3, 4))

    assert surgery.interactions("angles") == (InteractionSpec("angles", (2, 3, 4), 1),)
    reported = {
        (warning.code, warning.kind, warning.atoms) for warning in result.warnings
    }
    assert ("unresolved_parameter", "angles", (2, 3, 4)) in reported


def test_parameter_resolution_does_not_fill_existing_rows() -> None:
    """An append only resolves table parameters for the requested row."""
    atoms = tuple(
        AtomSpec(index, "c3", f"C{index}", charge_group=index, mass=12.011)
        for index in range(1, 5)
    )
    existing_angle = InteractionSpec("angles", (1, 2, 3), 1)
    surgery = TopologySurgery.from_records(
        atoms,
        (*_chain_bonds(4), existing_angle),
        parameter_sections=_PARAMETER_SECTIONS,
    )

    result = surgery.add_interaction("angles", (2, 3, 4))

    assert surgery.interactions("angles") == (
        existing_angle,
        InteractionSpec("angles", (2, 3, 4), 1, (109.5, 460.24)),
    )
    assert result.edit is not None
    assert result.edit.created == (
        InteractionSpec("angles", (2, 3, 4), 1, (109.5, 460.24)),
    )
    assert result.edit.removed == ()
    assert result.warnings == ()


def test_resolve_parameters_false_reports_only_created_rows() -> None:
    """Without table resolution only new rows are diagnosed as unresolved."""
    surgery = _chain_surgery()

    result = surgery.add_interaction("angles", (2, 3, 4), resolve_parameters=False)

    assert surgery.interactions("angles") == (InteractionSpec("angles", (2, 3, 4), 1),)
    assert [
        (warning.code, warning.kind, warning.atoms) for warning in result.warnings
    ] == [("unresolved_parameter", "angles", (2, 3, 4))]


def test_empty_terms_sequence_is_rejected() -> None:
    """An empty terms sequence cannot stand for an appended row."""
    surgery = _chain_surgery()

    with pytest.raises(ValueError, match="at least one BondedTerm"):
        surgery.add_interaction("angles", (2, 3, 4), terms=[])


def test_remove_atoms_renumbers_appended_improper() -> None:
    """Atom removal compacts and renumbers appended rows correctly."""
    surgery = _chain_surgery()
    surgery.add_interaction(
        "dihedrals",
        (3, 4, 5, 6),
        function=2,
        terms=BondedTerm(2, (180.0, 4.6)),
        resolve_parameters=False,
    )

    result = surgery.remove_atoms((1,))

    assert result.index_map.forward == {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    assert surgery.interactions("dihedrals") == (
        InteractionSpec("dihedrals", (2, 3, 4, 5), 2, (180.0, 4.6)),
    )


def test_cut_bond_elsewhere_preserves_same_component_improper() -> None:
    """Appended rows follow the sanitize and invariant-6 repair rules."""
    surgery = _chain_surgery()
    surgery.add_interaction(
        "angles",
        (3, 4, 5),
        terms=BondedTerm(1, (109.5, 460.24)),
        resolve_parameters=False,
    )
    surgery.add_interaction(
        "dihedrals",
        (3, 4, 5, 6),
        terms=BondedTerm(9, (0.0, 0.2, 3)),
        resolve_parameters=False,
    )
    surgery.add_interaction(
        "dihedrals",
        (3, 4, 5, 6),
        function=2,
        terms=BondedTerm(2, (180.0, 4.6)),
        resolve_parameters=False,
    )

    surgery.cut_bond(1, 2)
    assert surgery.interactions("angles") == (
        InteractionSpec("angles", (3, 4, 5), 1, (109.5, 460.24)),
    )
    assert {
        (item.atoms, item.function) for item in surgery.interactions("dihedrals")
    } == {((3, 4, 5, 6), 9), ((3, 4, 5, 6), 2)}

    surgery.cut_bond(4, 5)
    assert surgery.interactions("angles") == ()
    assert surgery.interactions("dihedrals") == ()


def test_explicit_pair_survives_pair_rebuild() -> None:
    """Non-graph pairs persist through later pair rebuilds."""
    surgery = _chain_surgery(5)
    surgery.cut_bond(3, 4)
    added = surgery.add_interaction(
        "pairs",
        (1, 3),
        terms=BondedTerm(1),
        resolve_parameters=False,
    )
    assert added.warnings[0].code == "pair_distance"

    surgery.connect_bond(3, 4, update_pairs=True, resolve_parameters=False)

    assert {tuple(sorted(item.atoms)) for item in surgery.interactions("pairs")} == {
        (1, 3),
        (1, 4),
        (2, 5),
    }


def test_appended_rows_export_round_trip(tmp_path: Path) -> None:
    """Appended rows survive component snapshots and ITP export."""
    surgery = TopologySurgery.from_records(
        _atoms(5),
        _chain_bonds(5),
        coordinates=tuple((0.15 * index, 0.0, 0.0) for index in range(5)),
    )
    result = surgery.add_interaction(
        "dihedrals",
        (2, 1, 3, 4),
        function=4,
        terms=BondedTerm(4, (180.0, 4.6, 2)),
        resolve_parameters=False,
    )

    snapshots = result.component_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].interactions_of("dihedrals") == (
        InteractionSpec("dihedrals", (2, 1, 3, 4), 4, (180.0, 4.6, 2)),
    )

    exported = result.export(tmp_path / "exported", format="itp")
    assert len(exported.components) == 1
    parsed = gml.Top(str(exported.components[0].topology_path), suppress=True)
    entries = parsed.molecules[0].get_subsection("dihedrals").entries_bonded
    assert len(entries) == 1
    assert tuple(int(atom) for atom in entries[0].atom_numbers) == (2, 1, 3, 4)
    assert int(entries[0].interaction_type) == 4
    assert [float(value) for value in entries[0].params_state_a] == pytest.approx(
        [180.0, 4.6, 2.0]
    )


def test_resolver_exception_rolls_back_the_append() -> None:
    """A raising resolver leaves interactions and the graph unchanged."""
    surgery = _chain_surgery()
    before = surgery.snapshot()
    before_graph = surgery.graph.signature()

    def resolver(request: ParameterRequest) -> BondedTerm:
        raise RuntimeError("resolver failed")

    with pytest.raises(RuntimeError, match="resolver failed"):
        surgery.add_interaction(
            "angles",
            (2, 3, 4),
            parameter_resolver=resolver,
            resolve_parameters=False,
        )

    assert surgery.snapshot() == before
    assert surgery.graph.signature() == before_graph


def test_later_duplicate_term_rolls_back_the_first_row() -> None:
    """A failure after one appended row restores the complete session."""
    surgery = _uniform_chain_surgery()
    surgery.add_interaction(
        "dihedrals",
        (1, 2, 3, 4),
        terms=BondedTerm(9, (0.0, 0.5, 1)),
        resolve_parameters=False,
    )
    before = surgery.snapshot()
    before_graph = surgery.graph.signature()

    with pytest.raises(ValueError, match="identical row"):
        surgery.add_interaction(
            "dihedrals",
            (1, 2, 3, 4),
            terms=[
                BondedTerm(9, (0.0, 0.2, 2)),
                BondedTerm(9, (0.0, 0.5, 1)),
            ],
            resolve_parameters=False,
        )

    assert surgery.snapshot() == before
    assert surgery.graph.signature() == before_graph


@pytest.mark.parametrize(
    "section",
    ("bonds", "constraints", "settles", "cmap", "exclusions", "bogus"),
)
def test_non_appendable_sections_are_rejected(section: str) -> None:
    """Connectivity and unsupported sections point at connect_bond."""
    surgery = _chain_surgery()

    with pytest.raises(ValueError, match="connect_bond"):
        surgery.add_interaction(section, (1, 2))


def test_atom_arity_and_number_validation() -> None:
    """Arity, range, distinctness, and atom types are validated."""
    surgery = _chain_surgery()

    with pytest.raises(ValueError, match="require 3 atoms"):
        surgery.add_interaction("angles", (1, 2))
    with pytest.raises(ValueError, match="require 4 atoms"):
        surgery.add_interaction("dihedrals", (1, 2, 3))
    with pytest.raises(ValueError, match="require 2 atoms"):
        surgery.add_interaction("pairs", (1, 2, 3))
    with pytest.raises(ValueError, match="outside the topology range"):
        surgery.add_interaction("angles", (1, 2, 99))
    with pytest.raises(ValueError, match="pairwise distinct"):
        surgery.add_interaction("angles", (2, 2, 3))
    with pytest.raises(TypeError, match="must be integers"):
        surgery.add_interaction("angles", (1, 2, "3"))
    with pytest.raises(ValueError, match="positive integer"):
        surgery.add_interaction("angles", (2, 3, 4), function=0)
