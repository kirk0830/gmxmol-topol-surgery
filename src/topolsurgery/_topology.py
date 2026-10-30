"""Internal gromologist topology inspection and graph helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import gromologist as gml

from topolsurgery.graph import AtomicGraph
from topolsurgery.model import (
    INTERACTION_ORDER,
    _interaction_connectivity_edges,
    _interaction_ownership_edges,
)

IMPROPER_FUNCTIONS = frozenset({"2", "4"})
TermParams = tuple[int | float | str, ...]
TermTemplate = tuple[int, TermParams, TermParams]
ParameterTemplates = dict[
    tuple[str, tuple[str, ...]],
    tuple[TermTemplate, ...],
]
ExactParameterTemplates = dict[
    tuple[str, tuple[int, ...]],
    tuple[TermTemplate, ...],
]


@dataclass(frozen=True)
class ParameterLibrary:
    """Explicit parameter rows indexed by exact atoms and type patterns."""

    exact: ExactParameterTemplates
    by_type: ParameterTemplates


@dataclass(frozen=True)
class FlatTopology:
    """A topology normalized to one working molecule."""

    top: "gml.Top"
    molecule_name: str


def molecule_nrexcl(molecule: "gml.SectionMol") -> int:
    """Read ``nrexcl`` from a moleculetype header."""
    subsection = molecule.get_subsection("moleculetype")
    for entry in subsection.entries:
        if entry.content and len(entry.content) >= 2:
            return int(entry.content[1])
    raise ValueError(f"molecule {molecule.mol_name!r} has no readable nrexcl")


def flatten_topology(topology: "gml.Top") -> FlatTopology:
    """Materialize system instances into one molecule for atom-level surgery.

    Parameters
    ----------
    topology : gromologist.Top
        Parsed topology, possibly containing repeated molecule copies or
        several active molecule definitions.

    Returns
    -------
    FlatTopology
        A clone whose atom numbering follows the original system order.

    Raises
    ------
    ValueError
        The topology has no unambiguous active molecule.
    """
    top = topology.clone()
    if not top.molecules:
        raise ValueError("topology contains no [ moleculetype ] section")
    _audit_system_level_sections(top)

    initial_system = top.system
    system_names = [name for name, _count in initial_system]
    nrexcls = {name: molecule_nrexcl(top.get_molecule(name)) for name in system_names}
    if len(set(nrexcls.values())) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in sorted(nrexcls.items()))
        raise ValueError(
            "cannot flatten moleculetypes with different nrexcl values "
            f"({detail}); normalize nrexcl or process them separately"
        )
    if len(system_names) != len(set(system_names)):
        _consolidate_system_rows(top)
    for name in dict.fromkeys(name for name, _count in initial_system):
        molecule = top.get_molecule(name)
        if molecule.nmols > 1:
            top.explicit_multiple_molecules(name)

    while len(top.system) > 1:
        base_name = top.system[0][0]
        other_name = top.system[1][0]
        top.merge_molecules(base_name, other_name)

    if top.system:
        molecule_name = top.system[0][0]
    else:
        if len(top.molecules) != 1:
            raise ValueError(
                "an ITP input without [ molecules ] must contain exactly "
                "one [ moleculetype ]"
            )
        molecule_name = top.molecules[0].mol_name

    for name in [
        molecule.mol_name
        for molecule in tuple(top.molecules)
        if molecule.mol_name != molecule_name
    ]:
        top.remove_molecule(name, remove_from_system=False)

    molecule = top.get_molecule(molecule_name)
    expected = tuple(range(1, molecule.natoms + 1))
    actual = tuple(atom.num for atom in molecule.atoms)
    if actual != expected:
        raise ValueError(
            f"molecule {molecule_name!r} atom rows must be ordered and "
            f"contiguously numbered as 1..{molecule.natoms}; got {actual}"
        )

    if top.system:
        rewrite_system(top, molecule_name)
    return FlatTopology(
        top=top,
        molecule_name=molecule_name,
    )


def _audit_system_level_sections(top: "gml.Top") -> None:
    """Reject system-level records that flattened surgery cannot preserve."""
    for section in top.sections:
        if isinstance(section, (gml.SectionMol, gml.SectionParam)):
            continue
        unsupported = sorted(
            subsection.header
            for subsection in section.subsections
            if subsection.header not in ("system", "molecules")
        )
        if unsupported:
            raise ValueError(
                "system-level atom-addressed sections are not supported by "
                f"flattened surgery: {unsupported}; process each moleculetype "
                "separately or remove these sections"
            )
        try:
            molecules = section.get_subsection("molecules")
        except KeyError:
            continue
        if any(entry.condition_frames for entry in molecules.entries):
            raise ValueError(
                "conditional [ molecules ] entries cannot be flattened "
                "without losing their conditionals"
            )


def _consolidate_system_rows(top: "gml.Top") -> None:
    """Merge contiguous repeated ``[ molecules ]`` names."""
    section = next(
        (section for section in top.sections if section.has_subsection("molecules")),
        None,
    )
    if section is None:
        return
    subsection = section.get_subsection("molecules")
    ordered_counts: dict[str, int] = {}
    last_name: str | None = None
    for name, count in top.system:
        if name in ordered_counts and name != last_name:
            raise ValueError(
                "duplicate molecule names in [ molecules ] must be "
                "contiguous for flattened surgery"
            )
        ordered_counts[name] = ordered_counts.get(name, 0) + int(count)
        last_name = name
    comments = [entry for entry in subsection.entries if entry.is_comment]
    subsection.entries = comments
    for name, count in ordered_counts.items():
        subsection.add_entry(gml.Entry(f"{name} {count}", subsection))


def rewrite_system(top: "gml.Top", molecule_name: str) -> None:
    """Replace the system composition with one copy of *molecule_name*.

    Parameters
    ----------
    top : gromologist.Top
        Topology to mutate.
    molecule_name : str
        The sole molecule name written to ``[ molecules ]``.

    Returns
    -------
    None
    """
    system_sections = [
        section for section in top.sections if section.has_subsection("molecules")
    ]
    if not system_sections:
        return
    subsection = system_sections[-1].get_subsection("molecules")
    comments = [entry for entry in subsection.entries if entry.is_comment]
    subsection.entries = comments
    subsection.add_entry(gml.Entry(f"{molecule_name} 1", subsection))


def subset_molecule(
    molecule: "gml.SectionMol",
    atom_numbers: Sequence[int],
) -> dict[int, int]:
    """Keep one atom subset and compactly renumber all molecule records."""
    requested = tuple(int(atom) for atom in atom_numbers)
    if not requested:
        raise ValueError("a component must contain at least one atom")
    if len(requested) != len(set(requested)):
        raise ValueError("component atom numbers must be unique")
    keep = set(requested)
    unknown = keep - {atom.num for atom in molecule.atoms}
    if unknown:
        raise ValueError(f"component references unknown atoms {sorted(unknown)}")
    ordered_old = tuple(atom.num for atom in molecule.atoms if atom.num in keep)
    mapping = {
        old_number: new_number
        for new_number, old_number in enumerate(ordered_old, start=1)
    }

    settle_groups = settle_atom_triples(molecule)
    for subsection in molecule.subsections:
        if subsection.header == "atoms":
            kept_atoms = []
            for entry in subsection.entries:
                if isinstance(entry, gml.EntryAtom):
                    if entry.num in mapping:
                        entry.num = mapping[entry.num]
                        kept_atoms.append(entry)
                else:
                    kept_atoms.append(entry)
            subsection.entries = kept_atoms
            continue
        kept_entries = []
        for entry in subsection.entries:
            if not isinstance(entry, gml.EntryBonded):
                kept_entries.append(entry)
                continue
            if subsection.header == "settles":
                oxygen = int(entry.atom_numbers[0])
                group = next(
                    (triple for triple in settle_groups if triple[0] == oxygen),
                    None,
                )
                if group is None or not set(group).issubset(keep):
                    continue
            if not set(entry.atom_numbers).issubset(keep):
                continue
            entry.atom_numbers = tuple(mapping[atom] for atom in entry.atom_numbers)
            kept_entries.append(entry)
        subsection.entries = kept_entries
    molecule.update_dicts()
    molecule._check_correct()
    return mapping


def bonded_entries(
    molecule: "gml.SectionMol",
    header: str,
) -> list["gml.EntryBonded"]:
    """Return parsed entries of one bonded subsection."""
    try:
        subsection = molecule.get_subsection(header)
    except KeyError:
        return []
    return list(subsection.entries_bonded)


def ensure_bonded_subsection(
    molecule: "gml.SectionMol",
    header: str,
) -> "gml.SubsectionBonded":
    """Return *header*, creating an empty interaction subsection if absent."""
    molecule.init_subsection(header)
    subsection = molecule.get_subsection(header)
    if not isinstance(subsection, gml.SubsectionBonded):
        raise TypeError(f"[ {header} ] is not a bonded subsection")
    return subsection


def append_bonded_entry(
    molecule: "gml.SectionMol",
    header: str,
    atom_numbers: Sequence[int],
    function: int | str,
    parameters: Sequence[int | float | str] = (),
    second_parameters: Sequence[int | float | str] = (),
) -> "gml.EntryBonded":
    """Append one atom-addressed interaction row.

    Parameters
    ----------
    molecule : gromologist.SectionMol
        Molecule whose interaction table is edited.
    header : str
        GROMACS subsection name, such as ``"bonds"``.
    atom_numbers : Sequence[int]
        1-based atoms in interaction order.
    function : int or str
        GROMACS interaction function.
    parameters : Sequence[int | float | str]
        Optional parameter columns after the function.
    second_parameters : Sequence[int | float | str]
        Optional state-B parameter columns.

    Returns
    -------
    gromologist.EntryBonded
        The inserted interaction record.
    """
    subsection = ensure_bonded_subsection(molecule, header)
    fields = [*(str(value) for value in atom_numbers), str(function)]
    fields.extend(str(value) for value in parameters)
    fields.extend(str(value) for value in second_parameters)
    try:
        entry = gml.EntryBonded(" ".join(fields), subsection)
    except Exception as exc:
        raise ValueError(
            f"invalid {header} interaction for atoms {tuple(atom_numbers)} "
            f"function {function}: {exc}"
        ) from exc
    if len(entry.params_state_a) != len(parameters) or len(entry.params_state_b) != len(
        second_parameters
    ):
        raise ValueError(
            f"{header} function {function} parameter columns do not match "
            "the declared A/B fields; GROMACS must not infer state B from "
            "overflow columns"
        )
    subsection.add_entry(entry)
    return entry


def edge_rows(molecule: "gml.SectionMol") -> tuple[tuple[int, int], ...]:
    """Return connectivity edges in topology-file order.

    A ``[ settles ]`` row names only its oxygen. GROMACS assigns the next
    two atoms in the molecule to that rigid water group, so their O-H
    dependencies are included as protected graph connectivity.
    """
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    atom_numbers = set(range(1, molecule.natoms + 1))
    for header in INTERACTION_ORDER:
        for entry in bonded_entries(molecule, header):
            for edge in _interaction_connectivity_edges(
                header,
                entry.atom_numbers,
                atom_numbers=atom_numbers,
            ):
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
    return tuple(edges)


def settle_atom_triples(
    molecule: "gml.SectionMol",
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(oxygen, hydrogen_one, hydrogen_two)`` SETTLE groups."""
    triples = []
    for entry in bonded_entries(molecule, "settles"):
        if len(entry.atom_numbers) != 1:
            raise ValueError("each [ settles ] row must name exactly one oxygen atom")
        oxygen = int(entry.atom_numbers[0])
        hydrogen_one = oxygen + 1
        hydrogen_two = oxygen + 2
        if oxygen < 1 or hydrogen_two > molecule.natoms:
            raise ValueError(
                f"SETTLE oxygen {oxygen} requires atoms "
                f"{hydrogen_one} and {hydrogen_two} in the molecule"
            )
        triples.append((oxygen, hydrogen_one, hydrogen_two))
    return tuple(triples)


def component_dependency_pairs(
    molecule: "gml.SectionMol",
) -> tuple[tuple[int, int], ...]:
    """Return non-chemical ownership dependencies for component grouping."""
    pairs = []
    for header in INTERACTION_ORDER:
        for entry in bonded_entries(molecule, header):
            pairs.extend(
                _interaction_ownership_edges(
                    header,
                    entry.atom_numbers,
                )
            )
    return tuple(pairs)


def component_groups(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
) -> tuple[tuple[int, ...], ...]:
    """Partition atoms while honoring virtual-site ownership dependencies."""
    dependencies = component_dependency_pairs(molecule)
    if not dependencies:
        return graph.components
    return AtomicGraph.from_edges(
        graph.nodes,
        (*graph.edges, *dependencies),
    ).components


def remove_settles_intersecting_atoms(
    molecule: "gml.SectionMol",
    removed_atoms: set[int],
) -> None:
    """Delete any SETTLE group whose oxygen or hydrogen is removed."""
    triples = settle_atom_triples(molecule)
    if not triples:
        return
    subsection = molecule.get_subsection("settles")
    keep = []
    for entry in subsection.entries:
        if isinstance(entry, gml.EntryBonded):
            oxygen = int(entry.atom_numbers[0])
            group = {oxygen, oxygen + 1, oxygen + 2}
            if group.intersection(removed_atoms):
                continue
        keep.append(entry)
    subsection.entries = keep


def canonical_pair(pair: Sequence[int]) -> tuple[int, int]:
    """Return a pair in ascending atom order."""
    first, second = _canonical_atoms(pair)
    return (first, second) if first < second else (second, first)


def canonical_angle(angle: Sequence[int]) -> tuple[int, int, int]:
    """Return an angle with its center fixed and ends unordered."""
    first, center, last = _canonical_atoms(angle)
    return (min(first, last), center, max(first, last))


def canonical_dihedral(dihedral: Sequence[int]) -> tuple[int, int, int, int]:
    """Return a torsion with global reversal removed."""
    values = _canonical_atoms(dihedral)
    reversed_values = tuple(reversed(values))
    return min(values, reversed_values)


def _canonical_atoms(values: Sequence[int]) -> tuple[int, ...]:
    """Freeze integer atom numbers without lossy scalar coercion."""
    atoms = tuple(values)
    if any(not isinstance(atom, int) or isinstance(atom, bool) for atom in atoms):
        raise TypeError("interaction atom numbers must be integers")
    return atoms


def canonical_type_pattern(
    header: str,
    atom_types: Sequence[str],
) -> tuple[str, ...]:
    """Normalize atom-type order for one interaction family."""
    types = tuple(str(atom_type) for atom_type in atom_types)
    if header in ("bonds", "pairs"):
        return tuple(sorted(types))
    if header == "angles":
        first, center, last = types
        return (min(first, last), center, max(first, last))
    if header == "dihedrals":
        reversed_types = tuple(reversed(types))
        return min(types, reversed_types)
    return types


def collect_parameter_templates(
    molecule: "gml.SectionMol",
) -> ParameterLibrary:
    """Index explicit bonded parameters by function and atom-type pattern.

    Multiple rows under one key are retained, which is required for
    GROMACS function 9 torsions whose Fourier terms occupy separate rows.
    """
    atom_types = {atom.num: atom.type for atom in molecule.atoms}
    by_type: dict[
        tuple[str, tuple[str, ...]],
        list[TermTemplate],
    ] = {}
    exact: dict[
        tuple[str, tuple[int, ...]],
        list[TermTemplate],
    ] = {}
    for header in ("bonds", "angles", "dihedrals"):
        for entry in bonded_entries(molecule, header):
            if not entry.params_state_a:
                continue
            if header == "dihedrals" and entry.interaction_type in IMPROPER_FUNCTIONS:
                continue
            types = canonical_type_pattern(
                header,
                [atom_types[atom] for atom in entry.atom_numbers],
            )
            template = (
                int(entry.interaction_type),
                tuple(entry.params_state_a),
                tuple(entry.params_state_b),
            )
            candidates = by_type.setdefault((header, types), [])
            if template not in candidates:
                candidates.append(template)
            atom_key = canonical_interaction(header, entry.atom_numbers)
            exact_candidates = exact.setdefault((header, atom_key), [])
            if template not in exact_candidates:
                exact_candidates.append(template)
    return ParameterLibrary(
        exact={key: tuple(parameters) for key, parameters in exact.items()},
        by_type={key: tuple(parameters) for key, parameters in by_type.items()},
    )


def lookup_parameter_templates(
    templates: ParameterLibrary,
    header: str,
    atoms: Sequence[int],
    atom_types: Sequence[str],
    preferred_function: int,
) -> tuple[TermTemplate, ...]:
    """Return unambiguous exact rows, then one matching type template.

    Exact atom histories preserve multi-term function-9 torsions. Type-only
    matching is intentionally conservative: two different rows for the same
    atom-type pattern are ambiguous and are not guessed.
    """
    exact_key = (header, canonical_interaction(header, atoms))
    exact_candidates = templates.exact.get(exact_key, ())
    selected = _select_parameter_candidates(
        exact_candidates,
        header,
        preferred_function,
    )
    if selected:
        return selected

    key = (header, canonical_type_pattern(header, atom_types))
    candidates = templates.by_type.get(key, ())
    return (candidates[0],) if len(candidates) == 1 else ()


def _select_parameter_candidates(
    candidates: Sequence[TermTemplate],
    header: str,
    preferred_function: int,
) -> tuple[TermTemplate, ...]:
    """Select only unambiguous exact-atom parameter rows."""
    if not candidates:
        return ()
    preferred = tuple(
        candidate for candidate in candidates if candidate[0] == preferred_function
    )
    functions = {candidate[0] for candidate in candidates}
    if preferred:
        selected = preferred
    elif len(functions) == 1:
        selected = tuple(candidates)
    else:
        return ()
    if len(selected) == 1:
        return selected
    if header != "dihedrals":
        return ()
    if any(candidate[0] != 9 or len(candidate[1]) % 3 for candidate in selected):
        return ()
    multiplicities = [
        candidate[1][index]
        for candidate in selected
        for index in range(2, len(candidate[1]), 3)
    ]
    return selected if len(multiplicities) == len(set(multiplicities)) else ()


def remap_parameter_templates(
    templates: ParameterLibrary,
    atom_mapping: Mapping[int, int],
) -> ParameterLibrary:
    """Renumber exact-atom parameter templates after atom removal."""
    exact: dict[
        tuple[str, tuple[int, ...]],
        tuple[TermTemplate, ...],
    ] = {}
    for (header, atoms), parameters in templates.exact.items():
        mapped = tuple(atom_mapping.get(atom) for atom in atoms)
        if any(atom is None for atom in mapped):
            continue
        canonical_atoms = canonical_interaction(
            header,
            tuple(int(atom) for atom in mapped if atom is not None),
        )
        exact[(header, canonical_atoms)] = parameters
    return ParameterLibrary(exact=exact, by_type=templates.by_type)


def merge_parameter_libraries(
    first: ParameterLibrary,
    second: ParameterLibrary,
) -> ParameterLibrary:
    """Merge explicit templates, retaining first-seen row order."""
    exact: dict[
        tuple[str, tuple[int, ...]],
        tuple[TermTemplate, ...],
    ] = {}
    for source in (first, second):
        for key, parameters in source.exact.items():
            current = list(exact.get(key, ()))
            for parameter in parameters:
                if parameter not in current:
                    current.append(parameter)
            exact[key] = tuple(current)

    by_type: dict[
        tuple[str, tuple[str, ...]],
        tuple[TermTemplate, ...],
    ] = {}
    for source in (first, second):
        for key, parameters in source.by_type.items():
            current = list(by_type.get(key, ()))
            for parameter in parameters:
                if parameter not in current:
                    current.append(parameter)
            by_type[key] = tuple(current)
    return ParameterLibrary(exact=exact, by_type=by_type)


def canonical_interaction(
    header: str,
    atoms: Sequence[int],
) -> tuple[int, ...]:
    """Canonicalize an interaction's atom ordering."""
    if header in ("bonds", "pairs"):
        return canonical_pair(atoms)
    if header == "angles":
        return canonical_angle(atoms)
    if header == "dihedrals":
        return canonical_dihedral(atoms)
    return _canonical_atoms(atoms)


def preferred_function(
    molecule: "gml.SectionMol",
    header: str,
    default: int,
    *,
    proper_only: bool = False,
) -> int:
    """Return the most common interaction function in one subsection."""
    functions = [
        entry.interaction_type
        for entry in bonded_entries(molecule, header)
        if not proper_only or entry.interaction_type not in IMPROPER_FUNCTIONS
    ]
    if not functions:
        return default
    return int(Counter(functions).most_common(1)[0][0])


def remove_exact_edges(
    molecule: "gml.SectionMol",
    headers: Sequence[str],
    edge: tuple[int, int],
) -> int:
    """Remove every interaction row matching *edge* from *headers*."""
    removed = 0
    for header in headers:
        try:
            subsection = molecule.get_subsection(header)
        except KeyError:
            continue
        keep = []
        for entry in subsection.entries:
            if (
                isinstance(entry, gml.EntryBonded)
                and len(entry.atom_numbers) == 2
                and canonical_pair(entry.atom_numbers) == edge
            ):
                removed += 1
            else:
                keep.append(entry)
        subsection.entries = keep
    return removed


def sanitize_interactions(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
) -> int:
    """Drop angle, torsional, and cmap rows invalidated by the graph."""
    return (
        _sanitize_angles(molecule, graph)
        + _sanitize_dihedrals(molecule, graph)
        + _sanitize_cmaps(molecule, graph)
    )


def _sanitize_angles(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
) -> int:
    """Drop angle rows without both incident graph edges."""
    try:
        subsection = molecule.get_subsection("angles")
    except KeyError:
        return 0
    removed = 0
    keep = []
    for entry in subsection.entries:
        if isinstance(entry, gml.EntryBonded):
            atoms = entry.atom_numbers
            if len(atoms) != 3 or not graph.has_path(atoms):
                removed += 1
                continue
        keep.append(entry)
    subsection.entries = keep
    return removed


def _sanitize_dihedrals(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
) -> int:
    """Drop invalid propers and component-spanning impropers."""
    component_of = graph.component_of
    removed = 0
    try:
        dihedral_subsection = molecule.get_subsection("dihedrals")
    except KeyError:
        return 0
    keep = []
    for entry in dihedral_subsection.entries:
        if isinstance(entry, gml.EntryBonded):
            atoms = entry.atom_numbers
            if entry.interaction_type in IMPROPER_FUNCTIONS:
                valid = (
                    len(atoms) == 4
                    and all(graph.has_node(atom) for atom in atoms)
                    and len({component_of[atom] for atom in atoms}) == 1
                )
            else:
                valid = len(atoms) == 4 and graph.has_path(atoms)
            if not valid:
                removed += 1
                continue
        keep.append(entry)
    dihedral_subsection.entries = keep
    return removed


def _sanitize_cmaps(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
) -> int:
    """Drop CMAP rows whose five-atom path no longer exists."""
    try:
        subsection = molecule.get_subsection("cmap")
    except KeyError:
        return 0
    removed = 0
    keep = []
    for entry in subsection.entries:
        if isinstance(entry, gml.EntryBonded):
            atoms = entry.atom_numbers
            if (
                len(atoms) != 5
                or not all(graph.has_node(atom) for atom in atoms)
                or not graph.has_path(atoms)
            ):
                removed += 1
                continue
        keep.append(entry)
    subsection.entries = keep
    return removed


def rebuild_pairs(
    molecule: "gml.SectionMol",
    graph: "AtomicGraph",
    *,
    create: bool,
    previous_expected: set[tuple[int, int]] | None = None,
    append_term: Callable[[Sequence[int], int], None] | None = None,
) -> int:
    """Rewrite 1-4 pairs from the current shortest-path graph.

    Graph-generated pairs from *previous_expected* that no longer have
    distance three are removed. Explicit records that were not graph pairs
    are preserved while their atoms remain in one connected component.

    Parameters
    ----------
    molecule : gromologist.SectionMol
        Molecule whose pair section is edited.
    graph : AtomicGraph
        Current connectivity.
    create : bool
        Create an absent ``[ pairs ]`` section.
    previous_expected : set[tuple[int, int]], optional
        Graph-distance-three pairs before this operation.
    append_term : Callable, optional
        Caller-owned pair appender used for parameter resolution.

    Returns
    -------
    int
        Number of newly appended graph pairs.
    """
    try:
        subsection = molecule.get_subsection("pairs")
    except KeyError:
        if not create:
            return 0
        subsection = ensure_bonded_subsection(molecule, "pairs")

    expected = graph.pairs_at_distance(3)
    default_function = preferred_function(molecule, "pairs", 1)
    component_of = graph.component_of
    retained: dict[tuple[int, int], "gml.EntryBonded"] = {}
    for entry in subsection.entries:
        if not isinstance(entry, gml.EntryBonded):
            continue
        if len(entry.atom_numbers) != 2:
            continue
        pair = canonical_pair(entry.atom_numbers)
        if not all(graph.has_node(atom) for atom in pair):
            continue
        was_graph_pair = previous_expected is not None and pair in previous_expected
        same_component = component_of[pair[0]] == component_of[pair[1]]
        if (
            pair in expected or (not was_graph_pair and same_component)
        ) and pair not in retained:
            retained[pair] = entry

    subsection.entries = [
        entry for entry in subsection.entries if not isinstance(entry, gml.EntryBonded)
    ]
    for pair in sorted(retained):
        subsection.add_entry(retained[pair])

    added = 0
    for pair in sorted(expected - retained.keys()):
        if append_term is None:
            append_bonded_entry(molecule, "pairs", pair, default_function)
        else:
            append_term(pair, default_function)
        added += 1
    return added


def resolve_parameter_tables(
    molecule: "gml.SectionMol",
) -> list[tuple[str, int, tuple[int, ...]]]:
    """Fill missing explicit parameters from the parsed force-field tables.

    Returns
    -------
    list[tuple[str, int, tuple[int, ...]]]
        ``(section, function, atoms)`` records that remain without explicit
        parameters.
    """
    parameter_headers = {
        "bonds": "bondtypes",
        "angles": "angletypes",
        "dihedrals": "dihedraltypes",
    }
    try:
        parameters = molecule.top.parameters
    except RuntimeError:
        parameters = None
    if parameters is not None:
        available = {subsection.header for subsection in parameters.subsections}
        for header, table in parameter_headers.items():
            if table in available:
                molecule.add_ff_params(add_section=header)

    missing: list[tuple[str, int, tuple[int, ...]]] = []
    for header in ("bonds", "angles", "dihedrals"):
        for entry in bonded_entries(molecule, header):
            if not entry.params_state_a:
                missing.append(
                    (
                        header,
                        int(entry.interaction_type),
                        tuple(int(atom) for atom in entry.atom_numbers),
                    )
                )
    return missing
