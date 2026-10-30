"""Conversion between public records and gromologist topology objects."""

from __future__ import annotations

import math
from collections import Counter
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

import gromologist as gml

from topolsurgery._topology import molecule_nrexcl
from topolsurgery.gro import GroAtom, GroFile
from topolsurgery.model import (
    INTERACTION_ARITY,
    INTERACTION_ORDER,
    AtomSpec,
    InteractionKind,
    InteractionSpec,
    ParameterValue,
    TopologySnapshot,
)


def parse_records(
    atoms: Sequence[AtomSpec],
    interactions: Sequence[InteractionSpec],
    *,
    name: str,
    nrexcl: int,
    parameter_sections: str = "",
) -> "gml.Top":
    """Build a parsed gromologist topology from public atom/interaction rows."""
    text = render_topology_text(
        atoms,
        interactions,
        name=name,
        nrexcl=nrexcl,
        parameter_sections=parameter_sections,
    )
    with TemporaryDirectory(prefix="topolsurgery_records_") as directory:
        path = Path(directory) / f"{_safe_name(name)}.top"
        path.write_text(text, encoding="utf-8")
        return gml.Top(str(path), suppress=True)


def render_topology_text(
    atoms: Sequence[AtomSpec],
    interactions: Sequence[InteractionSpec],
    *,
    name: str,
    nrexcl: int,
    parameter_sections: str = "",
) -> str:
    """Render public records as a complete, parseable GROMACS TOP text."""
    atoms = tuple(atoms)
    interactions = tuple(interactions)
    if not atoms:
        raise ValueError("from_records requires at least one atom")
    validate_molecule_name(name)
    if not isinstance(nrexcl, int) or isinstance(nrexcl, bool) or nrexcl < 0:
        raise ValueError("nrexcl must be a non-negative integer")
    ordered_atoms = tuple(sorted(atoms, key=lambda atom: atom.number))
    expected = tuple(range(1, len(ordered_atoms) + 1))
    if tuple(atom.number for atom in ordered_atoms) != expected:
        raise ValueError("AtomSpec.number values must be unique and contiguous from 1")
    atom_count = len(ordered_atoms)
    for interaction in interactions:
        if interaction.kind not in INTERACTION_ORDER:
            raise ValueError(f"unsupported interaction kind {interaction.kind!r}")
        expected_arity = INTERACTION_ARITY[interaction.kind]
        if expected_arity is not None and len(interaction.atoms) != expected_arity:
            raise ValueError(
                f"{interaction.kind} requires {expected_arity} atoms, "
                f"got {len(interaction.atoms)}"
            )
        for atom in interaction.atoms:
            if not 1 <= atom <= atom_count:
                raise ValueError(
                    f"{interaction.kind} references atom {atom}, "
                    f"outside 1..{atom_count}"
                )
        _validate_parameter_columns(interaction)

    lines: list[str] = []
    if parameter_sections.strip():
        lines.extend((parameter_sections.rstrip(), ""))
    lines.extend((f"[ moleculetype ]\n{name} {nrexcl}", "[ atoms ]"))
    for atom in ordered_atoms:
        charge_group = atom.number if atom.charge_group is None else atom.charge_group
        lines.append(
            f"{atom.number} {atom.atom_type} {atom.residue_number} "
            f"{atom.residue_name} {atom.atom_name} {charge_group} "
            f"{atom.charge} {atom.mass}"
        )

    by_kind = {
        kind: [interaction for interaction in interactions if interaction.kind == kind]
        for kind in INTERACTION_ORDER
    }
    for kind in INTERACTION_ORDER:
        rows = by_kind[kind]
        if not rows:
            continue
        lines.extend(("", f"[ {kind} ]"))
        lines.extend(_render_interaction(interaction) for interaction in rows)

    lines.extend(
        (
            "",
            "[ system ]",
            "topolsurgery",
            "",
            "[ molecules ]",
            f"{name} 1",
        )
    )
    return "\n".join(lines) + "\n"


def validate_molecule_name(name: str) -> None:
    """Require one non-empty GROMACS token."""
    if (
        not isinstance(name, str)
        or not name
        or any(character.isspace() for character in name)
    ):
        raise ValueError("molecule name must be a non-empty token")


def _validate_parameter_columns(interaction: InteractionSpec) -> None:
    """Reject parameter-column overflow that could imply an implicit B state."""
    if interaction.kind == "exclusions":
        return
    assert interaction.function is not None
    expected = len(
        gml.EntryBonded.fstr_suff.get(
            (interaction.kind, str(interaction.function)),
            (),
        )
    )
    if interaction.parameters:
        if not expected or len(interaction.parameters) != expected:
            raise ValueError(
                f"{interaction.kind} function {interaction.function} "
                f"expects {expected or 'a known number of'} first-state "
                f"parameters, got {len(interaction.parameters)}"
            )
    if interaction.second_parameters and (
        not expected
        or len(interaction.parameters) != expected
        or len(interaction.second_parameters) != expected
    ):
        raise ValueError(
            f"{interaction.kind} function {interaction.function} "
            "must provide matching explicit A and B parameter columns"
        )


def build_gro(
    atoms: Sequence[AtomSpec],
    coordinates: Sequence[Sequence[float]],
    *,
    velocities: Sequence[Sequence[float]] | None = None,
    box: Sequence[float] = (0.0, 0.0, 0.0),
    title: str = "topolsurgery",
) -> GroFile:
    """Build coordinates matching public atom records."""
    if len(coordinates) != len(atoms):
        raise ValueError("coordinates must have one row per atom")
    if velocities is not None and len(velocities) != len(atoms):
        raise ValueError("velocities must have one row per atom")
    if len(box) not in (3, 9):
        raise ValueError("box must contain 3 or 9 values")
    gro_atoms = []
    for index, atom in enumerate(atoms):
        position_values = tuple(float(value) for value in coordinates[index])
        if len(position_values) != 3:
            raise ValueError("each coordinate row must contain x, y, z")
        x, y, z = position_values
        position = (x, y, z)
        if not all(math.isfinite(value) for value in position):
            raise ValueError("coordinates must be finite")
        velocity = None
        if velocities is not None:
            velocity_values = tuple(float(value) for value in velocities[index])
            if len(velocity_values) != 3:
                raise ValueError("each velocity row must contain x, y, z")
            vx, vy, vz = velocity_values
            velocity = (vx, vy, vz)
            if not all(math.isfinite(value) for value in velocity):
                raise ValueError("velocities must be finite")
        gro_atoms.append(
            GroAtom(
                residue_number=atom.residue_number,
                residue_name=atom.residue_name,
                atom_name=atom.atom_name,
                serial=atom.number,
                position=position,
                velocity=velocity,
            )
        )
    normalized_box = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in normalized_box):
        raise ValueError("box values must be finite")
    return GroFile(
        title=title,
        atoms=tuple(gro_atoms),
        box=normalized_box,
    )


def snapshot_molecule(
    molecule: "gml.SectionMol",
    *,
    name: str | None = None,
    bond_orders: Mapping[tuple[int, int], int] | None = None,
) -> TopologySnapshot:
    """Return an immutable public view of one gromologist molecule."""
    return TopologySnapshot(
        name=molecule.mol_name if name is None else name,
        nrexcl=molecule_nrexcl(molecule),
        atoms=atoms_from_molecule(molecule),
        interactions=interactions_from_molecule(molecule),
        bond_orders={} if bond_orders is None else bond_orders,
        unsupported_sections=tuple(
            sorted(
                {
                    subsection.header
                    for subsection in molecule.subsections
                    if subsection.header
                    not in INTERACTION_ORDER + ("moleculetype", "atoms")
                }
            )
        ),
        lossy_features=tuple(
            feature
            for feature, present in (
                (
                    "conditional_entries",
                    any(
                        entry.condition_frames
                        for subsection in molecule.subsections
                        for entry in subsection.entries
                        if hasattr(entry, "condition_frames")
                    ),
                ),
                (
                    "alchemical_atom_state",
                    any(atom.type_b is not None for atom in molecule.atoms),
                ),
            )
            if present
        ),
    )


def atoms_from_molecule(
    molecule: "gml.SectionMol",
) -> tuple[AtomSpec, ...]:
    """Convert gromologist atom entries to public atom records."""
    return tuple(
        AtomSpec(
            number=atom.num,
            atom_type=atom.type,
            atom_name=atom.atomname,
            residue_number=atom.resid,
            residue_name=atom.resname,
            charge_group=int(atom.content[5]),
            charge=float(atom.charge),
            mass=0.0 if atom.mass is None else float(atom.mass),
        )
        for atom in molecule.atoms
    )


def interactions_from_molecule(
    molecule: "gml.SectionMol",
    kind: InteractionKind | None = None,
) -> tuple[InteractionSpec, ...]:
    """Convert one or all bonded sections to public interaction records."""
    kinds = INTERACTION_ORDER if kind is None else (kind,)
    interactions: list[InteractionSpec] = []
    for section_kind in kinds:
        try:
            subsection = molecule.get_subsection(section_kind)
        except KeyError:
            continue
        interactions.extend(
            _interaction_from_entry(section_kind, entry)
            for entry in subsection.entries_bonded
        )
    return tuple(interactions)


def diff_interactions(
    before: Sequence[InteractionSpec],
    after: Sequence[InteractionSpec],
) -> tuple[tuple[InteractionSpec, ...], tuple[InteractionSpec, ...]]:
    """Return created and removed records as multisets preserving input order."""
    before_counter = Counter(before)
    after_counter = Counter(after)
    removed_counts = before_counter - after_counter
    removed = []
    for interaction in before:
        if removed_counts[interaction]:
            removed_counts[interaction] -= 1
            removed.append(interaction)

    created_counts = after_counter - before_counter
    created: list[InteractionSpec] = []
    for interaction in after:
        if created_counts[interaction]:
            created_counts[interaction] -= 1
            created.append(interaction)
    return tuple(created), tuple(removed)


def diff_after_atom_removal(
    before: Sequence[InteractionSpec],
    after: Sequence[InteractionSpec],
    atom_mapping: Mapping[int, int],
) -> tuple[tuple[InteractionSpec, ...], tuple[InteractionSpec, ...]]:
    """Diff interaction records while accounting for atom renumbering.

    Records that reference a removed atom are reported in their original
    numbering. Surviving records are renumbered before comparison, so a
    compacted but otherwise unchanged row is not reported as a deletion
    and recreation.
    """
    paired: list[tuple[InteractionSpec, InteractionSpec]] = []
    removed: list[InteractionSpec] = []
    for interaction in before:
        if any(atom not in atom_mapping for atom in interaction.atoms):
            removed.append(interaction)
            continue
        paired.append(
            (
                interaction,
                InteractionSpec(
                    kind=interaction.kind,
                    atoms=tuple(atom_mapping[atom] for atom in interaction.atoms),
                    function=interaction.function,
                    parameters=interaction.parameters,
                    second_parameters=interaction.second_parameters,
                ),
            )
        )

    mapped_before = [mapped for _original, mapped in paired]
    mapped_counts = Counter(mapped_before)
    after_counts = Counter(after)
    for original, mapped in paired:
        if mapped_counts[mapped] == 0 or after_counts[mapped] == 0:
            removed.append(original)
            continue
        mapped_counts[mapped] -= 1
        after_counts[mapped] -= 1
    created_counts = Counter(after) - Counter(mapped_before)
    created = []
    for interaction in after:
        if created_counts[interaction]:
            created_counts[interaction] -= 1
            created.append(interaction)
    return tuple(created), tuple(removed)


def _render_interaction(interaction: InteractionSpec) -> str:
    """Render one public interaction row."""
    fields = [str(atom) for atom in interaction.atoms]
    if interaction.kind != "exclusions":
        if interaction.function is None:
            raise ValueError(f"{interaction.kind} interaction requires a function")
        fields.append(str(interaction.function))
        fields.extend(_render_parameter(value) for value in interaction.parameters)
        fields.extend(
            _render_parameter(value) for value in interaction.second_parameters
        )
    return " ".join(fields)


def _interaction_from_entry(
    kind: InteractionKind,
    entry: "gml.EntryBonded",
) -> InteractionSpec:
    """Convert one parsed gromologist interaction entry."""
    function = None if kind == "exclusions" else int(entry.interaction_type)
    return InteractionSpec(
        kind=kind,
        atoms=tuple(int(atom) for atom in entry.atom_numbers),
        function=function,
        parameters=tuple(_public_parameter(value) for value in entry.params_state_a),
        second_parameters=tuple(
            _public_parameter(value) for value in entry.params_state_b
        ),
    )


def _public_parameter(value: object) -> ParameterValue:
    """Convert numpy/scalar parameter values to public scalar types."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean interaction parameters are not valid")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("interaction parameters must be finite")
        return number
    raise TypeError(f"unsupported interaction parameter type {type(value).__name__}")


def _render_parameter(value: ParameterValue) -> str:
    """Render a public parameter value without changing its precision."""
    return value if isinstance(value, str) else repr(value)


def _safe_name(name: str) -> str:
    """Return a filesystem-safe temporary topology stem."""
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )
    return cleaned or "topology"
