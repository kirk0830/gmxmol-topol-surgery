"""Public data objects returned by topology surgery and export."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from topolsurgery.graph import AtomicGraph
    from topolsurgery.gro import GroFile

EditOperation = Literal["cut", "connect", "remove_atoms"]
SurgeryOperation = Literal["cut", "connect"]
OutputFormat = Literal["itp", "top"]
InteractionKind = Literal[
    "bonds",
    "pairs",
    "angles",
    "dihedrals",
    "pairs_nb",
    "position_restraints",
    "dihedral_restraints",
    "virtual_sites2",
    "virtual_sites3",
    "cmap",
    "constraints",
    "settles",
    "exclusions",
]
INTERACTION_ORDER: tuple[InteractionKind, ...] = (
    "bonds",
    "pairs",
    "angles",
    "dihedrals",
    "pairs_nb",
    "position_restraints",
    "dihedral_restraints",
    "virtual_sites2",
    "virtual_sites3",
    "cmap",
    "constraints",
    "settles",
    "exclusions",
)
INTERACTION_KINDS = frozenset(INTERACTION_ORDER)
INTERACTION_ARITY: dict[InteractionKind, int | None] = {
    "bonds": 2,
    "pairs": 2,
    "angles": 3,
    "dihedrals": 4,
    "pairs_nb": 2,
    "position_restraints": 1,
    "dihedral_restraints": 4,
    "virtual_sites2": 3,
    "virtual_sites3": 4,
    "cmap": 5,
    "constraints": 2,
    "settles": 1,
    "exclusions": None,
}
_CONNECTIVITY_ROLES = {
    "bonds": "pair",
    "constraints": "pair",
    "settles": "settle",
}
_OWNERSHIP_ROLES = {
    "virtual_sites2": "single_site",
    "virtual_sites3": "single_site",
}
ParameterValue = int | float | str


def _interaction_connectivity_edges(
    kind: InteractionKind,
    atoms: Sequence[int],
    *,
    atom_numbers: set[int],
) -> tuple[tuple[int, int], ...]:
    """Return chemical graph edges contributed by one interaction."""
    role = _CONNECTIVITY_ROLES.get(kind)
    if role is None:
        return ()
    values = tuple(atoms)
    if role == "pair":
        if len(values) != 2:
            raise ValueError(f"{kind} requires 2 atoms, got {values}")
        edges = ((min(values), max(values)),)
    else:
        if len(values) != 1:
            raise ValueError(f"{kind} requires 1 atom, got {values}")
        oxygen = values[0]
        edges = ((oxygen, oxygen + 1), (oxygen, oxygen + 2))
    unknown = {atom for edge in edges for atom in edge} - atom_numbers
    if unknown:
        raise ValueError(
            f"{kind} interaction references atoms outside topology: {sorted(unknown)}"
        )
    return edges


def _interaction_ownership_edges(
    kind: InteractionKind,
    atoms: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Return component-ownership dependencies for one interaction."""
    if kind not in _OWNERSHIP_ROLES:
        return ()
    values = tuple(atoms)
    expected = INTERACTION_ARITY[kind]
    if expected is None or len(values) != expected:
        raise ValueError(f"{kind} requires {expected} atoms, got {values}")
    site = values[0]
    return tuple((site, atom) for atom in values[1:])


def _validate_token(value: str, label: str) -> str:
    """Validate one non-empty whitespace-free public token."""
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be one non-empty token")
    return value


def _validate_token_tuple(
    values: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Validate and freeze a sequence of unique public tokens."""
    frozen = tuple(_validate_token(value, label) for value in values)
    if len(frozen) != len(set(frozen)):
        raise ValueError(f"{label} values must be unique")
    return frozen


def _validate_atom_numbers(
    values: Sequence[int],
    label: str,
    *,
    allow_empty: bool = False,
    require_sorted: bool = False,
) -> tuple[int, ...]:
    """Validate, freeze, and optionally order-check 1-based atom numbers."""
    frozen = tuple(values)
    if not frozen and not allow_empty:
        raise ValueError(f"{label} must contain at least one atom")
    if any(
        not isinstance(atom, int) or isinstance(atom, bool) or atom < 1
        for atom in frozen
    ):
        raise TypeError(f"{label} atoms must be positive integers")
    if len(frozen) != len(set(frozen)):
        raise ValueError(f"{label} atoms must be unique")
    if require_sorted and frozen != tuple(sorted(frozen)):
        raise ValueError(f"{label} atoms must be in ascending order")
    return frozen


def _validate_parameter_values(
    values: Sequence[ParameterValue],
    label: str,
) -> tuple[ParameterValue, ...]:
    """Validate and freeze GROMACS parameter columns."""
    frozen = tuple(values)
    normalized: list[ParameterValue] = []
    for value in frozen:
        if isinstance(value, str):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{label} string values must be single tokens")
            normalized.append(value)
        elif isinstance(value, Integral) and not isinstance(value, bool):
            normalized.append(int(value))
        elif isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label} numeric values must be finite")
            normalized.append(number)
        else:
            raise TypeError(f"{label} must contain finite numbers or tokens")
    return tuple(normalized)


@dataclass(frozen=True)
class AtomSpec:
    """One force-field atom record accepted and returned by the public API."""

    number: int
    atom_type: str
    atom_name: str
    residue_number: int = 1
    residue_name: str = "MOL"
    charge_group: int | None = None
    charge: float = 0.0
    mass: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed atom records at the public boundary."""
        if not isinstance(self.number, int) or isinstance(self.number, bool):
            raise TypeError("AtomSpec.number must be an integer")
        if self.number < 1:
            raise ValueError("AtomSpec.number must be positive")
        if not isinstance(self.residue_number, int) or isinstance(
            self.residue_number, bool
        ):
            raise TypeError("AtomSpec.residue_number must be an integer")
        if self.residue_number < 0:
            raise ValueError("AtomSpec.residue_number must be non-negative")
        for field_name in ("atom_type", "atom_name", "residue_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AtomSpec.{field_name} must be non-empty")
            if any(character.isspace() for character in value):
                raise ValueError(
                    f"AtomSpec.{field_name} must be a single GROMACS token"
                )
        if self.charge_group is not None and (
            not isinstance(self.charge_group, int)
            or isinstance(self.charge_group, bool)
            or self.charge_group < 1
        ):
            raise ValueError("AtomSpec.charge_group must be a positive integer")
        for field_name in ("charge", "mass"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"AtomSpec.{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"AtomSpec.{field_name} must be finite")
        if self.mass < 0:
            raise ValueError("AtomSpec.mass cannot be negative")


@dataclass(frozen=True)
class InteractionSpec:
    """One atom-addressed GROMACS interaction record."""

    kind: InteractionKind
    atoms: tuple[int, ...]
    function: int | None = 1
    parameters: tuple[ParameterValue, ...] = ()
    second_parameters: tuple[ParameterValue, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze one public interaction record."""
        if self.kind not in INTERACTION_KINDS:
            raise ValueError(f"unsupported interaction kind {self.kind!r}")
        atoms = tuple(self.atoms)
        if not atoms:
            raise ValueError(f"{self.kind} interaction must reference atoms")
        if any(
            not isinstance(atom, int) or isinstance(atom, bool) or atom < 1
            for atom in atoms
        ):
            raise ValueError(f"{self.kind} interaction atoms must be positive integers")
        if len(atoms) != len(set(atoms)):
            raise ValueError(f"{self.kind} interaction cannot repeat an atom")
        expected_arity = INTERACTION_ARITY[self.kind]
        if expected_arity is not None and len(atoms) != expected_arity:
            raise ValueError(
                f"{self.kind} requires {expected_arity} atoms, got {len(atoms)}"
            )
        if self.kind == "exclusions" and len(atoms) < 2:
            raise ValueError("exclusions require at least two atoms")
        if self.kind == "exclusions":
            if self.function is not None:
                raise ValueError("exclusions records have no function column")
            if self.parameters or self.second_parameters:
                raise ValueError("exclusions records have no parameter columns")
        elif (
            not isinstance(self.function, int)
            or isinstance(self.function, bool)
            or self.function < 1
        ):
            raise ValueError(
                f"{self.kind} interaction requires a positive integer function"
            )
        parameters = _validate_parameter_values(
            self.parameters,
            "interaction parameters",
        )
        second_parameters = _validate_parameter_values(
            self.second_parameters,
            "interaction second-state parameters",
        )
        if second_parameters and not parameters:
            raise ValueError("second-state parameters require first-state parameters")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "second_parameters", second_parameters)


@dataclass(frozen=True)
class BondedTerm:
    """GROMACS function and parameter columns for one generated term."""

    function: int
    parameters: tuple[ParameterValue, ...] = ()
    second_parameters: tuple[ParameterValue, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze one generated parameter row."""
        if (
            not isinstance(self.function, int)
            or isinstance(self.function, bool)
            or self.function < 1
        ):
            raise ValueError("BondedTerm.function must be a positive integer")
        parameters = _validate_parameter_values(
            self.parameters,
            "bonded parameters",
        )
        second_parameters = _validate_parameter_values(
            self.second_parameters,
            "bonded second-state parameters",
        )
        if second_parameters and not parameters:
            raise ValueError("second-state parameters require first-state parameters")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "second_parameters", second_parameters)


@dataclass(frozen=True)
class ParameterRequest:
    """Context handed to a caller-supplied interaction parameter resolver."""

    kind: InteractionKind
    atoms: tuple[int, ...]
    atom_types: tuple[str, ...]
    default_function: int


class ParameterResolver(Protocol):
    """Resolve one generated interaction.

    Returning ``None`` or an empty sequence means "not handled"; the
    surgery then falls back to explicit topology templates and finally to
    an unparameterized row. An empty result never suppresses the mandatory
    graph-derived interaction.
    """

    def __call__(
        self,
        request: ParameterRequest,
    ) -> BondedTerm | Sequence[BondedTerm] | None:
        """Return one or more terms, or ``None`` when not handled."""


ParameterOverrides = Mapping[
    tuple[int, ...],
    BondedTerm | Sequence[BondedTerm],
]


@dataclass(frozen=True)
class IndexMap:
    """Old-to-new 1-based atom mapping after atom removal."""

    forward: Mapping[int, int]
    backward: Mapping[int, int]

    def __post_init__(self) -> None:
        """Freeze both directions of the renumbering."""
        forward = dict(self.forward)
        backward = dict(self.backward)
        if any(
            not isinstance(old, int)
            or isinstance(old, bool)
            or old < 1
            or not isinstance(new, int)
            or isinstance(new, bool)
            or new < 1
            for old, new in forward.items()
        ):
            raise ValueError("IndexMap entries must be positive integers")
        if sorted(forward.values()) != list(range(1, len(forward) + 1)):
            raise ValueError("IndexMap forward values must be contiguous")
        if len(set(forward.values())) != len(forward):
            raise ValueError("IndexMap forward values must be unique")
        if backward != {new: old for old, new in forward.items()}:
            raise ValueError("IndexMap forward and backward mappings disagree")
        object.__setattr__(
            self,
            "forward",
            MappingProxyType(forward),
        )
        object.__setattr__(
            self,
            "backward",
            MappingProxyType(backward),
        )

    def map_atom(self, old_number: int) -> int:
        """Translate an old atom number to the compacted numbering."""
        return self.forward[old_number]

    def unmap_atom(self, new_number: int) -> int:
        """Translate a compacted atom number back to the old numbering."""
        return self.backward[new_number]


@dataclass(frozen=True)
class TopologySnapshot:
    """Immutable public view of the current working topology."""

    name: str
    nrexcl: int
    atoms: tuple[AtomSpec, ...]
    interactions: tuple[InteractionSpec, ...]
    bond_orders: Mapping[tuple[int, int], int] = field(default_factory=dict)
    unsupported_sections: tuple[str, ...] = ()
    lossy_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the public snapshot representation."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(character.isspace() for character in self.name)
        ):
            raise ValueError("snapshot name must be one non-empty token")
        if (
            not isinstance(self.nrexcl, int)
            or isinstance(self.nrexcl, bool)
            or self.nrexcl < 0
        ):
            raise ValueError("snapshot nrexcl must be a non-negative integer")
        atoms = tuple(self.atoms)
        interactions = tuple(self.interactions)
        if not atoms:
            raise ValueError("TopologySnapshot requires at least one atom")
        if any(not isinstance(atom, AtomSpec) for atom in atoms):
            raise TypeError("snapshot atoms must contain AtomSpec records")
        if any(
            not isinstance(interaction, InteractionSpec) for interaction in interactions
        ):
            raise TypeError(
                "snapshot interactions must contain InteractionSpec records"
            )
        atom_numbers = tuple(atom.number for atom in atoms)
        if atom_numbers != tuple(range(1, len(atoms) + 1)):
            raise ValueError("TopologySnapshot atom numbers must be ordered 1..N")
        atom_number_set = set(atom_numbers)
        for interaction in interactions:
            if any(atom not in atom_number_set for atom in interaction.atoms):
                raise ValueError(
                    f"{interaction.kind} references an atom outside snapshot"
                )
        from topolsurgery.graph import AtomicGraph

        graph = AtomicGraph.from_edges(
            atom_numbers,
            _connectivity_edges(
                interactions,
                atom_numbers=set(atom_numbers),
            ),
            bond_orders=self.bond_orders,
        )
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "interactions", interactions)
        object.__setattr__(
            self,
            "bond_orders",
            MappingProxyType(dict(graph.bond_orders)),
        )
        object.__setattr__(
            self,
            "unsupported_sections",
            _validate_token_tuple(
                self.unsupported_sections,
                "unsupported section",
            ),
        )
        object.__setattr__(
            self,
            "lossy_features",
            _validate_token_tuple(
                self.lossy_features,
                "lossy feature",
            ),
        )

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in this snapshot."""
        return len(self.atoms)

    @property
    def graph(self) -> "AtomicGraph":
        """Return chemical connectivity, including implicit SETTLE edges."""
        from topolsurgery.graph import AtomicGraph

        return AtomicGraph.from_edges(
            tuple(atom.number for atom in self.atoms),
            _connectivity_edges(
                self.interactions,
                atom_numbers={atom.number for atom in self.atoms},
            ),
            bond_orders=self.bond_orders,
        )

    @property
    def component_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return component ownership including virtual-site dependencies."""
        from topolsurgery.graph import AtomicGraph

        atom_numbers = tuple(atom.number for atom in self.atoms)
        return AtomicGraph.from_edges(
            atom_numbers,
            (
                *_connectivity_edges(
                    self.interactions,
                    atom_numbers=set(atom_numbers),
                ),
                *_component_dependency_edges(self.interactions),
            ),
        ).components

    def interactions_of(
        self,
        kind: InteractionKind,
    ) -> tuple[InteractionSpec, ...]:
        """Return only interactions of one GROMACS subsection."""
        if kind not in INTERACTION_KINDS:
            raise ValueError(f"unknown interaction kind {kind!r}")
        return tuple(
            interaction for interaction in self.interactions if interaction.kind == kind
        )


@dataclass(frozen=True)
class TopologyEdit:
    """Created and removed interaction rows for one surgery."""

    operation: EditOperation
    atoms: tuple[int, ...]
    created: tuple[InteractionSpec, ...]
    removed: tuple[InteractionSpec, ...]

    def __post_init__(self) -> None:
        """Validate one public structural edit record."""
        if self.operation not in ("cut", "connect", "remove_atoms"):
            raise ValueError(f"unknown topology edit operation {self.operation!r}")
        atoms = _validate_atom_numbers(
            self.atoms,
            "topology edit",
            require_sorted=True,
        )
        if self.operation in ("cut", "connect") and len(atoms) != 2:
            raise ValueError(
                "cut/connect topology edits must reference exactly two atoms"
            )
        created = tuple(self.created)
        removed = tuple(self.removed)
        if any(not isinstance(item, InteractionSpec) for item in created + removed):
            raise TypeError(
                "topology edit created/removed rows must be InteractionSpec records"
            )
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "created", created)
        object.__setattr__(self, "removed", removed)


@dataclass(frozen=True)
class SurgeryWarning:
    """Structured non-fatal surgery diagnostic."""

    code: str
    message: str
    kind: InteractionKind | None = None
    atoms: tuple[int, ...] = ()
    blocking: bool = True

    def __post_init__(self) -> None:
        """Validate a public diagnostic record."""
        _validate_token(self.code, "warning code")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("warning message must be non-empty")
        if self.kind is not None and self.kind not in INTERACTION_KINDS:
            raise ValueError(f"unknown warning interaction kind {self.kind!r}")
        if not isinstance(self.blocking, bool):
            raise TypeError("warning blocking must be a boolean")
        atoms = _validate_atom_numbers(
            self.atoms,
            "warning",
            allow_empty=True,
        )
        object.__setattr__(self, "atoms", atoms)

    def __str__(self) -> str:
        """Render the warning for CLI and log output."""
        return self.message


class _TopologyResultMixin:
    """Read-only behavior shared by topology operation results."""

    components: tuple["Component", ...]
    _top: Any
    _gro: "GroFile | None"
    _bond_orders: Mapping[tuple[int, int], int]

    @property
    def component_count(self) -> int:
        """Return the number of result ownership groups."""
        return len(self.components)

    @property
    def topology(self) -> TopologySnapshot:
        """Return the complete result topology as immutable records."""
        return _snapshot_result_topology(self._top, self._bond_orders)

    @property
    def graph(self) -> "AtomicGraph":
        """Return the result topology's connectivity graph."""
        return self.topology.graph

    def export(
        self,
        outdir: str | Path,
        *,
        format: OutputFormat = "itp",
        prefix: str = "component",
        overwrite: bool = False,
    ) -> ExportResult:
        """Write each ownership group as a GRO/topology pair."""
        from topolsurgery.export import export_result

        return export_result(
            self,
            outdir,
            format=format,
            prefix=prefix,
            overwrite=overwrite,
        )

    def component_snapshots(self) -> tuple[TopologySnapshot, ...]:
        """Return components with local 1-based atom numbering."""
        from topolsurgery.export import component_snapshots

        return component_snapshots(self)


@dataclass(frozen=True)
class AtomRemovalResult(_TopologyResultMixin):
    """Atom-removal result with numbering and component provenance."""

    requested_atoms: tuple[int, ...]
    removed_atoms: tuple[AtomSpec, ...]
    index_map: IndexMap
    components: tuple["Component", ...]
    warnings: tuple[SurgeryWarning, ...] = ()
    edit: TopologyEdit | None = None
    operation: Literal["remove_atoms"] = "remove_atoms"
    _gro: "GroFile | None" = field(default=None, repr=False, compare=False)
    _top: Any = field(default=None, repr=False, compare=False)
    _bond_orders: Mapping[tuple[int, int], int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _parameter_templates: Any = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class Component:
    """One ownership group in the post-operation topology.

    Ownership groups use chemical connectivity plus virtual-site dependencies.
    They may contain one or more chemical components when a virtual site must
    stay with its host atoms.

    Attributes
    ----------
    index : int
        1-based component index in stable atom-order traversal.
    molecule_name : str
        Suggested molecule name derived from the input molecule.
    atom_numbers : tuple[int, ...]
        1-based atom numbers in the working flat topology.
    """

    index: int
    molecule_name: str
    atom_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate component identity and membership."""
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 1
        ):
            raise ValueError("component index must be positive")
        if (
            not isinstance(self.molecule_name, str)
            or not self.molecule_name
            or any(character.isspace() for character in self.molecule_name)
        ):
            raise ValueError("component molecule name must be one token")
        atom_numbers = _validate_atom_numbers(
            self.atom_numbers,
            "component",
            require_sorted=True,
        )
        object.__setattr__(self, "atom_numbers", atom_numbers)


@dataclass(frozen=True)
class ExportedComponent:
    """Files written for one component ownership group."""

    index: int
    molecule_name: str
    atom_numbers: tuple[int, ...]
    gro_path: Path
    topology_path: Path

    def __post_init__(self) -> None:
        """Validate exported component identity and file paths."""
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 1
        ):
            raise ValueError("exported component index must be positive")
        _validate_token(self.molecule_name, "exported component molecule name")
        atom_numbers = _validate_atom_numbers(
            self.atom_numbers,
            "exported component",
            require_sorted=True,
        )
        if not isinstance(self.gro_path, Path):
            raise TypeError("exported GRO path must be a pathlib.Path")
        if not isinstance(self.topology_path, Path):
            raise TypeError("exported topology path must be a pathlib.Path")
        if self.gro_path.suffix.lower() != ".gro":
            raise ValueError("exported GRO path must use the .gro suffix")
        if self.topology_path.suffix.lower() not in (".itp", ".top"):
            raise ValueError("exported topology path must use .itp or .top")
        object.__setattr__(self, "atom_numbers", atom_numbers)


@dataclass(frozen=True)
class ExportResult:
    """The complete set of component files produced by one export."""

    format: OutputFormat
    components: tuple[ExportedComponent, ...]

    def __post_init__(self) -> None:
        """Validate one complete component export result."""
        if self.format not in ("itp", "top"):
            raise ValueError("export format must be 'itp' or 'top'")
        components = tuple(self.components)
        if not components:
            raise ValueError("export result must contain at least one component")
        if any(
            not isinstance(component, ExportedComponent) for component in components
        ):
            raise TypeError("export result must contain ExportedComponent records")
        indices = tuple(component.index for component in components)
        if indices != tuple(range(1, len(components) + 1)):
            raise ValueError("exported component indices must be ordered 1..N")
        atom_numbers = [
            atom for component in components for atom in component.atom_numbers
        ]
        if len(atom_numbers) != len(set(atom_numbers)):
            raise ValueError("exported components cannot overlap in atom membership")
        expected_suffix = f".{self.format}"
        if any(
            component.topology_path.suffix.lower() != expected_suffix
            for component in components
        ):
            raise ValueError(
                "exported topology path suffixes must match the export format"
            )
        object.__setattr__(self, "components", components)

    @property
    def gro_paths(self) -> tuple[Path, ...]:
        """Return all coordinate paths in component order."""
        return tuple(component.gro_path for component in self.components)

    @property
    def topology_paths(self) -> tuple[Path, ...]:
        """Return all topology paths in component order."""
        return tuple(component.topology_path for component in self.components)


@dataclass(frozen=True)
class SurgeryResult(_TopologyResultMixin):
    """The graph partition produced by one completed surgery.

    Attributes
    ----------
    operation : Literal["cut", "connect"]
        The operation that produced this partition.
    atom_pair : tuple[int, int]
        The requested 1-based atom pair, normalized to ascending order.
    components : tuple[Component, ...]
        Component ownership groups after the operation.
    warnings : tuple[SurgeryWarning, ...]
        Non-fatal parameter or topology diagnostics.
    edit : TopologyEdit | None
        Interaction rows created and removed by the operation.
    """

    operation: SurgeryOperation
    atom_pair: tuple[int, int]
    components: tuple[Component, ...]
    warnings: tuple[SurgeryWarning, ...] = ()
    edit: TopologyEdit | None = None
    _top: Any = field(default=None, repr=False, compare=False)
    _gro: "GroFile | None" = field(default=None, repr=False, compare=False)
    _bond_orders: Mapping[tuple[int, int], int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _parameter_templates: Any = field(
        default=None,
        repr=False,
        compare=False,
    )


def _snapshot_result_topology(
    topology: Any,
    bond_orders: Mapping[tuple[int, int], int],
) -> TopologySnapshot:
    """Convert the private result payload to an immutable public snapshot."""
    if topology is None or not topology.molecules:
        raise ValueError("this result has no in-memory topology payload")
    from topolsurgery._records import snapshot_molecule

    return snapshot_molecule(
        topology.molecules[0],
        bond_orders=bond_orders,
    )


def _connectivity_edges(
    interactions: Sequence[InteractionSpec],
    *,
    atom_numbers: set[int],
) -> tuple[tuple[int, int], ...]:
    """Extract ordinary and implicit SETTLE connectivity edges."""
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for interaction in interactions:
        for edge in _interaction_connectivity_edges(
            interaction.kind,
            interaction.atoms,
            atom_numbers=atom_numbers,
        ):
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return tuple(edges)


def _component_dependency_edges(
    interactions: Sequence[InteractionSpec],
) -> tuple[tuple[int, int], ...]:
    """Return virtual-site ownership edges used only for grouping."""
    dependencies = []
    for interaction in interactions:
        dependencies.extend(
            _interaction_ownership_edges(
                interaction.kind,
                interaction.atoms,
            )
        )
    return tuple(dependencies)
