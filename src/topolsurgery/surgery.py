"""Single-bond surgery over a normalized gromologist topology."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import gromologist as gml

from topolsurgery._records import (
    atoms_from_molecule,
    build_gro,
    diff_after_atom_removal,
    diff_interactions,
    interactions_from_molecule,
    parse_records,
    snapshot_molecule,
    validate_molecule_name,
)
from topolsurgery._topology import (
    IMPROPER_FUNCTIONS,
    append_bonded_entry,
    bonded_entries,
    canonical_angle,
    canonical_dihedral,
    canonical_interaction,
    canonical_pair,
    collect_parameter_templates,
    component_groups,
    edge_rows,
    flatten_topology,
    lookup_parameter_templates,
    merge_parameter_libraries,
    molecule_nrexcl,
    preferred_function,
    rebuild_pairs,
    remap_parameter_templates,
    remove_exact_edges,
    remove_settles_intersecting_atoms,
    resolve_parameter_tables,
    rewrite_system,
    sanitize_interactions,
    settle_atom_triples,
)
from topolsurgery.graph import AtomicGraph
from topolsurgery.gro import GroFile
from topolsurgery.model import (
    INTERACTION_ORDER,
    AtomRemovalResult,
    AtomSpec,
    BondedTerm,
    Component,
    IndexMap,
    InteractionKind,
    InteractionSpec,
    ParameterOverrides,
    ParameterRequest,
    ParameterResolver,
    SurgeryResult,
    SurgeryWarning,
    TopologyEdit,
    TopologySnapshot,
)


def _set_default_file_mode(path: Path) -> None:
    """Apply the permissions a normally created file would receive."""
    mask = os.umask(0)
    os.umask(mask)
    os.chmod(path, 0o666 & ~mask)


def _restore_on_error(
    method: Callable[..., Any],
) -> Callable[..., Any]:
    """Restore the complete session state when a public edit fails."""

    @wraps(method)
    def wrapped(self: TopologySurgery, *args: Any, **kwargs: Any) -> Any:
        previous_top = self._top.clone()
        previous_bond_orders = dict(self._bond_orders)
        previous_templates = self._parameter_templates
        previous_structure = self._gro
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self._top = previous_top
            self._bond_orders = previous_bond_orders
            self._parameter_templates = previous_templates
            self._gro = previous_structure
            raise

    return wrapped


class TopologySurgery:
    """Perform and inspect one GROMACS bond operation at a time.

    Parameters
    ----------
    topology : gromologist.Top
        Parsed topology. It is cloned before normalization.
    structure : GroFile, optional
        Coordinates in the same atom order as the flattened topology.
        Omit for topology-only operations.
    """

    def __init__(
        self,
        topology: gml.Top,
        structure: GroFile | None = None,
        *,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
        validate_structure: bool = True,
    ):
        flat = flatten_topology(topology)
        molecule = flat.top.get_molecule(flat.molecule_name)
        self._top = flat.top
        self._molecule_name = flat.molecule_name
        self._validate_structure = validate_structure
        self._parameter_templates = collect_parameter_templates(molecule)
        graph = AtomicGraph.from_edges(
            range(1, molecule.natoms + 1),
            edge_rows(molecule),
            bond_orders=bond_orders,
        )
        self._bond_orders = dict(graph.bond_orders)
        self._gro: GroFile | None = None
        if structure is not None:
            self.attach_structure(
                structure,
                validate_identity=validate_structure,
            )

    @classmethod
    def from_files(
        cls,
        topology: str | Path,
        structure: str | Path | GroFile | None = None,
        *,
        gmx_dir: str | Path | None = None,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
        validate_structure: bool = True,
    ) -> TopologySurgery:
        """Read a GROMACS topology and coordinate file.

        Parameters
        ----------
        topology : str | Path
            ``.top`` or molecule ``.itp`` file.
        structure : str | Path | GroFile, optional
            Matching ``.gro`` file or parsed structure.
        gmx_dir : str | Path, optional
            Explicit GROMACS share directory for include resolution.
        bond_orders : Mapping, optional
            Optional chemical bond-order metadata for graph edges.
        validate_structure : bool
            Compare TOP and GRO residue/atom names when coordinates are
            supplied.

        Returns
        -------
        TopologySurgery
            Ready for one or more sequential single-bond operations.
        """
        parsed = gml.Top(
            str(topology),
            gmx_dir=None if gmx_dir is None else str(gmx_dir),
            suppress=True,
        )
        parsed_structure = (
            None
            if structure is None
            else structure
            if isinstance(structure, GroFile)
            else GroFile.read(structure)
        )
        return cls(
            parsed,
            parsed_structure,
            bond_orders=bond_orders,
            validate_structure=validate_structure,
        )

    @classmethod
    def from_gromologist(
        cls,
        topology: gml.Top,
        structure: GroFile | None = None,
        *,
        bond_orders: Mapping[tuple[int, int], int] | None = None,
        validate_structure: bool = True,
    ) -> TopologySurgery:
        """Create the public API around an in-memory gromologist topology."""
        return cls(
            topology,
            structure,
            bond_orders=bond_orders,
            validate_structure=validate_structure,
        )

    @classmethod
    def from_result(
        cls,
        result: SurgeryResult | AtomRemovalResult,
    ) -> TopologySurgery:
        """Create an independent session from a completed operation result."""
        if result._top is None:
            raise ValueError("this result has no in-memory topology payload")
        session = cls(
            result._top,
            result._gro,
            bond_orders=result._bond_orders,
            validate_structure=False,
        )
        if result._parameter_templates is not None:
            session._parameter_templates = result._parameter_templates
        return session

    @classmethod
    def from_records(
        cls,
        atoms: Iterable[AtomSpec],
        interactions: Iterable[InteractionSpec] = (),
        *,
        name: str = "MOL",
        nrexcl: int = 3,
        coordinates: Sequence[Sequence[float]] | None = None,
        velocities: Sequence[Sequence[float]] | None = None,
        box: Sequence[float] = (0.0, 0.0, 0.0),
        parameter_sections: str = "",
        bond_orders: Mapping[tuple[int, int], int] | None = None,
    ) -> TopologySurgery:
        """Construct a working topology directly from caller-owned records.

        Parameters
        ----------
        atoms : Sequence[AtomSpec]
            Contiguous 1-based atom records.
        interactions : Sequence[InteractionSpec]
            Existing bonded records.
        name : str
            Working molecule name.
        nrexcl : int
            GROMACS exclusion depth.
        coordinates, velocities : Sequence[Sequence[float]], optional
            Per-atom xyz data in GROMACS nanometres. Coordinates enable
            component and GRO export.
        box : Sequence[float]
            Three or nine GROMACS box values.
        parameter_sections : str
            Optional inline sections such as ``[ atomtypes ]``,
            ``[ bondtypes ]`` and ``[ angletypes ]`` placed before the
            moleculetype.

        Returns
        -------
        TopologySurgery
            A topology-only or coordinate-enabled surgery session.
        """
        input_atoms = tuple(atoms)
        ordered_atoms = tuple(sorted(input_atoms, key=lambda atom: atom.number))
        if coordinates is None and velocities is not None:
            raise ValueError(
                "velocities require coordinates when constructing a session"
            )
        interaction_rows = tuple(interactions)
        ordered_coordinates = None
        if coordinates is not None:
            if len(coordinates) != len(input_atoms):
                raise ValueError("coordinates must have one row per input atom")
            ordered_coordinates = tuple(
                row
                for _atom, row in sorted(
                    zip(input_atoms, coordinates, strict=True),
                    key=lambda item: item[0].number,
                )
            )
        ordered_velocities = None
        if velocities is not None:
            if len(velocities) != len(input_atoms):
                raise ValueError("velocities must have one row per input atom")
            ordered_velocities = tuple(
                row
                for _atom, row in sorted(
                    zip(input_atoms, velocities, strict=True),
                    key=lambda item: item[0].number,
                )
            )
        topology = parse_records(
            ordered_atoms,
            interaction_rows,
            name=name,
            nrexcl=nrexcl,
            parameter_sections=parameter_sections,
        )
        structure = (
            None
            if ordered_coordinates is None
            else build_gro(
                ordered_atoms,
                ordered_coordinates,
                velocities=ordered_velocities,
                box=box,
                title=name,
            )
        )
        return cls(
            topology,
            structure,
            bond_orders=bond_orders,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TopologySnapshot,
        *,
        coordinates: Sequence[Sequence[float]] | None = None,
        velocities: Sequence[Sequence[float]] | None = None,
        box: Sequence[float] = (0.0, 0.0, 0.0),
        parameter_sections: str = "",
        bond_orders: Mapping[tuple[int, int], int] | None = None,
    ) -> TopologySurgery:
        """Create an independent session from immutable public records."""
        if snapshot.unsupported_sections or snapshot.lossy_features:
            raise ValueError(
                "snapshot cannot represent topology features "
                f"sections={list(snapshot.unsupported_sections)}, "
                f"features={list(snapshot.lossy_features)}; use from_result() "
                "or retain the source gromologist topology"
            )
        return cls.from_records(
            snapshot.atoms,
            snapshot.interactions,
            name=snapshot.name,
            nrexcl=snapshot.nrexcl,
            coordinates=coordinates,
            velocities=velocities,
            box=box,
            parameter_sections=parameter_sections,
            bond_orders=(snapshot.bond_orders if bond_orders is None else bond_orders),
        )

    @property
    def atom_count(self) -> int:
        """Return the number of atoms in the working topology."""
        return self._molecule().natoms

    @property
    def name(self) -> str:
        """Return the current working molecule name."""
        return self._molecule_name

    @property
    def nrexcl(self) -> int:
        """Return the current molecule's exclusion depth."""
        return molecule_nrexcl(self._molecule())

    @property
    def atoms(self) -> tuple[AtomSpec, ...]:
        """Return a public snapshot of all current atoms."""
        return atoms_from_molecule(self._molecule())

    def interactions(
        self,
        kind: InteractionKind | None = None,
    ) -> tuple[InteractionSpec, ...]:
        """Return current interaction records, optionally for one kind."""
        if kind is not None and kind not in INTERACTION_ORDER:
            raise ValueError(f"unknown interaction kind {kind!r}")
        return interactions_from_molecule(self._molecule(), kind)

    def snapshot(self) -> TopologySnapshot:
        """Return an immutable public topology snapshot."""
        return snapshot_molecule(
            self._molecule(),
            name=self._molecule_name,
            bond_orders=self._bond_orders,
        )

    def to_gromologist(self) -> gml.Top:
        """Return a defensive clone of the parsed gromologist topology."""
        return self._top.clone()

    def clone(self) -> TopologySurgery:
        """Return an independent surgery session at the current state."""
        cloned = type(self)(
            self._top.clone(),
            self._gro,
            bond_orders=self._bond_orders,
            validate_structure=self._validate_structure,
        )
        cloned._parameter_templates = self._parameter_templates
        return cloned

    def attach_structure(
        self,
        structure: GroFile,
        *,
        validate_identity: bool = True,
    ) -> None:
        """Attach coordinates in the current flat atom order."""
        if len(structure.atoms) != self.atom_count:
            raise ValueError(
                "coordinate/topology atom count mismatch after flattening: "
                f"GRO has {len(structure.atoms)} atoms, topology has "
                f"{self.atom_count}. Repeated molecules are ordered by "
                "[ molecules ] and the GRO must follow that order."
            )
        if validate_identity:
            top_atoms = self.atoms
            for top_atom, gro_atom in zip(top_atoms, structure.atoms, strict=True):
                if (
                    top_atom.residue_name != gro_atom.residue_name
                    or top_atom.atom_name != gro_atom.atom_name
                ):
                    raise ValueError(
                        f"TOP/GRO atom identity mismatch at atom "
                        f"{top_atom.number}: TOP="
                        f"{top_atom.residue_name}."
                        f"{top_atom.atom_name}, GRO="
                        f"{gro_atom.residue_name}."
                        f"{gro_atom.atom_name}"
                    )
        self._gro = structure
        self._validate_structure = validate_identity

    @property
    def bonds(self) -> tuple[tuple[int, int], ...]:
        """Return all ``[ bonds ]`` atom pairs in file order."""
        return tuple(
            tuple(entry.atom_numbers)
            for entry in bonded_entries(self._molecule(), "bonds")
        )

    @property
    def components(self) -> tuple[tuple[int, ...], ...]:
        """Return component ownership groups, including virtual-site hosts."""
        return component_groups(self._molecule(), self.graph)

    @property
    def graph(self) -> AtomicGraph:
        """Return the current connectivity as an immutable ``AtomicGraph``."""
        molecule = self._molecule()
        return AtomicGraph.from_edges(
            range(1, molecule.natoms + 1),
            edge_rows(molecule),
            bond_orders=self._bond_orders,
        )

    @property
    def bond_orders(self) -> Mapping[tuple[int, int], int]:
        """Return a defensive view of caller-owned chemical bond orders."""
        return dict(self._bond_orders)

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["itp", "top"] | None = None,
        molecule_name: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write the current combined topology without component splitting.

        Parameters
        ----------
        path : str | Path
            Destination file.
        format : {"itp", "top"}, optional
            Output format. Inferred from the path suffix when omitted.
        molecule_name : str, optional
            Rename only the written moleculetype.
        overwrite : bool
            Replace an existing destination when ``True``.

        Returns
        -------
        Path
            The written path.
        """
        destination = Path(path)
        output_format = format or (
            destination.suffix.lstrip(".").lower()
            if destination.suffix.lower() in (".itp", ".top")
            else None
        )
        if output_format not in ("itp", "top"):
            raise ValueError("save format must be 'itp' or 'top', or use that suffix")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"destination already exists: {destination}; pass "
                "overwrite=True to replace it"
            )
        output_top = self._top.clone()
        molecule = output_top.get_molecule(self._molecule_name)
        if molecule_name is not None:
            validate_molecule_name(molecule_name)
            molecule.rename(molecule_name)
            if output_top.system:
                rewrite_system(output_top, molecule_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=f".{output_format}",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        committed = False
        try:
            if output_format == "itp":
                molecule.save_itp(str(temporary_path))
            else:
                if not output_top.system:
                    raise ValueError(
                        "TOP export requires [ system ] and [ molecules ]; "
                        "use format='itp' for a molecule-only topology"
                    )
                output_top.save_top(str(temporary_path))
            _set_default_file_mode(temporary_path)
            if overwrite:
                os.replace(temporary_path, destination)
                committed = True
            else:
                os.link(temporary_path, destination)
                committed = True
                try:
                    temporary_path.unlink()
                except BaseException:
                    destination.unlink(missing_ok=True)
                    committed = False
                    raise
        except BaseException:
            if not committed:
                temporary_path.unlink(missing_ok=True)
            raise
        return Path(destination)

    @_restore_on_error
    def remove_atoms(
        self,
        atom_numbers: Sequence[int],
    ) -> AtomRemovalResult:
        """Remove atoms and every interaction that references them.

        Parameters
        ----------
        atom_numbers : Sequence[int]
            Unique 1-based atom numbers in the current flat topology.

        Returns
        -------
        AtomRemovalResult
            Removed atom records, the survivors' index map, and the new
            connected components.
        """
        self._assert_supported_edit()
        requested = tuple(sorted(atom_numbers))
        if not requested:
            raise ValueError("atom_numbers must contain at least one atom")
        if len(requested) != len(set(requested)):
            raise ValueError("atom_numbers contains duplicates")
        for atom_number in requested:
            if not isinstance(atom_number, int) or isinstance(atom_number, bool):
                raise TypeError("atom numbers must be integers")
            if not 1 <= atom_number <= self.atom_count:
                raise ValueError(f"atom {atom_number} is outside 1..{self.atom_count}")
        if len(requested) == self.atom_count:
            raise ValueError("cannot remove every atom from a molecule")

        before_interactions = self.interactions()
        previous_pairs = self.graph.distance_three_pairs()
        before_atoms = {atom.number: atom for atom in self.atoms}
        removed_set = set(requested)
        survivors = tuple(
            number
            for number in range(1, self.atom_count + 1)
            if number not in removed_set
        )
        forward = {
            old_number: new_number
            for new_number, old_number in enumerate(survivors, start=1)
        }
        index_map = IndexMap(
            forward=forward,
            backward={
                new_number: old_number for old_number, new_number in forward.items()
            },
        )

        molecule = self._molecule()
        remove_settles_intersecting_atoms(molecule, removed_set)
        for atom_number in reversed(requested):
            molecule.del_atom(atom_number, del_in_pdb=False)
        molecule.update_dicts()
        self._parameter_templates = remap_parameter_templates(
            self._parameter_templates,
            forward,
        )
        self._parameter_templates = merge_parameter_libraries(
            self._parameter_templates,
            collect_parameter_templates(molecule),
        )
        self._bond_orders = {
            canonical_pair(
                (
                    forward[pair[0]],
                    forward[pair[1]],
                )
            ): order
            for pair, order in self._bond_orders.items()
            if pair[0] in forward and pair[1] in forward
        }
        graph = self.graph
        sanitize_interactions(molecule, graph)
        rebuild_pairs(
            molecule,
            graph,
            create=False,
            previous_expected={
                canonical_pair((forward[first], forward[second]))
                for first, second in previous_pairs
                if first in forward and second in forward
            },
        )
        molecule.update_dicts()
        if self._gro is not None:
            self._gro = self._gro.select(survivors)
        created, removed_interactions = diff_after_atom_removal(
            before_interactions,
            self.interactions(),
            forward,
        )
        return AtomRemovalResult(
            requested_atoms=requested,
            removed_atoms=tuple(before_atoms[number] for number in requested),
            index_map=index_map,
            components=self._public_components(),
            edit=TopologyEdit(
                operation="remove_atoms",
                atoms=requested,
                created=created,
                removed=removed_interactions,
            ),
            _gro=self._gro,
            _top=self._top.clone(),
            _bond_orders=dict(self._bond_orders),
            _parameter_templates=self._parameter_templates,
        )

    @_restore_on_error
    def cut_bond(self, atom_i: int, atom_j: int) -> SurgeryResult:
        """Cut exactly one bond or constraint and refresh related terms.

        Parameters
        ----------
        atom_i, atom_j : int
            1-based atom numbers, either endpoint order.

        Returns
        -------
        SurgeryResult
            The complete graph partition after the cut.

        Raises
        ------
        ValueError
            Either atom is invalid or no bond/constraint joins the pair.
        """
        self._assert_supported_edit()
        molecule = self._molecule()
        before_interactions = self.interactions()
        edge = self._validate_edge(atom_i, atom_j)
        previous_graph = self.graph
        previous_edges = set(previous_graph.edges)
        if edge not in previous_edges:
            raise ValueError(
                f"no [ bonds ] or [ constraints ] record joins atoms "
                f"{edge[0]} and {edge[1]}"
            )
        settle_edges = {
            pair
            for oxygen, hydrogen_one, hydrogen_two in settle_atom_triples(molecule)
            for pair in (
                (oxygen, hydrogen_one),
                (oxygen, hydrogen_two),
            )
        }
        if edge in settle_edges:
            raise ValueError(
                f"edge {edge[0]}-{edge[1]} is an implicit SETTLE dependency; "
                "replace the target SETTLE group with explicit bonds or "
                "constraints before cutting it"
            )

        previous_expected = set(previous_graph.distance_three_pairs())
        removed = remove_exact_edges(molecule, ("bonds", "constraints"), edge)
        if removed == 0:
            raise ValueError(f"could not remove the edge {edge[0]}-{edge[1]}")
        self._bond_orders.pop(edge, None)
        graph = self.graph
        sanitize_interactions(molecule, graph)
        rebuild_pairs(
            molecule,
            graph,
            create=False,
            previous_expected=previous_expected,
        )
        molecule.update_dicts()
        return self._result(
            "cut",
            edge,
            before_interactions=before_interactions,
        )

    @_restore_on_error
    def connect_bond(
        self,
        atom_i: int,
        atom_j: int,
        *,
        bond_function: int = 1,
        bond_parameters: Sequence[int | float | str] = (),
        generate_angles: bool = True,
        generate_dihedrals: bool = True,
        angle_function: int | None = None,
        dihedral_function: int | None = None,
        bond_order: int | None = None,
        update_pairs: bool | None = None,
        resolve_parameters: bool = True,
        parameter_resolver: ParameterResolver | None = None,
        parameter_overrides: ParameterOverrides | None = None,
    ) -> SurgeryResult:
        """Connect exactly one atom pair and add graph-derived interactions.

        Parameters
        ----------
        atom_i, atom_j : int
            1-based atoms to connect, either endpoint order.
        bond_function : int
            GROMACS ``[ bonds ]`` function, default 1.
        bond_parameters : Sequence[int | float | str]
            Explicit bond parameter columns. Empty means resolve from the
            parsed ``[ bondtypes ]`` tables when possible.
        generate_angles : bool
            Add angles centered on both new bond endpoints.
        generate_dihedrals : bool
            Add proper torsions whose graph path contains the new bond.
        angle_function : int, optional
            Function for generated angles. ``None`` follows the most common
            existing angle function, falling back to 1.
        dihedral_function : int, optional
            Function for generated proper torsions. ``None`` follows the
            most common existing proper function, falling back to 9.
        bond_order : int, optional
            Optional graph-only chemical bond order (1-4). GROMACS bond
            functions are stored separately in ``bond_function``.
        update_pairs : bool, optional
            Refresh graph-distance-three pairs. ``None`` updates/creates
            pairs only when a ``[ pairs ]`` section already exists.
        resolve_parameters : bool
            Try to fill missing explicit parameters from force-field tables.
        parameter_resolver : ParameterResolver, optional
            Caller-owned resolver consulted for each generated bond, angle,
            proper torsion, and new pair before topology templates.
        parameter_overrides : Mapping, optional
            Exact atom-sequence overrides. A value is one ``BondedTerm`` or
            a sequence of terms for multi-term torsions.

        Returns
        -------
        SurgeryResult
            The complete graph partition after the connection.

        Raises
        ------
        ValueError
            Either atom is invalid or the pair is already connected.
        """
        self._assert_supported_edit()
        edge = self._validate_edge(atom_i, atom_j)
        if bond_order is not None and (
            not isinstance(bond_order, int)
            or isinstance(bond_order, bool)
            or not 1 <= bond_order <= 4
        ):
            raise ValueError("bond_order must be an integer between 1 and 4")
        molecule = self._molecule()
        before_interactions = self.interactions()
        previous_graph = self.graph
        previous_edges = set(previous_graph.edges)
        if edge in previous_edges:
            raise ValueError(f"atoms {edge[0]} and {edge[1]} are already connected")

        graph = previous_graph.with_edge(
            edge[0],
            edge[1],
            bond_order=bond_order,
        )
        self._insert_connection(
            molecule,
            edge,
            bond_function=bond_function,
            bond_parameters=bond_parameters,
            bond_order=bond_order,
            parameter_resolver=parameter_resolver,
            parameter_overrides=parameter_overrides,
        )
        if generate_angles:
            self._add_connection_angles(
                molecule,
                graph,
                edge,
                angle_function=angle_function,
                parameter_resolver=parameter_resolver,
                parameter_overrides=parameter_overrides,
            )
        if generate_dihedrals:
            self._add_connection_dihedrals(
                molecule,
                graph,
                edge,
                dihedral_function=dihedral_function,
                parameter_resolver=parameter_resolver,
                parameter_overrides=parameter_overrides,
            )
        self._refresh_connection_pairs(
            molecule,
            graph,
            previous_graph,
            update_pairs=update_pairs,
            parameter_resolver=parameter_resolver,
            parameter_overrides=parameter_overrides,
        )

        sanitize_interactions(molecule, graph)
        molecule.update_dicts()

        warnings = self._connection_warnings(
            before_interactions,
            resolve_parameters=resolve_parameters,
        )
        self._parameter_templates = merge_parameter_libraries(
            self._parameter_templates,
            collect_parameter_templates(molecule),
        )
        return self._result(
            "connect",
            edge,
            warnings=warnings,
            before_interactions=before_interactions,
        )

    def _insert_connection(
        self,
        molecule: gml.SectionMol,
        edge: tuple[int, int],
        *,
        bond_function: int,
        bond_parameters: Sequence[int | float | str],
        bond_order: int | None,
        parameter_resolver: ParameterResolver | None,
        parameter_overrides: ParameterOverrides | None,
    ) -> None:
        """Insert the requested bond and its graph metadata."""
        if bond_parameters:
            append_bonded_entry(
                molecule,
                "bonds",
                edge,
                bond_function,
                bond_parameters,
            )
        else:
            self._append_graph_term(
                molecule,
                "bonds",
                edge,
                bond_function,
                parameter_resolver=parameter_resolver,
                parameter_overrides=parameter_overrides,
            )
        if bond_order is not None:
            self._bond_orders[edge] = bond_order

    def _add_connection_angles(
        self,
        molecule: gml.SectionMol,
        graph: AtomicGraph,
        edge: tuple[int, int],
        *,
        angle_function: int | None,
        parameter_resolver: ParameterResolver | None,
        parameter_overrides: ParameterOverrides | None,
    ) -> None:
        """Add the missing angles whose two bonds contain the new edge."""
        existing = {
            canonical_angle(entry.atom_numbers)
            for entry in bonded_entries(molecule, "angles")
        }
        function = (
            angle_function
            if angle_function is not None
            else preferred_function(molecule, "angles", 1)
        )
        for angle in sorted(graph.angles_through_edge(edge)):
            if angle not in existing:
                self._append_graph_term(
                    molecule,
                    "angles",
                    angle,
                    function,
                    parameter_resolver=parameter_resolver,
                    parameter_overrides=parameter_overrides,
                )
                existing.add(angle)

    def _add_connection_dihedrals(
        self,
        molecule: gml.SectionMol,
        graph: AtomicGraph,
        edge: tuple[int, int],
        *,
        dihedral_function: int | None,
        parameter_resolver: ParameterResolver | None,
        parameter_overrides: ParameterOverrides | None,
    ) -> None:
        """Add missing proper torsions whose path contains the new edge."""
        existing = {
            canonical_dihedral(entry.atom_numbers)
            for entry in bonded_entries(molecule, "dihedrals")
            if entry.interaction_type not in IMPROPER_FUNCTIONS
        }
        function = (
            dihedral_function
            if dihedral_function is not None
            else preferred_function(
                molecule,
                "dihedrals",
                9,
                proper_only=True,
            )
        )
        for dihedral in sorted(graph.proper_dihedrals_through_edge(edge)):
            if dihedral not in existing:
                self._append_graph_term(
                    molecule,
                    "dihedrals",
                    dihedral,
                    function,
                    parameter_resolver=parameter_resolver,
                    parameter_overrides=parameter_overrides,
                )
                existing.add(dihedral)

    def _refresh_connection_pairs(
        self,
        molecule: gml.SectionMol,
        graph: AtomicGraph,
        previous_graph: AtomicGraph,
        *,
        update_pairs: bool | None,
        parameter_resolver: ParameterResolver | None,
        parameter_overrides: ParameterOverrides | None,
    ) -> None:
        """Apply the requested pair-refresh policy."""
        if update_pairs is None:
            try:
                molecule.get_subsection("pairs")
                should_update = True
            except KeyError:
                should_update = False
        else:
            should_update = update_pairs
        if not should_update:
            return
        rebuild_pairs(
            molecule,
            graph,
            create=True,
            previous_expected=set(previous_graph.distance_three_pairs()),
            append_term=lambda pair, function: self._append_graph_term(
                molecule,
                "pairs",
                pair,
                function,
                parameter_resolver=parameter_resolver,
                parameter_overrides=parameter_overrides,
            ),
        )

    def _connection_warnings(
        self,
        before_interactions: Sequence[InteractionSpec],
        *,
        resolve_parameters: bool,
    ) -> list[SurgeryWarning]:
        """Resolve tables when requested and report unresolved new rows."""
        created_interactions, _removed = diff_interactions(
            before_interactions,
            self.interactions(),
        )
        missing: set[tuple[str, int, tuple[int, ...]]] = set()
        if resolve_parameters:
            missing.update(resolve_parameter_tables(self._molecule()))
        else:
            missing.update(
                (
                    interaction.kind,
                    int(interaction.function or 0),
                    interaction.atoms,
                )
                for interaction in created_interactions
                if (
                    interaction.kind in ("bonds", "angles", "dihedrals")
                    or (interaction.kind == "pairs" and interaction.function != 1)
                )
                and not interaction.parameters
            )
        if not missing:
            return []
        return [
            SurgeryWarning(
                code="unresolved_parameter",
                kind=cast(InteractionKind, header),
                atoms=atoms,
                message=(
                    f"{header} interaction {atoms} has no explicit "
                    f"parameters (function {function})"
                ),
            )
            for header, function, atoms in sorted(missing)
        ]

    def _result(
        self,
        operation: Literal["cut", "connect"],
        edge: tuple[int, int],
        *,
        warnings: Sequence[SurgeryWarning] = (),
        before_interactions: Sequence[InteractionSpec] | None = None,
    ) -> SurgeryResult:
        """Snapshot the current graph partition."""
        public_components = self._public_components()
        edit = None
        if before_interactions is not None:
            created, removed = diff_interactions(
                before_interactions,
                self.interactions(),
            )
            edit = TopologyEdit(
                operation=operation,
                atoms=edge,
                created=created,
                removed=removed,
            )
        return SurgeryResult(
            operation=operation,
            atom_pair=edge,
            components=public_components,
            warnings=tuple(warnings),
            edit=edit,
            _top=self._top.clone(),
            _gro=self._gro,
            _bond_orders=dict(self._bond_orders),
            _parameter_templates=self._parameter_templates,
        )

    def _public_components(self) -> tuple[Component, ...]:
        """Return the current connected components with stable names."""
        components = component_groups(self._molecule(), self.graph)
        return tuple(
            Component(
                index=index,
                molecule_name=f"{self._molecule_name}_part{index}",
                atom_numbers=atoms,
            )
            for index, atoms in enumerate(components, start=1)
        )

    def _molecule(self) -> gml.SectionMol:
        """Return the sole working molecule."""
        return self._top.get_molecule(self._molecule_name)

    def _assert_supported_edit(self) -> None:
        """Reject edits whose atom-addressed records cannot be remapped."""
        snapshot = self.snapshot()
        if snapshot.unsupported_sections or snapshot.lossy_features:
            raise ValueError(
                "topology surgery cannot safely edit unsupported or lossy "
                f"features sections={list(snapshot.unsupported_sections)}, "
                f"features={list(snapshot.lossy_features)}; use save() "
                "without atom edits or remove those features first"
            )

    def _append_graph_term(
        self,
        molecule: gml.SectionMol,
        header: str,
        atoms: Sequence[int],
        default_function: int,
        *,
        parameter_resolver: ParameterResolver | None = None,
        parameter_overrides: ParameterOverrides | None = None,
    ) -> None:
        """Append a graph term through caller and topology parameter layers."""
        type_by_number = {atom.num: atom.type for atom in molecule.atoms}
        atom_types = [type_by_number[atom] for atom in atoms]
        override = self._find_parameter_override(
            header,
            atoms,
            parameter_overrides,
        )
        if override is not None:
            override_terms = self._normalize_terms(override, header)
            for term in override_terms:
                append_bonded_entry(
                    molecule,
                    header,
                    atoms,
                    term.function,
                    term.parameters,
                    term.second_parameters,
                )
            if override_terms:
                return

        if parameter_resolver is not None:
            resolved = parameter_resolver(
                ParameterRequest(
                    kind=cast(InteractionKind, header),
                    atoms=tuple(atoms),
                    atom_types=tuple(atom_types),
                    default_function=default_function,
                )
            )
            if resolved is not None:
                terms = self._normalize_terms(resolved, header)
                if terms:
                    for term in terms:
                        append_bonded_entry(
                            molecule,
                            header,
                            atoms,
                            term.function,
                            term.parameters,
                            term.second_parameters,
                        )
                    return

        templates = lookup_parameter_templates(
            self._parameter_templates,
            header,
            atoms,
            atom_types,
            default_function,
        )
        if templates:
            template_terms = self._normalize_terms(
                tuple(
                    BondedTerm(function, parameters, second_parameters)
                    for function, parameters, second_parameters in templates
                ),
                header,
            )
            for term in template_terms:
                append_bonded_entry(
                    molecule,
                    header,
                    atoms,
                    term.function,
                    term.parameters,
                    term.second_parameters,
                )
            return
        append_bonded_entry(molecule, header, atoms, default_function)

    @staticmethod
    def _find_parameter_override(
        header: str,
        atoms: Sequence[int],
        overrides: ParameterOverrides | None,
    ) -> BondedTerm | Sequence[BondedTerm] | None:
        """Find an override under any equivalent atom ordering."""
        if not overrides:
            return None
        expected_size = {
            "bonds": 2,
            "pairs": 2,
            "angles": 3,
            "dihedrals": 4,
        }.get(header, len(atoms))
        candidates = [
            (key, terms)
            for key, terms in overrides.items()
            if len(key) == expected_size
        ]
        if not candidates:
            return None
        canonical_atoms = canonical_interaction(header, atoms)
        for key, terms in candidates:
            if canonical_interaction(header, key) == canonical_atoms:
                return terms
        return None

    @staticmethod
    def _normalize_terms(
        terms: BondedTerm | Sequence[BondedTerm],
        header: str,
    ) -> tuple[BondedTerm, ...]:
        """Normalize one term or a term sequence."""
        normalized = (terms,) if isinstance(terms, BondedTerm) else tuple(terms)
        if not all(isinstance(term, BondedTerm) for term in normalized):
            raise TypeError("parameter resolver values must be BondedTerm records")
        if not normalized:
            return normalized
        if len(normalized) > 1 and header != "dihedrals":
            raise ValueError(
                f"{header} accepts one generated parameter term per atom "
                f"set; got {len(normalized)}"
            )
        if len(normalized) > 1 and any(term.function != 9 for term in normalized):
            raise ValueError("multiple generated dihedral terms require function 9")
        if header != "dihedrals" or normalized[0].function != 9:
            return normalized
        if any(len(term.parameters) % 3 for term in normalized):
            raise ValueError(
                "function-9 dihedral parameters must be phase/kd/pn triples"
            )
        state_a = [
            term.parameters[index]
            for term in normalized
            for index in range(2, len(term.parameters), 3)
        ]
        state_b_presence = [bool(term.second_parameters) for term in normalized]
        if any(state_b_presence) and not all(state_b_presence):
            raise ValueError(
                "multiple function-9 dihedral terms must all define state-B "
                "parameters or all omit them"
            )
        if any(len(term.second_parameters) % 3 for term in normalized):
            raise ValueError(
                "function-9 state-B dihedral parameters must be phase/kd/pn triples"
            )
        state_b = [
            term.second_parameters[index]
            for term in normalized
            for index in range(2, len(term.second_parameters), 3)
        ]
        for label, values in (("state-A", state_a), ("state-B", state_b)):
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in values
            ):
                raise ValueError(
                    f"function-9 {label} multiplicities must be positive integers"
                )
        if len(normalized) > 1 and not state_a:
            raise ValueError(
                "multiple function-9 dihedral terms must define multiplicities"
            )
        if state_a and len(state_a) != len(set(state_a)):
            raise ValueError(
                "multiple function-9 dihedral terms must have unique multiplicities"
            )
        if state_b and len(state_b) != len(set(state_b)):
            raise ValueError(
                "multiple function-9 state-B dihedral terms must have "
                "unique multiplicities"
            )
        return normalized

    def _validate_edge(self, atom_i: int, atom_j: int) -> tuple[int, int]:
        """Validate and normalize one requested atom pair."""
        if not isinstance(atom_i, int) or isinstance(atom_i, bool):
            raise TypeError("atom_i must be an integer")
        if not isinstance(atom_j, int) or isinstance(atom_j, bool):
            raise TypeError("atom_j must be an integer")
        first, second = atom_i, atom_j
        if first == second:
            raise ValueError("a surgery atom pair must contain two distinct atoms")
        maximum = self.atom_count
        for atom in (first, second):
            if not 1 <= atom <= maximum:
                raise ValueError(
                    f"atom {atom} is outside the topology range 1..{maximum}"
                )
        return canonical_pair((first, second))
