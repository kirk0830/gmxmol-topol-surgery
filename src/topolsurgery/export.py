"""Export post-surgery ownership groups as GRO plus ITP or TOP."""

from __future__ import annotations

import os
import re
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from topolsurgery._topology import rewrite_system, subset_molecule
from topolsurgery._records import snapshot_molecule
from topolsurgery.model import (
    AtomRemovalResult,
    ExportResult,
    ExportedComponent,
    OutputFormat,
    SurgeryResult,
    TopologySnapshot,
)

if TYPE_CHECKING:
    import gromologist as gml


def export_result(
    result: SurgeryResult | AtomRemovalResult,
    outdir: str | Path,
    *,
    format: OutputFormat = "itp",
    prefix: str = "component",
    overwrite: bool = False,
) -> ExportResult:
    """Write each component ownership group as one GRO/topology pair.

    Parameters
    ----------
    result : SurgeryResult | AtomRemovalResult
        Completed surgery snapshot.
    outdir : str | Path
        Destination directory.
    format : {"itp", "top"}
        Molecule ITP or a complete parsed TOP.
    prefix : str
        File and molecule-name prefix.
    overwrite : bool
        Allow replacement of existing output files. Default is ``False``.

    Returns
    -------
    ExportResult
        All paths in stable component order.

    Raises
    ------
    ValueError
        The format is unknown, export data is incomplete, or TOP output was
        requested from an ITP input without a system section.
    """
    if format not in ("itp", "top"):
        raise ValueError("format must be 'itp' or 'top'")
    if result._top is None or result._gro is None:
        raise ValueError("this result has no in-memory export payload")
    topology = result.topology
    if topology.unsupported_sections or topology.lossy_features:
        raise ValueError(
            "component export cannot safely split unsupported or lossy "
            f"topology features sections={list(topology.unsupported_sections)}, "
            f"features={list(topology.lossy_features)}; use save() on the "
            "combined topology"
        )
    if format == "top" and not result._top.system:
        raise ValueError(
            "TOP export requires an input with [ system ] and [ molecules ]; "
            "use format='itp' for a molecule-only ITP"
        )

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_prefix = _safe_stem(prefix)
    total = len(result.components)
    planned = [
        (
            component,
            f"{safe_prefix}{'' if total == 1 else f'_{component.index:03d}'}",
        )
        for component in result.components
    ]
    targets = [
        destination / f"{stem}.{suffix}"
        for _component, stem in planned
        for suffix in ("gro", format)
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output file(s) already exist; pass overwrite=True to replace: "
            + ", ".join(str(path) for path in existing)
        )

    exported: list[ExportedComponent] = []
    with TemporaryDirectory(
        prefix=f".{safe_prefix}.",
        dir=destination,
    ) as staging_name:
        staging = Path(staging_name)
        for component, stem in planned:
            component_top = _component_top(result, component.atom_numbers)
            molecule = component_top.molecules[0]
            molecule.rename(stem)
            if component_top.system:
                rewrite_system(component_top, stem)
            staged_gro = staging / f"{stem}.gro"
            result._gro.select(component.atom_numbers).write(
                staged_gro,
                title=f"{result.operation} component {component.index}",
            )
            staged_topology = staging / f"{stem}.{format}"
            if format == "itp":
                molecule.save_itp(str(staged_topology))
            else:
                component_top.save_top(str(staged_topology))
            exported.append(
                ExportedComponent(
                    index=component.index,
                    molecule_name=stem,
                    atom_numbers=component.atom_numbers,
                    gro_path=destination / f"{stem}.gro",
                    topology_path=destination / f"{stem}.{format}",
                )
            )
        export_record = ExportResult(format=format, components=tuple(exported))
        _commit_export(staging, exported, overwrite=overwrite)
    return export_record


def _commit_export(
    staging: Path,
    exported: Sequence[ExportedComponent],
    *,
    overwrite: bool,
) -> None:
    """Commit staged files with rollback on a placement failure."""
    backup_dir = staging / ".backup"
    backup_dir.mkdir()
    backups: list[tuple[Path, Path]] = []
    deployed: list[Path] = []
    try:
        for item in exported:
            for source, target in (
                (staging / item.gro_path.name, item.gro_path),
                (staging / item.topology_path.name, item.topology_path),
            ):
                if overwrite and target.exists():
                    backup = backup_dir / target.name
                    os.replace(target, backup)
                    backups.append((target, backup))
                if overwrite:
                    os.replace(source, target)
                    deployed.append(target)
                else:
                    os.link(source, target)
                    deployed.append(target)
                    source.unlink()
    except BaseException:
        for target in reversed(deployed):
            target.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise


def component_snapshots(
    result: SurgeryResult | AtomRemovalResult,
) -> tuple[TopologySnapshot, ...]:
    """Return each component ownership group without writing files.

    Component atom numbers and bond-order metadata are compacted to local
    1-based indices, matching the export contract.
    """
    if result._top is None:
        raise ValueError("this result has no in-memory topology payload")
    topology = result.topology
    if topology.unsupported_sections or topology.lossy_features:
        raise ValueError(
            "component snapshots cannot represent unsupported or lossy "
            f"topology features sections={list(topology.unsupported_sections)}, "
            f"features={list(topology.lossy_features)}"
        )
    snapshots = []
    for component in result.components:
        component_top = _component_top(result, component.atom_numbers)
        local_of = {old: new for new, old in enumerate(component.atom_numbers, start=1)}
        local_orders = {
            (
                min(local_of[first], local_of[second]),
                max(local_of[first], local_of[second]),
            ): order
            for (first, second), order in result._bond_orders.items()
            if first in local_of and second in local_of
        }
        snapshots.append(
            snapshot_molecule(
                component_top.molecules[0],
                name=component.molecule_name,
                bond_orders=local_orders,
            )
        )
    return tuple(snapshots)


def _component_top(
    result: SurgeryResult | AtomRemovalResult,
    atom_numbers: tuple[int, ...],
) -> "gml.Top":
    """Clone the surgery topology and retain one atom component."""
    component_top = result._top.clone()
    molecule = component_top.molecules[0]
    subset_molecule(molecule, atom_numbers)
    return component_top


def _safe_stem(prefix: str) -> str:
    """Return a conservative GROMACS file/molecule stem."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._")
    if not stem:
        raise ValueError("prefix must contain a letter or digit")
    return stem
