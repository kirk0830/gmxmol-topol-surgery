"""Minimal GROMACS ``.gro`` reader, subsetter, and writer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroAtom:
    """One fixed-column GROMACS coordinate atom record."""

    residue_number: int
    residue_name: str
    atom_name: str
    serial: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate fixed-column atom field domains."""
        if self.residue_number < 0:
            raise ValueError("GRO residue number must be non-negative")
        if self.serial < 1:
            raise ValueError("GRO atom serial must be positive")
        for label, value in (
            ("residue name", self.residue_name),
            ("atom name", self.atom_name),
        ):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"GRO {label} must be one non-empty token")
            if len(value) > 5:
                raise ValueError(f"GRO {label} cannot exceed 5 characters: {value!r}")
        if len(self.position) != 3 or not all(
            math.isfinite(value) for value in self.position
        ):
            raise ValueError("GRO positions must be three finite numbers")
        if self.velocity is not None and (
            len(self.velocity) != 3
            or not all(math.isfinite(value) for value in self.velocity)
        ):
            raise ValueError("GRO velocities must be three finite numbers")


@dataclass(frozen=True)
class GroFile:
    """A parsed GROMACS structure file."""

    title: str
    atoms: tuple[GroAtom, ...]
    box: tuple[float, ...]

    def __post_init__(self) -> None:
        """Freeze and validate a parsed structure."""
        atoms = tuple(self.atoms)
        box = tuple(float(value) for value in self.box)
        if not atoms:
            raise ValueError("GroFile requires at least one atom")
        if len(box) not in (3, 9):
            raise ValueError("GRO box must contain 3 or 9 values")
        if not all(math.isfinite(value) for value in box):
            raise ValueError("GRO box values must be finite")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "box", box)

    @classmethod
    def read(cls, path: str | Path) -> GroFile:
        """Read a GROMACS ``.gro`` file.

        Parameters
        ----------
        path : str | Path
            Coordinate file path.

        Returns
        -------
        GroFile
            Parsed title, atoms, and periodic box.

        Raises
        ------
        ValueError
            The atom count or a coordinate record is malformed.
        """
        source = Path(path)
        lines = source.read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            raise ValueError(f"{source}: a GRO file needs at least 3 lines")
        try:
            atom_count = int(lines[1].strip())
        except ValueError as exc:
            raise ValueError(
                f"{source}: second line must contain the atom count"
            ) from exc
        if len(lines) < atom_count + 3:
            raise ValueError(
                f"{source}: expected {atom_count} atoms and a box line, "
                f"found only {max(0, len(lines) - 2)} data lines"
            )
        atoms = tuple(
            _parse_atom_line(lines[index], source, atom_index)
            for atom_index, index in enumerate(range(2, atom_count + 2), 1)
        )
        try:
            box = tuple(float(value) for value in lines[atom_count + 2].split())
        except ValueError as exc:
            raise ValueError(f"{source}:{atom_count + 3}: malformed box line") from exc
        if len(box) not in (3, 9):
            raise ValueError(f"{source}:{atom_count + 3}: box must have 3 or 9 values")
        return cls(title=lines[0], atoms=atoms, box=box)

    def select(self, atom_numbers: Sequence[int]) -> GroFile:
        """Return a coordinate subset using the same 1-based numbering.

        Parameters
        ----------
        atom_numbers : Sequence[int]
            1-based source atom numbers in output order.

        Returns
        -------
        GroFile
            Selected coordinates with fresh 1-based serials.

        Raises
        ------
        IndexError
            An atom number is outside this structure.
        """
        invalid = [
            number for number in atom_numbers if not 1 <= number <= len(self.atoms)
        ]
        if invalid:
            raise IndexError(f"atom numbers outside 1..{len(self.atoms)}: {invalid}")
        if len(atom_numbers) != len(set(atom_numbers)):
            raise ValueError("atom_numbers contains duplicates")
        selected = tuple(self.atoms[number - 1] for number in atom_numbers)
        return GroFile(
            title=self.title,
            atoms=tuple(
                GroAtom(
                    residue_number=atom.residue_number,
                    residue_name=atom.residue_name,
                    atom_name=atom.atom_name,
                    serial=index,
                    position=atom.position,
                    velocity=atom.velocity,
                )
                for index, atom in enumerate(selected, start=1)
            ),
            box=self.box,
        )

    def write(
        self,
        path: str | Path,
        *,
        title: str | None = None,
    ) -> Path:
        """Write this structure to a GROMACS ``.gro`` file.

        Parameters
        ----------
        path : str | Path
            Destination path.
        title : str, optional
            Replacement first-line title.

        Returns
        -------
        Path
            The written path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [title if title is not None else self.title, str(len(self.atoms))]
        for index, atom in enumerate(self.atoms, start=1):
            x, y, z = atom.position
            line = (
                f"{atom.residue_number:5d}"
                f"{atom.residue_name:<5s}"
                f"{atom.atom_name:>5s}"
                f"{index:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
            )
            if atom.velocity is not None:
                vx, vy, vz = atom.velocity
                line += f"{vx:8.4f}{vy:8.4f}{vz:8.4f}"
            lines.append(line)
        lines.append("".join(f"{value:10.5f}" for value in self.box))
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination


def _parse_atom_line(line: str, path: Path, atom_index: int) -> GroAtom:
    """Parse one GRO atom line, preferring its fixed-column layout."""
    try:
        residue_number = int(line[0:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        serial = int(line[15:20])
        position = (
            float(line[20:28]),
            float(line[28:36]),
            float(line[36:44]),
        )
        velocity = None
        if len(line) >= 68 and line[44:68].strip():
            velocity = (
                float(line[44:52]),
                float(line[52:60]),
                float(line[60:68]),
            )
    except (ValueError, IndexError):
        fields = line.split()
        if len(fields) not in (7, 10):
            raise ValueError(
                f"{path}:{atom_index + 2}: malformed GRO atom record"
            ) from None
        residue_number = int(fields[0])
        residue_name = fields[1]
        atom_name = fields[2]
        serial = int(fields[3])
        values = tuple(float(value) for value in fields[4:])
        position = values[:3]
        velocity = values[3:6] if len(values) == 6 else None
    if not all(math.isfinite(value) for value in position):
        raise ValueError(f"{path}:{atom_index + 2}: non-finite atom coordinate")
    return GroAtom(
        residue_number=residue_number,
        residue_name=residue_name,
        atom_name=atom_name,
        serial=serial,
        position=position,
        velocity=velocity,
    )
