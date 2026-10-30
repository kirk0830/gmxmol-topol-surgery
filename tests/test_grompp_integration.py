"""Optional end-to-end validation with the GROMACS preprocessor."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from topolsurgery import AtomSpec, InteractionSpec, TopologySurgery

PARAMETER_SECTIONS = """\
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
"""


def _gmx_executable() -> str | None:
    """Return an explicitly requested or PATH-provided ``gmx`` executable."""
    requested = os.environ.get("TOPOLSURGERY_GMX")
    return requested or shutil.which("gmx")


def _assert_grompp(
    tmp_path: Path,
    gro_path: Path,
    topology_path: Path,
    label: str,
) -> None:
    """Preprocess one exported GRO/TOP pair and report full diagnostics."""
    executable = _gmx_executable()
    assert executable is not None
    mdp = Path(__file__).parent / "fixtures" / "minimal.mdp"
    completed = subprocess.run(
        (
            executable,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(gro_path),
            "-p",
            str(topology_path),
            "-o",
            str(tmp_path / f"{label}.tpr"),
            "-po",
            str(tmp_path / f"{label}.mdout.mdp"),
            "-maxwarn",
            "5",
        ),
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    _gmx_executable() is None,
    reason="gmx is not available; run in a GROMACS-enabled environment",
)
def test_exported_components_pass_grompp(tmp_path: Path) -> None:
    """Both component TOP/GRO pairs survive GROMACS preprocessing."""
    atoms = tuple(
        AtomSpec(index, "c3", f"C{index}", charge_group=index, mass=12.011)
        for index in range(1, 6)
    )
    interactions = (
        InteractionSpec("bonds", (1, 2), 1, (0.153, 251040.0)),
        InteractionSpec("bonds", (2, 3), 1, (0.153, 251040.0)),
        InteractionSpec("bonds", (3, 4), 1, (0.153, 251040.0)),
        InteractionSpec("bonds", (4, 5), 1, (0.153, 251040.0)),
        InteractionSpec("angles", (1, 2, 3), 1, (109.5, 460.24)),
        InteractionSpec("angles", (2, 3, 4), 1, (109.5, 460.24)),
        InteractionSpec("angles", (3, 4, 5), 1, (109.5, 460.24)),
        InteractionSpec("dihedrals", (1, 2, 3, 4), 9, (0.0, 0.0, 3)),
        InteractionSpec("dihedrals", (2, 3, 4, 5), 9, (0.0, 0.0, 3)),
        InteractionSpec("pairs", (1, 4), 1),
        InteractionSpec("pairs", (2, 5), 1),
    )
    surgery = TopologySurgery.from_records(
        atoms,
        interactions,
        name="CHAIN",
        coordinates=tuple((0.153 * (index - 1), 0.0, 0.0) for index in range(1, 6)),
        box=(3.0, 3.0, 3.0),
        parameter_sections=PARAMETER_SECTIONS,
    )
    exported = surgery.cut_bond(3, 4).export(
        tmp_path / "components",
        format="top",
        prefix="piece",
    )
    for component in exported.components:
        _assert_grompp(
            tmp_path,
            component.gro_path,
            component.topology_path,
            f"piece_{component.index}",
        )


@pytest.mark.skipif(
    _gmx_executable() is None,
    reason="gmx is not available; run in a GROMACS-enabled environment",
)
def test_reconnected_component_passes_grompp(tmp_path: Path) -> None:
    """A cut/reconnect cycle emits GROMACS-valid bond, angle, torsion, pairs."""
    atoms = tuple(
        AtomSpec(index, "c3", f"C{index}", charge_group=index, mass=12.011)
        for index in range(1, 5)
    )
    surgery = TopologySurgery.from_records(
        atoms,
        (
            InteractionSpec("bonds", (1, 2), 1),
            InteractionSpec("bonds", (2, 3), 1),
            InteractionSpec("bonds", (3, 4), 1),
        ),
        name="JOINED",
        coordinates=tuple((0.153 * (index - 1), 0.0, 0.0) for index in range(4)),
        box=(3.0, 3.0, 3.0),
        parameter_sections=PARAMETER_SECTIONS,
    )
    surgery.cut_bond(2, 3)
    result = surgery.connect_bond(
        2,
        3,
        bond_parameters=(0.153, 251040.0),
        update_pairs=True,
    )
    exported = result.export(
        tmp_path / "reconnected",
        format="top",
        prefix="joined",
    )

    component = exported.components[0]
    _assert_grompp(
        tmp_path,
        component.gro_path,
        component.topology_path,
        "joined",
    )


@pytest.mark.skipif(
    _gmx_executable() is None,
    reason="gmx is not available; run in a GROMACS-enabled environment",
)
def test_settle_water_component_passes_grompp(tmp_path: Path) -> None:
    """A SETTLE water component exports as one valid rigid molecule."""
    parameter_sections = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
OW OW 15.999 0.0 A 0.315061 0.636386
HW HW 1.008 0.0 A 0.0 0.0
c3 c3 12.011 0.0 A 0.339967 0.45773

[ bondtypes ]
c3 c3 1 0.1530 251040.0
"""
    atoms = (
        AtomSpec(1, "OW", "OW", 1, "SOL", 1, -0.834, 15.999),
        AtomSpec(2, "HW", "HW1", 1, "SOL", 1, 0.417, 1.008),
        AtomSpec(3, "HW", "HW2", 1, "SOL", 1, 0.417, 1.008),
        AtomSpec(4, "c3", "C1", 2, "M", 4, 0.0, 12.011),
        AtomSpec(5, "c3", "C2", 2, "M", 5, 0.0, 12.011),
    )
    surgery = TopologySurgery.from_records(
        atoms,
        (
            InteractionSpec("settles", (1,), 1, (0.09572, 0.15139)),
            InteractionSpec("bonds", (4, 5), 1, (0.153, 251040.0)),
        ),
        name="MIX",
        nrexcl=2,
        coordinates=(
            (0.0, 0.0, 0.0),
            (0.09572, 0.0, 0.0),
            (-0.024, 0.0927, 0.0),
            (1.0, 0.0, 0.0),
            (1.153, 0.0, 0.0),
        ),
        box=(3.0, 3.0, 3.0),
        parameter_sections=parameter_sections,
    )
    exported = surgery.cut_bond(4, 5).export(
        tmp_path / "water",
        format="top",
        prefix="piece",
    )
    water = next(item for item in exported.components if item.atom_numbers == (1, 2, 3))
    _assert_grompp(
        tmp_path,
        water.gro_path,
        water.topology_path,
        "water",
    )


@pytest.mark.skipif(
    _gmx_executable() is None,
    reason="gmx is not available; run in a GROMACS-enabled environment",
)
def test_tip4p_virtual_site_component_passes_grompp(tmp_path: Path) -> None:
    """Virtual-site ownership survives component extraction and grompp."""
    parameter_sections = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
OW OW 15.999 0.0 A 0.315061 0.636386
HW HW 1.008 0.0 A 0.0 0.0
MW MW 0.0 0.0 A 0.0 0.0
c3 c3 12.011 0.0 A 0.339967 0.45773

[ bondtypes ]
c3 c3 1 0.1530 251040.0
"""
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
        name="MIX",
        nrexcl=2,
        coordinates=(
            (0.0, 0.0, 0.0),
            (0.09572, 0.0, 0.0),
            (-0.024, 0.0927, 0.0),
            (0.01, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.153, 0.0, 0.0),
        ),
        box=(3.0, 3.0, 3.0),
        parameter_sections=parameter_sections,
    )
    exported = surgery.cut_bond(5, 6).export(
        tmp_path / "tip4p",
        format="top",
        prefix="piece",
    )
    water = next(
        item for item in exported.components if item.atom_numbers == (1, 2, 3, 4)
    )
    _assert_grompp(
        tmp_path,
        water.gro_path,
        water.topology_path,
        "tip4p",
    )
