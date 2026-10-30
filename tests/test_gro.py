"""Tests for the local GROMACS coordinate reader/writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from topolsurgery.gro import GroFile


def test_gro_subset_renumbers_and_preserves_velocity(tmp_path: Path) -> None:
    """Selected coordinates keep fixed data and receive local serials."""
    source = tmp_path / "in.gro"
    source.write_text(
        "input\n"
        "2\n"
        f"{4:5d}{'WAT':<5s}{'OW':>5s}{8:5d}"
        f"{1.25:8.3f}{2.50:8.3f}{3.75:8.3f}"
        f"{0.1:8.4f}{0.2:8.4f}{0.3:8.4f}\n"
        f"{5:5d}{'SOL':<5s}{'HW1':>5s}{9:5d}"
        f"{2.25:8.3f}{3.50:8.3f}{4.75:8.3f}\n"
        f"{3.0:10.5f}{3.0:10.5f}{3.0:10.5f}\n",
        encoding="utf-8",
    )

    parsed = GroFile.read(source)
    subset = parsed.select((2,))
    output = subset.write(tmp_path / "out.gro")
    roundtrip = GroFile.read(output)

    assert parsed.atoms[0].velocity == pytest.approx((0.1, 0.2, 0.3))
    assert roundtrip.atoms[0].residue_number == 5
    assert roundtrip.atoms[0].residue_name == "SOL"
    assert roundtrip.atoms[0].atom_name == "HW1"
    assert roundtrip.atoms[0].serial == 1
    assert roundtrip.atoms[0].position == pytest.approx((2.25, 3.50, 4.75))
    assert roundtrip.atoms[0].velocity is None
