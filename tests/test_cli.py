"""Command-line behavior, exit codes, and output protection."""

from __future__ import annotations

from pathlib import Path

from topolsurgery.cli import main


def _write_fixture(tmp_path: Path, *, two_fragments: bool = False) -> tuple[Path, Path]:
    """Write a three- or four-atom command-line fixture."""
    topology = tmp_path / "input.top"
    structure = tmp_path / "input.gro"
    if two_fragments:
        atoms = 4
        bonds = "1 2 1\n3 4 1"
    else:
        atoms = 3
        bonds = "1 2 1\n2 3 1"
    topology.write_text(
        "[ moleculetype ]\n"
        "MOL 3\n"
        "[ atoms ]\n"
        + "".join(
            f"{index} c3 1 M C{index} {index} 0.0 12.011\n"
            for index in range(1, atoms + 1)
        )
        + "[ bonds ]\n"
        + bonds
        + "\n[ system ]\nMOL\n[ molecules ]\nMOL 1\n",
        encoding="utf-8",
    )
    lines = ["MOL", str(atoms)]
    for index in range(1, atoms + 1):
        lines.append(
            f"{1:5d}{'M':<5s}{f'C{index}':>5s}{index:5d}"
            f"{0.15 * index:8.3f}{0.0:8.3f}{0.0:8.3f}"
        )
    lines.append(f"{3.0:10.5f}{3.0:10.5f}{3.0:10.5f}")
    structure.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return topology, structure


def test_cli_cut_success_and_overwrite_guard(tmp_path: Path, capsys) -> None:
    """A clean cut exits zero; a repeated write exits two."""
    topology, structure = _write_fixture(tmp_path)
    outdir = tmp_path / "cut"
    argv = [
        "cut",
        str(topology),
        str(structure),
        "--atom-i",
        "2",
        "--atom-j",
        "3",
        "--outdir",
        str(outdir),
        "--prefix",
        "part",
    ]
    assert main(argv) == 0
    assert len(list(outdir.glob("*.gro"))) == 2
    assert len(list(outdir.glob("*.itp"))) == 2
    assert main(argv) == 2
    assert "overwrite" in capsys.readouterr().err


def test_cli_blocks_unresolved_connect_unless_allowed(
    tmp_path: Path,
    capsys,
) -> None:
    """Blocking parameter warnings use a nonzero exit without exports."""
    topology, structure = _write_fixture(tmp_path, two_fragments=True)
    outdir = tmp_path / "joined"
    argv = [
        "connect",
        str(topology),
        str(structure),
        "--atom-i",
        "2",
        "--atom-j",
        "3",
        "--outdir",
        str(outdir),
        "--no-resolve-parameters",
        "--update-pairs",
    ]
    assert main(argv) == 2
    assert not outdir.exists()
    assert "unresolved" in capsys.readouterr().err

    allowed = [*argv, "--allow-unresolved"]
    assert main(allowed) == 0
    assert len(list(outdir.glob("*.gro"))) == 1
    assert len(list(outdir.glob("*.itp"))) == 1


def test_cli_reports_invalid_input_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    """Expected input failures are converted to a concise exit code."""
    topology, structure = _write_fixture(tmp_path)
    code = main(
        [
            "cut",
            str(topology),
            str(structure),
            "--atom-i",
            "1",
            "--atom-j",
            "3",
            "--outdir",
            str(tmp_path / "bad"),
        ]
    )
    assert code == 2
    assert "no [ bonds ]" in capsys.readouterr().err
